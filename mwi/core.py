"""
Core functions
"""
import os
import asyncio
import json
import re
import signal
import sys
import time
import shutil
import zipfile
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import date, datetime, timedelta
from os import path
from typing import Callable, Dict, List, Optional, Union
from urllib.parse import urlparse, urljoin, quote, unquote

import aiohttp # type: ignore
import nltk # type: ignore
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize # type: ignore
from peewee import IntegrityError, JOIN
import trafilatura # type: ignore
try:
    from playwright.async_api import async_playwright  # noqa: F401  (availability probe)
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: Playwright not available. Dynamic media extraction will be skipped.")

import settings
from . import link_context
from . import model
from .export import Export
from .platform_heuristics import PLATFORM_HEURISTICS as _PLATFORM_HEURISTICS


# Circuit breaker and fetch pipeline live in mwi.fetcher; re-exported here
# for backwards-compat (tests reach into core._ArchiveOrgBreaker).
from mwi.fetcher import _ArchiveOrgBreaker, fetch_html, FetchResult  # noqa: E402,F401

# SerpAPI routing lives in mwi.serpapi_router; re-exported here for backwards-compat
# (controller and tests still reach core.SerpApiError / core.fetch_serpapi_url_list).
# Note: the new multi-API search router lives under mwi/search/ (a separate package).
from mwi.serpapi_router import SearchError, SearchRequest, run_search as _run_search  # noqa: E402

# Legacy alias preserved so `except core.SerpApiError` keeps working.
SerpApiError = SearchError


def _maybe_truncate_html(raw_html: str) -> str:
    """Truncate raw HTML to settings.fullhtml_max_size_kb if needed.

    Sprint-html E, hardened by sprint-multilang D-3: the cap is enforced
    in UTF-8 **bytes**, not characters — a multi-byte page can exceed the
    byte cap while staying under it in characters. Returns the original
    string when the cap is 0, missing or the body fits within the limit.
    Logs a one-line warning so operators can audit which URLs hit the cap.
    """
    cap_kb = getattr(settings, 'fullhtml_max_size_kb', 0) or 0
    if cap_kb <= 0 or not raw_html:
        return raw_html
    cap_bytes = cap_kb * 1024
    # UTF-8 encodes a character in at most 4 bytes: below cap/4 characters
    # the page necessarily fits — skip the encode cost for most pages.
    if len(raw_html) * 4 <= cap_bytes:
        return raw_html
    encoded = raw_html.encode('utf-8')
    if len(encoded) <= cap_bytes:
        return raw_html
    print(f"[fullhtml] Truncating HTML "
          f"({len(encoded) / 1024:.1f} KB > {cap_kb} KB cap)")
    return encoded[:cap_bytes].decode('utf-8', errors='ignore')


def _cleanup_nltk_resource(resource: str) -> bool:
    """Remove corrupted NLTK resource artifacts so they can be re-downloaded."""
    removed = False
    # Look for both directory and zip variants in every configured nltk data path
    candidates = []
    for base in list(nltk.data.path):
        tokenizers_dir = path.join(base, 'tokenizers')
        candidates.extend([
            path.join(tokenizers_dir, resource),
            path.join(tokenizers_dir, f"{resource}.zip"),
            path.join(tokenizers_dir, f"{resource}.pickle"),
        ])
    for candidate in candidates:
        if path.exists(candidate):
            try:
                if path.isdir(candidate):
                    shutil.rmtree(candidate)
                else:
                    os.remove(candidate)
                removed = True
            except Exception:
                pass
    return removed

def _ensure_nltk_tokenizers() -> bool:
    """Ensure required NLTK tokenizers are available with resilient SSL and local cache.

    This function attempts to download and configure NLTK tokenizers ('punkt' and
    'punkt_tab') needed for text processing. It configures SSL certificates for
    platforms with certificate store issues and caches data in the project folder.

    Returns:
        bool: True if tokenizers are found or successfully downloaded, False otherwise.

    Notes:
        - Caches NLTK data in the project's data_location/nltk_data directory.
        - Configures SSL using certifi to handle certificate issues on Windows/macOS.
        - Silently handles errors to allow fallback tokenizers to be used.
    """
    # Prefer caching into the project data folder
    try:
        nltk_dir = path.join(getattr(settings, 'data_location', 'data'), 'nltk_data')
        os.makedirs(nltk_dir, exist_ok=True)
        if nltk_dir not in nltk.data.path:
            nltk.data.path.append(nltk_dir)
    except Exception:
        pass

    # Harden SSL on platforms with broken cert stores (notably some Windows/macOS setups)
    try:
        import ssl  # type: ignore
        import certifi  # type: ignore
        os.environ.setdefault('SSL_CERT_FILE', certifi.where())
        # Ensure urllib uses certifi's CA bundle
        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())  # type: ignore[attr-defined]
    except Exception:
        # If certifi/ssl tweak fails, we still attempt standard downloads
        pass

    ok = True
    for resource in ('punkt', 'punkt_tab'):
        try:
            nltk.data.find(f'tokenizers/{resource}')
            continue
        except Exception as exc:
            # If the resource exists but is corrupted, purge artifacts before re-downloading
            if isinstance(exc, zipfile.BadZipFile):
                _cleanup_nltk_resource(resource)
            try:
                nltk.download(resource, quiet=True)
                nltk.data.find(f'tokenizers/{resource}')
            except Exception:
                ok = False
    return ok

_NLTK_OK = _ensure_nltk_tokenizers()
if not _NLTK_OK:
    print("Warning: NLTK 'punkt'/'punkt_tab' not available; using a simple tokenizer fallback.")

def _simple_word_tokenize(text: str) -> List[str]:
    """Provide a simple unicode-aware fallback tokenizer.

    This lightweight tokenizer extracts letter sequences from the input text
    in any script (Latin, Cyrillic, Arabic, Greek, CJK, ...) and returns them
    as lowercase tokens.

    Args:
        text: The text to tokenize. Non-string inputs are converted to strings.

    Returns:
        list: A list of lowercase word tokens containing only letter sequences.

    Notes:
        - Used when NLTK's punkt tokenizer is unavailable, or for languages
          without a punkt model (see `_PUNKT_LANGS`).
        - `[^\\W\\d_]` matches unicode letters only (no digits, punctuation
          or underscore).
    """
    if not isinstance(text, str):
        text = str(text)
    # Keep only letter sequences, all unicode scripts included
    return re.findall(r"[^\W\d_]+", text.lower(), re.UNICODE)


async def extract_dynamic_medias(url: str, expression: model.Expression) -> list:
    """Extract media URLs from a webpage using a headless browser.

    This function uses Playwright to execute JavaScript on a webpage and capture
    dynamically generated media URLs including images, videos, and audio files.
    It also handles lazy-loaded images using common data attributes.

    Args:
        url: The URL of the webpage to extract media from.
        expression: The Expression database object to associate extracted media with.

    Returns:
        list: A list of media URLs (strings) found after JavaScript execution.

    Notes:
        - Requires Playwright to be installed and available.
        - Returns empty list if Playwright is not available.
        - Waits for network idle and additional 3 seconds for dynamic content.
        - Detects lazy-loaded images via data-src, data-lazy-src, etc.
        - Automatically saves new media to the database.
        - Resolves relative URLs to absolute URLs.
    """
    if not PLAYWRIGHT_AVAILABLE:
        print(f"Playwright not available, skipping dynamic media extraction for {url}")
        return []

    dynamic_medias = []

    try:
        # Borrow a page from the shared BrowserPool (sprint-403 Sprint 3b).
        # The pool launches Chromium once per process and lends fresh
        # contexts; user-agent is injected at context creation.
        from mwi.browser_pool import BrowserPool
        async with BrowserPool.get().page() as page:
            # Navigate to the page. ``domcontentloaded`` returns as soon
            # as the DOM is parsed (typically 1-2s) — much more reliable
            # than ``networkidle`` on sites with persistent trackers /
            # ads / polling JS, which would otherwise time out after 30s.
            await page.goto(url, wait_until='domcontentloaded', timeout=30000)

            # Best-effort wait for the network to settle so lazy-loaded
            # media has a chance to appear. Bounded to 5s; a timeout here
            # is non-fatal — we still extract whatever rendered.
            try:
                await page.wait_for_load_state('networkidle', timeout=5000)
            except Exception:
                pass

            # Small extra delay for late JS rendering.
            await page.wait_for_timeout(2000)

            # Extract media elements after JavaScript execution
            media_selectors = {
                'img': 'img[src]',
                'video': 'video[src], video source[src]',
                'audio': 'audio[src], audio source[src]'
            }

            for media_type, selector in media_selectors.items():
                elements = await page.query_selector_all(selector)

                for element in elements:
                    src = await element.get_attribute('src')
                    if src:
                        # Resolve relative URLs to absolute
                        resolved_url = resolve_url(url, src)

                        # Check if this is a valid media type
                        if media_type == 'img':
                            IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg")
                            if resolved_url.lower().endswith(IMAGE_EXTENSIONS):
                                dynamic_medias.append({
                                    'url': resolved_url,
                                    'type': media_type
                                })
                        elif media_type in ['video', 'audio']:
                            dynamic_medias.append({
                                'url': resolved_url,
                                'type': media_type
                            })

            # Look for lazy-loaded images and other dynamic content
            # Check for data-src, data-lazy-src, and other common lazy loading attributes
            lazy_img_selectors = [
                'img[data-src]',
                'img[data-lazy-src]',
                'img[data-original]',
                'img[data-url]'
            ]

            for selector in lazy_img_selectors:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    for attr in ['data-src', 'data-lazy-src', 'data-original', 'data-url']:
                        src = await element.get_attribute(attr)
                        if src:
                            resolved_url = resolve_url(url, src)
                            IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg")
                            if resolved_url.lower().endswith(IMAGE_EXTENSIONS):
                                dynamic_medias.append({
                                    'url': resolved_url,
                                    'type': 'img'
                                })
                            break  # Stop at first found attribute

        print(f"Dynamic media extraction found {len(dynamic_medias)} media items for {url}")
        
        # Save found media to database
        for media_info in dynamic_medias:
            # Check if media doesn't already exist in database
            if not model.Media.select().where(
                (model.Media.expression == expression) & 
                (model.Media.url == media_info['url'])
            ).exists():
                media = model.Media.create(
                    expression=expression, 
                    url=media_info['url'], 
                    type=media_info['type']
                )
                media.save()
        
        return [media['url'] for media in dynamic_medias]
        
    except Exception as e:
        print(f"Error during dynamic media extraction for {url}: {e}")
        return []


def resolve_url(base_url: str, relative_url: str) -> str:
    """Resolve a relative URL to an absolute URL using the base URL.

    This function converts relative URLs to absolute URLs using urllib's urljoin.
    URLs are normalized to lowercase for consistency.

    Args:
        base_url: The base URL (typically the page URL containing the link).
        relative_url: The relative or absolute URL to resolve.

    Returns:
        str: The resolved absolute URL in lowercase.

    Notes:
        - Already absolute URLs (starting with http:// or https://) are returned as-is.
        - All returned URLs are converted to lowercase for consistency.
        - Errors during resolution are logged and the relative URL is returned.
    """
    try:
        # If already absolute, return as is (but lowercase for consistency)
        if relative_url.startswith(('http://', 'https://')):
            return relative_url.lower()
        
        # Use urljoin to properly resolve relative URLs
        resolved_url = urljoin(base_url, relative_url)
        return resolved_url.lower()
    except Exception as e:
        print(f"Error resolving URL '{relative_url}' with base '{base_url}': {e}")
        return relative_url.lower()


def confirm(message: str) -> bool:
    """Confirm an action by requesting user input.

    This function displays a message to the user and expects 'Y' as confirmation.

    Args:
        message: The confirmation message to display to the user.

    Returns:
        bool: True if the user enters 'Y', False otherwise.

    Notes:
        - Only the exact character 'Y' (uppercase) confirms the action.
        - Any other input (including 'y', 'yes', etc.) returns False.
    """
    return input(message) == 'Y'


def check_args(args: Namespace, mandatory) -> bool:
    """Validate that all required arguments are present in parsed input arguments.

    This function checks whether mandatory arguments are present and non-None in
    the provided Namespace object from argparse.

    Args:
        args: The argparse Namespace object containing parsed arguments.
        mandatory: A string or list of strings representing required argument names.

    Returns:
        bool: True if all mandatory arguments are present and non-None.

    Raises:
        ValueError: If any mandatory argument is missing or None.

    Notes:
        - Accepts either a single string or list of strings for mandatory args.
        - Converts args Namespace to dictionary for validation.
    """
    args_dict = vars(args)
    if isinstance(mandatory, str):
        mandatory = [mandatory]
    for arg in mandatory:
        if arg not in args_dict or args_dict[arg] is None:
            raise ValueError('Argument "%s" is required' % arg)
    return True


def split_arg(arg: str) -> list:
    """Split an argument string using comma separator and return a filtered list.

    This function splits a comma-separated string into individual arguments,
    stripping whitespace and filtering out empty strings.

    Args:
        arg: A comma-separated string of arguments.

    Returns:
        list: A list of non-empty strings with leading/trailing whitespace removed.

    Notes:
        - Empty strings after splitting are filtered out.
        - Each element is stripped of leading and trailing whitespace.
    """
    args = arg.split(",")
    return [a.strip() for a in args if a]


def get_dryrun(args: Namespace) -> bool:
    """Read the dry-run flag whichever spelling was used.

    Two flags coexist in the CLI (sprint-multilang, D-5): `--dryrun`
    (store_true, dest `dryrun`) and `--dry-run` (str 'TRUE', dest
    `dry_run`). This helper accepts both so every command honours
    either spelling.

    Returns:
        bool: True when a dry run was requested.
    """
    raw = getattr(args, 'dryrun', None)
    if raw is None or raw is False:
        raw = getattr(args, 'dry_run', None)
    if isinstance(raw, str):
        return raw.strip().upper() == 'TRUE'
    return bool(raw)


def get_arg_option(name: str, args: Namespace, set_type, default):
    """Retrieve and type-cast an optional argument value with a default fallback.

    This function extracts an optional argument from the parsed args, applies a
    type conversion function, and returns a default value if the argument is missing.

    Args:
        name: The name of the argument to retrieve.
        args: The argparse Namespace object containing parsed arguments.
        set_type: A callable to convert the argument value to the desired type.
        default: The default value to return if the argument is missing or None.

    Returns:
        The argument value converted using set_type, or the default value.

    Notes:
        - Returns default if argument is not present or is None.
        - set_type can be any callable (int, str, bool, custom function, etc.).
    """
    args_dict = vars(args)
    if (name in args_dict) and (args_dict[name] is not None):
        return set_type(args_dict[name])
    return default


def fetch_serpapi_url_list(
    api_key: str,
    query: str,
    engine: str = 'google',
    lang: str = 'fr',
    gl: Optional[str] = None,
    datestart: Optional[str] = None,
    dateend: Optional[str] = None,
    timestep: str = 'week',
    sleep_seconds: float = 1.0,
    progress_hook: Optional[Callable[[Optional[date], Optional[date], int], None]] = None,
    window_results_hook: Optional[Callable[[Optional[date], Optional[date], list], None]] = None
) -> List[Dict[str, Optional[Union[str, int]]]]:
    """Backwards-compatible wrapper around `mwi.serpapi_router.run_search`.

    Engine-specific behaviour, pagination, date windowing and HTTP plumbing now
    live in `mwi.serpapi_router`. This wrapper preserves the original signature
    so the controller and the legacy CLI tests (which patch
    `controller.core.fetch_serpapi_url_list`) keep working unchanged.

    Raises:
        SerpApiError: alias of `mwi.serpapi_router.SearchError`.
    """
    return _run_search(SearchRequest(
        api_key=api_key,
        query=query,
        engine=engine,
        lang=lang,
        gl=gl,
        datestart=datestart,
        dateend=dateend,
        timestep=timestep,
        sleep_seconds=sleep_seconds,
        progress_hook=progress_hook,
        window_results_hook=window_results_hook,
    ))


def parse_serp_result_date(value: Optional[str]) -> Optional[datetime]:
    """Parse the ``date`` field returned by SerpAPI organic results.

    The payload is inconsistent (absolute dates like ``Apr 2, 2024`` or
    relative strings such as ``2 days ago``). We normalise the string and try a
    series of lightweight parsing strategies. When parsing fails we return
    ``None`` so callers can skip the update gracefully.
    """

    if not value:
        return None

    normalized = value.strip()
    if not normalized:
        return None

    # Remove common prefixes and punctuation quirks.
    normalized = re.sub(r'(?i)^(updated|publié[e]?)[:\s-]+', '', normalized)
    normalized = normalized.replace('·', ' ')
    normalized = normalized.replace('\u2013', ' ').replace('\u2014', ' ')
    normalized = re.sub(r'\s+', ' ', normalized).strip(' .-')
    normalized = re.sub(r'(?i)\b([A-Za-z]{3,9})\.', r'\1', normalized)
    normalized = re.sub(r'(?<=\d)(st|nd|rd|th)', '', normalized, flags=re.IGNORECASE)

    # Try ISO formats first (SerpAPI sometimes emits them directly).
    iso_candidate = normalized.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(iso_candidate)
    except ValueError:
        pass

    for fmt in (
        '%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d',
        '%m/%d/%Y', '%d/%m/%Y', '%d.%m.%Y', '%m.%d.%Y',
        '%b %d, %Y', '%B %d, %Y', '%d %b %Y', '%d %B %Y'
    ):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue

    lowered = normalized.lower()
    if lowered in {'today'}:
        return datetime.now()
    if lowered in {'yesterday'}:
        return datetime.now() - timedelta(days=1)

    relative_match = re.match(r'(?i)^(?:about\s+)?(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago$', normalized)
    if relative_match:
        amount = int(relative_match.group(1))
        unit = relative_match.group(2).lower()

        delta_kwargs = {
            'minute': {'minutes': amount},
            'hour': {'hours': amount},
            'day': {'days': amount},
            'week': {'weeks': amount},
            'month': {'days': amount * 30},
            'year': {'days': amount * 365},
        }.get(unit)

        if delta_kwargs:
            return datetime.now() - timedelta(**delta_kwargs)

    return None


def prefer_earlier_datetime(
    current_value: Optional[datetime],
    candidate: Optional[datetime]
) -> Optional[datetime]:
    """Return the earliest non-null datetime between current and candidate values.

    This utility function compares two optional datetime values and returns the
    earlier one, handling None values gracefully.

    Args:
        current_value: The current datetime value, or None.
        candidate: The candidate datetime value to compare, or None.

    Returns:
        Optional[datetime]: The earlier of the two datetimes, or the non-None value
            if only one is provided, or None if both are None.

    Notes:
        - If both values are None, returns None.
        - If one value is None, returns the other value.
        - If both values are non-None, returns the earlier datetime.
    """

    if candidate is None:
        return current_value
    if current_value is None:
        return candidate
    return candidate if candidate < current_value else current_value


def fetch_seorank_for_url(url: str, api_key: str) -> Optional[dict]:
    """Call the SEO Rank API for a single URL with retry logic and return the JSON payload.

    This function makes an HTTP request to the SEO Rank API to fetch metrics
    (Moz, SimilarWeb, Facebook) for a given URL. Implements exponential backoff
    retry logic for transient failures.

    Args:
        url: The URL to fetch SEO metrics for.
        api_key: The API key for authenticating with the SEO Rank service.

    Returns:
        Optional[dict]: A dictionary containing the JSON response from the API,
            or None if all retry attempts fail.

    Notes:
        - Automatically unwraps archive.org URLs to get the original URL for ranking.
        - Uses URL-safe encoding for the URL parameter.
        - Default base URL is 'https://seo-rank.my-addr.com/api2/moz+sr+fb'.
        - Configurable via settings.seorank_api_base_url.
        - Timeout is configurable via settings.seorank_timeout (default 30s).
        - Implements retry with exponential backoff (configurable via settings).
        - Max retries: settings.seorank_max_retries (default 3).
        - Backoff multiplier: settings.seorank_retry_backoff (default 2.0).
        - Prints detailed error messages including retry attempts.
    """
    base_url = getattr(settings, 'seorank_api_base_url', '').strip()
    if not base_url:
        base_url = 'https://seo-rank.my-addr.com/api2/moz+sr+fb'

    timeout = getattr(settings, 'seorank_timeout', 30)
    max_retries = getattr(settings, 'seorank_max_retries', 3)
    backoff_multiplier = getattr(settings, 'seorank_retry_backoff', 2.0)

    # Unwrap archive.org URLs to get the original URL that Google indexes
    original_url = unwrap_archive_url(url)

    # API expects the raw URL path; keep common URL separators unescaped
    safe_url = quote(original_url, safe=':/?&=%')
    request_url = f"{base_url.rstrip('/')}/{api_key}/{safe_url}"

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(request_url, timeout=timeout)

            if response.status_code != 200:
                if attempt < max_retries:
                    print(f"[seorank] Status {response.status_code} for {url} (attempt {attempt}/{max_retries})")
                else:
                    print(f"[seorank] Failed with status {response.status_code} for {url} after {max_retries} attempts")
                    return None
            else:
                # Success - parse JSON
                try:
                    return response.json()
                except ValueError as exc:
                    snippet = response.text[:120].replace('\n', ' ')
                    print(f"[seorank] JSON decoding failed for {url}: {exc} (body preview: {snippet})")
                    return None

        except requests.Timeout as exc:
            if attempt < max_retries:
                wait_time = backoff_multiplier ** (attempt - 1)
                print(f"[seorank] Timeout for {url} (attempt {attempt}/{max_retries}), retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"[seorank] Timeout for {url} after {max_retries} attempts: {exc}")
                return None

        except requests.RequestException as exc:
            if attempt < max_retries:
                wait_time = backoff_multiplier ** (attempt - 1)
                print(f"[seorank] HTTP error for {url} (attempt {attempt}/{max_retries}): {exc}, retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"[seorank] HTTP error for {url} after {max_retries} attempts: {exc}")
                return None

    return None


def update_seorank_for_land(
    land: model.Land,
    api_key: str,
    limit: int = 0,
    depth: Optional[int] = None,
    http_status: Optional[str] = '200',
    min_relevance: int = 1,
    force_refresh: bool = False,
) -> tuple[int, int]:
    """Fetch SEO Rank data for land expressions and store the raw JSON payload.

    This function processes expressions in a land, fetches SEO metrics from the
    SEO Rank API, and stores the results in the database. Provides detailed
    progress tracking and statistics.

    Args:
        land: The Land object containing expressions to process.
        api_key: The API key for the SEO Rank service.
        limit: Maximum number of expressions to process (0 for unlimited).
        depth: Optional crawl depth filter (None for all depths).
        http_status: Filter by HTTP status code ('200', 'all', or specific code).
        min_relevance: Minimum relevance score filter (default 1).
        force_refresh: Whether to refresh existing seorank data (default False).

    Returns:
        tuple[int, int]: A tuple of (processed_count, updated_count) indicating
            how many expressions were processed and how many were successfully updated.

    Notes:
        - By default, only processes expressions without existing seorank data.
        - Use force_refresh=True to re-fetch data for all expressions.
        - Adds configurable delay between requests (settings.seorank_request_delay).
        - Filters by HTTP status (defaults to '200' if not specified).
        - Skips expressions with None relevance if min_relevance > 0.
        - Prints progress updates every 100 URLs with time estimates.
        - Displays detailed statistics upon completion.
    """
    expressions = model.Expression.select().where(model.Expression.land == land)

    if depth is not None:
        expressions = expressions.where(model.Expression.depth == depth)

    # HTTP filter: default to 200 unless user explicitly requests otherwise.
    if http_status:
        http_status_normalized = http_status.strip().lower()
        if http_status_normalized not in ('all', 'any', 'none', ''):
            expressions = expressions.where(model.Expression.http_status == http_status.strip())

    if min_relevance is not None and min_relevance > 0:
        expressions = expressions.where(model.Expression.relevance >= min_relevance)

    if not force_refresh:
        expressions = expressions.where(model.Expression.seorank.is_null(True))

    expressions = expressions.order_by(model.Expression.id)
    if limit and limit > 0:
        expressions = expressions.limit(limit)

    # Get total count for progress tracking
    total_count = expressions.count()
    print(f"[seorank] Starting SEO Rank enrichment for land '{land.name}'")
    print(f"[seorank] Total URLs to process: {total_count}")

    if total_count == 0:
        print("[seorank] No expressions to process.")
        return 0, 0

    processed = 0
    updated = 0
    failed = 0
    sleep_seconds = max(0.0, float(getattr(settings, 'seorank_request_delay', 0)))

    start_time = time.time()

    for expression in expressions:
        processed += 1
        payload = fetch_seorank_for_url(str(expression.url), api_key)
        if payload is not None:
            expression.seorank = json.dumps(payload)
            expression.save(only=[model.Expression.seorank])
            updated += 1
        else:
            failed += 1

        # Progress update every 100 URLs
        if processed % 100 == 0 or processed == total_count:
            elapsed = time.time() - start_time
            avg_time_per_url = elapsed / processed
            remaining_urls = total_count - processed
            eta_seconds = remaining_urls * avg_time_per_url

            # Format ETA
            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds/60:.0f}m"
            else:
                eta_str = f"{eta_seconds/3600:.1f}h"

            success_rate = (updated / processed * 100) if processed > 0 else 0
            print(f"[seorank] Progress: {processed}/{total_count} ({processed/total_count*100:.1f}%) | "
                  f"Success: {updated} | Failed: {failed} | Success rate: {success_rate:.1f}% | "
                  f"ETA: {eta_str}")

        if sleep_seconds:
            time.sleep(sleep_seconds)

    # Final statistics
    total_time = time.time() - start_time
    success_rate = (updated / processed * 100) if processed > 0 else 0

    print(f"\n[seorank] ===== SEO Rank Enrichment Complete =====")
    print(f"[seorank] Land: {land.name}")
    print(f"[seorank] Total processed: {processed}")
    print(f"[seorank] Successful: {updated} ({success_rate:.1f}%)")
    print(f"[seorank] Failed: {failed} ({100-success_rate:.1f}%)")
    print(f"[seorank] Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"[seorank] Average time per URL: {total_time/processed:.2f}s")
    print(f"[seorank] ==========================================\n")

    return processed, updated


# ISO 639-1 -> NLTK Snowball stemmer names (sprint-multilang, D1)
_SNOWBALL_LANGS = {
    'ar': 'arabic',  'da': 'danish',   'de': 'german',    'en': 'english',
    'es': 'spanish', 'fi': 'finnish',  'fr': 'french',    'hu': 'hungarian',
    'it': 'italian', 'nl': 'dutch',    'no': 'norwegian', 'pt': 'portuguese',
    'ro': 'romanian', 'ru': 'russian', 'sv': 'swedish',
}
_stemmers: dict = {}

# ISO 639-1 -> punkt model names shipped in nltk_data/tokenizers/punkt_tab
# (sprint-multilang, D2). Languages without a punkt model (ar, hu, ro, ...)
# fall back to _simple_word_tokenize, which is unicode-aware.
_PUNKT_LANGS = {
    'cs': 'czech', 'da': 'danish', 'de': 'german', 'el': 'greek',
    'en': 'english', 'es': 'spanish', 'et': 'estonian', 'fi': 'finnish',
    'fr': 'french', 'it': 'italian', 'ml': 'malayalam', 'nl': 'dutch',
    'no': 'norwegian', 'pl': 'polish', 'pt': 'portuguese', 'ru': 'russian',
    'sl': 'slovene', 'sv': 'swedish', 'tr': 'turkish',
}


def stem_word(word: str, lang: str = 'fr') -> str:
    """Stem a word with the NLTK Snowball stemmer for the given language.

    Args:
        word: The word to stem.
        lang: ISO 639-1 language code (e.g. 'fr', 'en', 'de'). Regional
            variants ('en-US') are reduced to their base code. Defaults
            to 'fr' to preserve the historical French-only behaviour.

    Returns:
        str: The stemmed word in lowercase. For languages without a
            Snowball stemmer, returns the lowercased word unchanged
            (better no stemming than wrong-language stemming).

    Notes:
        - Stemmer instances are cached per language in `_stemmers`.
        - SnowballStemmer is imported lazily to preserve the `_NLTK_OK`
          fallback path when NLTK data is unavailable.
    """
    name = _SNOWBALL_LANGS.get((lang or 'fr').split('-')[0].strip().lower())
    if name is None:
        return word.lower()
    if name not in _stemmers:
        from nltk.stem.snowball import SnowballStemmer  # type: ignore
        _stemmers[name] = SnowballStemmer(name)
    return str(_stemmers[name].stem(word.lower()))


def is_language_compatible(expression_lang: str, land_langs: str) -> bool:
    """Check if an expression's language is compatible with the land's accepted languages.

    This function parses comma-separated language lists and performs flexible language
    code matching. It handles language variants (e.g., 'fr' matches 'fr-FR') and
    returns True if at least one language from the expression is compatible with the
    land's accepted languages.

    Args:
        expression_lang: The language code from the expression (e.g., 'fr', 'en-US').
            May be empty or None.
        land_langs: Comma-separated list of accepted language codes for the land
            (e.g., 'fr,en,de'). May be empty or None.

    Returns:
        bool: True if the expression language is compatible with at least one land
            language, or if expression_lang is empty/None (to avoid blocking pages
            without lang attribute). False if there's a language mismatch.

    Examples:
        >>> is_language_compatible('fr', 'fr,en')
        True
        >>> is_language_compatible('fr-FR', 'fr')
        True
        >>> is_language_compatible('de', 'fr,en')
        False
        >>> is_language_compatible('', 'fr')
        True  # Empty lang doesn't block
        >>> is_language_compatible('en-US', 'en,fr')
        True

    Notes:
        - Language codes are compared case-insensitively.
        - Supports both exact matches ('fr' == 'fr') and variant matches ('fr-FR' matches 'fr').
        - If expression_lang is empty or None, returns True to avoid blocking pages
          that don't have a lang attribute in their HTML.
        - Whitespace in language codes is automatically stripped.
    """
    # If expression has no language info, don't block it
    # (some pages may not have lang attribute in HTML)
    if not expression_lang or not expression_lang.strip():
        return True

    # If land has no language restriction, accept all
    if not land_langs or not land_langs.strip():
        return True

    # Parse language lists (comma-separated)
    expr_langs = [lang.strip().lower() for lang in expression_lang.split(',') if lang.strip()]
    land_lang_list = [lang.strip().lower() for lang in land_langs.split(',') if lang.strip()]

    # Check if any expression language is compatible with any land language
    for expr_lang in expr_langs:
        for land_lang in land_lang_list:
            # Exact match
            if expr_lang == land_lang:
                return True
            # Variant match: 'fr-FR' matches 'fr', 'en-US' matches 'en'
            expr_base = expr_lang.split('-')[0]
            land_base = land_lang.split('-')[0]
            if expr_base == land_base:
                return True

    return False


def detect_content_language(text: str, html_lang: str = '') -> str:
    """Detect language from extracted text content, with HTML lang as fallback.

    Uses langdetect on the actual text content rather than relying on the
    <html lang> attribute, which can be wrong (e.g., Twitter/X always
    returns lang='en' regardless of tweet language).

    Args:
        text: The extracted readable text content.
        html_lang: The lang attribute from the HTML tag (fallback).

    Returns:
        str: Detected language code (e.g., 'fr', 'en'), or html_lang if
            detection fails, or empty string if nothing is available.
    """
    if not text or len(text.strip()) < 20:
        return html_lang

    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0  # reproducible results
        detected = detect(text)
        return detected
    except Exception:
        return html_lang


class TimeoutException(Exception):
    """Custom exception for timeout handling."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for SIGALRM timeout."""
    raise TimeoutException("Operation timed out")


def _fetch_url_with_retry_and_timeout(url: str, timeout: int = 10, max_retries: int = 3, backoff_delays: List[int] = None) -> Optional[str]:
    """Fetch URL with timeout and retry logic using exponential backoff.

    Uses signal.alarm() on Unix/macOS for proper timeout enforcement, and
    ThreadPoolExecutor as fallback on Windows.

    Args:
        url: The URL to fetch
        timeout: Timeout in seconds for each fetch attempt
        max_retries: Maximum number of retry attempts (default: 3)
        backoff_delays: List of delays between retries in seconds (default: [1, 2, 5])

    Returns:
        Optional[str]: The downloaded HTML content, or None if all attempts failed
    """
    if backoff_delays is None:
        backoff_delays = [1, 2, 5]

    # Check if signal.alarm is available (Unix/macOS)
    use_signal = hasattr(signal, 'alarm') and sys.platform != 'win32'

    for attempt in range(max_retries):
        try:
            if use_signal:
                # Unix/macOS: Use signal.alarm for reliable timeout
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout)
                try:
                    result = trafilatura.fetch_url(url)
                    signal.alarm(0)  # Cancel the alarm
                    if result:
                        return result
                finally:
                    signal.alarm(0)  # Ensure alarm is cancelled
                    signal.signal(signal.SIGALRM, old_handler)  # Restore previous handler
            else:
                # Windows: Use ThreadPoolExecutor as fallback
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(trafilatura.fetch_url, url)
                    result = future.result(timeout=timeout)
                    if result:
                        return result

        except TimeoutException:
            print(f"  Attempt {attempt + 1}/{max_retries} timed out after {timeout}s for {url}", flush=True)
        except FuturesTimeoutError:
            print(f"  Attempt {attempt + 1}/{max_retries} timed out after {timeout}s for {url}", flush=True)
        except Exception as e:
            print(f"  Attempt {attempt + 1}/{max_retries} failed for {url}: {e}", flush=True)

        # Apply backoff delay before next retry (but not after the last attempt)
        if attempt < max_retries - 1:
            delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
            print(f"  Waiting {delay}s before retry...", flush=True)
            time.sleep(delay)

    return None


def crawl_domains(limit: int = 0, http: Optional[str] = None):
    """Crawl domains to retrieve metadata using a multi-strategy pipeline.

    This function processes domains through a three-stage pipeline: Trafilatura,
    Archive.org, and direct HTTP requests. It extracts title, description, and
    keywords from each domain's homepage.

    Args:
        limit: Maximum number of domains to process (0 for unlimited).
        http: Optional HTTP status code filter for recrawling specific domains.

    Returns:
        int: The number of successfully processed domains.

    Notes:
        - Pipeline order: (1) Trafilatura, (2) Archive.org, (3) Direct requests.
        - Trafilatura tries HTTPS then HTTP automatically.
        - Archive.org provides historical snapshots if live site is unavailable.
        - Direct requests attempt both HTTPS and HTTP protocols.
        - Extracts metadata using BeautifulSoup and Trafilatura.
        - Updates domain.fetched_at, domain.http_status, and metadata fields.
        - Sets custom error codes (ERR_TRAFI, ERR_ARCHIVE, etc.) for failures.
    """
    domains_query = model.Domain.select()
    if limit > 0:
        domains_query = domains_query.limit(limit)
    if http is not None: # If http is specified, we are likely recrawling specific statuses
        if str(http).strip().upper() == 'ERR':
            # Match every failure status (sprint-multilang, D-2): the real
            # values are prefixed codes (ERR_TRAFI, ERR_ARCHIVE, ERR_UNKNOWN,
            # ERR_ALL_FAILED, ...) plus ARC_NO_HTML / REQ_NO_HTML / 000.
            domains_query = domains_query.where(
                (model.Domain.http_status.startswith('ERR')) |
                (model.Domain.http_status.in_(['000', 'ARC_NO_HTML', 'REQ_NO_HTML'])))
        else:
            domains_query = domains_query.where(model.Domain.http_status == http)
    else: # Default: crawl domains not yet fetched
        domains_query = domains_query.where(model.Domain.fetched_at.is_null())

    processed_count = 0
    for domain in domains_query:
        domain_url_https = f"https://{domain.name}"
        domain_url_http = f"http://{domain.name}"
        html_content = None
        effective_url = None
        source_method = None
        final_status_code = None

        # Reset fields that might be populated by previous attempts
        domain.title = None
        domain.description = None
        domain.keywords = None

        # Attempt 1: Trafilatura (tries HTTPS then HTTP internally if not specified)
        print(f"Attempting Trafilatura for {domain.name} ({domain_url_https})")
        try:
            # Use retry and timeout wrapper to prevent infinite blocking
            downloaded = _fetch_url_with_retry_and_timeout(domain_url_https, timeout=settings.default_timeout)
            if not downloaded: # Try HTTP if HTTPS failed
                 downloaded = _fetch_url_with_retry_and_timeout(domain_url_http, timeout=settings.default_timeout)

            if downloaded:
                html_content = downloaded
                # fetch_url doesn't directly give status or final url, assume 200 if content received
                # For effective_url, we can try to get it from metadata later or use the input.
                # For now, let's assume the input URL that worked.
                # We need to determine if HTTPS or HTTP was successful for effective_url
                try:
                    # A bit of a hack: check if https version gives content
                    # This is imperfect as trafilatura might have its own redirect logic
                    requests.get(domain_url_https, timeout=2, allow_redirects=False).raise_for_status()
                    effective_url = domain_url_https
                except:
                    effective_url = domain_url_http

                final_status_code = "200" # Assume success if trafilatura returned content
                source_method = "TRAFILATURA"
                print(f"Trafilatura success for {domain.name} (URL: {effective_url})")
            else:
                print(f"Trafilatura failed to fetch content for {domain.name}")
        except Exception as e_trafi:
            print(f"Trafilatura exception for {domain.name}: {e_trafi}")
            final_status_code = "ERR_TRAFI"

        # Attempt 2: Archive.org (if Trafilatura failed)
        if not html_content:
            print(f"Attempting Archive.org for {domain.name}")
            try:
                # Prefer HTTPS for archive.org lookup, but it handles redirects
                archive_data_url = f"http://archive.org/wayback/available?url={domain_url_https}"
                archive_response = requests.get(archive_data_url, timeout=settings.default_timeout)
                archive_response.raise_for_status()
                archive_data = archive_response.json()
                
                archived_snapshot = archive_data.get('archived_snapshots', {}).get('closest', {})
                if archived_snapshot and archived_snapshot.get('available') and archived_snapshot.get('url'):
                    effective_url = archived_snapshot['url']
                    print(f"Found archived URL: {effective_url}")
                    archived_content_response = requests.get(
                        effective_url,
                        headers={"User-Agent": settings.user_agent},
                        timeout=settings.default_timeout
                    )
                    # We don't use raise_for_status() here as archive.org might return non-200 for the page itself
                    # but still provide content. The status code of the *archived page* is what matters.
                    final_status_code = str(archived_snapshot.get('status', '200')) # Use archived status
                    
                    if 'text/html' in archived_content_response.headers.get('Content-Type', '').lower():
                        html_content = archived_content_response.text
                        source_method = "ARCHIVE_ORG"
                        print(f"Archive.org success for {domain.name} (Status: {final_status_code})")
                    else:
                        print(f"Archive.org content for {domain.name} not HTML: {archived_content_response.headers.get('Content-Type')}")
                        if not final_status_code or final_status_code == '200': # If status was ok but not html
                            final_status_code = "ARC_NO_HTML"
                else:
                    print(f"No suitable archive found for {domain.name}")
                    final_status_code = "ERR_ARCHIVE_NF" # Not Found
            except requests.exceptions.Timeout:
                print(f"Archive.org timeout for {domain.name}")
                final_status_code = "ERR_ARCHIVE_TO"
            except requests.exceptions.RequestException as e_arc_req:
                print(f"Archive.org request exception for {domain.name}: {e_arc_req}")
                final_status_code = "ERR_ARCHIVE_REQ"
            except Exception as e_archive:
                print(f"Archive.org general exception for {domain.name}: {e_archive}")
                final_status_code = "ERR_ARCHIVE"

        # Attempt 3: Direct Requests (if Trafilatura and Archive.org failed)
        if not html_content:
            print(f"Attempting direct requests for {domain.name}")
            urls_to_try = [domain_url_https, domain_url_http]
            for current_url_to_try in urls_to_try:
                try:
                    response = requests.get(
                        current_url_to_try,
                        headers={"User-Agent": settings.user_agent},
                        timeout=settings.default_timeout,
                        allow_redirects=True # Allow redirects to find the final page
                    )
                    final_status_code = str(response.status_code)
                    effective_url = response.url # URL after redirects

                    if response.ok and 'text/html' in response.headers.get('Content-Type', '').lower():
                        html_content = response.text
                        source_method = "REQUESTS"
                        print(f"Direct request success for {domain.name} (URL: {effective_url}, Status: {final_status_code})")
                        break # Success, no need to try other URL
                    else:
                        print(f"Direct request for {current_url_to_try} failed or not HTML. Status: {final_status_code}, Content-Type: {response.headers.get('Content-Type')}")
                        if response.ok and not ('text/html' in response.headers.get('Content-Type', '').lower()):
                             final_status_code = "REQ_NO_HTML" # Mark as non-HTML success
                except requests.exceptions.Timeout:
                    print(f"Direct request timeout for {current_url_to_try}")
                    final_status_code = "000"
                except requests.exceptions.RequestException as e_req:
                    print(f"Direct request exception for {current_url_to_try}: {e_req}")
                    final_status_code = "000"
                except Exception as e_direct: # Catch any other unexpected errors
                    print(f"Direct request general exception for {current_url_to_try}: {e_direct}")
                    final_status_code = "ERR_UNKNOWN"
                if not html_content and not final_status_code: # If all attempts failed without setting a status
                    final_status_code = "ERR_ALL_FAILED"


        domain.fetched_at = model.datetime.datetime.now()
        domain.http_status = str(final_status_code) if final_status_code else "ERR_NO_STATUS"

        if html_content and source_method:
            try:
                process_domain_content(domain, html_content, effective_url or domain_url_https, source_method)
                print(f"Domain {domain.name} processed successfully via {source_method}.")
                processed_count += 1
            except Exception as e_proc:
                print(f"Error processing content for domain {domain.name}: {e_proc}")
                domain.http_status = "ERR_PROCESS" # Mark as processing error
        else:
            print(f"Failed to fetch HTML for domain {domain.name} after all attempts. Final status: {domain.http_status}")
            # Ensure some basic info if all fails
            domain.title = None # Set to None as per the initial request

        try:
            domain.save()
        except Exception as e_save:
            print(f"CRITICAL: Failed to save domain {domain.name}: {e_save}")
            # Potentially log this more severely or handle retry

    return processed_count


def process_domain_content(domain: model.Domain, html_content: str, effective_url: str, source_method: str):
    """Process and extract metadata from domain HTML content.

    This function extracts title, description, and keywords from HTML content
    using both BeautifulSoup and Trafilatura, combining results with priority.

    Args:
        domain: The Domain database object to update with extracted metadata.
        html_content: The HTML content of the domain's homepage.
        effective_url: The URL from which content was fetched (live or archived).
        source_method: Source indicator ('TRAFILATURA', 'ARCHIVE_ORG', 'REQUESTS').

    Notes:
        - Combines metadata from BeautifulSoup and Trafilatura extraction.
        - Trafilatura results take priority over BeautifulSoup results.
        - Extracts Open Graph, Twitter Card, and schema.org metadata.
        - Falls back to generic title if no metadata is found.
        - Updates domain.title, domain.description, and domain.keywords fields.
        - Prints extraction results for debugging.
    """
    # 1. Use BeautifulSoup based helpers (og: twitter: schema: etc.)
    soup = BeautifulSoup(html_content, 'html.parser')
    bs_title = get_title(soup)
    bs_description = get_description(soup)
    bs_keywords = get_keywords(soup)

    # 2. Use Trafilatura's metadata extraction
    trafi_title = None
    trafi_description = None
    trafi_keywords_list = None
    
    try:
        # Ensure html_content is not None or empty before passing to trafilatura
        if html_content:
            meta_object = trafilatura.extract_metadata(html_content)
            if meta_object:
                trafi_title = meta_object.title
                trafi_description = meta_object.description
                if meta_object.tags: # Trafilatura uses 'tags'
                     trafi_keywords_list = meta_object.tags
        else:
            print(f"HTML content is empty for {domain.name} ({effective_url}), skipping trafilatura metadata.")
            
    except Exception as e_t_meta:
        print(f"Error during trafilatura.extract_metadata for {domain.name} ({effective_url}): {e_t_meta}")

    # 3. Combine results
    # Prioritize Trafilatura if available, then BS, then existing (if any, though usually None at this stage for domain)
    final_title = trafi_title or bs_title
    final_description = trafi_description or bs_description
    
    final_keywords_str = None
    if trafi_keywords_list:
        final_keywords_str = ", ".join(trafi_keywords_list)
    elif bs_keywords: # Only use bs_keywords if trafilatura didn't provide any
        final_keywords_str = bs_keywords
    
    print(f"Metadata from '{source_method}' for {domain.name} (URL: {effective_url}):\n"
          f"  BS: title={bool(bs_title)}, desc={bool(bs_description)}, keyw={bool(bs_keywords)}\n"
          f"  Trafi: title={bool(trafi_title)}, desc={bool(trafi_description)}, tags={bool(trafi_keywords_list)}")

    domain.title = str(final_title).strip() if final_title else None # type: ignore
    domain.description = str(final_description).strip() if final_description else None # type: ignore
    domain.keywords = str(final_keywords_str).strip() if final_keywords_str else None # type: ignore
    
    # Fallback title if still nothing
    domain.title = domain.title or f"Website: {domain.name}"
    
    print(f"Final domain metadata for {domain.name}: title='{(domain.title or '')[:50]}...', "
          f"description='{(domain.description or '')[:50]}...', keywords='{(domain.keywords or '')[:50]}...'")


def get_meta_content(soup: BeautifulSoup, name: str) -> str:
    """Extract content from a named meta tag.

    This function searches for a meta tag with the specified name attribute
    and returns its content value.

    Args:
        soup: A BeautifulSoup object containing parsed HTML.
        name: The value of the 'name' attribute to search for.

    Returns:
        str: The content of the meta tag, or empty string if not found.

    Notes:
        - Only returns string content values.
        - Strips leading and trailing whitespace from content.
        - Logs found content to console for debugging (first 30 chars).
    """
    tag = soup.find('meta', attrs={'name': name})
    if tag and tag.has_attr('content'): # type: ignore
        content = tag['content'] # type: ignore
        if isinstance(content, str):
            print(f"Found meta content for {name}: {content[:30]}...")
            return content.strip()
    return ""


def extract_metadata(url: str) -> dict:
    """Extract metadata from a webpage using multiple fallback sources.

    This function fetches a webpage and extracts title, description, and keywords
    using helper functions that check multiple meta tag sources.

    Args:
        url: The URL of the webpage to extract metadata from.

    Returns:
        dict: A dictionary with keys 'title', 'description', and 'keywords'.
            Values are strings or None if extraction fails.

    Notes:
        - Automatically adds 'https://' protocol if missing from URL.
        - Uses get_title(), get_description(), and get_keywords() helpers.
        - Returns None values for all fields if request fails.
        - Timeout is set to 5 seconds.
        - Logs extraction progress and errors to console.
    """
    try:
        # Ensure URL has a protocol
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        print(f"Extracting metadata from {url}")
        response = requests.get(url, headers={"User-Agent": settings.user_agent}, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = get_title(soup)
        description = get_description(soup)
        keywords = get_keywords(soup)
        
        print(f"Extracted metadata: title={bool(title)}, description={bool(description)}, keywords={bool(keywords)}")
        
        return {
            'title': title,
            'description': description,
            'keywords': keywords
        }
    except Exception as e:
        print(f"Error extracting metadata from {url}: {str(e)}")
        return {'title': None, 'description': None, 'keywords': None}


def get_title(soup: BeautifulSoup) -> str:
    """Extract page title using a fallback chain of meta tag sources.

    This function searches for the page title in multiple locations with priority
    given to Open Graph, then Twitter Card, then schema.org, and finally standard HTML title.

    Args:
        soup: A BeautifulSoup object containing parsed HTML.

    Returns:
        str: The page title, or empty string if no title is found.

    Notes:
        - Priority order: og:title > twitter:title > schema.org title > <title> tag.
        - Returns the first non-empty title found in the priority order.
        - Strips leading and trailing whitespace from the title.
        - Empty string is returned if no title source is found.
    """
    # Open Graph title (highest priority)
    og_title = soup.find('meta', attrs={'property': 'og:title'})
    if og_title and og_title.has_attr('content'): # type: ignore
        content = og_title['content'] # type: ignore
        if isinstance(content, str):
            return content.strip()
    
    # Twitter title
    twitter_title = soup.find('meta', attrs={'name': 'twitter:title'})
    if twitter_title and twitter_title.has_attr('content'): # type: ignore
        content = twitter_title['content'] # type: ignore
        if isinstance(content, str):
            return content.strip()
    
    # Schema.org title
    schema_title = soup.find('meta', attrs={'itemprop': 'title'})
    if schema_title and schema_title.has_attr('content'): # type: ignore
        content = schema_title['content'] # type: ignore
        if isinstance(content, str):
            return content.strip()
    
    # Standard HTML title (lowest priority)
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    
    return ""


def get_description(soup: BeautifulSoup) -> str:
    """Extract page description using a fallback chain of meta tag sources.

    This function searches for the page description in multiple meta tag locations
    with priority to standard meta description, then Open Graph, Twitter Card,
    and schema.org.

    Args:
        soup: A BeautifulSoup object containing parsed HTML.

    Returns:
        str: The page description, or empty string if no description is found.

    Notes:
        - Priority order: meta[name="description"] > og:description >
          twitter:description > schema.org description.
        - Returns the first non-empty description found in the priority order.
        - Strips leading and trailing whitespace from the description.
        - Empty string is returned if no description source is found.
    """
    # Standard meta description
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc and meta_desc.has_attr('content'): # type: ignore
        content = meta_desc['content'] # type: ignore
        if isinstance(content, str):
            return content.strip()
    
    # Open Graph description
    og_desc = soup.find('meta', attrs={'property': 'og:description'})
    if og_desc and og_desc.has_attr('content'): # type: ignore
        content = og_desc['content'] # type: ignore
        if isinstance(content, str):
            return content.strip()
    
    # Twitter description
    twitter_desc = soup.find('meta', attrs={'name': 'twitter:description'})
    if twitter_desc and twitter_desc.has_attr('content'): # type: ignore
        content = twitter_desc['content'] # type: ignore
        if isinstance(content, str):
            return content.strip()
    
    # Schema.org description
    schema_desc = soup.find('meta', attrs={'itemprop': 'description'})
    if schema_desc and schema_desc.has_attr('content'): # type: ignore
        content = schema_desc['content'] # type: ignore
        if isinstance(content, str):
            return content.strip()
    
    return ""


def get_keywords(soup: BeautifulSoup) -> str:
    """Extract page keywords using a fallback chain of meta tag sources.

    This function searches for page keywords in multiple meta tag locations with
    priority to standard meta keywords, then Open Graph and Twitter Card variations.

    Args:
        soup: A BeautifulSoup object containing parsed HTML.

    Returns:
        str: A comma-separated string of keywords, or empty string if not found.

    Notes:
        - Priority order: meta[name="keywords"] > og:keywords > twitter:keywords.
        - Returns the first non-empty keywords found in the priority order.
        - Strips leading and trailing whitespace from the keywords string.
        - Empty string is returned if no keywords source is found.
        - Note: og:keywords and twitter:keywords are rare in practice.
    """
    # Standard meta keywords
    meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
    if meta_keywords and meta_keywords.has_attr('content'): # type: ignore
        content = meta_keywords['content'] # type: ignore
        if isinstance(content, str):
            return content.strip()
    
    # Open Graph keywords (rare but check)
    og_keywords = soup.find('meta', attrs={'property': 'og:keywords'})
    if og_keywords and og_keywords.has_attr('content'): # type: ignore
        content = og_keywords['content'] # type: ignore
        if isinstance(content, str):
            return content.strip()
    
    # Twitter keywords (rare but check)
    twitter_keywords = soup.find('meta', attrs={'name': 'twitter:keywords'})
    if twitter_keywords and twitter_keywords.has_attr('content'): # type: ignore
        content = twitter_keywords['content'] # type: ignore
        if isinstance(content, str):
            return content.strip()
    
    return ""


async def crawl_land(land: model.Land, limit: int = 0, http: Optional[str] = None, depth: Optional[int] = None, store_html: bool = False, retry_status: Optional[list] = None, issue_mode: Optional[bool] = None) -> tuple:
    """Asynchronously crawl all expressions in a land.

    This function orchestrates the crawling process for a land, processing
    expressions depth by depth with concurrent HTTP requests and media analysis.

    Args:
        land: The Land object to crawl.
        limit: Maximum number of expressions to crawl (0 for unlimited).
        http: Optional HTTP status filter for recrawling specific expressions.
        depth: Optional depth filter to crawl only expressions at this depth level.
        store_html: When True, persist raw HTML in expression.html.
        retry_status: Optional list of HTTP status codes to retry. When provided,
            selects expressions whose http_status matches any code in the list,
            ignoring fetched_at. Takes precedence over http. Useful to backfill
            the cascade fallback (sprint-403 Sprint 5) on previously crawled URLs.

    Returns:
        tuple: A tuple of (total_processed, total_errors) counts.

    Notes:
        - Processes expressions depth by depth in ascending order.
        - Uses asynchronous HTTP requests for efficient crawling.
        - Includes media analysis if enabled in settings.
        - Respects parallel connection limits from settings.
        - Updates expression metadata, content, relevance, and links.
        - Handles different event loop policies for Windows vs Unix.
    """
    print(f"Crawling land {land.id}") # type: ignore
    dictionary = get_land_dictionary(land)

    total_processed = 0
    total_errors = 0
    # Track how many expressions we attempted to crawl to enforce --limit
    total_attempted = 0

    def _base_filter():
        """Build the WHERE filter for expression selection.

        Priority: retry_status > http > fetched_at IS NULL.
        """
        if retry_status:
            return (model.Expression.land == land,
                    model.Expression.http_status.in_(retry_status))
        if http is None:
            return (model.Expression.land == land,
                    model.Expression.fetched_at.is_null(True))
        return (model.Expression.land == land,
                model.Expression.http_status == http)

    try:
        # If depth is specified, only process that depth
        if depth is not None:
            depths_to_process = [depth]
        else:
            depths_query = (model.Expression
                            .select(model.Expression.depth)
                            .where(*_base_filter())
                            .distinct()
                            .order_by(model.Expression.depth))
            depths_to_process = [d.depth for d in depths_query]

        for current_depth in depths_to_process:
            print(f"Processing depth {current_depth}")

            expressions = model.Expression.select().where(
                *_base_filter(),
                model.Expression.depth == current_depth,
            )

            expression_count = expressions.count()
            if expression_count == 0:
                continue

            batch_size = settings.parallel_connections
            batch_count = -(-expression_count // batch_size)
            last_batch_size = expression_count % batch_size
            current_offset = 0

            for current_batch in range(batch_count):
                print(f"Batch {current_batch + 1}/{batch_count} for depth {current_depth}")
                # Determine base batch limit from remaining rows in this depth
                batch_limit = last_batch_size if (current_batch + 1 == batch_count and last_batch_size != 0) else batch_size
                # Enforce the global --limit on attempts (not only successes)
                if limit > 0:
                    remaining = max(0, limit - total_attempted)
                    if remaining == 0:
                        return total_processed, total_errors
                    effective_batch_limit = min(batch_limit, remaining)
                else:
                    effective_batch_limit = batch_limit

                current_expressions_query = expressions.limit(effective_batch_limit).offset(current_offset)
                current_batch_expressions = list(current_expressions_query)
                attempted_in_batch = len(current_batch_expressions)

                if attempted_in_batch == 0:
                    break

                connector = aiohttp.TCPConnector(limit=settings.parallel_connections, ssl=False)
                async with aiohttp.ClientSession(connector=connector, max_field_size=16384) as session:
                    tasks = [
                        crawl_expression_with_media_analysis(expr, dictionary, session, store_html=store_html, issue_mode=issue_mode)
                        for expr in current_batch_expressions
                    ]
                    # return_exceptions=True: a single expression raising
                    # (e.g. a malformed link reaching urlparse) must never
                    # sink the whole batch. Exceptions fall into the error
                    # bucket (errors = attempted - processed).
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    processed_in_batch = 0
                    for expr, res in zip(current_batch_expressions, results):
                        if isinstance(res, Exception):
                            print(f"Error crawling #{expr.id} ({expr.url}): {res}")  # type: ignore
                        elif res:
                            processed_in_batch += 1
                    total_processed += processed_in_batch
                    total_errors += (attempted_in_batch - processed_in_batch)
                    total_attempted += attempted_in_batch

                current_offset += attempted_in_batch

                if limit > 0 and total_attempted >= limit:
                    return total_processed, total_errors

        return total_processed, total_errors
    finally:
        # Shutdown the shared BrowserPool if it was started during this crawl
        # (sprint-403 Sprint 3b). Idempotent — does nothing when the pool
        # was never instantiated.
        try:
            from mwi.browser_pool import BrowserPool, PLAYWRIGHT_AVAILABLE
            if PLAYWRIGHT_AVAILABLE and BrowserPool._instance is not None:
                await BrowserPool.get().shutdown()
                BrowserPool.reset()
        except Exception as e:
            print(f"BrowserPool shutdown warning: {e}")

def _extract_content_and_links(raw_html, expression, source_method: str = "aiohttp"):
    """Run Trafilatura then BeautifulSoup on ``raw_html`` and persist medias.

    Returns ``(content, links)`` where ``content`` is the markdown body
    (or ``None`` if extraction yielded nothing useful) and ``links`` is a
    list of outgoing URLs found in the body.

    Side-effects: writes to ``expression.readable`` and inserts ``Media``
    rows for embedded images. ``source_method`` is only used to phrase
    the success log line ("aiohttp" vs "archive_org").
    """
    if not raw_html:
        return None, []

    content = None
    links: list = []

    # 2a. Trafilatura
    try:
        extracted_content = trafilatura.extract(
            raw_html, include_links=True, include_comments=False,
            include_images=True, output_format='markdown',
        )
        readable_html = trafilatura.extract(
            raw_html, include_links=True, include_comments=False,
            include_images=True, output_format='html',
        )
        if extracted_content and len(extracted_content) > 100:
            media_lines = []
            if readable_html:
                soup_readable = BeautifulSoup(readable_html, 'html.parser')
                for tag, label in [('img', 'IMAGE'), ('video', 'VIDEO'), ('audio', 'AUDIO')]:
                    for element in soup_readable.find_all(tag):
                        src = element.get('src')
                        if src:
                            if tag == 'img':
                                media_lines.append(f"![{label}]({src})")
                            else:
                                media_lines.append(f"[{label}: {src}]")
            content = extracted_content
            if media_lines:
                content += "\n\n" + "\n".join(media_lines)
            if readable_html:
                soup_readable = BeautifulSoup(readable_html, 'html.parser')
                extract_medias(soup_readable, expression)
            img_md_links = re.findall(r'!\[.*?\]\((.*?)\)', content)
            for img_url in img_md_links:
                resolved_img_url = resolve_url(str(expression.url), img_url)
                if not model.Media.select().where(
                    (model.Media.expression == expression) &
                    (model.Media.url == resolved_img_url)
                ).exists():
                    model.Media.create(expression=expression, url=resolved_img_url, type='img')
            links = extract_md_links(content, str(expression.url))
            expression.readable = content # type: ignore
            if source_method == "archive_org":
                print(f"Archive.org + Trafilatura succeeded for {expression.url}")
            else:
                print(f"Trafilatura succeeded on fetched HTML for {expression.url}")
    except Exception as e:
        print(f"Trafilatura failed on raw HTML for {expression.url}: {e}")

    # 2b. BeautifulSoup fallback on the same HTML
    if not content:
        try:
            soup = BeautifulSoup(raw_html, 'html.parser')
            clean_html(soup)
            text_content = get_readable(soup)
            if text_content and len(text_content) > 100:
                content = text_content
                # Resolve relative hrefs before is_crawlable (which requires
                # http(s)://): otherwise every relative link is dropped.
                hrefs = [a.get('href') for a in soup.find_all('a')]
                urls = [urljoin(str(expression.url), h)
                        for h in hrefs if isinstance(h, str) and h]
                links = [u for u in urls if is_crawlable(u)]
                expression.readable = content # type: ignore
                print(f"BeautifulSoup fallback succeeded for {expression.url}")
        except Exception as e:
            print(f"BeautifulSoup fallback failed for {expression.url}: {e}")

    return content, links


async def crawl_expression_with_media_analysis(expression: model.Expression, dictionary, session: aiohttp.ClientSession, store_html: bool = False, issue_mode: Optional[bool] = None):
    """Crawl and process an expression with integrated media analysis.

    This function fetches an expression's URL, extracts content using Trafilatura,
    analyzes media elements, and updates the expression in the database.

    Args:
        expression: The Expression database object to process.
        dictionary: The land's word dictionary for relevance scoring.
        session: An aiohttp ClientSession for making HTTP requests.

    Returns:
        int: 1 if content was successfully processed, 0 on failure.

    Notes:
        - Fetches URL content via direct HTTP request.
        - Extracts main content using Trafilatura with markdown and HTML formats.
        - Analyzes embedded media (images, videos, audio) in readable content.
        - Falls back to raw HTML content processing if Trafilatura fails.
        - Updates expression.http_status, expression.content, and expression.relevance.
        - Automatically calls analyze_media() for media extraction and analysis.
        - Sets fetched_at timestamp on the expression.
    """
    print(f"Crawling expression #{expression.id} with media analysis: {expression.url}") # type: ignore
    expression.fetched_at = model.datetime.datetime.now() # type: ignore

    # Step 1+3 (fetch + URL-based fallbacks): delegated to mwi.fetcher
    fetch_result = await fetch_html(str(expression.url), session=session)
    expression.http_status = fetch_result.status_code # type: ignore
    expression.fetch_method = fetch_result.method_used # type: ignore
    raw_html = fetch_result.html

    # Persist raw HTML *before* extraction — invariant set by sprint-html
    # Sprint A: storage must not depend on the success of Trafilatura/BS4.
    # If the cascade returned any HTML, archive it, even when extraction
    # later yields no readable content (CF interstitial, JS-only sites).
    # Sprint E: truncate at settings.fullhtml_max_size_kb (default 5 MB)
    # to protect the SQLite WAL cache from pathological oversized pages.
    if store_html and raw_html:
        expression.html = _maybe_truncate_html(raw_html)  # type: ignore

    # Step 2: extract content from whichever raw_html we got (live or archived)
    content, links = _extract_content_and_links(
        raw_html, expression, source_method=fetch_result.method_used,
    )

    if not content and fetch_result.html is None:
        print(f"All extraction methods failed for {expression.url}. Final status: {expression.http_status}")

    # Step 4: post-processing (language, relevance, links, media) when content available
    if content:
        soup = BeautifulSoup(raw_html if raw_html else content, 'html.parser')
        expression.title = str(get_title(soup) or expression.url) # type: ignore
        expression.description = str(get_description(soup)) if get_description(soup) else None # type: ignore
        expression.keywords = str(get_keywords(soup)) if get_keywords(soup) else None # type: ignore
        html_lang = str(soup.html.get('lang', '')) if soup.html else ''
        expression.lang = detect_content_language(content, html_lang)  # type: ignore
        # Check language compatibility BEFORE any relevance calculation
        if expression.lang and not is_language_compatible(expression.lang, expression.land.lang):  # type: ignore
            expression.relevance = 0  # type: ignore
            print(f"Language mismatch for expression #{expression.id}: lang='{expression.lang}' not in land langs='{expression.land.lang}'")  # type: ignore
        else:
            # Compute relevance with OpenRouter gate when enabled
            try:
                from .llm_openrouter import is_relevant_via_openrouter  # local import to avoid overhead when disabled
                if getattr(settings, 'openrouter_enabled', False) and settings.openrouter_api_key and settings.openrouter_model:
                    verdict = is_relevant_via_openrouter(expression.land, expression, issue_mode=issue_mode)  # type: ignore
                    if verdict is False:
                        expression.relevance = 0  # type: ignore
                    else:
                        expression.relevance = expression_relevance(dictionary, expression)  # type: ignore
                else:
                    expression.relevance = expression_relevance(dictionary, expression)  # type: ignore
            except Exception as e:
                print(f"OpenRouter gate error for {expression.url}: {e}")
                expression.relevance = expression_relevance(dictionary, expression)  # type: ignore
        expression.readable_at = model.datetime.datetime.now() # type: ignore
        if expression.relevance is not None and expression.relevance > 0: # type: ignore
            expression.approved_at = model.datetime.datetime.now() # type: ignore
        model.ExpressionLink.delete().where(model.ExpressionLink.source == expression.id).execute() # type: ignore

        # Extract dynamic media using headless browser (only for approved expressions)
        if (expression.relevance is not None and expression.relevance > 0 and # type: ignore
            settings.dynamic_media_extraction and PLAYWRIGHT_AVAILABLE):
            try:
                print(f"Attempting dynamic media extraction for #{expression.id}") # type: ignore
                dynamic_media_urls = await extract_dynamic_medias(str(expression.url), expression)
                if dynamic_media_urls:
                    print(f"Dynamic extraction found {len(dynamic_media_urls)} additional media items for #{expression.id}") # type: ignore
                else:
                    print(f"No dynamic media found for #{expression.id}") # type: ignore
            except Exception as e:
                print(f"Dynamic media extraction failed for #{expression.id}: {e}") # type: ignore
        elif expression.relevance is not None and expression.relevance > 0 and settings.dynamic_media_extraction and not PLAYWRIGHT_AVAILABLE: # type: ignore
            print(f"Dynamic media extraction requested but Playwright not available for #{expression.id}") # type: ignore

        if expression.relevance is not None and expression.relevance > 0 and expression.depth is not None and expression.depth < 3 and links: # type: ignore
            print(f"Linking {len(links)} expressions to #{expression.id}") # type: ignore
            # sprint link-context: locate each link in the raw DOM (soup reused, no re-parse)
            dom_map = link_context.extract_link_dom_map(
                raw_html, str(expression.url), soup=soup) if raw_html else {}
            for link in links:
                info = link_context.lookup_link_info(dom_map, link)
                ctx = link_context.extract_md_paragraph(content, link)
                if ctx is None and info is not None:
                    ctx = info.block_text
                link_expression(expression.land, expression, link, # type: ignore
                                context=ctx,
                                dom=info.dom if info else None,
                                dom_html=info.dom_html if info else None)
        expression.save()
        return 1
    else:
        if fetch_result.html is not None:
            # We had HTML but extraction yielded nothing usable
            print(f"All extraction methods failed for {expression.url}. Final status: {expression.http_status}")
        expression.save()
        return 0

async def consolidate_land(
    land: model.Land,
    limit: int = 0,
    depth: Optional[int] = None,
    min_relevance: int = 0,
    llm_revalidate: bool = False,
    issue_mode: Optional[bool] = None,
) -> tuple:
    """Consolidate a land by reprocessing expressions and rebuilding relationships.

    This function recalculates relevance scores, extracts links and media from
    existing content, and repairs the expression graph after external modifications.

    Args:
        land: The Land object to consolidate.
        limit: Maximum number of expressions to process (0 for unlimited).
        depth: Optional depth filter to process only expressions at this depth.
        min_relevance: Minimum relevance threshold for expressions to process.

    Returns:
        tuple: A tuple of (total_processed, total_errors) counts.

    Notes:
        - Only processes expressions that have been approved or have readable content.
        - Recalculates relevance scores based on current land dictionary.
        - Respects existing LLM verdicts: an expression with validllm='non'
          keeps relevance=0 (it is never resurrected by the lexical recompute).
        - With llm_revalidate=True, re-runs the OpenRouter gate (issue_mode
          honoured, None => settings.openrouter_issue_mode) and refreshes
          validllm/validmodel before applying the verdict.
        - Deletes and recreates all expression links from content.
        - Resolves URL variants (http/https, www, trailing slash) onto the
          EXISTING corpus expression via the 3-key ladder (sprint
          dedup-selfloops) instead of creating duplicate rows; a link to a
          variant of the page itself never creates a self-loop.
        - Extracts and analyzes media from existing content.
        - Useful for repairing data after manual content edits or dictionary updates.
        - Does not re-fetch URLs; works with existing content in the database.
    """
    print(f"Consolidating land {land.id}") # type: ignore
    dictionary = get_land_dictionary(land)

    # Select expressions to process
    query = model.Expression.select().where(
        model.Expression.land == land,
        (
            model.Expression.approved_at.is_null(False) |
            model.Expression.readable_at.is_null(False)
        )
    )
    if depth is not None:
        query = query.where(model.Expression.depth == depth)
    if limit > 0:
        query = query.limit(limit)
    if min_relevance > 0:
        query = query.where(model.Expression.relevance >= min_relevance)

    total_processed = 0
    total_errors = 0

    # 3-key URL index of the whole land: readable links are resolved onto
    # existing expressions (variant-proof) before any creation. Expressions
    # created during the run are appended to the index.
    url_index = link_context.build_url_index(
        model.Expression.select(model.Expression.id, model.Expression.url)
        .where(model.Expression.land == land).tuples())

    batch_size = settings.parallel_connections
    expression_count = query.count()
    batch_count = -(-expression_count // batch_size)
    last_batch_size = expression_count % batch_size
    current_offset = 0

    for current_batch in range(batch_count):
        print(f"Consolidation batch {current_batch + 1}/{batch_count}")
        batch_limit = last_batch_size if (current_batch + 1 == batch_count and last_batch_size != 0) else batch_size
        current_expressions = query.limit(batch_limit).offset(current_offset)

        for expr in current_expressions:
            try:
                # 1. Supprimer anciens liens et médias
                model.ExpressionLink.delete().where(model.ExpressionLink.source == expr.id).execute()
                model.Media.delete().where(model.Media.expression == expr.id).execute()

                # 2. Re-validation LLM optionnelle (--llm=true), puis recalcul
                #    lexical, puis respect du verdict LLM existant.
                if (llm_revalidate and expr.readable and
                        len(str(expr.readable)) >= getattr(settings, 'openrouter_readable_min_chars', 0)):
                    from .llm_openrouter import is_relevant_via_openrouter
                    verdict = is_relevant_via_openrouter(land, expr, issue_mode=issue_mode)
                    if verdict is True:
                        expr.validllm = 'oui'  # type: ignore
                        expr.validmodel = settings.openrouter_model  # type: ignore
                    elif verdict is False:
                        expr.validllm = 'non'  # type: ignore
                        expr.validmodel = settings.openrouter_model  # type: ignore
                    # verdict None (désactivé/budget/erreur) : ne pas toucher validllm
                try:
                    new_rel = expression_relevance(dictionary, expr)  # type: ignore
                except Exception as e:
                    print(f"Error recalculating relevance for {expr.url}: {e}")
                    new_rel = expr.relevance or 0  # type: ignore
                # Respecter le verdict LLM : ne pas ressusciter une page rejetée
                # ('non' => 0). Traçabilité scientifique.
                if getattr(expr, 'validllm', None) == 'non':
                    new_rel = 0
                expr.relevance = new_rel  # type: ignore
                expr.save()

                # 3. Extraire les liens sortants du contenu lisible
                links = []
                if expr.readable:
                    # Extraction des liens markdown (relatifs résolus via urljoin)
                    links = extract_md_links(expr.readable, str(expr.url))
                    # Extraction des liens HTML (fallback) — résoudre les hrefs
                    # relatifs avant is_crawlable, sinon perte sèche.
                    soup = BeautifulSoup(expr.readable, 'html.parser')
                    hrefs = [a.get('href') for a in soup.find_all('a')]
                    urls = [urljoin(str(expr.url), h)
                            for h in hrefs if isinstance(h, str) and h]
                    links += [u for u in urls if is_crawlable(u) and u not in links]
                nb_links = len(set(links))

                # 4. Ajouter les documents manquants et recréer les liens
                # sprint link-context: backfill context/dom/dom_html depuis le
                # HTML stocké (--fullhtml) quand il est disponible
                stored_html = getattr(expr, 'html', None)
                dom_map = link_context.extract_link_dom_map(
                    stored_html, str(expr.url)) if stored_html else {}
                for url in set(links):
                    # variant-proof: resolve onto an existing corpus fiche
                    # (http/https, www, trailing slash absorbed) before
                    # falling back to creation.
                    target_id = link_context.resolve_url_in_index(url_index, url)
                    if target_id is None:
                        if not is_crawlable(url):
                            continue
                        target_expr = add_expression(land, url, expr.depth + 1 if expr.depth is not None else 1)
                        if not target_expr:
                            continue
                        target_id = target_expr.id  # type: ignore
                        link_context.add_to_url_index(
                            url_index, target_id, str(target_expr.url))
                    if target_id == expr.id:
                        continue  # self-citation (permalink/variant) -> no self-loop
                    info = link_context.lookup_link_info(dom_map, url)
                    ctx = link_context.extract_md_paragraph(expr.readable, url)
                    if ctx is None and info is not None:
                        ctx = info.block_text
                    try:
                        model.ExpressionLink.create(
                            source_id=expr.id, # type: ignore
                            target_id=target_id,
                            context=ctx,
                            dom=info.dom if info else None,
                            dom_html=info.dom_html if info else None)
                    except IntegrityError:
                        pass

                # 5. Extraire et recréer les médias
                nb_media = 0
                if expr.readable:
                    soup = BeautifulSoup(expr.readable, 'html.parser')
                    extract_medias(soup, expr)
                    nb_media = model.Media.select().where(model.Media.expression == expr.id).count()

                print(f"Expression #{expr.id}: {nb_links} liens extraits, {nb_media} médias extraits.")

                total_processed += 1
            except Exception as e:
                print(f"Error consolidating expression {expr.id}: {e}")
                total_errors += 1

        current_offset += batch_size

        if limit > 0 and total_processed >= limit:
            return total_processed, total_errors

    return total_processed, total_errors


async def crawl_expression(expression: model.Expression, dictionary, session: aiohttp.ClientSession, store_html: bool = False, issue_mode: Optional[bool] = None):
    """Crawl and process an expression using a multi-stage fallback pipeline.

    This function fetches and processes web content through a sophisticated pipeline
    with multiple extraction methods and fallback strategies.

    Args:
        expression: The Expression database object to process.
        dictionary: The land's word dictionary for relevance scoring.
        session: An aiohttp ClientSession for making HTTP requests.

    Returns:
        int: 1 if content was successfully processed, 0 on failure.

    Notes:
        - Pipeline stages: Direct fetch -> Trafilatura -> BeautifulSoup -> Mercury -> Archive.org
        - Preserves original HTTP status code throughout the pipeline.
        - Extracts links from content for building the expression graph.
        - Calculates relevance score using the land's dictionary.
        - Updates expression.http_status, expression.content, and expression.relevance.
        - Sets fetched_at timestamp on the expression.
        - Deprecated in favor of crawl_expression_with_media_analysis.
    """
    print(f"Crawling expression #{expression.id}: {expression.url}") # type: ignore
    expression.fetched_at = model.datetime.datetime.now() # type: ignore

    fetch_result = await fetch_html(str(expression.url), session=session)
    expression.http_status = fetch_result.status_code # type: ignore
    expression.fetch_method = fetch_result.method_used # type: ignore
    raw_html = fetch_result.html

    # Persist raw HTML *before* extraction (sprint-html Sprint A + E)
    if store_html and raw_html:
        expression.html = _maybe_truncate_html(raw_html)  # type: ignore

    content, links = _extract_content_and_links(
        raw_html, expression, source_method=fetch_result.method_used,
    )

    if content:
        soup = BeautifulSoup(raw_html if raw_html else content, 'html.parser')
        expression.title = str(get_title(soup) or expression.url) # type: ignore
        expression.description = str(get_description(soup)) if get_description(soup) else None # type: ignore
        expression.keywords = str(get_keywords(soup)) if get_keywords(soup) else None # type: ignore
        html_lang = str(soup.html.get('lang', '')) if soup.html else ''
        expression.lang = detect_content_language(content, html_lang)  # type: ignore
        if expression.lang and not is_language_compatible(expression.lang, expression.land.lang):  # type: ignore
            expression.relevance = 0  # type: ignore
            print(f"Language mismatch for expression #{expression.id}: lang='{expression.lang}' not in land langs='{expression.land.lang}'")  # type: ignore
        else:
            try:
                from .llm_openrouter import is_relevant_via_openrouter
                if getattr(settings, 'openrouter_enabled', False) and settings.openrouter_api_key and settings.openrouter_model:
                    verdict = is_relevant_via_openrouter(expression.land, expression, issue_mode=issue_mode)  # type: ignore
                    if verdict is False:
                        expression.relevance = 0  # type: ignore
                    else:
                        expression.relevance = expression_relevance(dictionary, expression)  # type: ignore
                else:
                    expression.relevance = expression_relevance(dictionary, expression)  # type: ignore
            except Exception as e:
                print(f"OpenRouter gate error for {expression.url}: {e}")
                expression.relevance = expression_relevance(dictionary, expression)  # type: ignore
        expression.readable_at = model.datetime.datetime.now() # type: ignore
        if expression.relevance is not None and expression.relevance > 0: # type: ignore
            expression.approved_at = model.datetime.datetime.now() # type: ignore
        model.ExpressionLink.delete().where(model.ExpressionLink.source == expression.id).execute() # type: ignore

        if (expression.relevance is not None and expression.relevance > 0 and # type: ignore
            settings.dynamic_media_extraction and PLAYWRIGHT_AVAILABLE):
            try:
                print(f"Attempting dynamic media extraction for #{expression.id}") # type: ignore
                dynamic_media_urls = await extract_dynamic_medias(str(expression.url), expression)
                if dynamic_media_urls:
                    print(f"Dynamic extraction found {len(dynamic_media_urls)} additional media items for #{expression.id}") # type: ignore
                else:
                    print(f"No dynamic media found for #{expression.id}") # type: ignore
            except Exception as e:
                print(f"Dynamic media extraction failed for #{expression.id}: {e}") # type: ignore
        elif expression.relevance is not None and expression.relevance > 0 and settings.dynamic_media_extraction and not PLAYWRIGHT_AVAILABLE: # type: ignore
            print(f"Dynamic media extraction requested but Playwright not available for #{expression.id}") # type: ignore

        if expression.relevance is not None and expression.relevance > 0 and expression.depth is not None and expression.depth < 3 and links: # type: ignore
            print(f"Linking {len(links)} expressions to #{expression.id}") # type: ignore
            # sprint link-context: locate each link in the raw DOM (soup reused, no re-parse)
            dom_map = link_context.extract_link_dom_map(
                raw_html, str(expression.url), soup=soup) if raw_html else {}
            for link in links:
                info = link_context.lookup_link_info(dom_map, link)
                ctx = link_context.extract_md_paragraph(content, link)
                if ctx is None and info is not None:
                    ctx = info.block_text
                link_expression(expression.land, expression, link, # type: ignore
                                context=ctx,
                                dom=info.dom if info else None,
                                dom_html=info.dom_html if info else None)
        expression.save()
        return 1
    else:
        print(f"All extraction methods failed for {expression.url}. Final status: {expression.http_status}")
        expression.save()
        return 0

async def analyze_media(expression: model.Expression, session: aiohttp.ClientSession) -> list:
    """Analyze and extract detailed metadata for media associated with an expression.

    This function processes all media items linked to an expression, fetching and
    analyzing their properties using the MediaAnalyzer.

    Args:
        expression: The Expression database object whose media should be analyzed.
        session: An aiohttp ClientSession for downloading media files.

    Returns:
        list: A list of successfully analyzed media items.

    Notes:
        - Uses MediaAnalyzer for comprehensive media analysis.
        - Extracts metadata like dimensions, colors, format, EXIF data, etc.
        - Calculates perceptual hashes for duplicate detection.
        - Updates Media database records with analysis results.
        - Requires settings.media_analysis to be enabled.
        - Returns empty list if no media is found or analysis fails.
    """
    from .media_analyzer import MediaAnalyzer
    media_settings = {
        'user_agent': settings.user_agent,
        'min_width': getattr(settings, 'media_min_width', 100),
        'min_height': getattr(settings, 'media_min_height', 100),
        'max_file_size': getattr(settings, 'media_max_file_size', 10 * 1024 * 1024),
        'download_timeout': getattr(settings, 'media_download_timeout', 30),
        'max_retries': getattr(settings, 'media_max_retries', 2),
        'analyze_content': getattr(settings, 'media_analyze_content', False),
        'extract_colors': getattr(settings, 'media_extract_colors', True),
        'extract_exif': getattr(settings, 'media_extract_exif', True),
        'n_dominant_colors': getattr(settings, 'media_n_dominant_colors', 5)
    }
    analyzer = MediaAnalyzer(session, media_settings)
    analyzed_medias = []

    # Get all media URLs associated with this expression
    media_items = model.Media.select().where(model.Media.expression == expression)

    for media in media_items:
        try:
            analysis_result = await analyzer.analyze_image(media.url)
            if analysis_result:
                # Update media record with analysis results
                media.width = analysis_result.get('width')
                media.height = analysis_result.get('height')
                media.file_size = analysis_result.get('file_size')
                media.format = analysis_result.get('format')
                media.color_mode = analysis_result.get('color_mode')
                media.dominant_colors = json.dumps(analysis_result.get('dominant_colors', []))
                media.has_transparency = analysis_result.get('has_transparency')
                media.aspect_ratio = analysis_result.get('aspect_ratio')
                media.exif_data = json.dumps(analysis_result.get('exif_data', {}))
                media.image_hash = analysis_result.get('image_hash')
                media.content_tags = json.dumps(analysis_result.get('content_tags', []))
                media.nsfw_score = analysis_result.get('nsfw_score')
                media.analyzed_at = model.datetime.datetime.now()
                media.analysis_error = None
                media.save()

                analyzed_medias.append(media.url)
                print(f"Analyzed media: {media.url}")
            else:
                print(f"No analysis result for media: {media.url}")
        except Exception as e:
            print(f"Error analyzing media {media.url}: {e}")
            # Update media record with error
            media.analysis_error = str(e)
            media.analyzed_at = model.datetime.datetime.now()
            media.save()

    return analyzed_medias


def extract_md_links(md_content: str, base_url: Optional[str] = None):
    """Extract hyperlink targets from Markdown content.

    Thin backward-compatible wrapper over the unified parser
    ``link_context.extract_markdown_links`` (sprint EXTRACTLINKS-2026-06).
    The single parser fixes the legacy greedy regex on every axis: balanced
    parentheses (no overflow/truncation), ``<url>`` autolinks, image
    exclusion, and — via the new optional `base_url` — resolution of
    *relative* markdown links through ``urljoin``.

    Args:
        md_content: A string containing Markdown-formatted content.
        base_url: Source page URL; when given, relative targets are resolved
            to absolute (otherwise only absolute targets are kept).

    Returns:
        list: A list of URL strings (duplicates preserved; caller dedups).
    """
    return link_context.extract_markdown_links(md_content or "", base_url)


def add_expression(land: model.Land, url: str, depth=0) -> Union[model.Expression, bool]:
    """Add a new expression (URL) to a land or retrieve existing one.

    This function creates a new expression in the database if it doesn't exist,
    or retrieves the existing expression for the same URL in the land.

    Args:
        land: The Land object to add the expression to.
        url: The URL of the expression.
        depth: The crawl depth level (default 0 for seed URLs).

    Returns:
        Union[model.Expression, bool]: The Expression object if successfully added
            or retrieved, False if the URL is not crawlable.

    Notes:
        - Applies the full URL normalization pipeline (mwi.url_normalizer)
          before insertion. This includes anchor removal, recursive archive
          unwrapping, host lowercasing, tracker stripping and (if enabled)
          scheme / www / mobile-subdomain canonicalization.
        - When normalization changes the URL, the original form is recorded
          in `Expression.original_url` for provenance.
        - Checks if URL is crawlable using is_crawlable().
        - Creates or retrieves the associated Domain object.
        - Returns existing expression if URL already exists in the land.
        - Returns False for non-crawlable URLs (PDFs, images, etc.).
    """
    from mwi.url_normalizer import normalize_url
    raw_url = url
    url = normalize_url(url)
    if is_crawlable(url):
        domain_name = get_domain_name(url)
        domain = model.Domain.get_or_create(name=domain_name)[0]
        expression = model.Expression.get_or_none(
            model.Expression.url == url,
            model.Expression.land == land)
        if expression is None:
            expression = model.Expression.create(
                land=land,
                domain=domain,
                url=url,
                original_url=(raw_url if raw_url != url else None),
                depth=depth)
        return expression
    return False


def unwrap_archive_url(url: str) -> str:
    """Extract the original URL from an archive.org or ghostarchive.org URL.

    Backwards-compatible wrapper around mwi.url_normalizer._unwrap_archive
    (recursive). Kept for callers outside the canonical pipeline.
    """
    from mwi.url_normalizer import _unwrap_archive
    return _unwrap_archive(url)


def domain_from_url(url: str) -> str:
    """Extract the logical domain from a URL via the platform heuristics table.

    Unwraps archive URLs, then looks the host up in the platform heuristics
    table (``mwi.platform_heuristics.PLATFORM_HEURISTICS``, overridable via
    ``settings.platform_heuristics``). If the host matches a listed platform
    that has a ``url`` regex, the captured editorial entity is returned (e.g.
    ``youtube.com/@channel``, ``twitter.com/handle``); otherwise the bare
    netloc is returned. Hosts absent from the table always resolve to their
    netloc — only listed platforms are ever refined. Pure: no I/O.
    """
    unwrapped_url = unwrap_archive_url(url)
    netloc = urlparse(unwrapped_url).netloc
    rule = _platform_rule(_host_key(netloc))
    if rule and rule.get('url'):
        matches = re.findall(rule['url'], unwrapped_url)
        if matches:
            return _normalize_entity(matches[0], rule)
    return netloc


def _normalize_entity(entity: str, rule: dict) -> str:
    """Apply optional per-platform normalizations to a captured entity.

    - ``lower``: lowercase the entity — for platforms whose handle is
      case-insensitive (twitter/x, instagram...), so ``JLMelenchon`` and
      ``jlmelenchon`` collapse to one node.
    - ``alias``: replace the platform host with a canonical alias and drop any
      subdomain, unifying e.g. ``twitter.com`` / ``www.twitter.com`` / ``x.com``
      under a single ``x.com/<handle>`` node.
    - ``prefix``: prepend a fixed ``host/`` to a capture that is only the entity
      segment (used when a single regex must extract the author from two URL
      forms, e.g. ``blogs.mediapart.fr/<author>/blog`` and the legacy
      ``blogs.mediapart.fr/blog/<author>``).

    The captured entity is URL-decoded first (``unquote``) so a percent-encoded
    accented slug — e.g. ``s%C3%A9bastien-melenchon`` — becomes
    ``sébastien-melenchon`` instead of being truncated at the ``%``.
    """
    entity = unquote(entity)
    prefix = rule.get('prefix')
    if prefix:
        entity = prefix + entity
    if rule.get('lower'):
        entity = entity.lower()
    alias = rule.get('alias')
    if alias and '/' in entity:
        entity = alias + '/' + entity.split('/', 1)[1]
    return entity


# Backwards-compatible alias: get_domain_name is the historical name that
# link_expression, update_heuristic, normalize_pipeline and the controller
# still call. domain_from_url is the canonical name (sprint-heuristique).
get_domain_name = domain_from_url


# --- Platform heuristics table (sprint-heuristique) --------------------------
#
# One unified table, host suffix -> {"url": regex|None, "html": signal}. The URL
# rule extracts the editorial entity from the path (channel / handle / subreddit
# / author...) where it is present; the HTML signal recovers it from the page
# for opaque hosts (heuristic update --html), then re-resolves through the URL
# rule. Only hosts in the table are ever re-grouped by heuristic update; every
# other host keeps its bare netloc. The 163-entry default lives in
# mwi/platform_heuristics.py; override via settings.platform_heuristics.


def _platform_heuristics() -> dict:
    """Return the active platform heuristics table (settings override wins)."""
    override = getattr(settings, 'platform_heuristics', None)
    return override if override else _PLATFORM_HEURISTICS


def _host_key(netloc_or_url: str) -> str:
    """Lowercased host without a leading 'www.' (accepts a netloc or a URL)."""
    host = netloc_or_url
    if '/' in host:
        host = urlparse(host).netloc
    host = host.lower()
    return host[4:] if host.startswith('www.') else host


def _platform_rule(host: str):
    """Return the platform rule for a host (label-suffix match), or None."""
    if not host:
        return None
    for suffix, rule in _platform_heuristics().items():
        if host == suffix or host.endswith('.' + suffix):
            return rule
    return None


def _is_opaque(url: str) -> bool:
    """True when the URL host is a listed platform (in the heuristics table)."""
    return _platform_rule(_host_key(url)) is not None


def _opaque_platforms() -> frozenset:
    """Compat: the set of listed platform suffixes (tooling/tests)."""
    return frozenset(_platform_heuristics().keys())


# --- HTML signal extractors (sprint-heuristique) -----------------------------
# Each maps a declarative signal name (the "html" field of a platform rule) to a
# function that recovers the editorial URL that signal points to, resolved
# against base_url, or None. domain_from_html applies the ONE signal declared
# for the page's platform — no fixed cascade.

def _meta_url(soup: BeautifulSoup, key: str) -> str:
    """Return a meta tag's content by property= (og:) or name= (twitter:)."""
    tag = soup.find('meta', attrs={'property': key}) \
        or soup.find('meta', attrs={'name': key})
    if tag and tag.has_attr('content'):  # type: ignore
        content = tag['content']  # type: ignore
        if isinstance(content, str):
            return content.strip()
    return ""


def _ldjson_url(soup: BeautifulSoup, base_url: str, keys) -> Optional[str]:
    """First url/@id under any of ``keys`` (e.g. author, publisher) in JSON-LD.

    Walks every ``application/ld+json`` script, flattening ``@graph``. Resolves
    the result against ``base_url``. Never raises: malformed blocks are skipped.
    """
    def _url_of(node) -> Optional[str]:
        if isinstance(node, dict):
            for k in ('url', '@id'):
                v = node.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        elif isinstance(node, list):
            for item in node:
                found = _url_of(item)
                if found:
                    return found
        return None

    for script in soup.find_all('script', attrs={'type': 'application/ld+json'}):
        raw = script.string or script.get_text() or ""
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            continue
        nodes = data.get('@graph', [data]) if isinstance(data, dict) else data
        if isinstance(nodes, dict):
            nodes = [nodes]
        if not isinstance(nodes, list):
            continue
        for key in keys:
            for node in nodes:
                if isinstance(node, dict):
                    found = _url_of(node.get(key))
                    if found:
                        return urljoin(base_url, found)
    return None


def _sig_canonical(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    link = soup.find('link', attrs={'rel': 'canonical'})
    if link and link.has_attr('href'):  # type: ignore
        href = link['href']  # type: ignore
        if isinstance(href, str) and href.strip():
            return urljoin(base_url, href.strip())
    return None


def _sig_og_url(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    for key in ('og:url', 'twitter:url'):
        content = _meta_url(soup, key)
        if content:
            return urljoin(base_url, content)
    return None


def _sig_rel_author(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    a = soup.find('a', attrs={'rel': 'author'})
    if a and a.has_attr('href'):  # type: ignore
        href = a['href']  # type: ignore
        if isinstance(href, str) and href.strip():
            return urljoin(base_url, href.strip())
    return None


def _sig_ldjson_author(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    return _ldjson_url(soup, base_url, ('author',))


def _sig_ldjson_publisher(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    return _ldjson_url(soup, base_url, ('publisher',))


def _sig_itemprop_author(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    """Author URL from schema.org microdata (``itemprop``).

    YouTube ``/watch`` pages carry no JSON-LD; the channel lives in a
    ``<span itemprop="author"><link itemprop="url" href=".../@Channel"></span>``
    Person block. Reads the nested ``itemprop="url"`` href, falling back to the
    author element's own ``href`` (``<a itemprop="author" href=…>`` sites).
    """
    author = soup.find(attrs={'itemprop': 'author'})
    if author is None:
        return None
    link = author.find(attrs={'itemprop': 'url'})  # type: ignore
    for node in (link, author):
        if node is not None and node.has_attr('href'):  # type: ignore
            href = node['href']  # type: ignore
            if isinstance(href, str) and href.strip():
                return urljoin(base_url, href.strip())
    return None


def _sig_dailymotion_channel(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    """Dailymotion channel from the embedded SSR state JSON.

    Video pages return a 200 JS shell with no JSON-LD/canonical/microdata, but
    the uploader lives in a ``"channel":{…,"name":"<screenname>"}`` blob inside
    a ``<script>``. ``name`` is the URL slug (``dailymotion.com/<name>``), so
    the recovered URL re-resolves through the platform rule to the channel.
    """
    for script in soup.find_all('script'):
        text = script.string or ''
        if '"channel"' not in text:
            continue
        match = re.search(r'"channel":\{[^{}]*?"name":"([^"]+)"', text)
        if match:
            return urljoin(base_url, '/' + match.group(1))
    return None


# dc_creator / citation_author are name-only meta tags (an author NAME, not a
# URL). Without a URL to re-resolve they cannot refine the domain, so they fall
# back to canonical (the record URL) — which resolves to the host, i.e. no harm.
_SIGNAL_EXTRACTORS = {
    'canonical': _sig_canonical,
    'og_url': _sig_og_url,
    'rel_author': _sig_rel_author,
    'ldjson_author': _sig_ldjson_author,
    'ldjson_publisher': _sig_ldjson_publisher,
    'itemprop_author': _sig_itemprop_author,
    'dailymotion_channel': _sig_dailymotion_channel,
    'dc_creator': _sig_canonical,
    'citation_author': _sig_canonical,
}


def editorial_url_from_html(html: str, base_url: str,
                            signal: str = 'canonical') -> Optional[str]:
    """Recover the editorial URL a platform's declarative ``signal`` points to.

    Applies exactly ONE signal (the platform's ``html`` field) — not a fixed
    cascade — so each platform resolves via the signal the annex assigned it
    (a video's channel via ``ldjson_author``, a custom-domain blog via
    ``canonical``, etc.). Returns ``None`` when the signal yields nothing; the
    caller then falls back to the URL heuristic.
    """
    if not html:
        return None
    soup = BeautifulSoup(html, 'html.parser')
    extractor = _SIGNAL_EXTRACTORS.get(signal, _sig_canonical)
    return extractor(soup, base_url)


def domain_from_html(url: str, html: str) -> Optional[str]:
    """Resolve the logical domain from the page HTML.

    Looks up the page host's platform rule, applies its declared HTML signal via
    :func:`editorial_url_from_html` to recover a better editorial URL, then
    re-resolves it through :func:`domain_from_url`. Returns ``None`` when the
    host is unlisted or the signal yields nothing, so the caller can fall back
    to the URL heuristic.
    """
    rule = _platform_rule(_host_key(url))
    if not rule:
        return None
    editorial_url = editorial_url_from_html(html, url, rule.get('html', 'canonical'))
    return domain_from_url(editorial_url) if editorial_url else None


def _needs_html(url: str) -> bool:
    """True when HTML resolution could actually help for this URL.

    Two conditions: (1) the platform declares an HTML signal (not a subdomain
    host and not unlisted), and (2) the URL alone did NOT extract the entity —
    ``domain_from_url`` still returns the bare netloc. When a url rule already
    captured the entity (e.g. ``blogs.mediapart.fr/jacqueslancier``), the URL is
    authoritative and no fetch is needed.
    """
    rule = _platform_rule(_host_key(url))
    if not rule or not rule.get('html'):
        return False
    netloc = urlparse(unwrap_archive_url(url)).netloc
    return domain_from_url(url) == netloc


def resolve_domain(expression: model.Expression, *,
                   html: Optional[str] = None) -> str:
    """Resolve an expression's logical domain, HTML-aware for opaque hosts.

    Pure orchestrator — no network, no database. For a non-opaque host the URL
    heuristic is authoritative and no HTML is consulted. For an opaque host,
    when HTML is available (stored under ``--fullhtml`` on the expression, or
    fetched by the batch driver and passed as ``html``) the HTML signal takes
    precedence, falling back to the URL heuristic when it yields nothing. The
    caller owns fetching.
    """
    url = str(expression.url)
    url_domain = domain_from_url(url)
    if not _needs_html(url):
        # Entity already extracted from the URL (path/subdomain), no HTML signal,
        # or unlisted host: the URL resolution is authoritative.
        return url_domain
    effective_html = html if html is not None else getattr(expression, 'html', None)
    if not effective_html:
        return url_domain
    return domain_from_html(url, effective_html) or url_domain


def remove_anchor(url: str) -> str:
    """Remove the anchor (fragment) from a URL.

    This function strips the hash fragment identifier from a URL, returning
    only the base URL without the anchor portion.

    Args:
        url: The URL to process.

    Returns:
        str: The URL without the anchor portion, or the original URL if no anchor exists.

    Notes:
        - Removes everything after and including the '#' character.
        - Returns the original URL if no '#' is found.
        - Useful for deduplicating URLs that differ only by anchor.
    """
    anchor_pos = url.find('#')
    return url[:anchor_pos] if anchor_pos > 0 else url


def link_expression(land: model.Land, source_expression: model.Expression, url: str, *,
                    context: Optional[str] = None, dom: Optional[str] = None,
                    dom_html: Optional[str] = None) -> bool:
    """Create a link from a source expression to a target expression.

    This function adds a new expression for the target URL and creates a directed
    link in the expression graph from source to target.

    Args:
        land: The Land object containing both expressions.
        source_expression: The Expression object that contains the link.
        url: The URL of the target expression to link to.
        context: Markdown paragraph of the source readable containing the link
            (sprint link-context). Optional, keyword-only.
        dom: CSS-like path from the document root to the parent of the <a> tag.
            Optional, keyword-only.
        dom_html: outerHTML of the closest block ancestor of the <a> tag,
            truncated. Optional, keyword-only.

    Returns:
        bool: True if the link was successfully created, False otherwise.

    Notes:
        - Automatically creates target expression with depth = source.depth + 1.
        - Creates ExpressionLink record in the database.
        - Handles IntegrityError silently (link already exists).
        - Returns False if target URL is not crawlable.
        - Builds the directed graph structure for land crawling.
    """
    target_expression = add_expression(land, url, source_expression.depth + 1) # type: ignore
    if target_expression:
        try:
            model.ExpressionLink.create(
                source_id=source_expression.id, # type: ignore
                target_id=target_expression.id, # type: ignore
                context=context,
                dom=dom,
                dom_html=dom_html)
            return True
        except IntegrityError:
            pass
    return False


def is_crawlable(url: str):
    """Check whether a URL is valid and suitable for crawling.

    This function validates URLs to ensure they are HTTP/HTTPS and do not point
    to binary files or documents that should not be crawled.

    Args:
        url: The URL to validate.

    Returns:
        bool: True if the URL is crawlable, False otherwise.

    Notes:
        - Requires URL to start with 'http://' or 'https://'.
        - Excludes URLs ending with image extensions (.jpg, .png, etc.).
        - Excludes URLs ending with document extensions (.pdf, .doc, etc.).
        - Excludes URLs ending with data files (.csv, .xlsx, etc.).
        - Returns False for any URL that raises an exception during parsing.
    """
    try:
        if not url or not url.startswith(('http://', 'https://')):
            return False
        # Test the extension on the PATH only (not the whole URL): a query
        # string or fragment must not smuggle a binary past the filter
        # (…/doc.pdf?dl=1) nor make an editorial URL look binary (…/article#x).
        path = urlparse(url).path.lower()
        exclude_ext = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg',
                       '.ico', '.pdf', '.txt', '.csv', '.xls', '.xlsx', '.doc',
                       '.docx', '.ppt', '.pptx', '.zip', '.mp4', '.webm',
                       '.mp3', '.wav')
        return not path.endswith(exclude_ext)
    except Exception:
        return False


def extract_medias(content, expression: model.Expression):
    """Extract media references from HTML or Markdown content and save to database.

    This function identifies and extracts image, video, and audio URLs from content,
    resolves them to absolute URLs, and creates Media database records.

    Args:
        content: Either a BeautifulSoup object or string containing HTML/Markdown.
        expression: The Expression database object to associate media with.

    Notes:
        - Supports HTML <img>, <video>, and <audio> tags.
        - Supports Markdown image syntax ![alt](url).
        - Supports custom markers like [IMAGE: url], [VIDEO: url], [AUDIO: url].
        - Handles srcset attribute for responsive images.
        - Handles <source> elements within video/audio tags.
        - Resolves relative URLs to absolute URLs using expression.url.
        - Validates file extensions before creating media records.
        - Prevents duplicate media entries for the same expression and URL.
    """
    print(f"Extracting media from #{expression.id}") # type: ignore

    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg")

    def has_allowed_extension(url: str, extensions: tuple) -> bool:
        if not url:
            return False
        if url.lower().startswith('data:'):
            return True
        path_only = urlparse(url).path.lower()
        return any(path_only.endswith(ext) for ext in extensions)

    raw_representation = str(content)
    soup = content if hasattr(content, 'find_all') else BeautifulSoup(raw_representation, 'html.parser')

    collected_urls = {
        media.url for media in
        model.Media.select(model.Media.url).where(model.Media.expression == expression)
    }

    def register_media(raw_url: str, media_type: str):
        if not raw_url:
            return
        clean_url = raw_url.strip()

        if media_type == 'img' and not has_allowed_extension(clean_url, IMAGE_EXTENSIONS):
            return

        resolved_url = resolve_url(str(expression.url), clean_url)
        if resolved_url in collected_urls:
            return

        collected_urls.add(resolved_url)
        if not model.Media.select().where(
            (model.Media.expression == expression) &
            (model.Media.url == resolved_url)
        ).exists():
            media = model.Media.create(expression=expression, url=resolved_url, type=media_type)
            media.save()

    for tag in ['img', 'video', 'audio']:
        for element in soup.find_all(tag):
            primary_src = element.get('src')
            if primary_src:
                register_media(primary_src, tag)

            if tag == 'img':
                srcset = element.get('srcset')
                if srcset:
                    for candidate in srcset.split(','):
                        candidate_url = candidate.strip().split(' ')[0]
                        register_media(candidate_url, 'img')
            if tag in ('video', 'audio'):
                for source in element.find_all('source'):
                    register_media(source.get('src'), tag)

    markdown_text = raw_representation if raw_representation else soup.get_text(separator='\n')

    for match in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', markdown_text):
        register_media(match, 'img')

    for label, url in re.findall(r'\[(IMAGE|VIDEO|AUDIO):\s*([^\]]+)\]', markdown_text, flags=re.IGNORECASE):
        media_type = label.lower()
        if media_type == 'image':
            media_type = 'img'
        register_media(url, media_type)


def get_readable(content):
    """Extract readable text from HTML content while preserving media references.

    This function extracts clean text from HTML while converting media elements
    into text markers that can be later extracted.

    Args:
        content: A BeautifulSoup object containing HTML content.

    Returns:
        str: Clean text with media elements replaced by markers like [IMAGE: url].

    Notes:
        - Replaces <img> tags with [IMAGE: url] markers.
        - Replaces <video> tags with [VIDEO: url] markers.
        - Replaces <audio> tags with [AUDIO: url] markers.
        - Removes extra whitespace and empty lines.
        - Preserves URLs for media extraction via extract_medias().
        - Returns text with one non-empty line per line.
    """
    # Insérer des marqueurs pour les médias avant d'extraire le texte
    for tag, label in [('img', 'IMAGE'), ('video', 'VIDEO'), ('audio', 'AUDIO')]:
        for element in content.find_all(tag):
            src = element.get('src')
            if src:
                marker = f"[{label}: {src}]"
                # Remplacer la balise entière par le marqueur
                element.replace_with(marker)
    text = content.get_text(separator=' ')
    lines = text.split("\n")
    text_lines = [line.strip() for line in lines if len(line.strip()) > 0]
    return "\n".join(text_lines)


def clean_html(soup):
    """Remove non-valuable DOM elements from HTML for content extraction.

    This function removes elements that typically don't contain useful content
    like scripts, navigation, footers, etc.

    Args:
        soup: A BeautifulSoup object containing HTML to clean.

    Returns:
        BeautifulSoup: The same soup object with unwanted elements removed.

    Notes:
        - Removes script, style, iframe, and form tags.
        - Removes footer, nav, menu, social, and modal elements.
        - Modifies the soup object in place.
        - Useful for extracting main content before text analysis.
    """
    remove_selectors = ('script', 'style', 'iframe', 'form', 'footer', '.footer',
                        'nav', '.nav', '.menu', '.social', '.modal')
    for selector in remove_selectors:
        for tag in soup.select(selector):
            tag.decompose()


def get_land_dictionary(land: model.Land):
    """Retrieve the word dictionary associated with a land.

    This function fetches all Word objects that are part of a land's dictionary,
    used for relevance scoring and text analysis.

    Args:
        land: The Land object whose dictionary to retrieve.

    Returns:
        A Peewee query object containing Word records associated with the land.

    Notes:
        - Returns a query that can be further filtered or iterated.
        - Words are linked to lands via the LandDictionary join table.
        - Dictionary words are stemmed forms used for matching content.
    """
    return model.Word.select() \
        .join(model.LandDictionary, JOIN.LEFT_OUTER) \
        .where(model.LandDictionary.land == land)


def land_relevance(land: model.Land):
    """Calculate and update relevance scores for all expressions in a land.

    This function recalculates relevance scores for all non-null content expressions
    in a land based on the current land dictionary.

    Args:
        land: The Land object whose expressions should be scored.

    Notes:
        - Only processes expressions with non-null content.
        - Uses expression_relevance() to calculate individual scores.
        - Updates expression.relevance field in the database.
        - Relevance is based on dictionary word frequency in title and content.
        - Higher scores indicate better match with land's topic/theme.
    """
    words = get_land_dictionary(land)
    select = model.Expression.select() \
        .where(model.Expression.land == land, model.Expression.readable.is_null(False))
    row_count = select.count()
    if row_count > 0:
        print(f"Updating relevances for {row_count} expressions, it may take some time.")
        for expression in select:
            expression.relevance = expression_relevance(words, expression) # type: ignore
            expression.save()


def _resolve_text_lang(expression: model.Expression) -> str:
    """Resolve the language to use for tokenizing/stemming an expression.

    Sprint-multilang (D4). Returns the expression's detected language when it
    belongs to the land's accepted languages, otherwise the land's primary
    language. Always returns a base ISO 639-1 code (e.g. 'en', never 'en-US').
    """
    land_langs = [l.strip().lower() for l in
                  str(expression.land.lang or 'fr').split(',') if l.strip()]
    expr_lang = str(expression.lang or '').split('-')[0].strip().lower()
    if expr_lang:
        for land_lang in land_langs:
            if land_lang.split('-')[0] == expr_lang:
                return expr_lang
    return land_langs[0].split('-')[0] if land_langs else 'fr'


def expression_relevance(dictionary, expression: model.Expression) -> int:
    """Calculate expression relevance score based on land dictionary word matches.

    This function computes a weighted relevance score by counting dictionary word
    occurrences in the expression's title and readable content.

    Args:
        dictionary: A Peewee query result containing Word objects from the land dictionary.
        expression: The Expression object to score.

    Returns:
        int: The relevance score (title matches × 10 + content matches × 1).

    Notes:
        - Title matches are weighted 10x more than content matches.
        - Tokenization and stemming use the language resolved by
          `_resolve_text_lang` (expression language if accepted by the land,
          otherwise the land's primary language). Sprint-multilang.
        - The text is matched against ALL dictionary lemmas (multilingual
          union, see Word.lang).
        - Matches whole words only (word boundaries).
        - Returns 0 if title or readable content is missing.
        - Falls back to the unicode tokenizer if NLTK or the language's punkt
          model is unavailable.
    """
    lemmas = [w.lemma for w in dictionary]
    title_relevance = [0]
    content_relevance = [0]
    text_lang = _resolve_text_lang(expression)
    punkt_lang = _PUNKT_LANGS.get(text_lang)

    def get_relevance(text, weight) -> list:
        if not isinstance(text, str): # Ensure text is a string
            text = str(text)
        if _NLTK_OK and punkt_lang:
            tokens = word_tokenize(text, language=punkt_lang)
        else:
            tokens = _simple_word_tokenize(text)
        stems = [stem_word(w, text_lang) for w in tokens]
        stemmed_text = " ".join(stems)
        return [sum(weight for _ in re.finditer(r'\b%s\b' % re.escape(lemma), stemmed_text)) for lemma in lemmas]

    try:
        title_relevance = get_relevance(expression.title, 10) # type: ignore
        content_relevance = get_relevance(expression.readable, 1) # type: ignore
    except Exception as e:
        print(f"Error computing relevance: {e}")
        pass
    return sum(title_relevance) + sum(content_relevance)


def export_land(land: model.Land, export_type: str, minimum_relevance: int,
                fullhtml: bool = False):
    """Export land data to a file in the specified format.

    This function creates an export file containing land data filtered by
    minimum relevance threshold.

    Args:
        land: The Land object to export.
        export_type: The export format (e.g., 'pagecsv', 'pagegexf', 'corpus').
        minimum_relevance: Minimum relevance score filter for expressions.
        fullhtml: When True and export_type == 'nodelinkcsv', also emit the
            raw-HTML link network files (*fullhtml.csv). Ignored otherwise.

    Notes:
        - Output filename includes land name, export type, and timestamp.
        - Files are saved to settings.data_location directory.
        - Supports formats: pagecsv, fullpagecsv, nodecsv, pagegexf, nodegexf,
          mediacsv, corpus, pseudolinks, pseudolinkspage, pseudolinksdomain,
          nodelinkcsv, nodesjson (domain force-graph JSON), pagesjson (page
          force-graph JSON), htmldump.
        - Prints success message with record count or error message.
        - See Export class for format-specific details.
    """
    date_tag = model.datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = path.join(settings.data_location, 'export_land_%s_%s_%s') \
               % (land.name, export_type, date_tag)
    export = Export(export_type, land, minimum_relevance, fullhtml=fullhtml)
    count = export.write(export_type, filename)
    if count > 0:
        print("Successfully exported %s records to %s" % (count, filename))
    else:
        print("No records to export, check crawling state or lower minimum relevance threshold")


def export_tags(land: model.Land, export_type: str, minimum_relevance: int):
    """Export tag data to a CSV file.

    This function exports tag-related data from a land, such as tag co-occurrence
    matrix or tagged content.

    Args:
        land: The Land object to export tags from.
        export_type: The export format (e.g., 'matrix', 'content').
        minimum_relevance: Minimum relevance score filter for expressions.

    Notes:
        - Output filename includes land name, export type, and timestamp.
        - Always exports to CSV format.
        - Files are saved to settings.data_location directory.
        - Prints success or error message.
        - See Export class for tag export details.
    """
    date_tag = model.datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = path.join(settings.data_location, 'export_tags_%s_%s_%s.csv') \
               % (land.name, export_type, date_tag)
    export = Export(export_type, land, minimum_relevance)
    res = export.export_tags(filename)
    if res == 1:
        print("Successfully exported %s" % filename)
    else:
        print("Error exporting %s" % filename)


def update_heuristic(land: Optional[model.Land] = None, *,
                     use_html: bool = False, fetch_missing: bool = False,
                     limit: Optional[int] = None,
                     html_map: Optional[Dict[str, str]] = None,
                     only_listed: bool = True, dry_run: bool = False,
                     minrel: Optional[int] = None, chunk: int = 500) -> int:
    """Re-group expressions of LISTED platforms from the heuristics table.

    Scope / behaviour:

    - ``only_listed`` (default ``True``): only expressions whose host is a
      **listed platform** (in the platform heuristics table) are ever
      reassigned; every other host keeps its bare netloc, untouched. This is the
      safe default — it never re-baselines the whole corpus. Set ``False`` for a
      full recompute (used by the standalone reconstruction script).
    - ``land``: restrict to one land (``None`` = all expressions).
    - ``limit``: cap the number of expressions processed.
    - ``use_html``: for listed hosts, resolve via the page HTML
      (``Expression.html`` or a pre-fetched ``html_map``) instead of the URL.
    - ``html_map``: ``{url: html}`` of volatile HTML fetched by the caller (see
      :func:`fetch_missing_opaque_html`).
    - ``fetch_missing``: accepted for call symmetry; the network fetch is done by
      the caller and passed via ``html_map``.
    - ``dry_run``: report what WOULD change (count + sample) and write nothing.

    Writes are batched in chunked transactions (WAL single-writer friendly).
    Returns the number of expressions reassigned (or that would be, in dry-run).
    """
    html_map = html_map or {}
    query = model.Expression.select().order_by(model.Expression.id)
    if land is not None:
        query = query.where(model.Expression.land == land)
    if minrel is not None:
        query = query.where(model.Expression.relevance >= minrel)
    if limit:
        query = query.limit(limit)

    domains = {x['id']: x for x in model.Domain.select().dicts()}

    pending = []  # (expression, new_domain_name) for rows whose domain changed
    for expression in query:
        url = str(expression.url)
        listed = _is_opaque(url)
        if only_listed and not listed:
            continue  # heuristic touches only listed platforms — never global
        if use_html and listed:
            new_domain = resolve_domain(
                expression, html=(expression.html or html_map.get(url)))
        else:
            new_domain = domain_from_url(url)
        current = (domains.get(expression.domain_id) or {}).get('name')
        if new_domain != current:
            pending.append((expression, new_domain))

    # --- change report (bilan): old -> new transitions + samples ------------
    def _old_name(expr) -> Optional[str]:
        return (domains.get(expr.domain_id) or {}).get('name')

    verb = "would change (dry-run — nothing written)" if dry_run else "updated"
    print(f"{len(pending)} domain(s) {verb}")
    if pending:
        counts: Dict[tuple, int] = {}
        for expr, new_domain in pending:
            key = (_old_name(expr), new_domain)
            counts[key] = counts.get(key, 0) + 1
        top = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[:25]
        print("  transitions (old -> new : count):")
        for (old, new), n in top:
            print(f"    {n:>5}  {old!r} -> {new!r}")
        print(f"  samples ({min(20, len(pending))}):")
        for expr, new_domain in pending[:20]:
            print(f"    {_old_name(expr)!r} -> {new_domain!r}   {expr.url}")

    if dry_run:
        return len(pending)

    updated = 0
    dom_cache: Dict[str, model.Domain] = {}
    for start in range(0, len(pending), chunk):
        batch = pending[start:start + chunk]
        with model.DB.atomic():
            for expression, new_domain in batch:
                to_domain = dom_cache.get(new_domain)
                if to_domain is None:
                    to_domain, _ = model.Domain.get_or_create(name=new_domain)
                    dom_cache[new_domain] = to_domain
                expression.domain = to_domain
                expression.save()
                updated += 1
    return updated


async def fetch_missing_opaque_html(land: Optional[model.Land] = None,
                                    limit: Optional[int] = None,
                                    minrel: Optional[int] = None) -> Dict[str, str]:
    """Fetch HTML for opaque-host expressions that have no stored html.

    Selects expressions with ``html IS NULL`` on an opaque platform (optionally
    land-scoped, capped at ``limit`` fetches) and retrieves each through the
    full fetch cascade (aiohttp → curl_cffi → archive.org; Playwright only when
    opted in) over a single shared aiohttp session, bounded by
    ``settings.parallel_connections``. Returns ``{url: html}`` for successful
    fetches only. Network only — no DB writes — so it is safe to run before the
    write transaction of :func:`update_heuristic`. A per-URL failure is logged
    and skipped (never raises for a single URL).
    """
    query = (model.Expression
             .select()
             .where(model.Expression.html.is_null(True))
             .order_by(model.Expression.id))  # deterministic selection (JOSS)
    if land is not None:
        query = query.where(model.Expression.land == land)
    if minrel is not None:
        query = query.where(model.Expression.relevance >= minrel)
    targets = []
    for expression in query:
        url = str(expression.url)
        if _needs_html(url):  # only fetch where the URL didn't resolve the entity
            targets.append(url)
            if limit and len(targets) >= limit:
                break  # stop scanning early: no full-table walk on big lands
    if not targets:
        return {}

    html_map: Dict[str, str] = {}
    concurrency = max(1, int(settings.parallel_connections))
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        async def _one(u: str):
            async with semaphore:
                try:
                    result = await fetch_html(u, session=session)
                except Exception as exc:  # network resilience: log & continue
                    print(f"[heuristic] volatile fetch failed for {u}: {exc}")
                    return
                if result.html:
                    html_map[u] = result.html
        await asyncio.gather(*[_one(u) for u in targets])
    return html_map


def prune_orphan_expressions(land: model.Land, *, dry_run: bool = False,
                             maxrel: int = 0, chunk: int = 500):
    """Delete (or, in dry-run, count) uncrawled orphan expressions of a land.

    An orphan is an expression that is uncrawled (``fetched_at IS NULL``), not a
    seed (``depth > 0``, which also excludes ``depth IS NULL``), and has no
    incoming ``ExpressionLink``. Such an expression is an unreachable leaf in the
    crawl frontier: nothing points to it anymore, so keeping it in the queue is
    pointless. Because an uncrawled expression never has outgoing links (links are
    only created at crawl/readable time), deleting it creates no new orphan — a
    single pass is exact, no transitive closure is needed.

    In dry-run the orphan set is *projected* as it would be AFTER a pending
    ``--maxrel`` deletion (sources falling under ``maxrel`` are treated as already
    gone), so the preview equals the real run. The real run is meant to be called
    AFTER the ``--maxrel`` DELETE, when the cascade has already removed the deleted
    parents' links, so the plain "no incoming link" query is exact.

    Args:
        land: The land whose orphan expressions are considered.
        dry_run: When True, nothing is deleted; the projected set is only counted.
        maxrel: The ``--maxrel`` threshold used to project survivors in dry-run.
        chunk: Batch size for the chunked DELETE (SQLITE_MAX_VARIABLE_NUMBER).

    Returns:
        Tuple[int, list]: count of orphans (deleted, or projected in dry-run) and
        a sample of up to 20 ``(url, depth)`` tuples (empty list on the real run).
    """
    if dry_run:
        if maxrel > 0:
            doomed = ((model.Expression.relevance < maxrel)
                      & model.Expression.fetched_at.is_null(False))
            survivors = (model.Expression
                         .select(model.Expression.id)
                         .where((model.Expression.land == land) & ~doomed))
        else:
            survivors = (model.Expression
                         .select(model.Expression.id)
                         .where(model.Expression.land == land))
        inbound_ok = (model.ExpressionLink
                      .select(model.ExpressionLink.target_id)
                      .where(model.ExpressionLink.source_id.in_(survivors)))
        rows = list(model.Expression
                    .select()
                    .where((model.Expression.land == land)
                           & model.Expression.fetched_at.is_null(True)
                           & (model.Expression.depth > 0)
                           & model.Expression.id.not_in(inbound_ok)))
        return len(rows), [(str(r.url), r.depth) for r in rows[:20]]

    inbound = model.ExpressionLink.select(model.ExpressionLink.target_id)
    ids = [e.id for e in (model.Expression
                          .select(model.Expression.id)
                          .where((model.Expression.land == land)
                                 & model.Expression.fetched_at.is_null(True)
                                 & (model.Expression.depth > 0)
                                 & model.Expression.id.not_in(inbound)))]
    deleted = 0
    with model.DB.atomic():
        for start in range(0, len(ids), chunk):
            batch = ids[start:start + chunk]
            deleted += (model.Expression
                        .delete()
                        .where(model.Expression.id.in_(batch))
                        .execute())
    return deleted, []


def delete_media(land: model.Land, max_width: int = 0, max_height: int = 0, max_size: int = 0):
    """Delete all media associated with expressions in a land.

    This function removes all Media records linked to expressions in the specified
    land from the database.

    Args:
        land: The Land object whose media should be deleted.
        max_width: Unused parameter (kept for compatibility).
        max_height: Unused parameter (kept for compatibility).
        max_size: Unused parameter (kept for compatibility).

    Notes:
        - Deletes all media for all expressions in the land.
        - Currently ignores size/dimension filters.
        - Performs database DELETE operation (irreversible).
        - Does not delete actual media files, only database records.
    """
    expressions = model.Expression.select().where(model.Land == land)
    model.Media.delete().where(model.Media.expression << expressions)

async def medianalyse_land(land: model.Land,
                           depth: Optional[int] = None,
                           minrel: Optional[int] = None) -> dict:
    """Analyze all media items associated with expressions in a land.

    This async function processes all media in a land, extracting metadata and
    performing analysis using the MediaAnalyzer.

    Args:
        land: The Land object whose media should be analyzed.
        depth: When set, only analyze media of expressions with
            Expression.depth <= depth (sprint-multilang, B1).
        minrel: When set, only analyze media of expressions with
            Expression.relevance >= minrel (sprint-multilang, B1).

    Returns:
        dict: A dictionary containing analysis results with processed count.

    Notes:
        - Processes media from all expressions in the land asynchronously.
        - Uses MediaAnalyzer for comprehensive media analysis.
        - Extracts dimensions, colors, format, EXIF data, perceptual hashes.
        - Updates Media database records with analysis results.
        - Respects media filtering settings (min dimensions, max file size).
        - Returns statistics about the analysis process.
    """
    from .media_analyzer import MediaAnalyzer
    
    processed_count = 0
    
    async with aiohttp.ClientSession() as session:
        analyzer = MediaAnalyzer(session, {
            'user_agent': settings.user_agent,
            'min_width': settings.media_min_width,
            'min_height': settings.media_min_height,
            'max_file_size': settings.media_max_file_size,
            'download_timeout': settings.media_download_timeout,
            'max_retries': settings.media_max_retries,
            'analyze_content': settings.media_analyze_content,
            'extract_colors': settings.media_extract_colors,
            'extract_exif': settings.media_extract_exif,
            'n_dominant_colors': settings.media_n_dominant_colors
        })
        
        medias = model.Media.select().join(model.Expression).where(model.Expression.land == land)
        if depth is not None:
            medias = medias.where(model.Expression.depth <= depth)
        if minrel is not None:
            medias = medias.where(model.Expression.relevance >= minrel)

        for media in medias:
            print(f'Analyse de {media.url}')
            result = await analyzer.analyze_image(media.url)
            
            for field, value in result.items():
                if hasattr(media, field):
                    setattr(media, field, value)
            
            media.analyzed_at = model.datetime.datetime.now()
            media.save()
            processed_count += 1

    return {'processed': processed_count}
