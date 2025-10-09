"""
Core functions
"""
import os
import asyncio
import calendar
import json
import random
import re
import time
import shutil
import zipfile
from argparse import Namespace
from datetime import date, datetime, timedelta
from os import path
from typing import Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse, urljoin, quote, parse_qs

import aiohttp # type: ignore
import nltk # type: ignore
import requests
from bs4 import BeautifulSoup
from nltk.stem.snowball import FrenchStemmer # type: ignore
from nltk.tokenize import word_tokenize # type: ignore
from peewee import IntegrityError, JOIN, SQL
import trafilatura # type: ignore
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: Playwright not available. Dynamic media extraction will be skipped.")

import settings
from . import model
from .export import Export

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
    """Provide a simple fallback tokenizer for French when NLTK data is unavailable.

    This lightweight tokenizer extracts sequences of letters (including accented
    characters) from the input text and returns them as lowercase tokens.

    Args:
        text: The text to tokenize. Non-string inputs are converted to strings.

    Returns:
        list: A list of lowercase word tokens containing only letter sequences.

    Notes:
        - Used as a fallback when NLTK's punkt tokenizer is unavailable.
        - Supports basic Latin-1 accented characters (À-ÖØ-öø-ÿ).
        - Removes all non-letter characters (numbers, punctuation, etc.).
    """
    if not isinstance(text, str):
        text = str(text)
    # Keep only letter sequences (incl. basic Latin-1 accents)
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text.lower())


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
        async with async_playwright() as p:
            # Launch browser in headless mode
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Set user agent to match the one used in regular crawling
            await page.set_extra_http_headers({"User-Agent": settings.user_agent})
            
            # Navigate to the page
            await page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Wait for additional time to let dynamic content load
            await page.wait_for_timeout(3000)
            
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
            
            # Close browser
            await browser.close()
            
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


class SerpApiError(Exception):
    """Raised when a SerpAPI request or response fails."""


def fetch_serpapi_url_list(
    api_key: str,
    query: str,
    engine: str = 'google',
    lang: str = 'fr',
    datestart: Optional[str] = None,
    dateend: Optional[str] = None,
    timestep: str = 'week',
    sleep_seconds: float = 1.0,
    progress_hook: Optional[Callable[[Optional[date], Optional[date], int], None]] = None
) -> List[Dict[str, Optional[Union[str, int]]]]:
    """Query SerpAPI for organic results and return URL metadata.

    The function mirrors the behaviour of the original R helper: it optionally
    walks a date range in fixed windows, paginates through search result pages,
    and collects the ``organic_results`` payload.

    Args:
        api_key: SerpAPI secret key (required).
        query: Search query sent to SerpAPI.
        engine: SerpAPI engine used to fetch results (``google|bing|duckduckgo``).
        lang: Language code that maps to the engine-specific locale parameters.
        datestart: Optional lower bound (``YYYY-MM-DD``) for the search window.
        dateend: Optional upper bound (``YYYY-MM-DD``) for the search window.
        timestep: Window size when iterating between ``datestart`` and
            ``dateend`` (``day`` | ``week`` | ``month``).
        sleep_seconds: Base delay between HTTP calls to avoid rate limits.
        progress_hook: Optional callable invoked after each date window with
            the start date, end date and number of fetched results.

    Returns:
        A list of dictionaries containing ``position``, ``title``, ``link`` and
        ``date`` keys extracted from the SerpAPI response.

    Raises:
        SerpApiError: if the query is empty, the date range is invalid, or the
            HTTP request/response is not usable.
    """

    normalized_query = (query or '').strip()
    if not normalized_query:
        raise SerpApiError('Query must be a non-empty string')

    engine = (engine or 'google').strip().lower() or 'google'
    allowed_engines = {'google', 'bing', 'duckduckgo'}
    if engine not in allowed_engines:
        raise SerpApiError(f'Unsupported SerpAPI engine "{engine}"')

    lang = (lang or 'fr').strip().lower() or 'fr'
    timestep = (timestep or 'week').strip().lower() or 'week'

    if bool(datestart) ^ bool(dateend):
        raise SerpApiError('Both datestart and dateend must be provided together')

    date_capable_engines = {'google', 'duckduckgo'}
    if (datestart or dateend) and engine not in date_capable_engines:
        raise SerpApiError('Date filtering is only supported with the google or duckduckgo engines')

    normalized_start: Optional[date] = None
    normalized_end: Optional[date] = None
    if datestart and dateend:
        normalized_start = _parse_serpapi_date(datestart)
        normalized_end = _parse_serpapi_date(dateend)
        if normalized_start > normalized_end:
            raise SerpApiError('datestart must be earlier than or equal to dateend')

    date_windows: List[Tuple[Optional[date], Optional[date]]] = []
    if engine in date_capable_engines and normalized_start and normalized_end:
        date_windows = list(_build_serpapi_windows(datestart, dateend, timestep))
    if not date_windows:
        date_windows = [(normalized_start, normalized_end)]  # Always run at least once.

    aggregated: List[Dict[str, Optional[Union[str, int]]]] = []

    base_url = getattr(settings, 'serpapi_base_url', 'https://serpapi.com/search')
    timeout = getattr(settings, 'serpapi_timeout', 15)
    jitter_floor, jitter_ceil = 0.8, 1.2
    page_size = _serpapi_page_size(engine)

    for window_start, window_end in date_windows:
        # Pagination resets for every date window so we can cover the full range.
        start_index = 0
        window_count = 0
        while True:
            params: Dict[str, Union[str, int]] = {
                'api_key': api_key,
                'engine': engine,
                'q': normalized_query,
            }
            params.update(_build_serpapi_params(
                engine,
                lang,
                start_index,
                page_size,
                window_start=window_start,
                window_end=window_end,
                use_date_filter=bool(window_start and window_end)
            ))

            if engine == 'google' and window_start and window_end:
                # Google accepts the date constraint through the tbs parameter.
                params['tbs'] = _build_serpapi_tbs(window_start, window_end)

            try:
                response = requests.get(base_url, params=params, timeout=timeout)
            except requests.RequestException as exc:  # pragma: no cover - network failure
                raise SerpApiError(f'HTTP error during SerpAPI request: {exc}') from exc

            if response.status_code != 200:
                snippet = response.text[:200]
                raise SerpApiError(
                    f'SerpAPI request failed with status {response.status_code}: {snippet}'
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise SerpApiError('Invalid JSON payload returned by SerpAPI') from exc

            if 'error' in payload:
                message = str(payload.get('error', '')).strip()
                lowered = message.lower()
                if engine == 'duckduckgo' and "hasn't returned any results" in lowered:
                    # DuckDuckGo uses an error payload when no results exist for the window.
                    break
                raise SerpApiError(f"SerpAPI error: {message}")

            organic_results = payload.get('organic_results') or []
            if not organic_results:
                break

            for entry in organic_results:
                # Keep a compact structure: position/title/link/date match the R helper output.
                aggregated.append({
                    'position': entry.get('position'),
                    'title': entry.get('title'),
                    'link': entry.get('link'),
                    'date': entry.get('date'),
                })
                window_count += 1

            serp_pagination = payload.get('serpapi_pagination') or {}
            next_link = serp_pagination.get('next_link') or serp_pagination.get('next')
            has_next_page = bool(next_link)

            if not has_next_page:
                break

            next_offset_raw = serp_pagination.get('next_offset')
            next_index: Optional[int] = None
            if next_offset_raw is not None:
                try:
                    next_index = int(next_offset_raw)
                except (TypeError, ValueError):
                    next_index = None

            if next_index is None and isinstance(next_link, str):
                try:
                    parsed = urlparse(next_link)
                    query_params = parse_qs(parsed.query)
                except Exception:  # pragma: no cover - defensive
                    query_params = {}

                for key in ('start', 'first', 'offset'):
                    values = query_params.get(key)
                    if not values:
                        continue
                    try:
                        candidate = int(values[0])
                    except (TypeError, ValueError):
                        continue
                    if candidate > start_index:
                        next_index = candidate
                        break

            if next_index is not None and next_index > start_index:
                start_index = next_index
                continue

            increment = len(organic_results)
            if increment <= 0:
                break
            start_index += increment

            # Light jitter mirrors the original R helper and reduces rate-limit risks.
            effective_sleep = max(0.0, float(sleep_seconds)) * random.uniform(jitter_floor, jitter_ceil)
            if effective_sleep > 0:
                time.sleep(effective_sleep)

        if progress_hook:
            progress_hook(window_start, window_end, window_count)

    return aggregated


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


def _serpapi_page_size(engine: str) -> int:
    """Return the maximum page size for a given SerpAPI search engine.

    Different search engines support different maximum result counts per page.
    This function returns the appropriate page size for each engine.

    Args:
        engine: The search engine name (e.g., 'google', 'bing', 'duckduckgo').

    Returns:
        int: The maximum page size (100 for Google, 50 for others).

    Notes:
        - Google supports up to 100 results per page.
        - Other engines (Bing, DuckDuckGo) default to 50 results per page.
    """
    if engine == 'google':
        return 100
    return 50


def _build_serpapi_params(
    engine: str,
    lang: str,
    start_index: int,
    page_size: int,
    window_start: Optional[date] = None,
    window_end: Optional[date] = None,
    use_date_filter: bool = False
) -> Dict[str, Union[str, int]]:
    """Build engine-specific query parameters for SerpAPI requests.

    This function constructs the appropriate query parameters for different search
    engines (Google, Bing, DuckDuckGo) with language settings and pagination.

    Args:
        engine: The search engine name ('google', 'bing', or 'duckduckgo').
        lang: Language code for localization (e.g., 'fr', 'en').
        start_index: The starting index for pagination (0-based).
        page_size: Maximum number of results per page.
        window_start: Optional start date for date-filtered searches.
        window_end: Optional end date for date-filtered searches.
        use_date_filter: Whether date filtering is being applied.

    Returns:
        Dict[str, Union[str, int]]: A dictionary of query parameters specific to
            the chosen search engine.

    Notes:
        - Google uses 'start', 'num', 'gl', 'hl', 'lr' parameters.
        - Bing uses 'mkt', 'count', 'first' parameters (1-indexed).
        - DuckDuckGo uses 'kl', 'start', 'm', 'df' parameters.
        - Date filters are only applied for DuckDuckGo via 'df' parameter.
    """
    normalized_lang = (lang or 'fr').strip().lower() or 'fr'

    if engine == 'google':
        params: Dict[str, Union[str, int]] = {
            'google_domain': _serpapi_google_domain(normalized_lang),
            'gl': normalized_lang,
            'hl': normalized_lang,
            'lr': f'lang_{normalized_lang}',
            'safe': 'off',
            'start': start_index,
        }
        if not use_date_filter:
            params['num'] = page_size
        return params

    if engine == 'bing':
        return {
            'mkt': _serpapi_bing_market(normalized_lang),
            'count': page_size,
            'first': start_index + 1,
        }

    if engine == 'duckduckgo':
        params = {
            'kl': _serpapi_duckduckgo_region(normalized_lang),
            'start': start_index,
            'm': page_size,
        }
        if window_start and window_end:
            params['df'] = f"{window_start.isoformat()}..{window_end.isoformat()}"
        return params

    return {}


def _build_serpapi_windows(
    datestart: Optional[str],
    dateend: Optional[str],
    timestep: str
) -> List[Tuple[date, date]]:
    """Generate inclusive date windows for time-based search iteration.

    This function splits a date range into smaller windows based on the specified
    timestep, matching the behavior of the original R helper implementation.

    Args:
        datestart: Start date in YYYY-MM-DD format, or None.
        dateend: End date in YYYY-MM-DD format, or None.
        timestep: Window size ('day', 'week', or 'month').

    Returns:
        List[Tuple[date, date]]: A list of (start_date, end_date) tuples representing
            inclusive date windows, or empty list if dates are not provided.

    Raises:
        SerpApiError: If datestart is later than dateend.

    Notes:
        - Returns empty list if either datestart or dateend is None.
        - Each window is inclusive of both start and end dates.
        - Last window may be shorter to fit within the overall date range.
    """

    if not datestart or not dateend:
        return []

    start_date = _parse_serpapi_date(datestart)
    end_date = _parse_serpapi_date(dateend)
    if start_date > end_date:
        raise SerpApiError('datestart must be earlier than or equal to dateend')

    current_start = start_date
    step = timestep.lower()
    windows: List[Tuple[date, date]] = []

    while current_start <= end_date:
        next_start = _advance_date(current_start, step)
        window_end = min(end_date, next_start - timedelta(days=1))
        windows.append((current_start, window_end))
        current_start = next_start

    return windows


def _advance_date(current: date, timestep: str) -> date:
    """Advance a date by the specified timestep increment.

    This function adds a time increment to the current date based on the timestep
    parameter, handling month boundaries correctly.

    Args:
        current: The date to advance.
        timestep: The increment type ('day', 'week', or 'month').

    Returns:
        date: The advanced date.

    Raises:
        SerpApiError: If timestep is not one of 'day', 'week', or 'month'.

    Notes:
        - 'day' advances by 1 day.
        - 'week' advances by 7 days.
        - 'month' advances by 1 month, handling month-end edge cases correctly.
    """

    if timestep == 'day':
        return current + timedelta(days=1)
    if timestep == 'week':
        return current + timedelta(weeks=1)
    if timestep == 'month':
        year = current.year + (current.month // 12)
        month = current.month % 12 + 1
        day = min(current.day, calendar.monthrange(year, month)[1])
        return date(year, month, day)
    raise SerpApiError('timestep must be one of: day, week, month')


def _parse_serpapi_date(value: str) -> date:
    """Parse a YYYY-MM-DD string into a date object and validate the format.

    This function converts a date string in ISO format (YYYY-MM-DD) to a Python
    date object, raising an error if the format is invalid.

    Args:
        value: A date string in YYYY-MM-DD format.

    Returns:
        date: The parsed date object.

    Raises:
        SerpApiError: If the date string is not in valid YYYY-MM-DD format.

    Notes:
        - Only accepts ISO format (YYYY-MM-DD).
        - Validates both format and date validity (e.g., rejects 2023-02-30).
    """

    try:
        return datetime.strptime(value, '%Y-%m-%d').date()
    except ValueError as exc:
        raise SerpApiError(f'Invalid date "{value}" — expected YYYY-MM-DD') from exc


def _build_serpapi_tbs(start: date, end: date) -> str:
    """Build the Google tbs parameter encoding a closed date range.

    This function constructs the Google-specific 'tbs' (to-be-searched) parameter
    that encodes a custom date range for search filtering.

    Args:
        start: The start date of the range (inclusive).
        end: The end date of the range (inclusive).

    Returns:
        str: A formatted tbs parameter string for Google Search API.

    Notes:
        - Format: 'cdr:1,cd_min:MM/DD/YYYY,cd_max:MM/DD/YYYY'
        - 'cdr:1' enables custom date range filtering.
        - Dates are formatted as MM/DD/YYYY for Google's API.
    """

    return 'cdr:1,cd_min:{},cd_max:{}'.format(
        start.strftime('%m/%d/%Y'),
        end.strftime('%m/%d/%Y')
    )


def _serpapi_google_domain(lang: str) -> str:
    """Return a Google domain TLD matching the language code.

    This function maps language codes to appropriate Google domain TLDs for
    localized search results.

    Args:
        lang: A language code (e.g., 'fr', 'en').

    Returns:
        str: The Google domain TLD (e.g., 'google.fr', 'google.com').

    Notes:
        - 'fr' maps to 'google.fr' for French results.
        - 'en' maps to 'google.com' for English results.
        - Defaults to 'google.com' for unrecognized language codes.
    """

    lang = lang.lower()
    if lang == 'fr':
        return 'google.fr'
    if lang == 'en':
        return 'google.com'
    return 'google.com'


def _serpapi_bing_market(lang: str) -> str:
    """Return the Bing market code matching the requested language.

    This function maps language codes to Bing market identifiers for localized
    search results.

    Args:
        lang: A language code (e.g., 'fr', 'en').

    Returns:
        str: The Bing market code (e.g., 'fr-FR', 'en-US').

    Notes:
        - 'fr' maps to 'fr-FR' for French market.
        - 'en' maps to 'en-US' for English/US market.
        - Defaults to 'en-US' for unrecognized language codes.
    """

    mapping = {
        'fr': 'fr-FR',
        'en': 'en-US',
    }
    return mapping.get(lang, 'en-US')


def _serpapi_duckduckgo_region(lang: str) -> str:
    """Return the DuckDuckGo region code for the given language.

    This function maps language codes to DuckDuckGo region identifiers for
    localized search results.

    Args:
        lang: A language code (e.g., 'fr', 'en').

    Returns:
        str: The DuckDuckGo region code (e.g., 'fr-fr', 'us-en').

    Notes:
        - 'fr' maps to 'fr-fr' for French region.
        - 'en' maps to 'us-en' for US English region.
        - Defaults to 'us-en' for unrecognized language codes.
    """

    mapping = {
        'fr': 'fr-fr',
        'en': 'us-en',
    }
    return mapping.get(lang, 'us-en')


def fetch_seorank_for_url(url: str, api_key: str) -> Optional[dict]:
    """Call the SEO Rank API for a single URL and return the JSON payload.

    This function makes an HTTP request to the SEO Rank API to fetch metrics
    (Moz, SimilarWeb, Facebook) for a given URL.

    Args:
        url: The URL to fetch SEO metrics for.
        api_key: The API key for authenticating with the SEO Rank service.

    Returns:
        Optional[dict]: A dictionary containing the JSON response from the API,
            or None if the request fails.

    Notes:
        - Uses URL-safe encoding for the URL parameter.
        - Default base URL is 'https://seo-rank.my-addr.com/api2/moz+sr+fb'.
        - Configurable via settings.seorank_api_base_url.
        - Timeout is configurable via settings.seorank_timeout (default 15s).
        - Prints error messages for HTTP failures or JSON parsing errors.
    """
    base_url = getattr(settings, 'seorank_api_base_url', '').strip()
    if not base_url:
        base_url = 'https://seo-rank.my-addr.com/api2/moz+sr+fb'

    # API expects the raw URL path; keep common URL separators unescaped
    safe_url = quote(url, safe=':/?&=%')
    request_url = f"{base_url.rstrip('/')}/{api_key}/{safe_url}"
    try:
        response = requests.get(request_url, timeout=getattr(settings, 'seorank_timeout', 15))
    except requests.RequestException as exc:
        print(f"[seorank] HTTP error for {url}: {exc}")
        return None

    if response.status_code != 200:
        print(f"[seorank] Unexpected status {response.status_code} for {url}")
        return None

    try:
        return response.json()
    except ValueError as exc:
        snippet = response.text[:120].replace('\n', ' ')
        print(f"[seorank] JSON decoding failed for {url}: {exc} (body preview: {snippet})")
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
    SEO Rank API, and stores the results in the database.

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

    processed = 0
    updated = 0
    sleep_seconds = max(0.0, float(getattr(settings, 'seorank_request_delay', 0)))

    for expression in expressions:
        processed += 1
        payload = fetch_seorank_for_url(str(expression.url), api_key)
        if payload is not None:
            expression.seorank = json.dumps(payload)
            expression.save(only=[model.Expression.seorank])
            updated += 1
        if sleep_seconds:
            time.sleep(sleep_seconds)

    return processed, updated


def stem_word(word: str) -> str:
    """Stem a word using NLTK Snowball FrenchStemmer.

    This function reduces a French word to its root form (stem) using the NLTK
    Snowball stemmer for French. The stemmer is cached as a function attribute.

    Args:
        word: The word to stem.

    Returns:
        str: The stemmed word in lowercase.

    Notes:
        - Uses NLTK's Snowball French Stemmer.
        - The stemmer instance is cached in the function's attributes.
        - Input is automatically converted to lowercase before stemming.
        - Designed specifically for French language processing.
    """
    if not hasattr(stem_word, "stemmer"):
        setattr(stem_word, "stemmer", FrenchStemmer())
    # The following line uses getattr which is safe
    return str(getattr(stem_word, "stemmer").stem(word.lower()))


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
            # Trafilatura's fetch_url tries to find a working scheme
            downloaded = trafilatura.fetch_url(domain_url_https)
            if not downloaded: # Try HTTP if HTTPS failed
                 downloaded = trafilatura.fetch_url(domain_url_http)

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


async def crawl_land(land: model.Land, limit: int = 0, http: Optional[str] = None, depth: Optional[int] = None) -> tuple:
    """Asynchronously crawl all expressions in a land.

    This function orchestrates the crawling process for a land, processing
    expressions depth by depth with concurrent HTTP requests and media analysis.

    Args:
        land: The Land object to crawl.
        limit: Maximum number of expressions to crawl (0 for unlimited).
        http: Optional HTTP status filter for recrawling specific expressions.
        depth: Optional depth filter to crawl only expressions at this depth level.

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

    # If depth is specified, only process that depth
    if depth is not None:
        depths_to_process = [depth]
    else:
        # Get distinct depths in ascending order for expressions not yet fetched or matching http filter
        if http is None:
            depths_query = model.Expression.select(model.Expression.depth).where(
                model.Expression.land == land,
                model.Expression.fetched_at.is_null(True)
            ).distinct().order_by(model.Expression.depth)
        else:
            depths_query = model.Expression.select(model.Expression.depth).where(
                model.Expression.land == land,
                model.Expression.http_status == http
            ).distinct().order_by(model.Expression.depth)
        depths_to_process = [d.depth for d in depths_query]

    for current_depth in depths_to_process:
        print(f"Processing depth {current_depth}")

        if http is None:
            expressions = model.Expression.select().where(
                model.Expression.land == land,
                model.Expression.fetched_at.is_null(True),
                model.Expression.depth == current_depth
            )
        else:
            expressions = model.Expression.select().where(
                model.Expression.land == land,
                model.Expression.http_status == http,
                model.Expression.depth == current_depth
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
            async with aiohttp.ClientSession(connector=connector) as session:
                tasks = [
                    crawl_expression_with_media_analysis(expr, dictionary, session)
                    for expr in current_batch_expressions
                ]
                results = await asyncio.gather(*tasks)
                processed_in_batch = sum(results)
                total_processed += processed_in_batch
                total_errors += (attempted_in_batch - processed_in_batch)
                total_attempted += attempted_in_batch

            current_offset += attempted_in_batch

            if limit > 0 and total_attempted >= limit:
                return total_processed, total_errors

    return total_processed, total_errors

async def crawl_expression_with_media_analysis(expression: model.Expression, dictionary, session: aiohttp.ClientSession):
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
    content = None
    raw_html = None
    links = []
    status_code_str = "000"  # Default to client error
    expression.fetched_at = model.datetime.datetime.now() # type: ignore

    # Step 1: Direct HTTP request to get status and content
    try:
        async with session.get(expression.url,
                               headers={"User-Agent": settings.user_agent},
                               timeout=aiohttp.ClientTimeout(total=15)) as response:
            status_code_str = str(response.status)
            if response.status == 200 and 'html' in response.headers.get('content-type', ''):
                raw_html = await response.text()
            else:
                print(f"Direct request for {expression.url} returned status {status_code_str}")

    except aiohttp.ClientError as e:
        print(f"ClientError for {expression.url}: {e}. Status: 000.")
        status_code_str = "000"
    except Exception as e:
        print(f"Generic exception during initial fetch for {expression.url}: {e}")
        status_code_str = "ERR"

    expression.http_status = str(status_code_str) # type: ignore

    # Step 2: Try to extract content if we got HTML from the direct request
    if raw_html:
        # 2a. Trafilatura on the fetched HTML
        try:
            extracted_content = trafilatura.extract(raw_html, include_links=True, include_comments=False, include_images=True, output_format='markdown')
            readable_html = trafilatura.extract(raw_html, include_links=True, include_comments=False, include_images=True, output_format='html')
            if extracted_content and len(extracted_content) > 100:
                # Extraction des médias du readable HTML (corps du texte)
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
                # Enregistrer les médias du readable dans la table Media
                # 1. Depuis le HTML (si balises <img> présentes)
                if readable_html:
                    soup_readable = BeautifulSoup(readable_html, 'html.parser')
                    extract_medias(soup_readable, expression)
                # 2. Depuis le markdown (pour les images converties en markdown)
                img_md_links = re.findall(r'!\[.*?\]\((.*?)\)', content)
                for img_url in img_md_links:
                    # Résoudre l'URL relative en URL absolue
                    resolved_img_url = resolve_url(str(expression.url), img_url)
                    # Vérifier si déjà présent (éviter doublons)
                    if not model.Media.select().where((model.Media.expression == expression) & (model.Media.url == resolved_img_url)).exists():
                        model.Media.create(expression=expression, url=resolved_img_url, type='img')
                links = extract_md_links(content)
                expression.readable = content # type: ignore
                print(f"Trafilatura succeeded on fetched HTML for {expression.url}")
        except Exception as e:
            print(f"Trafilatura failed on raw HTML for {expression.url}: {e}")

        # 2b. BeautifulSoup as a fallback on the same HTML
        if not content:
            try:
                soup = BeautifulSoup(raw_html, 'html.parser')
                clean_html(soup)
                text_content = get_readable(soup)
                if text_content and len(text_content) > 100:
                    content = text_content
                    urls = [a.get('href') for a in soup.find_all('a') if is_crawlable(a.get('href'))]
                    links = urls
                    expression.readable = content # type: ignore
                    print(f"BeautifulSoup fallback succeeded for {expression.url}")
            except Exception as e:
                print(f"BeautifulSoup fallback failed for {expression.url}: {e}")

    # Step 3: If no content yet (e.g., non-200 status, or parsing failed), try URL-based fallbacks
    if not content:
        # 3b. Archive.org (if Mercury also fails)
        if not content:
            try:
                print(f"Trying URL-based fallback: archive.org for {expression.url}")
                archive_data_url = f"http://archive.org/wayback/available?url={expression.url}"
                archive_response = await asyncio.to_thread(lambda: requests.get(archive_data_url, timeout=10))
                archive_response.raise_for_status()
                archive_data = archive_response.json()
                archived_url = archive_data.get('archived_snapshots', {}).get('closest', {}).get('url')
                if archived_url:
                    downloaded = await asyncio.to_thread(trafilatura.fetch_url, archived_url)
                    if downloaded:
                        raw_html = downloaded
                        extracted_content = trafilatura.extract(downloaded, include_links=True, include_images=True, output_format='markdown')
                        readable_html = trafilatura.extract(downloaded, include_links=True, include_images=True, output_format='html')
                        if extracted_content and len(extracted_content) > 100:
                            # Extraction des médias du readable HTML (corps du texte archivé)
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
                            # Enregistrer les médias du readable archivé dans la table Media
                            # 1. Depuis le HTML (si balises <img> présentes)
                            if readable_html:
                                soup_readable = BeautifulSoup(readable_html, 'html.parser')
                                extract_medias(soup_readable, expression)
                            # 2. Depuis le markdown (pour les images converties en markdown)
                            img_md_links = re.findall(r'!\[.*?\]\((.*?)\)', content)
                            for img_url in img_md_links:
                                # Résoudre l'URL relative en URL absolue
                                resolved_img_url = resolve_url(str(expression.url), img_url)
                                if not model.Media.select().where((model.Media.expression == expression) & (model.Media.url == resolved_img_url)).exists():
                                    model.Media.create(expression=expression, url=resolved_img_url, type='img')
                            links = extract_md_links(content)
                            expression.readable = content # type: ignore
                            print(f"Archive.org + Trafilatura succeeded for {expression.url}")
            except Exception as e:
                print(f"Archive.org fallback failed for {expression.url}: {e}")

    # Final processing and saving
    if content:
        soup = BeautifulSoup(raw_html if raw_html else content, 'html.parser')
        expression.title = str(get_title(soup) or expression.url) # type: ignore
        expression.description = str(get_description(soup)) if get_description(soup) else None # type: ignore
        expression.keywords = str(get_keywords(soup)) if get_keywords(soup) else None # type: ignore
        expression.lang = str(soup.html.get('lang', '')) if soup.html else '' # type: ignore
        # Compute relevance with OpenRouter gate when enabled
        try:
            from .llm_openrouter import is_relevant_via_openrouter  # local import to avoid overhead when disabled
            if getattr(settings, 'openrouter_enabled', False) and settings.openrouter_api_key and settings.openrouter_model:
                verdict = is_relevant_via_openrouter(expression.land, expression)  # type: ignore
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
            for link in links:
                link_expression(expression.land, expression, link) # type: ignore
        expression.save()
        return 1
    else:
        print(f"All extraction methods failed for {expression.url}. Final status: {expression.http_status}")
        expression.save()
        return 0

async def consolidate_land(
    land: model.Land,
    limit: int = 0,
    depth: Optional[int] = None,
    min_relevance: int = 0,
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
        - Deletes and recreates all expression links from content.
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

                # 2. Recalculer la relevance sans appel OpenRouter
                try:
                    expr.relevance = expression_relevance(dictionary, expr)  # type: ignore
                except Exception as e:
                    print(f"Error recalculating relevance for {expr.url}: {e}")
                    expr.relevance = expr.relevance or 0  # type: ignore
                expr.save()

                # 3. Extraire les liens sortants du contenu lisible
                links = []
                if expr.readable:
                    # Extraction des liens markdown
                    links = extract_md_links(expr.readable)
                    # Extraction des liens HTML (fallback)
                    soup = BeautifulSoup(expr.readable, 'html.parser')
                    urls = [a.get('href') for a in soup.find_all('a') if is_crawlable(a.get('href'))]
                    links += [u for u in urls if u and u not in links]
                nb_links = len(set(links))

                # 4. Ajouter les documents manquants et recréer les liens
                for url in set(links):
                    if is_crawlable(url):
                        target_expr = add_expression(land, url, expr.depth + 1 if expr.depth is not None else 1)
                        if target_expr:
                            try:
                                model.ExpressionLink.create(
                                    source_id=expr.id, # type: ignore
                                    target_id=target_expr.id) # type: ignore
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


async def crawl_expression(expression: model.Expression, dictionary, session: aiohttp.ClientSession):
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
    content = None
    raw_html = None
    links = []
    status_code_str = "000"  # Default to client error
    expression.fetched_at = model.datetime.datetime.now() # type: ignore

    # Step 1: Direct HTTP request to get status and content
    try:
        async with session.get(expression.url,
                               headers={"User-Agent": settings.user_agent},
                               timeout=aiohttp.ClientTimeout(total=15)) as response:
            status_code_str = str(response.status)
            if response.status == 200 and 'html' in response.headers.get('content-type', ''):
                raw_html = await response.text()
            else:
                print(f"Direct request for {expression.url} returned status {status_code_str}")

    except aiohttp.ClientError as e:
        print(f"ClientError for {expression.url}: {e}. Status: 000.")
        status_code_str = "000"
    except Exception as e:
        print(f"Generic exception during initial fetch for {expression.url}: {e}")
        status_code_str = "ERR"

    expression.http_status = str(status_code_str) # type: ignore

    # Step 2: Try to extract content if we got HTML from the direct request
    if raw_html:
        # 2a. Trafilatura on the fetched HTML
        try:
            extracted_content = trafilatura.extract(raw_html, include_links=True, include_comments=False, include_images=True, output_format='markdown')
            readable_html = trafilatura.extract(raw_html, include_links=True, include_comments=False, include_images=True, output_format='html')
            if extracted_content and len(extracted_content) > 100:
                # Extraction des médias du readable HTML (corps du texte)
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
                # Enregistrer les médias du readable dans la table Media
                # 1. Depuis le HTML (si balises <img> présentes)
                if readable_html:
                    soup_readable = BeautifulSoup(readable_html, 'html.parser')
                    extract_medias(soup_readable, expression)
                # 2. Depuis le markdown (pour les images converties en markdown)
                img_md_links = re.findall(r'!\[.*?\]\((.*?)\)', content)
                for img_url in img_md_links:
                    # Résoudre l'URL relative en URL absolue
                    resolved_img_url = resolve_url(str(expression.url), img_url)
                    # Vérifier si déjà présent (éviter doublons)
                    if not model.Media.select().where((model.Media.expression == expression) & (model.Media.url == resolved_img_url)).exists():
                        model.Media.create(expression=expression, url=resolved_img_url, type='img')
                links = extract_md_links(content)
                expression.readable = content # type: ignore
                print(f"Trafilatura succeeded on fetched HTML for {expression.url}")
        except Exception as e:
            print(f"Trafilatura failed on raw HTML for {expression.url}: {e}")

        # 2b. BeautifulSoup as a fallback on the same HTML
        if not content:
            try:
                soup = BeautifulSoup(raw_html, 'html.parser')
                clean_html(soup)
                text_content = get_readable(soup)
                if text_content and len(text_content) > 100:
                    content = text_content
                    urls = [a.get('href') for a in soup.find_all('a') if is_crawlable(a.get('href'))]
                    links = urls
                    expression.readable = content # type: ignore
                    print(f"BeautifulSoup fallback succeeded for {expression.url}")
            except Exception as e:
                print(f"BeautifulSoup fallback failed for {expression.url}: {e}")

    # Step 3: If no content yet (e.g., non-200 status, or parsing failed), try URL-based fallbacks
    if not content:
        # 3b. Archive.org (if Mercury also fails)
        if not content:
            try:
                print(f"Trying URL-based fallback: archive.org for {expression.url}")
                archive_data_url = f"http://archive.org/wayback/available?url={expression.url}"
                archive_response = await asyncio.to_thread(lambda: requests.get(archive_data_url, timeout=10))
                archive_response.raise_for_status()
                archive_data = archive_response.json()
                archived_url = archive_data.get('archived_snapshots', {}).get('closest', {}).get('url')
                if archived_url:
                    downloaded = await asyncio.to_thread(trafilatura.fetch_url, archived_url)
                    if downloaded:
                        raw_html = downloaded
                        extracted_content = trafilatura.extract(downloaded, include_links=True, include_images=True, output_format='markdown')
                        readable_html = trafilatura.extract(downloaded, include_links=True, include_images=True, output_format='html')
                        if extracted_content and len(extracted_content) > 100:
                            # Extraction des médias du readable HTML (corps du texte archivé)
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
                            # Enregistrer les médias du readable archivé dans la table Media
                            # 1. Depuis le HTML (si balises <img> présentes)
                            if readable_html:
                                soup_readable = BeautifulSoup(readable_html, 'html.parser')
                                extract_medias(soup_readable, expression)
                            # 2. Depuis le markdown (pour les images converties en markdown)
                            img_md_links = re.findall(r'!\[.*?\]\((.*?)\)', content)
                            for img_url in img_md_links:
                                # Résoudre l'URL relative en URL absolue
                                resolved_img_url = resolve_url(str(expression.url), img_url)
                                if not model.Media.select().where((model.Media.expression == expression) & (model.Media.url == resolved_img_url)).exists():
                                    model.Media.create(expression=expression, url=resolved_img_url, type='img')
                            links = extract_md_links(content)
                            expression.readable = content # type: ignore
                            print(f"Archive.org + Trafilatura succeeded for {expression.url}")
            except Exception as e:
                print(f"Archive.org fallback failed for {expression.url}: {e}")

    # Final processing and saving
    if content:
        soup = BeautifulSoup(raw_html if raw_html else content, 'html.parser')
        expression.title = str(get_title(soup) or expression.url) # type: ignore
        expression.description = str(get_description(soup)) if get_description(soup) else None # type: ignore
        expression.keywords = str(get_keywords(soup)) if get_keywords(soup) else None # type: ignore
        expression.lang = str(soup.html.get('lang', '')) if soup.html else '' # type: ignore
        # Compute relevance with OpenRouter gate when enabled
        try:
            from .llm_openrouter import is_relevant_via_openrouter
            if getattr(settings, 'openrouter_enabled', False) and settings.openrouter_api_key and settings.openrouter_model:
                verdict = is_relevant_via_openrouter(expression.land, expression)  # type: ignore
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
            for link in links:
                link_expression(expression.land, expression, link) # type: ignore
        expression.save()
        return 1
    else:
        print(f"All extraction methods failed for {expression.url}. Final status: {expression.http_status}")
        expression.save()
        return 0

    # Step 4: Analyze media if enabled
    if settings.media_analysis:
        try:
            media_analysis_results = await analyze_media(expression, session)
            if media_analysis_results:
                print(f"Media analysis found {len(media_analysis_results)} media items for #{expression.id}") # type: ignore
            else:
                print(f"No media found for #{expression.id}") # type: ignore
        except Exception as e:
            print(f"Media analysis failed for #{expression.id}: {e}") # type: ignore

    return 1

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


def extract_md_links(md_content: str):
    """Extract URLs from Markdown content with proper parenthesis handling.

    This function finds all URLs in Markdown link syntax and removes unmatched
    closing parentheses from the end of URLs.

    Args:
        md_content: A string containing Markdown-formatted content.

    Returns:
        list: A list of URL strings extracted from the Markdown content.

    Notes:
        - Matches URLs in Markdown link format: (http://example.com)
        - Supports http, https, and ftp protocols.
        - Removes unmatched closing parentheses from URL ends.
        - Handles edge case where URLs contain parentheses in parameters.
    """
    matches = re.findall(r'\(((https?|ftp)://[^\s/$.?#].[^\s]*)\)', md_content)
    urls = []
    for match in matches:
        url = match[0]
        # Si l'URL se termine par une parenthèse fermante non appariée, on la retire
        if url.endswith(")") and url.count("(") <= url.count(")"):
            url = url[:-1]
        urls.append(url)
    return urls


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
        - Automatically removes URL anchors before processing.
        - Checks if URL is crawlable using is_crawlable().
        - Creates or retrieves the associated Domain object.
        - Returns existing expression if URL already exists in the land.
        - Returns False for non-crawlable URLs (PDFs, images, etc.).
    """
    url = remove_anchor(url)
    if is_crawlable(url):
        domain_name = get_domain_name(url)
        domain = model.Domain.get_or_create(name=domain_name)[0]
        expression = model.Expression.get_or_none(
            model.Expression.url == url,
            model.Expression.land == land)
        if expression is None:
            expression = model.Expression.create(land=land, domain=domain, url=url, depth=depth)
        return expression
    return False


def get_domain_name(url: str) -> str:
    """Extract the domain name from a URL with heuristic-based refinement.

    This function extracts the domain from a URL and applies configured heuristics
    to handle special cases like social media profile URLs or platform-specific patterns.

    Args:
        url: The URL to extract the domain from.

    Returns:
        str: The extracted domain name, possibly refined by heuristics.

    Notes:
        - Default extraction uses urllib's urlparse to get netloc.
        - Applies heuristics from settings.heuristics for special domains.
        - Heuristics can extract sub-paths (e.g., 'twitter.com/username').
        - Useful for grouping URLs by logical domain/account.
    """
    parsed = urlparse(url)
    domain_name = parsed.netloc
    for key, value in settings.heuristics.items():
        if domain_name.endswith(key):
            matches = re.findall(value, url)
            domain_name = matches[0] if matches else domain_name
    return domain_name


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


def link_expression(land: model.Land, source_expression: model.Expression, url: str) -> bool:
    """Create a link from a source expression to a target expression.

    This function adds a new expression for the target URL and creates a directed
    link in the expression graph from source to target.

    Args:
        land: The Land object containing both expressions.
        source_expression: The Expression object that contains the link.
        url: The URL of the target expression to link to.

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
                target_id=target_expression.id) # type: ignore
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
        parsed = urlparse(url)
        exclude_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.pdf',
                       '.txt', '.csv', '.xls', '.xlsx', '.doc', '.docx')

        return \
            (url is not None) \
            and url.startswith(('http://', 'https://')) \
            and (not url.endswith(exclude_ext))
    except:
        return False


def process_expression_content(expression: model.Expression, html: str, dictionary) -> model.Expression:
    """Process and extract metadata from HTML content for an expression.

    This function extracts title, description, keywords, language, and content
    from HTML, calculates relevance, and extracts media elements.

    Args:
        expression: The Expression database object to update.
        html: The raw HTML content to process.
        dictionary: The land's word dictionary for relevance scoring.

    Returns:
        model.Expression: The updated Expression object.

    Notes:
        - Extracts metadata using BeautifulSoup and enhanced extraction helpers.
        - Calculates relevance score based on land dictionary word matches.
        - Extracts language from HTML lang attribute.
        - Falls back to domain-based title if no title is found.
        - Extracts media elements (images, videos, audio) from content.
        - Uses Trafilatura to extract clean readable content.
        - Updates expression.title, description, keywords, lang, content, and relevance.
    """
    print(f"Processing expression #{expression.id}") # type: ignore
    soup = BeautifulSoup(html, 'html.parser')

    if soup.html is not None:
        expression.lang = str(soup.html.get('lang', '')) # type: ignore
    
    # Extract basic metadata from the soup object first
    expression.title = str(soup.title.string.strip()) if soup.title and soup.title.string else '' # type: ignore
    expression.description = str(get_meta_content(soup, 'description')) if get_meta_content(soup, 'description') else None # type: ignore
    expression.keywords = str(get_meta_content(soup, 'keywords')) if get_meta_content(soup, 'keywords') else None # type: ignore
    
    print(f"Initial metadata from HTML for expression {expression.id}: title={bool(expression.title)}, " # type: ignore
          f"description={bool(expression.description)}, keywords={bool(expression.keywords)}")
    
    # Try to enhance with more robust metadata extraction
    try:
        metadata = extract_metadata(str(expression.url)) # Ensure url is str
        
        # Only override if we got better metadata
        if metadata['title']:
            expression.title = str(metadata['title']) # type: ignore
        if metadata['description']:
            expression.description = str(metadata['description']) # type: ignore
        if metadata['keywords']:
            expression.keywords = str(metadata['keywords']) # type: ignore
            
        print(f"Enhanced metadata for expression {expression.id}: title={bool(metadata['title'])}, " # type: ignore
              f"description={bool(metadata['description'])}, keywords={bool(metadata['keywords'])}")
    except Exception as e:
        print(f"Error enhancing metadata for expression {expression.id}: {str(e)}") # type: ignore
    
    # Ensure title has at least an empty string, but leave description and keywords as null if not found
    domain_name = expression.domain.name if expression.domain else urlparse(str(expression.url)).netloc # Ensure url is str
    expression.title = str(expression.title or f"Content from {domain_name}") # type: ignore
    
    print(f"Final expression metadata to save: title={bool(expression.title)}, " # type: ignore
          f"description={bool(expression.description)}, keywords={bool(expression.keywords)}")

    clean_html(soup)

    if settings.archive is True:
        loc = path.join(settings.data_location, 'lands/%s/%s') \
              % (expression.land.id, expression.id) # Use .id instead of .get_id() # type: ignore
        with open(loc, 'w', encoding="utf-8") as html_file:
            html_file.write(html.strip())
        html_file.close()

    readable_content = get_readable(soup)
    if not readable_content.strip():
        expression.readable = f"<!-- RAW HTML -->\n{html}" # type: ignore
    else:
        expression.readable = readable_content # type: ignore

    # Check if page language matches land language
    if expression.lang and expression.land.lang and expression.lang != expression.land.lang:
        expression.relevance = 0 # type: ignore
    else:
        # Compute relevance with OpenRouter gate when enabled
        try:
            from .llm_openrouter import is_relevant_via_openrouter
            if getattr(settings, 'openrouter_enabled', False) and settings.openrouter_api_key and settings.openrouter_model:
                verdict = is_relevant_via_openrouter(expression.land, expression)  # type: ignore
                if verdict is False:
                    expression.relevance = 0  # type: ignore
                else:
                    expression.relevance = expression_relevance(dictionary, expression)  # type: ignore
            else:
                expression.relevance = expression_relevance(dictionary, expression)  # type: ignore
        except Exception as e:
            print(f"OpenRouter gate error for {expression.url}: {e}")
            expression.relevance = expression_relevance(dictionary, expression)  # type: ignore

    if expression.relevance is not None and expression.relevance > 0: # type: ignore
        print(f"Expression #{expression.id} approved") # type: ignore
        extract_medias(soup, expression)
        expression.approved_at = model.datetime.datetime.now() # type: ignore
        if expression.depth is not None and expression.depth < 3: # type: ignore
            urls = [a.get('href') for a in soup.find_all('a') if is_crawlable(a.get('href'))]
            print(f"Linking {len(urls)} expression to #{expression.id}") # type: ignore
            for url in urls:
                link_expression(expression.land, expression, url) # type: ignore

    return expression


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
    VIDEO_EXTENSIONS = (".mp4", ".webm", ".ogg", ".ogv", ".mov", ".avi", ".mkv")
    AUDIO_EXTENSIONS = (".mp3", ".wav", ".ogg", ".aac", ".flac", ".m4a")

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
        - Uses French word tokenization and stemming for matching.
        - Matches whole words only (word boundaries).
        - Returns 0 if title or readable content is missing.
        - Falls back to simple tokenizer if NLTK is unavailable.
    """
    lemmas = [w.lemma for w in dictionary]
    title_relevance = [0]
    content_relevance = [0]

    def get_relevance(text, weight) -> list:
        if not isinstance(text, str): # Ensure text is a string
            text = str(text)
        if _NLTK_OK:
            tokens = word_tokenize(text, language='french')
        else:
            tokens = _simple_word_tokenize(text)
        stems = [stem_word(w) for w in tokens]
        stemmed_text = " ".join(stems)
        return [sum(weight for _ in re.finditer(r'\b%s\b' % re.escape(lemma), stemmed_text)) for lemma in lemmas]

    try:
        title_relevance = get_relevance(expression.title, 10) # type: ignore
        content_relevance = get_relevance(expression.readable, 1) # type: ignore
    except Exception as e:
        print(f"Error computing relevance: {e}")
        pass
    return sum(title_relevance) + sum(content_relevance)


def export_land(land: model.Land, export_type: str, minimum_relevance: int):
    """Export land data to a file in the specified format.

    This function creates an export file containing land data filtered by
    minimum relevance threshold.

    Args:
        land: The Land object to export.
        export_type: The export format (e.g., 'pagecsv', 'pagegexf', 'corpus').
        minimum_relevance: Minimum relevance score filter for expressions.

    Notes:
        - Output filename includes land name, export type, and timestamp.
        - Files are saved to settings.data_location directory.
        - Supports formats: pagecsv, fullpagecsv, pagegexf, nodecsv, nodegexf, mediacsv, corpus.
        - Prints success message with record count or error message.
        - See Export class for format-specific details.
    """
    date_tag = model.datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = path.join(settings.data_location, 'export_land_%s_%s_%s') \
               % (land.name, export_type, date_tag)
    export = Export(export_type, land, minimum_relevance)
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


def update_heuristic():
    """Update expression domains based on current heuristic settings.

    This function recalculates domain names for all expressions using the current
    heuristics configuration and updates the database accordingly.

    Notes:
        - Iterates through all expressions in the database.
        - Applies current heuristics from settings.heuristics.
        - Creates new Domain records if needed.
        - Updates expression.domain foreign key relationships.
        - Prints count of updated expressions.
        - Useful after modifying heuristics configuration.
    """
    domains = list(model.Domain.select().dicts())
    domains = {x['id']: x for x in domains}
    expressions = model.Expression.select()
    updated = 0
    for expression in expressions:
        domain = get_domain_name(str(expression.url)) # Ensure url is str
        if domain != domains[expression.domain_id]['name']:
            to_domain, _ = model.Domain.get_or_create(name=domain)
            expression.domain = to_domain
            expression.save()
            updated += 1
    print(f"{updated} domain(s) updated")


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

async def medianalyse_land(land: model.Land) -> dict:
    """Analyze all media items associated with expressions in a land.

    This async function processes all media in a land, extracting metadata and
    performing analysis using the MediaAnalyzer.

    Args:
        land: The Land object whose media should be analyzed.

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
