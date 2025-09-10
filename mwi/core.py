"""
Core functions
"""
import asyncio
import json
import re
from argparse import Namespace
from os import path
from typing import Union, Optional
from urllib.parse import urlparse, urljoin

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

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


async def extract_dynamic_medias(url: str, expression: model.Expression) -> list:
    """
    Extract media URLs from a webpage using a headless browser to execute JavaScript
    and capture dynamically generated media URLs
    :param url: URL to extract media from
    :param expression: The expression object to associate media with
    :return: List of media URLs found after JavaScript execution
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
    """
    Resolve relative URL to absolute URL using the base URL
    :param base_url: The base URL (page URL)
    :param relative_url: The relative or absolute URL to resolve
    :return: Absolute URL
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
    """
    Confirms action by requesting right input from user
    :param message:
    :return:
    """
    return input(message) == 'Y'


def check_args(args: Namespace, mandatory) -> bool:
    """
    Returns True if all required args are in parsed input args
    :param args:
    :param mandatory:
    :return:
    """
    args_dict = vars(args)
    if isinstance(mandatory, str):
        mandatory = [mandatory]
    for arg in mandatory:
        if arg not in args_dict or args_dict[arg] is None:
            raise ValueError('Argument "%s" is required' % arg)
    return True


def split_arg(arg: str) -> list:
    """
    Splits arg string using comma separator and returns a filtered list
    :param arg:
    :return:
    """
    args = arg.split(",")
    return [a.strip() for a in args if a]


def get_arg_option(name: str, args: Namespace, set_type, default):
    """
    Returns value from optional argument
    :param name:
    :param args:
    :param set_type:
    :param default:
    :return:
    """
    args_dict = vars(args)
    if (name in args_dict) and (args_dict[name] is not None):
        return set_type(args_dict[name])
    return default


def stem_word(word: str) -> str:
    """
    Stems word with NLTK Snowball FrenchStemmer
    :param word:
    :return:
    """
    if not hasattr(stem_word, "stemmer"):
        setattr(stem_word, "stemmer", FrenchStemmer())
    # The following line uses getattr which is safe
    return str(getattr(stem_word, "stemmer").stem(word.lower()))


def crawl_domains(limit: int = 0, http: Optional[str] = None):
    """
    Crawl domains to retrieve info using a pipeline: Trafilatura, Archive.org, then direct requests.
    :param limit: Max number of domains to process.
    :param http: Filter by specific HTTP status (for recrawling).
    :return: Number of successfully processed domains.
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
    """
    Process domain info from HTML content obtained via the pipeline.
    Uses enhanced metadata extraction (BeautifulSoup helpers and Trafilatura).
    :param domain: The domain object to update.
    :param html_content: The HTML content of the domain's page.
    :param effective_url: The URL from which the content was actually fetched (could be live or archive).
    :param source_method: String indicating how content was obtained (e.g., "TRAFILATURA", "ARCHIVE_ORG", "REQUESTS").
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
    """
    Get named meta content property
    :param soup:
    :param name:
    :return:
    """
    tag = soup.find('meta', attrs={'name': name})
    if tag and tag.has_attr('content'): # type: ignore
        content = tag['content'] # type: ignore
        if isinstance(content, str):
            print(f"Found meta content for {name}: {content[:30]}...")
            return content.strip()
    return ""


def extract_metadata(url: str) -> dict:
    """
    Extract metadata from webpage with multiple fallback sources
    :param url: URL to extract metadata from
    :return: Dictionary with title, description, and keywords (None if not found)
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
    """
    Get page title with fallback chain
    :param soup: BeautifulSoup object
    :return: Title string or empty string if not found
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
    """
    Get page description with fallback chain
    :param soup: BeautifulSoup object
    :return: Description string or empty string if not found
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
    """
    Get page keywords with fallback chain
    :param soup: BeautifulSoup object
    :return: Keywords string or empty string if not found
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
    """
    Start land crawl
    :param land:
    :param limit:
    :param http:
    :param depth: Only crawl expressions at this depth (if not None)
    :return:
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
    """
    Crawl and process an expression with media analysis
    :param expression: The expression to process.
    :param dictionary: The land's word dictionary for relevance.
    :param session: aiohttp.ClientSession for requests.
    :return: 1 if content was processed, 0 on failure.
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

async def consolidate_land(land: model.Land, limit: int = 0, depth: Optional[int] = None) -> tuple:
    """
    Consolidate a land: for each expression, recalculate relevance, links, media, add missing docs, recreate links, replace old ones.
    :param land:
    :param limit:
    :param depth: Only process expressions at this depth (if not None)
    :return: (number of expressions consolidated, number of errors)
    """
    print(f"Consolidating land {land.id}") # type: ignore
    dictionary = get_land_dictionary(land)

    # Select expressions to process
    query = model.Expression.select().where(
        model.Expression.land == land,
        model.Expression.fetched_at.is_null(False)
    )
    if depth is not None:
        query = query.where(model.Expression.depth == depth)
    if limit > 0:
        query = query.limit(limit)

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

                # 2. Recalculer la relevance (avec garde-fou OpenRouter) et le contenu
                try:
                    from .llm_openrouter import is_relevant_via_openrouter
                    if getattr(settings, 'openrouter_enabled', False) and settings.openrouter_api_key and settings.openrouter_model:
                        verdict = is_relevant_via_openrouter(land, expr)  # type: ignore
                        if verdict is False:
                            expr.relevance = 0  # type: ignore
                        else:
                            expr.relevance = expression_relevance(dictionary, expr)  # type: ignore
                    else:
                        expr.relevance = expression_relevance(dictionary, expr)  # type: ignore
                except Exception as e:
                    print(f"OpenRouter gate error for {expr.url}: {e}")
                    expr.relevance = expression_relevance(dictionary, expr)  # type: ignore
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
    """
    Crawl and process an expression using a robust pipeline, while preserving the original HTTP status code.
    Pipeline: Direct Fetch -> [Trafilatura -> BS] -> Fallbacks [Mercury -> Archive]
    :param expression: The expression to process.
    :param dictionary: The land's word dictionary for relevance.
    :param session: aiohttp.ClientSession for requests.
    :return: 1 if content was processed, 0 on failure.
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
    """
    Analyze media for an expression using MediaAnalyzer
    :param expression: The expression to analyze media for
    :param session: aiohttp.ClientSession for requests.
    :return: List of analyzed media items
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
    """
    Extract URLs from Markdown content, en retirant toute parenthèse fermante finale non appariée
    :param md_content:
    :return:
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
    """
    Add expression to land
    :param land:
    :param url:
    :param depth:
    :return:
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
    """
    Returns domain from any url as sub.domain.ext or according to heuristics settings
    :param url:
    :return:
    """
    parsed = urlparse(url)
    domain_name = parsed.netloc
    for key, value in settings.heuristics.items():
        if domain_name.endswith(key):
            matches = re.findall(value, url)
            domain_name = matches[0] if matches else domain_name
    return domain_name


def remove_anchor(url: str) -> str:
    """
    Removes anchor from URL
    :param url:
    :return:
    """
    anchor_pos = url.find('#')
    return url[:anchor_pos] if anchor_pos > 0 else url


def link_expression(land: model.Land, source_expression: model.Expression, url: str) -> bool:
    """
    Link target expression to source expression
    :param land:
    :param source_expression:
    :param url:
    :return:
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
    """
    Checks whether an URL is valid for crawling
    :param url:
    :return:
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
    """
    Process expression fields from HTML content
    :param expression:
    :param html:
    :param dictionary:
    :return:
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
    """
    Extract media src (img, video) from html content
    :param content:
    :param expression:
    :return:
    """
    print(f"Extracting media from #{expression.id}") # type: ignore
    medias = []
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg")
    VIDEO_EXTENSIONS = (".mp4", ".webm", ".ogg", ".ogv", ".mov", ".avi", ".mkv")
    AUDIO_EXTENSIONS = (".mp3", ".wav", ".ogg", ".aac", ".flac", ".m4a")
    
    for tag in ['img', 'video', 'audio']:
        for element in content.find_all(tag):
            src = element.get('src')
            if src is None:
                continue
                
            is_valid_src = src not in medias
            if tag == 'img':
                is_valid_src = is_valid_src and src.lower().endswith(IMAGE_EXTENSIONS)
            elif tag == 'video':
                is_valid_src = is_valid_src and (src.lower().endswith(VIDEO_EXTENSIONS) or True)
            elif tag == 'audio':
                is_valid_src = is_valid_src and (src.lower().endswith(AUDIO_EXTENSIONS) or True)
            
            if is_valid_src:
                # Resolve relative URLs to absolute URLs
                resolved_url = resolve_url(str(expression.url), src)
                medias.append(resolved_url)
                
                # Check if media doesn't already exist in database
                if not model.Media.select().where(
                    (model.Media.expression == expression) & 
                    (model.Media.url == resolved_url)
                ).exists():
                    media = model.Media.create(expression=expression, url=resolved_url, type=tag)
                    media.save()


def get_readable(content):
    """
    Get readable part of HTML content, preserving media links as [IMAGE: url], [VIDEO: url], [AUDIO: url]
    :param content:
    :return:
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
    """
    Get rid of DOM objects with no valuable content
    :param soup:
    :return:
    """
    remove_selectors = ('script', 'style', 'iframe', 'form', 'footer', '.footer',
                        'nav', '.nav', '.menu', '.social', '.modal')
    for selector in remove_selectors:
        for tag in soup.select(selector):
            tag.decompose()


def get_land_dictionary(land: model.Land):
    """
    Get land dictionary
    :param land:
    :return:
    """
    return model.Word.select() \
        .join(model.LandDictionary, JOIN.LEFT_OUTER) \
        .where(model.LandDictionary.land == land)


def land_relevance(land: model.Land):
    """
    Start relevance computing according to land dictionary for each expression in land
    :param land:
    :return:
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
    """
    Compute expression relevance according to land dictionary
    :param dictionary:
    :param expression:
    :return:
    """
    lemmas = [w.lemma for w in dictionary]
    title_relevance = [0]
    content_relevance = [0]

    def get_relevance(text, weight) -> list:
        if not isinstance(text, str): # Ensure text is a string
            text = str(text)
        stems = [stem_word(w) for w in word_tokenize(text, language='french')]
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
    """
    Export land data, file extension is set according to export type
    :param land:
    :param export_type:
    :param minimum_relevance:
    :return:
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
    """
    Update domains according to heuristic settings
    :return:
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
    expressions = model.Expression.select().where(model.Land == land)
    model.Media.delete().where(model.Media.expression << expressions)

async def medianalyse_land(land: model.Land) -> dict:
    """
    Analyse les médias pour un land donné.
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
