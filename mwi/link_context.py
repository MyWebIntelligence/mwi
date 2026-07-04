"""Link-context helpers (sprint link-context, migration 012).

Computes, for each ``<a href>`` of a page, the metadata stored on
``ExpressionLink``:

- ``dom``       — CSS-like path from the document root down to the *parent*
                  of the ``<a>`` tag, e.g.
                  ``html > body > div#main.layout > article.post > p``.
- ``dom_html``  — outerHTML of the closest block-level ancestor, truncated
                  to ``settings.link_dom_html_max_chars``.
- ``context``   — markdown paragraph of the readable containing the link
                  (computed by :func:`extract_md_paragraph`); when the
                  readable does not carry the URL (BS4 fallback path), the
                  block ancestor's text (``block_text``) is used instead.

Design constraints:

- This module must NOT import ``mwi.core`` (it is imported *by* core and by
  ``readable_pipeline`` — keeping it leaf-level avoids import cycles).
- ``extract_link_dom_map`` must never raise: a context-extraction failure
  must never fail a crawl. It returns ``{}`` on any error.
- URL matching strategy: ``Expression.url`` is stored normalized by
  ``url_normalizer.normalize_url`` (idempotent), so map keys are the
  normalized absolute href, plus a relaxed key (lowercase, no trailing
  slash) absorbing residual divergences (e.g. legacy data lowercased by
  ``core.resolve_url``). ``core.resolve_url`` is intentionally NOT used
  here because it lowercases the URL path.
"""

import re
import warnings
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urldefrag, urljoin, urlparse

import settings

from .url_normalizer import normalize_url

BLOCK_TAGS = ('p', 'li', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
              'blockquote', 'figcaption', 'dd', 'dt')
FALLBACK_BLOCK_TAGS = ('div', 'section', 'article')
SKIP_HREF_PREFIXES = ('mailto:', 'javascript:', 'tel:', 'data:', '#')
MAX_CLASSES_PER_SEGMENT = 3

_VALID_TOKEN = re.compile(r'^[A-Za-z0-9_-]+$')


@dataclass
class LinkDomInfo:
    """DOM-level metadata for one ``<a href>`` occurrence in a page."""
    dom: str                      # CSS path root -> parent of the <a>
    dom_html: Optional[str]       # outerHTML of the block ancestor, truncated
    block_text: Optional[str]     # text of the block ancestor, truncated


def _context_cap() -> int:
    return getattr(settings, 'link_context_max_chars', 1000) or 1000


def _dom_html_cap() -> int:
    return getattr(settings, 'link_dom_html_max_chars', 4000) or 4000


def _segment(element) -> str:
    """Build one CSS-path segment: ``tag#id.cls1.cls2.cls3``."""
    part = element.name
    el_id = element.get('id')
    if isinstance(el_id, str) and _VALID_TOKEN.match(el_id):
        part += f'#{el_id}'
    classes = element.get('class') or []
    kept = [c for c in classes if isinstance(c, str) and _VALID_TOKEN.match(c)]
    for cls in kept[:MAX_CLASSES_PER_SEGMENT]:
        part += f'.{cls}'
    return part


def build_dom_path(a_tag) -> str:
    """CSS-like path from the document root to the direct parent of `a_tag`.

    The ``<a>`` itself is not included. ``[document]`` (the BeautifulSoup
    root) is skipped.
    """
    segments = []
    for parent in a_tag.parents:
        if parent.name is None or parent.name == '[document]':
            continue
        segments.append(_segment(parent))
    segments.reverse()
    return ' > '.join(segments)


def find_block_ancestor(a_tag):
    """Closest block-level ancestor of `a_tag`.

    Strict blocks (``BLOCK_TAGS``) win; the first ``div``/``section``/
    ``article`` met on the way up is kept as fallback. Returns None when
    neither exists (e.g. ``<a>`` directly under ``<body>``).
    """
    fallback = None
    for parent in a_tag.parents:
        if parent.name in BLOCK_TAGS:
            return parent
        if fallback is None and parent.name in FALLBACK_BLOCK_TAGS:
            fallback = parent
    return fallback


def _relaxed_key(normalized_url: str) -> str:
    return normalized_url.lower().rstrip('/')


def _same_page_norm(base_url: str) -> Optional[str]:
    """Normalized, fragment-stripped form of the source page URL, or None."""
    try:
        nofrag = urldefrag(base_url)[0]
        return normalize_url(nofrag) if nofrag else None
    except Exception:
        return None


def _is_same_page(absolute: str, base_norm: Optional[str]) -> bool:
    """True when `absolute` only navigates within the source page.

    A section / reference link written as an absolute URL + fragment
    (e.g. ``https://arxiv.org/html/2404.00600v2#S1``) collapses onto the
    source page once the fragment is stripped — it is in-page navigation, not
    an outbound hyperlink, and would otherwise flood the raw link network with
    self-loops (audit: ~96% of the raw weight). The fragment is stripped
    explicitly here, independently of the ``remove_anchor`` setting. A bare
    self-link (no fragment) is caught too. Never raises.
    """
    if not base_norm:
        return False
    try:
        nofrag = urldefrag(absolute)[0]
        return bool(nofrag) and normalize_url(nofrag) == base_norm
    except Exception:
        return False


def _quiet_soup(raw_html: str, parser: str):
    """BeautifulSoup parse with the noisy XMLParsedAsHTMLWarning silenced.

    Stored HTML is sometimes actually XML (sitemaps, RSS/Atom feeds); bs4
    then emits a multi-line warning. It is benign for link extraction, so we
    suppress it locally (no global filterwarnings side effect).
    """
    from bs4 import BeautifulSoup
    with warnings.catch_warnings():
        try:
            from bs4 import XMLParsedAsHTMLWarning
            warnings.simplefilter('ignore', XMLParsedAsHTMLWarning)
        except ImportError:
            pass
        return BeautifulSoup(raw_html, parser)


def extract_link_dom_map(raw_html: Optional[str], base_url: str,
                         soup=None) -> Dict[str, LinkDomInfo]:
    """Map each crawlable ``<a href>`` of `raw_html` to its LinkDomInfo.

    Keys are the normalized absolute URL plus a relaxed variant
    (lowercase, no trailing slash). First occurrence wins (consistent with
    the composite primary key on expressionlink: first link wins too).

    Never raises — returns ``{}`` on any failure. Reuses `soup` when the
    caller already parsed the page (crawl path).
    """
    try:
        if soup is None:
            if not raw_html:
                return {}
            soup = _quiet_soup(raw_html, 'html.parser')

        dom_html_cap = _dom_html_cap()
        context_cap = _context_cap()
        mapping: Dict[str, LinkDomInfo] = {}
        base_norm = _same_page_norm(base_url)

        for a_tag in soup.find_all('a', href=True):
            href = (a_tag.get('href') or '').strip()
            if not href or href.lower().startswith(SKIP_HREF_PREFIXES):
                continue
            if href.startswith(('http://', 'https://')):
                absolute = href
            else:
                absolute = urljoin(base_url, href)
                if not absolute.startswith(('http://', 'https://')):
                    continue

            if _is_same_page(absolute, base_norm):
                continue
            try:
                key = normalize_url(absolute)
            except Exception:
                continue

            block = find_block_ancestor(a_tag)
            info = LinkDomInfo(
                dom=build_dom_path(a_tag),
                dom_html=str(block)[:dom_html_cap] if block is not None else None,
                block_text=(block.get_text(' ', strip=True)[:context_cap]
                            if block is not None else None),
            )
            mapping.setdefault(key, info)
            mapping.setdefault(_relaxed_key(key), info)
        return mapping
    except Exception:
        return {}


def extract_all_links(raw_html: Optional[str], base_url: str,
                      soup=None) -> list:
    """All http(s) ``<a href>`` URLs of `raw_html`, **duplicates preserved**.

    Unlike :func:`extract_link_dom_map` (which deduplicates via
    ``setdefault``), this APPENDS every anchor so the caller can count
    multiplicity (edge weight). URLs are resolved to absolute with
    ``urljoin`` but **not** normalized — the caller applies
    ``url_normalizer.normalize_url`` to align with the stored
    ``Expression.url``. ``core.resolve_url`` is intentionally NOT used (it
    lowercases the URL path).

    Reuses the same filtering as ``extract_link_dom_map`` (skip
    ``mailto:``/``javascript:``/``tel:``/``data:``/``#``; drop anything that
    does not resolve to http(s)). Never raises — returns ``[]`` on any error.
    """
    links: list = []
    try:
        if soup is None:
            if not raw_html:
                return links
            try:
                soup = _quiet_soup(raw_html, 'lxml')
            except Exception:
                soup = _quiet_soup(raw_html, 'html.parser')

        base_norm = _same_page_norm(base_url)
        for a_tag in soup.find_all('a', href=True):
            href = (a_tag.get('href') or '').strip()
            if not href or href.lower().startswith(SKIP_HREF_PREFIXES):
                continue
            if href.startswith(('http://', 'https://')):
                absolute = href
            else:
                absolute = urljoin(base_url, href)
                if not absolute.startswith(('http://', 'https://')):
                    continue
            if _is_same_page(absolute, base_norm):
                continue
            links.append(absolute)
        return links
    except Exception:
        return links


def lookup_link_info(mapping: Dict[str, LinkDomInfo],
                     url: str) -> Optional[LinkDomInfo]:
    """Find the LinkDomInfo of `url` in a map built by extract_link_dom_map."""
    if not mapping or not url:
        return None
    try:
        key = normalize_url(url)
    except Exception:
        key = url
    return mapping.get(key) or mapping.get(_relaxed_key(key))


def extract_md_paragraph(markdown: Optional[str], url: Optional[str],
                         max_chars: Optional[int] = None) -> Optional[str]:
    """First markdown paragraph (blank-line delimited) containing `url`.

    Returns the stripped paragraph truncated to `max_chars` (defaults to
    ``settings.link_context_max_chars``), or None when the URL does not
    appear literally in the markdown (e.g. BS4-fallback readable without
    hrefs).
    """
    if not markdown or not url:
        return None
    cap = max_chars or _context_cap()
    paragraphs = re.split(r'\n\s*\n', markdown)
    for needle in (url, url.rstrip('/')):
        if not needle:
            continue
        for para in paragraphs:
            if needle in para:
                return para.strip()[:cap]
    return None


# ---------------------------------------------------------------------------
# Unified markdown link parser (sprint EXTRACTLINKS-2026-06)
#
# Single source of truth for extracting hyperlink targets from a markdown
# body, shared by the three ExpressionLink write paths (crawl, consolidate,
# readable pipeline). Replaces the legacy greedy ``core.extract_md_links``
# regex which (A1) overflowed on the last ')' of a line, (A2) truncated
# balanced parentheses, (A3) ignored ``<url>`` autolinks, (A4) leaked
# ``![alt](url)`` images, and (B) dropped every relative ``[t](/path)`` link.
#
# Empirically validated against the 18 real cases of the sprint Annexe B
# (two independent implementations converged) — see
# ``.claude/project/sprint-extractlinks.md`` §12.
# ---------------------------------------------------------------------------

# Scheme is matched case-insensitively for the gate (``HTTP://`` accepted).
# The URL itself is NEVER lowercased — unlike the legacy ``core.resolve_url``
# which destroys case-sensitive paths (e.g. ``/Runaround_(story)``).
_SCHEME_RE = re.compile(r'(?:https?|ftp)://', re.IGNORECASE)


def _read_url_token(s: str, start: int):
    """Read a URL token from `start` (the char just after the opening '(').

    Follows the depth of nested '(' so balanced parentheses of the URL are
    preserved; stops on whitespace or an unmatched ')' (which closes the
    markdown link). Returns ``(token, end_index)``. This single rule removes
    both A1 (overflow) and A2 (truncation): no greedy ``[^\\s]*``, no naive
    trailing-paren strip.

    A CommonMark title (``[t](url "title")``) is separated from the
    destination by whitespace, so the whitespace stop already strips it — and
    a *bare* quote (no preceding whitespace) is a legal destination character,
    so it is kept rather than truncating the URL (e.g. ``?q="x"``).
    """
    depth = 0
    i = start
    n = len(s)
    chars = []
    while i < n:
        c = s[i]
        if c.isspace():
            break
        if c == '(':
            depth += 1
        elif c == ')':
            if depth == 0:
                break
            depth -= 1
        chars.append(c)
        i += 1
    return ''.join(chars), i


def iter_markdown_link_tokens(md_content: Optional[str]):
    """Yield the RAW URL tokens (as written, RELATIVE links included) of a
    markdown body, in document order, duplicates preserved.

    Recognizes inline links ``[text](url)`` (images ``![alt](url)`` excluded)
    and autolinks ``<url>``. Parentheses are balanced (neither overflow nor
    truncation). Never raises.

    The RAW (unresolved) token is exposed because
    :func:`extract_md_paragraph` locates the paragraph on the *literal* href
    — the readable pipeline needs ``raw_url`` to keep that lookup working.
    """
    if not md_content or not isinstance(md_content, str):
        return
    s = md_content
    n = len(s)
    i = 0
    while i < n:
        c = s[i]
        # Autolink: <url>  (always absolute)
        if c == '<':
            close = s.find('>', i + 1)
            if close != -1:
                content = s[i + 1:close]
                if ' ' not in content and _SCHEME_RE.match(content):
                    yield content
                    i = close + 1
                    continue
            i += 1
            continue
        # Inline link [text](url); image when preceded by '!'
        if c == '[':
            is_image = i > 0 and s[i - 1] == '!'
            close = s.find(']', i + 1)
            if close != -1 and close + 1 < n and s[close + 1] == '(':
                token, end = _read_url_token(s, close + 2)
                if not is_image and token:
                    yield token
                i = end
                continue
            i += 1
            continue
        i += 1


def extract_markdown_links(md_content: Optional[str],
                           base_url: Optional[str] = None) -> list:
    """Resolved http(s)/ftp hyperlink targets of a markdown body.

    When `base_url` is provided, relative targets are resolved via
    ``urljoin``; otherwise only already-absolute targets are kept
    (backward-compatible). Images excluded, autolinks recognized,
    parentheses balanced, path case preserved, duplicates preserved (the
    caller deduplicates). Never raises — returns ``[]`` on empty input.
    """
    out: list = []
    try:
        for token in iter_markdown_link_tokens(md_content):
            resolved = urljoin(base_url, token) if base_url else token
            if _SCHEME_RE.match(resolved):
                out.append(resolved)
    except Exception:
        return out
    return out


# --------------------------------------------------------------------------- #
# Tolerant URL resolution — 3-key ladder (sprint dedup-selfloops)              #
# --------------------------------------------------------------------------- #
# Same ladder as the fullhtml export: exact normalize_url, then relaxed
# (lowercase, no trailing slash), then host+path (scheme/www-insensitive).
# Used by consolidate to resolve readable links onto EXISTING corpus
# expressions instead of creating URL-variant duplicates.

def host_path_key(url: str) -> Optional[str]:
    """Scheme-and-www-insensitive key: host(no www) + path + query.

    Absorbs http<->https / www<->bare divergences when force_https /
    strip_www are OFF. Returns None on failure / no host.
    """
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or '').lower()
        if host.startswith('www.'):
            host = host[4:]
        if not host:
            return None
        key = host + (parsed.path or '').rstrip('/')
        if parsed.query:
            key += '?' + parsed.query
        return key
    except Exception:
        return None


def add_to_url_index(index: tuple, eid: int, url: str) -> None:
    """Index one expression URL under the 3 keys.

    A key already mapped to a DIFFERENT id becomes None (ambiguous ->
    unusable for lookup, never a wrong match).
    """
    exact, relaxed, by_host_path = index
    try:
        norm = normalize_url(url) if url else url
    except Exception:
        norm = url
    if not norm:
        return
    for table, key in ((exact, norm),
                       (relaxed, norm.lower().rstrip('/')),
                       (by_host_path, host_path_key(norm))):
        if not key:
            continue
        if key in table:
            if table[key] != eid:
                table[key] = None
        else:
            table[key] = eid


def build_url_index(pairs) -> tuple:
    """Build the 3-key URL index from (expression_id, url) pairs."""
    index = ({}, {}, {})
    for eid, url in pairs:
        add_to_url_index(index, eid, url)
    return index


def resolve_url_in_index(index: tuple, href: str) -> Optional[int]:
    """Resolve a href to an indexed expression id. None on miss/ambiguous."""
    exact, relaxed, by_host_path = index
    try:
        norm = normalize_url(href)
    except Exception:
        norm = href
    if not norm:
        return None
    eid = exact.get(norm)
    if eid is not None:
        return eid
    eid = relaxed.get(norm.lower().rstrip('/'))
    if eid is not None:
        return eid
    key = host_path_key(norm)
    if key is not None:
        eid = by_host_path.get(key)
        if eid is not None:
            return eid
    return None
