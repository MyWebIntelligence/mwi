"""
Mercury Parser Readable Pipeline - Système autonome d'enrichissement.

Mercury Parser Readable Pipeline - Autonomous enrichment system.

This module provides a complete pipeline for extracting readable content from web pages
using Mercury Parser. It includes automatic fallback to Wayback Machine snapshots when
live extraction fails, configurable merge strategies for content fusion, and support
for media and link extraction from markdown content.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import aiohttp

from . import model
from .core import get_land_dictionary, prefer_earlier_datetime


class MergeStrategy(Enum):
    """Stratégies de fusion des données.

    Data merge strategies for combining Mercury Parser results with existing data.

    Attributes:
        MERCURY_PRIORITY: Mercury Parser data always overwrites existing data.
        PRESERVE_EXISTING: Keeps existing data if it's not empty.
        SMART_MERGE: Intelligent fusion based on field type and content quality.
    """
    MERCURY_PRIORITY = "mercury_priority"     # Mercury écrase toujours
    PRESERVE_EXISTING = "preserve_existing"   # Garde l'existant si non vide
    SMART_MERGE = "smart_merge"               # Fusion intelligente


@dataclass
class MercuryResult:
    """Structure des résultats Mercury Parser.

    Data structure for Mercury Parser extraction results.

    Attributes:
        title: Extracted page title.
        content: Main content in HTML format.
        markdown: Main content in Markdown format.
        lead_image_url: URL of the primary image.
        date_published: Publication date as string.
        author: Author name or byline.
        excerpt: Short excerpt or description.
        domain: Domain name of the source.
        word_count: Number of words in the content.
        direction: Text direction (ltr/rtl).
        total_pages: Total number of pages for paginated content.
        rendered_pages: Number of pages actually rendered.
        next_page_url: URL of the next page in pagination.
        media: List of extracted media items (images, videos).
        links: List of extracted hyperlinks.
        raw_response: Complete raw JSON response from Mercury Parser.
        error: Error message if extraction failed.
        extraction_timestamp: Timestamp when extraction occurred.
    """
    title: Optional[str] = None
    content: Optional[str] = None
    markdown: Optional[str] = None
    lead_image_url: Optional[str] = None
    date_published: Optional[str] = None
    author: Optional[str] = None
    excerpt: Optional[str] = None
    domain: Optional[str] = None
    word_count: Optional[int] = None
    direction: Optional[str] = None
    total_pages: Optional[int] = None
    rendered_pages: Optional[int] = None
    next_page_url: Optional[str] = None
    media: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)
    raw_response: Optional[Dict] = None
    error: Optional[str] = None
    extraction_timestamp: Optional[datetime] = None


@dataclass
class ExpressionUpdate:
    """Structure pour les mises à jour d'expression.

    Data structure for tracking expression updates.

    Attributes:
        expression_id: ID of the expression being updated.
        field_updates: Dictionary mapping field names to (old_value, new_value) tuples.
        media_additions: List of media items to add to the expression.
        link_additions: List of links to add to the expression.
        update_reason: Human-readable reason for the update.
    """
    expression_id: int
    field_updates: Dict[str, Tuple[Any, Any]]  # (old_value, new_value)
    media_additions: List[Dict[str, Any]]
    link_additions: List[Dict[str, Any]]
    update_reason: str


class MercuryReadablePipeline:
    """Pipeline autonome pour l'extraction readable avec Mercury Parser.

    Autonomous pipeline for readable content extraction using Mercury Parser.

    This class orchestrates the entire extraction process including fetching content,
    applying merge strategies, extracting media and links, and updating the database.
    It supports batch processing, automatic retries, and fallback to Wayback Machine.
    """

    def __init__(self,
                 mercury_path: str = "mercury-parser",
                 merge_strategy: MergeStrategy = MergeStrategy.SMART_MERGE,
                 batch_size: int = 10,
                 max_retries: int = 3,
                 llm_enabled: bool = False):
        """Initialize the Mercury Parser readable pipeline.

        Args:
            mercury_path: Path or command to invoke Mercury Parser CLI.
            merge_strategy: Strategy for merging Mercury data with existing data.
            batch_size: Number of expressions to process concurrently in each batch.
            max_retries: Maximum number of retry attempts for failed extractions.
            llm_enabled: Whether to use LLM-based relevance validation via OpenRouter.

        Notes:
            Statistics are tracked in self.stats dictionary including processed count,
            updates, errors, skipped items, and Wayback Machine usage.
        """
        self.mercury_path = mercury_path
        self.merge_strategy = merge_strategy
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
        self.llm_enabled = llm_enabled
        self.stats = {
            'processed': 0,
            'updated': 0,
            'errors': 0,
            'skipped': 0,
            'wayback_used': 0
        }

    async def process_land(self,
                          land: model.Land,
                          limit: Optional[int] = None,
                          depth: Optional[int] = None) -> Dict[str, Any]:
        """Point d'entrée principal du pipeline.

        Main entry point for the pipeline processing.

        Args:
            land: The land object to process.
            limit: Maximum number of expressions to process (None for unlimited).
            depth: Filter expressions by specific depth level (None for all depths).

        Returns:
            Dictionary containing pipeline statistics including processed count,
            updated count, errors, skipped items, and success rate.

        Notes:
            - Retrieves the land dictionary for relevance calculation
            - Fetches expressions that have been crawled but not yet readable-processed
            - Processes expressions in configurable batch sizes for efficiency
            - Updates statistics throughout processing
            - Returns comprehensive statistics via _get_pipeline_stats()
        """
        self.logger.info(f"Starting readable pipeline for land: {land.name}")

        # Récupération du dictionnaire du land pour le calcul de pertinence
        dictionary = get_land_dictionary(land)

        # Récupération des expressions à traiter
        expressions = self._get_expressions_to_process(land, limit, depth)

        # Traitement par batch
        total_expressions = len(expressions)
        for i in range(0, total_expressions, self.batch_size):
            batch = expressions[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total_expressions + self.batch_size - 1) // self.batch_size

            self.logger.info(f"Processing batch {batch_num}/{total_batches}")
            await self._process_batch(batch, dictionary)

        return self._get_pipeline_stats()

    def _get_expressions_to_process(self,
                                    land: model.Land,
                                    limit: Optional[int],
                                    depth: Optional[int]) -> List[model.Expression]:
        """Récupère les expressions à traiter selon les critères.

        Retrieve expressions to process based on filtering criteria.

        Args:
            land: The land object to retrieve expressions from.
            limit: Maximum number of expressions to retrieve (None for unlimited).
            depth: Filter expressions by specific depth level (None for all depths).

        Returns:
            List of Expression objects that match the criteria, ordered by fetch date
            and depth, with never-processed expressions prioritized.

        Notes:
            Only returns expressions that have been fetched (fetched_at not null) but
            not yet processed through the readable pipeline (readable_at is null).
        """
        query = model.Expression.select().where(
            (model.Expression.land == land) &
            (model.Expression.fetched_at.is_null(False)) &
            (model.Expression.readable_at.is_null(True))
        )

        # Filtre par profondeur si spécifié
        if depth is not None:
            query = query.where(model.Expression.depth == depth)

        # Ordre par priorité : d'abord celles jamais traitées, puis par date
        query = query.order_by(
            model.Expression.fetched_at.asc(nulls='first'),
            model.Expression.depth.asc()
        )

        if limit:
            query = query.limit(limit)

        return list(query)

    async def _process_batch(self,
                             expressions: List[model.Expression],
                             dictionary) -> None:
        """Traite un batch d'expressions en parallèle.

        Process a batch of expressions concurrently.

        Args:
            expressions: List of Expression objects to process in this batch.
            dictionary: Land dictionary for relevance calculation.

        Returns:
            None. Updates self.stats with processing results.

        Notes:
            Uses asyncio.gather to process all expressions in the batch concurrently.
            Exceptions from individual expressions are caught and logged without
            stopping the entire batch.
        """
        tasks = []
        for expression in expressions:
            task = self._process_single_expression(expression, dictionary)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Traitement des résultats
        for expression, result in zip(expressions, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error processing {expression.url}: {result}")
                self.stats['errors'] += 1
            else:
                self.stats['processed'] += 1

    async def _process_single_expression(self,
                                         expression: model.Expression,
                                         dictionary) -> Optional[ExpressionUpdate]:
        """Traite une expression unique avec Mercury Parser.

        Process a single expression with Mercury Parser extraction.

        Args:
            expression: The Expression object to process.
            dictionary: Land dictionary for relevance calculation.

        Returns:
            ExpressionUpdate object with field changes and additions, or None if
            extraction failed or no updates were needed.

        Raises:
            Exception: Re-raises any exception that occurs during processing after
                       logging the error.

        Notes:
            - Attempts Mercury Parser extraction with automatic Wayback fallback
            - Updates readable_at timestamp even on failure
            - Applies configured merge strategy for content fusion
            - Recalculates relevance if content was updated
            - Updates statistics counters (updated, skipped, errors)
        """
        try:
            print(f"🔄 Processing URL: {expression.url}")
            # Extraction avec Mercury Parser
            mercury_result = await self._extract_with_mercury(str(expression.url))

            if mercury_result.error:
                self.logger.warning(f"Mercury extraction failed for {expression.url}: {mercury_result.error}")
                setattr(expression, 'readable_at', datetime.now())
                expression.save()
                print(f"🕒 Marked readable attempt (failure) for {expression.url}")
                return None

            # Préparation de la mise à jour
            update = self._prepare_expression_update(expression, mercury_result)

            # Application des mises à jour (même si aucune modification pour timestamp)
            self._apply_updates(expression, update, dictionary)
            
            if not update.field_updates and not update.media_additions and not update.link_additions:
                self.logger.debug(f"No content updates needed for {expression.url}")
                self.stats['skipped'] += 1
                print(f"⏩ Skipped URL (no changes): {expression.url}")
            else:
                self.stats['updated'] += 1

            return update

        except Exception as e:
            self.logger.error(f"Failed to process {expression.url}: {e}")
            raise

    async def _extract_with_mercury(self, url: str) -> MercuryResult:
        """Extraction Mercury + fallback Wayback si nécessaire.

        Extract content using Mercury Parser with automatic Wayback Machine fallback.

        Args:
            url: The URL to extract content from.

        Returns:
            MercuryResult object containing extracted content or error information.

        Notes:
            - First attempts extraction from the live URL
            - If extraction fails, searches for the earliest Wayback Machine snapshot
            - Attempts extraction from Wayback snapshot if available
            - Adds metadata to result indicating Wayback source and snapshot details
            - Updates wayback_used counter in statistics when fallback is successful
        """
        primary_result = await self._run_mercury(url)

        if not primary_result.error:
            return primary_result

        error_message = primary_result.error
        self.logger.warning(f"Mercury extraction failed for {url}: {error_message}")
        print(f"⚠️ Mercury failed for {url}: {error_message}")

        snapshot = await self._fetch_wayback_first_snapshot(url)
        if not snapshot:
            print(f"🚫 No Wayback snapshot available for {url}")
            return primary_result

        snapshot_url, snapshot_timestamp = snapshot
        self.logger.info(f"Found Wayback snapshot {snapshot_url} for {url}")
        print(f"📼 Wayback snapshot found ({snapshot_timestamp}) for {url}")

        wayback_result = await self._run_mercury(snapshot_url)
        if wayback_result.error:
            self.logger.warning(
                f"Mercury failed on Wayback snapshot {snapshot_url} for {url}: {wayback_result.error}"
            )
            print(
                f"❌ Mercury failed on Wayback snapshot {snapshot_url}: {wayback_result.error}"
            )
            return primary_result

        if wayback_result.raw_response is None:
            wayback_result.raw_response = {}
        wayback_result.raw_response['source'] = 'wayback'
        wayback_result.raw_response['wayback_timestamp'] = snapshot_timestamp
        wayback_result.raw_response['wayback_snapshot_url'] = snapshot_url
        wayback_result.raw_response['original_url'] = url

        self.stats['wayback_used'] += 1
        print(f"✅ Mercury succeeded via Wayback snapshot for {url}")
        return wayback_result

    async def _run_mercury(self, url: str) -> MercuryResult:
        """Exécute Mercury Parser et retourne le résultat brut.

        Execute Mercury Parser CLI and return the raw result.

        Args:
            url: The URL to extract content from.

        Returns:
            MercuryResult object populated with extracted data or error information.

        Notes:
            - Executes Mercury Parser as a subprocess with markdown and media extraction
            - Implements exponential backoff retry logic (max_retries attempts)
            - Parses JSON output and populates all MercuryResult fields
            - Sets extraction_timestamp to track when extraction occurred
            - Returns result with error field populated if all attempts fail
            - Handles JSON decode errors and subprocess execution failures
        """
        result = MercuryResult(extraction_timestamp=datetime.now())

        for attempt in range(self.max_retries):
            try:
                proc = await asyncio.create_subprocess_shell(
                    f'{self.mercury_path} "{url}" --format=markdown --extract-media --extract-links',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await proc.communicate()

                if proc.returncode != 0:
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    result.error = error_msg
                    break

                data = json.loads(stdout.decode())
                result.raw_response = data

                result.title = data.get('title')
                result.content = data.get('content')
                result.markdown = data.get('markdown', data.get('content'))
                result.lead_image_url = data.get('lead_image_url')
                result.date_published = data.get('date_published')
                result.author = data.get('author')
                result.excerpt = data.get('excerpt')
                result.domain = data.get('domain')
                result.word_count = data.get('word_count')
                result.direction = data.get('direction')
                result.total_pages = data.get('total_pages')
                result.rendered_pages = data.get('rendered_pages')
                result.next_page_url = data.get('next_page_url')

                self._extract_media_and_links(data, result)

                return result

            except json.JSONDecodeError as e:
                result.error = f"Invalid JSON response: {e}"
                break
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                result.error = str(e)
                break

        return result

    async def _fetch_wayback_first_snapshot(self, url: str) -> Optional[Tuple[str, str]]:
        """Récupère la première snapshot Wayback disponible pour l'URL donnée.

        Fetch the first available Wayback Machine snapshot for a given URL.

        Args:
            url: The URL to search for in the Wayback Machine.

        Returns:
            Tuple of (snapshot_url, timestamp) if a snapshot is found, None otherwise.
            The snapshot_url is the full Wayback Machine URL for accessing the archived page.
            The timestamp is the capture date in Wayback format (YYYYMMDDhhmmss).

        Notes:
            - Uses Wayback CDX API to search for snapshots
            - First tries to find snapshots with HTTP 200 status code
            - Falls back to any available snapshot if 200 status search fails
            - Returns the earliest available snapshot
            - Uses 10-second timeout for API requests
            - Handles various failure modes gracefully (HTTP errors, JSON parse errors)
        """
        base_url = "https://web.archive.org/cdx/search/cdx"
        params_common = {
            'url': url,
            'output': 'json',
            'limit': '1',
            'matchType': 'exact'
        }
        queries = [
            {**params_common, 'filter': 'statuscode:200'},
            params_common
        ]

        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for query in queries:
                try:
                    async with session.get(base_url, params=query) as response:
                        if response.status != 200:
                            self.logger.debug(
                                f"Wayback lookup HTTP {response.status} for {url} with params {query}"
                            )
                            continue
                        try:
                            payload = await response.json(content_type=None)
                        except Exception as e:
                            self.logger.debug(f"Wayback lookup JSON parse error for {url}: {e}")
                            continue

                        if not isinstance(payload, list) or len(payload) < 2:
                            continue

                        snapshot_row = payload[1]
                        if len(snapshot_row) < 2:
                            continue

                        timestamp = snapshot_row[1]
                        original = snapshot_row[2] if len(snapshot_row) > 2 else url
                        snapshot_url = f"https://web.archive.org/web/{timestamp}/{original}"
                        return snapshot_url, timestamp
                except aiohttp.ClientError as e:
                    self.logger.debug(f"Wayback lookup client error for {url}: {e}")
                except Exception as e:
                    self.logger.debug(f"Wayback lookup unexpected error for {url}: {e}")

        return None

    def _extract_media_and_links(self, data: Dict, result: MercuryResult) -> None:
        """Extrait les médias et liens du résultat Mercury.

        Extract media items and links from Mercury Parser JSON response.

        Args:
            data: The raw JSON dictionary from Mercury Parser.
            result: MercuryResult object to populate with extracted media and links.

        Returns:
            None. Modifies result.media and result.links lists in place.

        Notes:
            - Extracts images from 'images' field with src, alt, and title attributes
            - Extracts videos from 'videos' field with src and poster attributes
            - Extracts links from 'links' field with href, text, and title attributes
            - Handles both string URLs and dictionary objects for flexibility
        """
        # Extraction des images
        if 'images' in data:
            for img in data.get('images', []):
                media_item = {
                    'type': 'image',
                    'url': img.get('src', img),
                    'alt': img.get('alt', '') if isinstance(img, dict) else '',
                    'title': img.get('title', '') if isinstance(img, dict) else ''
                }
                result.media.append(media_item)

        # Extraction des vidéos
        if 'videos' in data:
            for video in data.get('videos', []):
                media_item = {
                    'type': 'video',
                    'url': video.get('src', video),
                    'poster': video.get('poster', '') if isinstance(video, dict) else ''
                }
                result.media.append(media_item)

        # Extraction des liens
        if 'links' in data:
            for link in data.get('links', []):
                link_item = {
                    'url': link.get('href', link),
                    'text': link.get('text', '') if isinstance(link, dict) else '',
                    'title': link.get('title', '') if isinstance(link, dict) else ''
                }
                result.links.append(link_item)

    def _prepare_expression_update(self,
                                   expression: model.Expression,
                                   mercury_result: MercuryResult) -> ExpressionUpdate:
        """Prépare les mises à jour en appliquant la stratégie de fusion.

        Prepare expression updates by applying the configured merge strategy.

        Args:
            expression: The Expression object to update.
            mercury_result: MercuryResult containing extracted data from Mercury Parser.

        Returns:
            ExpressionUpdate object with field updates, media additions, and link additions.

        Notes:
            - Applies merge strategy to title, description, readable, lang, and published_at
            - Tracks old and new values for each updated field
            - Extracts media and links from the final merged markdown content
            - Only includes fields where Mercury provided non-null values
            - Media and links are extracted from markdown after merge, ensuring consistency
        """
        update = ExpressionUpdate(
            expression_id=expression.get_id(),
            field_updates={},
            media_additions=[],
            link_additions=[],
            update_reason=f"Mercury extraction at {mercury_result.extraction_timestamp}"
        )

        # Mapping des champs à vérifier
        field_mapping = {
            'title': mercury_result.title,
            'description': mercury_result.excerpt,
            'readable': mercury_result.markdown,
            'lang': mercury_result.direction,
            'published_at': self._parse_date(mercury_result.date_published)
        }

        # Application de la stratégie de fusion pour chaque champ
        for field_name, mercury_value in field_mapping.items():
            if mercury_value is None:
                continue

            current_value = getattr(expression, field_name, None)
            new_value = self._apply_merge_strategy(current_value, mercury_value, field_name)

            if new_value != current_value:
                update.field_updates[field_name] = (current_value, new_value)

        # On détermine le markdown final (après fusion)
        readable_final = None
        if 'readable' in update.field_updates:
            readable_final = update.field_updates['readable'][1]
        else:
            readable_final = getattr(expression, 'readable', None)

        # Extraction des médias et liens à partir du markdown final
        update.media_additions = self._extract_media_from_markdown(readable_final, str(expression.url))
        update.link_additions = self._extract_links_from_markdown(readable_final, str(expression.url))

        return update

    def _apply_merge_strategy(self,
                              current_value: Any,
                              mercury_value: Any,
                              field_name: str) -> Any:
        """Applique la stratégie de fusion configurée.

        Apply the configured merge strategy to determine the final value.

        Args:
            current_value: The existing value in the database.
            mercury_value: The new value from Mercury Parser.
            field_name: Name of the field being merged (affects smart merge logic).

        Returns:
            The final value to use after applying the merge strategy.

        Notes:
            Merge logic:
            - If current value is empty: use Mercury value
            - If Mercury value is empty: keep current value
            - If both have values: apply strategy (MERCURY_PRIORITY, PRESERVE_EXISTING, or SMART_MERGE)
            - SMART_MERGE uses field-specific logic (see _smart_merge method)
        """
        # Si la base est vide, on prend Mercury
        if not current_value:
            return mercury_value

        # Si Mercury est vide, on garde la base
        if not mercury_value:
            return current_value

        # Les deux ont des valeurs, on applique la stratégie
        if self.merge_strategy == MergeStrategy.MERCURY_PRIORITY:
            return mercury_value
        elif self.merge_strategy == MergeStrategy.PRESERVE_EXISTING:
            return current_value
        elif self.merge_strategy == MergeStrategy.SMART_MERGE:
            return self._smart_merge(current_value, mercury_value, field_name)

        return current_value

    def _smart_merge(self, current_value: Any, mercury_value: Any, field_name: str) -> Any:
        """Fusion intelligente selon le type de champ.

        Intelligent merge logic with field-specific strategies.

        Args:
            current_value: The existing value in the database.
            mercury_value: The new value from Mercury Parser.
            field_name: Name of the field being merged.

        Returns:
            The final value selected based on field-specific heuristics.

        Notes:
            Field-specific strategies:
            - published_at: Prefers earlier datetime (oldest publication date)
            - title: Prefers longer, more informative title
            - readable: Prefers Mercury (cleaner extraction)
            - description: Prefers longer description
            - Other fields: Mercury value takes priority by default
        """
        if field_name == 'published_at':
            if mercury_value is None:
                return current_value
            if not isinstance(mercury_value, datetime):
                return current_value
            current_dt = current_value if isinstance(current_value, datetime) else None
            return prefer_earlier_datetime(current_dt, mercury_value)

        if field_name == 'title':
            # Préfère le titre le plus long et informatif
            if len(str(mercury_value)) > len(str(current_value)):
                return mercury_value
            return current_value

        elif field_name == 'readable':
            # Pour le contenu, préfère Mercury qui est généralement plus propre
            return mercury_value

        elif field_name == 'description':
            # Garde la description la plus longue
            if len(str(mercury_value)) > len(str(current_value)):
                return mercury_value
            return current_value

        else:
            # Par défaut, Mercury a priorité pour les autres champs
            return mercury_value

    def _extract_media_from_markdown(self, markdown: Optional[str], base_url: str) -> List[Dict[str, Any]]:
        """Extrait les médias (images, vidéos) à partir du markdown final.

        Extract media items (images, videos) from markdown content.

        Args:
            markdown: The markdown content to parse for media.
            base_url: Base URL for resolving relative media URLs.

        Returns:
            List of dictionaries containing media metadata (type, url, alt, title).

        Notes:
            - Parses markdown image syntax: ![alt](url "title")
            - Converts relative URLs to absolute using base_url
            - Currently only extracts images (video extraction can be added)
            - Empty or None markdown returns empty list
        """
        import re
        from urllib.parse import urljoin

        if not markdown:
            return []

        media = []
        # Images: ![alt](url "title")
        img_pattern = r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+"([^"]*)")?\)'
        for match in re.finditer(img_pattern, markdown):
            alt, url, title = match.groups()
            url = urljoin(base_url, url)
            media.append({'type': 'img', 'url': url, 'alt': alt or '', 'title': title or ''})

        # Vidéos (liens markdown ou HTML <video> tags, à adapter si besoin)
        # Ici, on ne traite que les images pour le markdown standard

        return media

    def _extract_links_from_markdown(self, markdown: Optional[str], base_url: str) -> List[Dict[str, Any]]:
        """Extrait les liens à partir du markdown final.

        Extract hyperlinks from markdown content.

        Args:
            markdown: The markdown content to parse for links.
            base_url: Base URL for resolving relative link URLs.

        Returns:
            List of dictionaries containing link metadata (url, text, title).
            Duplicate URLs are automatically filtered out.

        Notes:
            - Parses markdown link syntax: [text](url "title")
            - Converts relative URLs to absolute using base_url
            - Deduplicates links by URL
            - Empty or None markdown returns empty list
        """
        import re
        from urllib.parse import urljoin

        if not markdown:
            return []

        links = []
        # Liens markdown: [text](url "title")
        link_pattern = r'\[([^\]]+)\]\(([^)\s]+)(?:\s+"([^"]*)")?\)'
        seen_urls = set()
        for match in re.finditer(link_pattern, markdown):
            text, url, title = match.groups()
            url = urljoin(base_url, url)
            if url not in seen_urls:
                seen_urls.add(url)
                links.append({'url': url, 'text': text or '', 'title': title or ''})

        return links

    def _apply_updates(self,
                       expression: model.Expression,
                       update: ExpressionUpdate,
                       dictionary) -> None:
        """Applique les mises à jour à la base de données.

        Apply all updates to the expression in the database.

        Args:
            expression: The Expression object to update.
            update: ExpressionUpdate containing all changes to apply.
            dictionary: Land dictionary for recalculating relevance.

        Returns:
            None. Updates expression and related database records in place.

        Notes:
            - Updates all changed fields on the expression
            - Always sets readable_at timestamp to now
            - Recalculates relevance if readable content changed
            - Optionally validates relevance using OpenRouter LLM if enabled
            - Sets approved_at if relevance is positive
            - Deletes existing media before adding new ones (ensures consistency)
            - Creates new ExpressionLink records for discovered links
            - Saves expression after all updates
        """
        # Mise à jour des champs de l'expression
        for field_name, (old_value, new_value) in update.field_updates.items():
            setattr(expression, field_name, new_value)
            self.logger.debug(f"Updated {field_name}: {old_value} -> {new_value}")

        # Mise à jour du timestamp
        setattr(expression, 'readable_at', datetime.now())
        print(f"🕒 Updated timestamp for URL {expression.url}: {expression.readable_at}")

        # Recalcul de la pertinence si le contenu a changé (avec garde-fou OpenRouter)
        if 'readable' in update.field_updates:
            relevance = None
            try:
                import settings
                relevance = self._calculate_relevance(dictionary, expression)
                if self.llm_enabled and getattr(settings, 'openrouter_enabled', False) and settings.openrouter_api_key and settings.openrouter_model:
                    from .llm_openrouter import is_relevant_via_openrouter
                    verdict = is_relevant_via_openrouter(expression.land, expression)
                    if verdict is False:
                        relevance = 0
            except Exception as e:
                print(f"OpenRouter gate error for {expression.url}: {e}")
                relevance = self._calculate_relevance(dictionary, expression)

            setattr(expression, 'relevance', relevance)
            if relevance and relevance > 0:
                setattr(expression, 'approved_at', datetime.now())

        # Sauvegarde de l'expression
        expression.save()

        # Suppression des anciens médias AVANT ajout des nouveaux (cohérence stricte)
        model.Media.delete().where(model.Media.expression == expression).execute()

        # Ajout des nouveaux médias
        for media_data in update.media_additions:
            model.Media.create(
                expression=expression,
                url=media_data['url'],
                type=media_data['type']
            )

        # Ajout des nouveaux liens
        self._update_expression_links(expression, update.link_additions)

    def _update_expression_links(self,
                                 expression: model.Expression,
                                 new_links: List[Dict[str, Any]]) -> None:
        """Met à jour les liens de l'expression.

        Update the expression's outgoing links in the database.

        Args:
            expression: The source Expression object.
            new_links: List of dictionaries containing link metadata (url, text, title).

        Returns:
            None. Updates ExpressionLink records in database.

        Notes:
            - Deletes all existing outgoing links from this expression
            - Creates target expressions for new links (one depth level deeper)
            - Creates ExpressionLink records connecting source to targets
            - Silently ignores link creation failures (duplicate constraints)
            - Uses the expression's land for creating target expressions
        """
        model.ExpressionLink.delete().where(
            model.ExpressionLink.source == expression
        ).execute()

        for link_data in new_links:
            target_expression = self._get_or_create_expression(
                expression.land,  # Utilise le land directement
                link_data['url'],
                int(expression.depth) + 1
            )

            if target_expression:
                try:
                    model.ExpressionLink.create(
                        source=expression,
                        target=target_expression
                    )
                except:
                    pass

    def _calculate_relevance(self, dictionary, expression: model.Expression) -> int:
        """Calcule la pertinence selon le dictionnaire du land.

        Calculate expression relevance score based on land dictionary.

        Args:
            dictionary: Land dictionary containing relevant terms.
            expression: Expression object to score.

        Returns:
            Integer relevance score (typically 0-100+ range).

        Notes:
            Delegates to core.expression_relevance for the actual calculation.
        """
        from .core import expression_relevance
        return expression_relevance(dictionary, expression)

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse une date depuis Mercury.

        Parse a date string from Mercury Parser into a datetime object.

        Args:
            date_str: Date string to parse (various formats supported).

        Returns:
            Datetime object if parsing succeeds, None otherwise.

        Notes:
            Attempts parsing with multiple common date formats:
            - ISO 8601 with milliseconds: %Y-%m-%dT%H:%M:%S.%fZ
            - Simple date: %Y-%m-%d
            - ISO 8601 without milliseconds: %Y-%m-%dT%H:%M:%SZ
            Returns None if the string is empty or all formats fail.
        """
        if not date_str:
            return None
        try:
            for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%d', '%Y-%m-%dT%H:%M:%SZ']:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            return None
        except:
            return None

    def _resolve_url(self, url: str, base_url: str) -> str:
        """Résout une URL relative en URL absolue.

        Resolve a relative URL to an absolute URL.

        Args:
            url: URL to resolve (may be relative or absolute).
            base_url: Base URL to resolve relative URLs against.

        Returns:
            Absolute URL string. Returns empty string if url is empty.

        Notes:
            - URLs starting with http://, https://, or data: are returned as-is
            - Relative URLs are resolved using urllib.parse.urljoin
        """
        from urllib.parse import urljoin

        if not url:
            return ''
        if url.startswith(('http://', 'https://', 'data:')):
            return url
        return urljoin(base_url, url)

    def _is_valid_link(self, url: str) -> bool:
        """Vérifie si un lien est valide pour l'ajout.

        Check if a link is valid for adding to the database.

        Args:
            url: URL to validate.

        Returns:
            True if the URL is crawlable, False otherwise.

        Notes:
            Delegates to core.is_crawlable for the actual validation logic.
        """
        from .core import is_crawlable
        return is_crawlable(url)

    def _get_or_create_expression(self,
                                  land: model.Land,
                                  url: str,
                                  depth: int) -> Optional[model.Expression]:
        """Récupère ou crée une expression.

        Retrieve an existing expression or create a new one.

        Args:
            land: The land to associate the expression with.
            url: URL of the expression.
            depth: Crawl depth level for the expression.

        Returns:
            Expression object if creation/retrieval succeeds, None otherwise.

        Notes:
            Delegates to core.add_expression which may return bool or Expression.
            Only returns Expression objects, converts other return types to None.
        """
        from .core import add_expression
        result = add_expression(land, url, depth)
        # add_expression peut retourner bool ou Expression
        if isinstance(result, model.Expression):
            return result
        return None

    def _get_pipeline_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du pipeline.

        Return pipeline execution statistics.

        Returns:
            Dictionary containing:
            - processed: Total number of expressions processed
            - updated: Number of expressions successfully updated
            - errors: Number of processing errors
            - skipped: Number of expressions skipped (no changes needed)
            - wayback_used: Number of times Wayback Machine was used
            - success_rate: Percentage of successful updates (0-100)

        Notes:
            Success rate is calculated as (updated / processed * 100).
            Returns 0% if no expressions were processed.
        """
        return {
            'processed': self.stats['processed'],
            'updated': self.stats['updated'],
            'errors': self.stats['errors'],
            'skipped': self.stats['skipped'],
            'wayback_used': self.stats['wayback_used'],
            'success_rate': (self.stats['updated'] / self.stats['processed'] * 100)
                           if self.stats['processed'] > 0 else 0
        }


async def run_readable_pipeline(land: model.Land,
                              limit: Optional[int] = None,
                              depth: Optional[int] = None,
                              merge_strategy: str = 'smart_merge',
                              llm_enabled: bool = False) -> Tuple[int, int]:
    """Point d'entrée pour le contrôleur.

    Entry point for the readable pipeline controller.

    Args:
        land: The land object to process expressions from.
        limit: Maximum number of expressions to process (None for unlimited).
        depth: Filter expressions by specific depth level (None for all depths).
        merge_strategy: Merge strategy name ('mercury_priority', 'preserve_existing',
                       or 'smart_merge').
        llm_enabled: Whether to enable LLM-based relevance validation via OpenRouter.

    Returns:
        Tuple of (processed_count, error_count) indicating pipeline execution results.

    Raises:
        Exception: Re-raises any exception that occurs during pipeline execution.

    Notes:
        - Creates and configures a MercuryReadablePipeline instance
        - Prints progress messages to console during execution
        - Maps merge strategy strings to MergeStrategy enum values
        - Defaults to SMART_MERGE if an invalid strategy is provided
        - Returns statistics about expressions processed, updated, and errors
    """
    strategy_map = {
        'mercury_priority': MergeStrategy.MERCURY_PRIORITY,
        'preserve_existing': MergeStrategy.PRESERVE_EXISTING,
        'smart_merge': MergeStrategy.SMART_MERGE
    }

    pipeline = MercuryReadablePipeline(
        merge_strategy=strategy_map.get(merge_strategy, MergeStrategy.SMART_MERGE),
        llm_enabled=llm_enabled
    )

    print(f"🚀 Starting readable pipeline for land: {land.name}")
    print(f"🔧 Merge strategy: {merge_strategy}")
    depth_display = depth if depth is not None else 'all'
    print(f"📦 Processing limit: {limit or 'unlimited'}, depth: {depth_display}")
    print(f"🤖 OpenRouter validation: {'enabled' if llm_enabled else 'disabled'}")

    try:
        stats = await pipeline.process_land(land, limit, depth)
        print(f"✅ Completed processing {stats['processed']} expressions")
        print(f"✔️ Updated: {stats['updated']}, Errors: {stats['errors']}, Skipped: {stats['skipped']}")
        if stats.get('wayback_used'):
            print(f"📼 Wayback snapshots used: {stats['wayback_used']}")
        return stats['processed'], stats['errors']
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        raise
