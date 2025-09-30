"""
Mercury Parser Readable Pipeline - Système autonome d'enrichissement
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
from .core import get_land_dictionary


class MergeStrategy(Enum):
    """Stratégies de fusion des données"""
    MERCURY_PRIORITY = "mercury_priority"     # Mercury écrase toujours
    PRESERVE_EXISTING = "preserve_existing"   # Garde l'existant si non vide
    SMART_MERGE = "smart_merge"               # Fusion intelligente


@dataclass
class MercuryResult:
    """Structure des résultats Mercury Parser"""
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
    """Structure pour les mises à jour d'expression"""
    expression_id: int
    field_updates: Dict[str, Tuple[Any, Any]]  # (old_value, new_value)
    media_additions: List[Dict[str, Any]]
    link_additions: List[Dict[str, Any]]
    update_reason: str


class MercuryReadablePipeline:
    """Pipeline autonome pour l'extraction readable avec Mercury Parser"""

    def __init__(self,
                 mercury_path: str = "mercury-parser",
                 merge_strategy: MergeStrategy = MergeStrategy.SMART_MERGE,
                 batch_size: int = 10,
                 max_retries: int = 3,
                 llm_enabled: bool = False):
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
        """
        Point d'entrée principal du pipeline

        Args:
            land: Land à traiter
            limit: Nombre maximum d'expressions à traiter
            depth: Profondeur maximale des expressions à traiter

        Returns:
            Statistiques du traitement
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
        """Récupère les expressions à traiter selon les critères"""
        query = model.Expression.select().where(
            (model.Expression.land == land) &
            (model.Expression.readable.is_null(True))
        )

        # Filtre par profondeur si spécifié
        if depth is not None:
            query = query.where(model.Expression.depth == depth)

        # Ordre par priorité : d'abord celles jamais traitées, puis par date
        query = query.order_by(
            model.Expression.readable_at.asc(nulls='first'),
            model.Expression.depth.asc()
        )

        if limit:
            query = query.limit(limit)

        return list(query)

    async def _process_batch(self,
                             expressions: List[model.Expression],
                             dictionary) -> None:
        """Traite un batch d'expressions en parallèle"""
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
        """
        Traite une expression unique avec Mercury Parser
        """
        try:
            print(f"🔄 Processing URL: {expression.url}")
            # Extraction avec Mercury Parser
            mercury_result = await self._extract_with_mercury(str(expression.url))

            if mercury_result.error:
                self.logger.warning(f"Mercury extraction failed for {expression.url}: {mercury_result.error}")
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
        """Extraction Mercury + fallback Wayback si nécessaire."""
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
        """Exécute Mercury Parser et retourne le résultat brut."""
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
        """Récupère la première snapshot Wayback disponible pour l'URL donnée."""
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
        """Extrait les médias et liens du résultat Mercury"""
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
        """
        Prépare les mises à jour en appliquant la stratégie de fusion.
        Extraction des médias et liens à partir du markdown final (readable) après fusion.
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
        """
        Applique la stratégie de fusion configurée

        Logique:
        - Si base vide -> remplit avec Mercury
        - Si Mercury plein et base pleine -> selon stratégie
        - Si base pleine et Mercury vide -> garde base
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
        """
        Fusion intelligente selon le type de champ
        """
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
        """
        Extrait les médias (images, vidéos) à partir du markdown final.
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
        """
        Extrait les liens à partir du markdown final.
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
        """
        Applique les mises à jour à la base de données
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
        """Met à jour les liens de l'expression"""
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
        """Calcule la pertinence selon le dictionnaire du land"""
        from .core import expression_relevance
        return expression_relevance(dictionary, expression)

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse une date depuis Mercury"""
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
        """Résout une URL relative en URL absolue"""
        from urllib.parse import urljoin

        if not url:
            return ''
        if url.startswith(('http://', 'https://', 'data:')):
            return url
        return urljoin(base_url, url)

    def _is_valid_link(self, url: str) -> bool:
        """Vérifie si un lien est valide pour l'ajout"""
        from .core import is_crawlable
        return is_crawlable(url)

    def _get_or_create_expression(self,
                                  land: model.Land,
                                  url: str,
                                  depth: int) -> Optional[model.Expression]:
        """Récupère ou crée une expression"""
        from .core import add_expression
        result = add_expression(land, url, depth)
        # add_expression peut retourner bool ou Expression
        if isinstance(result, model.Expression):
            return result
        return None

    def _get_pipeline_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du pipeline"""
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
    """
    Point d'entrée pour le contrôleur

    Returns:
        Tuple (processed_count, error_count)
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
    print(f"📦 Processing limit: {limit or 'unlimited'}, depth: {depth or 'all'}")
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
