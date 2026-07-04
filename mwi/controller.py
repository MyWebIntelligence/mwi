"""
Application controller
"""
import asyncio
import os
import sys

from peewee import JOIN, fn
import aiohttp

import settings
from . import core
from . import model


def _get_event_loop():
    """Return the current event loop, creating one if none is set.

    Since Python 3.12, `asyncio.get_event_loop()` raises RuntimeError when
    the policy has no current loop (e.g. after an `asyncio.run()` call
    elsewhere in the process). Controllers are synchronous entry points,
    so transparently provisioning a fresh loop is the correct behaviour.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError('event loop is closed')
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


class DbController:
    """
    Db controller class
    """

    @staticmethod
    def migrate(args: core.Namespace):
        """Execute database migrations to update schema.

        Attempts to import and run the MigrationManager from various locations
        (migrations, migration, .migrations, .migration). After running pending
        migrations, performs additional safety checks to ensure LLM-related
        columns exist in the expression table.

        Args:
            args: Namespace object containing command-line arguments.

        Returns:
            int: 1 if migration completed successfully, 0 if migration manager
                could not be located.

        Notes:
            This method tries multiple import strategies to locate the migration
            manager module. It also includes a safety net to add validllm and
            validmodel columns if they are missing from the expression table.
        """
        # Support 'migrations', 'migration' and hidden variants '.migrations' / '.migration'
        MigrationManager = None
        # Try standard package imports first
        try:
            from migrations.migrate import MigrationManager as _MM  # type: ignore
            MigrationManager = _MM
        except Exception:
            try:
                from migration.migrate import MigrationManager as _MM2  # type: ignore
                MigrationManager = _MM2
            except Exception:
                # Fallback to file-based dynamic import in common locations
                import importlib.util, os
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
                candidate_paths = [
                    os.path.join(repo_root, 'migrations', 'migrate.py'),
                    os.path.join(repo_root, 'migration', 'migrate.py'),
                    os.path.join(repo_root, '.migrations', 'migrate.py'),
                    os.path.join(repo_root, '.migration', 'migrate.py'),
                ]
                for module_path in candidate_paths:
                    if os.path.exists(module_path):
                        spec = importlib.util.spec_from_file_location('mywi_migrations_migrate', module_path)
                        migrate_module = importlib.util.module_from_spec(spec)
                        assert spec and spec.loader
                        spec.loader.exec_module(migrate_module)  # type: ignore
                        MigrationManager = getattr(migrate_module, 'MigrationManager', None)
                        if MigrationManager is not None:
                            break
        if MigrationManager is None:
            print("[migrate] Error: unable to locate migration manager module")
            return 0
        manager = MigrationManager()
        manager.run_pending_migrations()

        # Filet de sécurité ad hoc: s'assurer que les nouvelles colonnes LLM existent
        try:
            cols = [row[1] for row in model.DB.execute_sql("PRAGMA table_info('expression')").fetchall()]
            if 'validllm' not in cols:
                model.DB.execute_sql("ALTER TABLE expression ADD COLUMN validllm TEXT DEFAULT NULL")
                print("[migrate] Added missing column expression.validllm")
            if 'validmodel' not in cols:
                model.DB.execute_sql("ALTER TABLE expression ADD COLUMN validmodel TEXT DEFAULT NULL")
                print("[migrate] Added missing column expression.validmodel")
            # HTML storage feature columns
            if 'html' not in cols:
                model.DB.execute_sql("ALTER TABLE expression ADD COLUMN html TEXT DEFAULT NULL")
                print("[migrate] Added missing column expression.html")
            land_cols = [row[1] for row in model.DB.execute_sql("PRAGMA table_info('land')").fetchall()]
            if 'fullhtml' not in land_cols:
                model.DB.execute_sql("ALTER TABLE land ADD COLUMN fullhtml INTEGER DEFAULT 0")
                print("[migrate] Added missing column land.fullhtml")
            # URL provenance (sprint-normalise)
            if 'original_url' not in cols:
                model.DB.execute_sql(
                    "ALTER TABLE expression ADD COLUMN original_url TEXT DEFAULT NULL")
                print("[migrate] Added missing column expression.original_url")
            # Cascade fetch audit (sprint-403)
            if 'fetch_method' not in cols:
                model.DB.execute_sql(
                    "ALTER TABLE expression ADD COLUMN fetch_method TEXT DEFAULT NULL")
                print("[migrate] Added missing column expression.fetch_method")
            # Link-context (sprint link-context, migration 012)
            link_cols = [row[1] for row in model.DB.execute_sql(
                "PRAGMA table_info('expressionlink')").fetchall()]
            for col in ('context', 'dom', 'dom_html'):
                if col not in link_cols:
                    model.DB.execute_sql(
                        f"ALTER TABLE expressionlink ADD COLUMN {col} TEXT DEFAULT NULL")
                    print(f"[migrate] Added missing column expressionlink.{col}")
        except Exception as e:
            # Non bloquant: on loggue et on continue
            print(f"[migrate] Warning: columns check failed: {e}")
        return 1

    @staticmethod
    def setup(args: core.Namespace):
        """Create database schema from model definitions.

        This is a destructive operation that drops all existing tables before
        recreating them. Requires user confirmation before proceeding.

        Args:
            args: Namespace object containing command-line arguments.

        Returns:
            int: 1 if setup completed successfully, 0 if user cancelled.

        Notes:
            This method destroys all existing data. Tables created include:
            Land, Domain, Expression, ExpressionLink, Word, LandDictionary,
            Media, Paragraph, ParagraphEmbedding, ParagraphSimilarity, Tag,
            and TaggedContent. User must type 'Y' to confirm the operation.
        """
        tables = [
            model.Land,
            model.Domain,
            model.Expression,
            model.ExpressionLink,
            model.Word,
            model.LandDictionary,
            model.Media,
            # Embedding feature tables
            getattr(model, 'Paragraph', None),
            getattr(model, 'ParagraphEmbedding', None),
            getattr(model, 'ParagraphSimilarity', None),
            # Client tagging
            model.Tag,
            model.TaggedContent,
            # Multi-API search router (sprint-searchrouter)
            getattr(model, 'SearchQuery', None),
            getattr(model, 'SearchResultLog', None),
        ]

        if core.confirm("Warning, existing data will be lost, type 'Y' to proceed : "):
            # Filter out None entries (older versions)
            tables_clean = [t for t in tables if t is not None]
            model.DB.drop_tables(tables_clean)
            model.DB.create_tables(tables_clean)
            print("Model created, setup complete")
            return 1

    @staticmethod
    def fix_archive_domains(args: core.Namespace):
        """Fix expressions that have archive.org domains instead of their original domains.

        This migration identifies expressions with URLs from web.archive.org, ghostarchive.org,
        or archive.org, extracts the original domain from the archived URL, and updates the
        domain_id to point to the correct domain.

        Args:
            args: Namespace object containing command-line arguments. Optional arguments:
                - dryrun: If True, only displays what would be changed without modifying database
                - limit: Maximum number of expressions to process (for testing)

        Returns:
            int: 1 if migration completed successfully.

        Notes:
            - Handles URLs in format: https://web.archive.org/web/{timestamp}/{original_url}
            - Creates new Domain entries if the original domain doesn't exist
            - Resets seorank to NULL so it can be re-fetched with the correct unwrapped URL
            - Reports progress every 100 expressions
            - Can be run multiple times safely (idempotent)
        """
        dryrun = core.get_dryrun(args)
        limit = core.get_arg_option('limit', args, set_type=int, default=0)

        # Find all expressions with archive domains
        archive_domains = model.Domain.select().where(
            (model.Domain.name == 'web.archive.org') |
            (model.Domain.name == 'archive.org') |
            (model.Domain.name == 'ghostarchive.org')
        )

        archive_domain_ids = [d.id for d in archive_domains]

        if not archive_domain_ids:
            print("No archive.org domains found in database")
            return 1

        # Only select the columns we need to avoid UTF-8 decoding errors in 'readable' column
        query = model.Expression.select(
            model.Expression.id,
            model.Expression.url,
            model.Expression.domain_id
        ).where(
            model.Expression.domain_id.in_(archive_domain_ids)
        )

        if limit > 0:
            query = query.limit(limit)

        # Get total count first (without loading the rows)
        total = query.count()

        if total == 0:
            print("No expressions with archive.org domains found")
            return 1

        print(f"Found {total} expressions with archive.org domains")
        if dryrun:
            print("DRY RUN MODE - no changes will be made")

        updated = 0
        errors = 0
        domain_cache = {}

        # Iterate without loading all into memory to avoid encoding issues
        for i, expr in enumerate(query.iterator(), 1):
            try:
                # Extract the real domain name from the URL
                real_domain_name = core.get_domain_name(expr.url)

                # Skip if it's still an archive domain (shouldn't happen with the fix)
                if real_domain_name in ('web.archive.org', 'archive.org', 'ghostarchive.org'):
                    print(f"Warning: Could not extract original domain from {expr.url}")
                    errors += 1
                    continue

                # Get or create the real domain
                if real_domain_name in domain_cache:
                    real_domain = domain_cache[real_domain_name]
                else:
                    real_domain, created = model.Domain.get_or_create(name=real_domain_name)
                    domain_cache[real_domain_name] = real_domain
                    if created and not dryrun:
                        print(f"  Created new domain: {real_domain_name}")

                # Update the expression using a direct SQL update to avoid loading all columns
                # Also reset seorank to NULL so it can be re-fetched with the unwrapped URL
                if not dryrun:
                    model.Expression.update(
                        domain_id=real_domain.id,
                        seorank=None
                    ).where(
                        model.Expression.id == expr.id
                    ).execute()

                updated += 1

                if i % 100 == 0:
                    print(f"Progress: {i}/{total} expressions processed ({updated} updated, {errors} errors)")

            except Exception as e:
                print(f"Error processing expression {expr.id} ({expr.url}): {e}")
                errors += 1

        print(f"\nMigration completed:")
        print(f"  Total expressions: {total}")
        print(f"  Updated: {updated}")
        print(f"  Errors: {errors}")

        if dryrun:
            print("\nThis was a DRY RUN. Run without --dryrun to apply changes.")

        return 1


class LandController:
    """
    Land controller class
    """

    @staticmethod
    def medianalyse(args: core.Namespace):
        """Analyze media files for expressions in a land.

        Initiates media analysis for all media associated with the specified
        land using the core medianalyse_land function.

        Args:
            args: Namespace object containing command-line arguments. Required
                argument is 'name' (land name). Optional arguments 'depth'
                (max expression depth) and 'minrel' (minimum relevance) filter
                the expressions whose media get analyzed (sprint-multilang, B2).

        Returns:
            int: 1 if analysis completed successfully, 0 if land not found or
                an error occurred.

        Notes:
            On Windows, this method sets up a ProactorEventLoop for async
            operations. The actual analysis is delegated to the core module's
            medianalyse_land function.
        """
        core.check_args(args, 'name')
        land = model.Land.get_or_none(model.Land.name == args.name)
        if not land:
            print(f'Land "{args.name}" introuvable')
            return 0

        depth = core.get_arg_option('depth', args, set_type=int, default=None)
        minrel = core.get_arg_option('minrel', args, set_type=int, default=None)
        active_filters = []
        if depth is not None:
            active_filters.append(f'depth<={depth}')
        if minrel is not None:
            active_filters.append(f'relevance>={minrel}')
        suffix = f" [{', '.join(active_filters)}]" if active_filters else ''
        print(f'Début analyse média pour {args.name}{suffix}')
        loop = _get_event_loop()
        if sys.platform == 'win32':
            asyncio.set_event_loop(asyncio.ProactorEventLoop())

        try:
            result = loop.run_until_complete(
                core.medianalyse_land(land, depth=depth, minrel=minrel))
            print(f"Analyse terminée : {result['processed']} médias traités")
            return 1
        except Exception as e:
            print(f"Erreur lors de l'analyse : {str(e)}")
            return 0

    @staticmethod
    def _media_base_query(land):
        """Base Media query for a land (joined on Expression)."""
        return (model.Media.select()
                .join(model.Expression)
                .where(model.Expression.land == land))

    @staticmethod
    def _media_filter_criteria(args):
        """Resolve --minwidth/--minheight/--maxsize, falling back to settings.

        Returns:
            tuple: (min_width, min_height, max_file_size_bytes). A value of 0
                disables the corresponding constraint (same convention as
                Media.is_conforming).
        """
        min_width = core.get_arg_option('minwidth', args, set_type=int,
                                        default=getattr(settings, 'media_min_width', 0) or 0)
        min_height = core.get_arg_option('minheight', args, set_type=int,
                                         default=getattr(settings, 'media_min_height', 0) or 0)
        maxsize_mb = core.get_arg_option('maxsize', args, set_type=float, default=None)
        if maxsize_mb is not None:
            max_file_size = int(maxsize_mb * 1024 * 1024)
        else:
            max_file_size = getattr(settings, 'media_max_file_size', 0) or 0
        return min_width, min_height, max_file_size

    @staticmethod
    def media_stats(args: core.Namespace):
        """Display aggregate media statistics for a land (sprint-multilang, C/P7).

        Prints totals (analyzed / errors), the distribution by format, by
        dimension buckets and by file-size buckets, and the number of
        duplicate groups detected through the perceptual image hash.

        Args:
            args: Namespace object containing command-line arguments. Required
                argument is 'name' (land name).

        Returns:
            int: 1 on success, 0 if land not found.
        """
        core.check_args(args, 'name')
        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print('Land "%s" not found' % args.name)
            return 0
        from peewee import fn

        base = LandController._media_base_query(land)
        total = base.count()
        analyzed = base.where(model.Media.analyzed_at.is_null(False)).count()
        errors = base.where(model.Media.analysis_error.is_null(False)).count()
        print(f'Media stats for land "{land.name}":')
        print(f'  Total: {total} — analyzed: {analyzed} — errors: {errors}')

        fmt_rows = (model.Media
                    .select(model.Media.format, fn.COUNT(model.Media.id).alias('n'))
                    .join(model.Expression)
                    .where(model.Expression.land == land)
                    .group_by(model.Media.format)
                    .order_by(fn.COUNT(model.Media.id).desc()))
        print('  By format:')
        for row in fmt_rows:
            print(f'    {row.format or "unknown"}: {row.n}')

        # Buckets are mutually exclusive: 'unknown' captures any NULL
        # dimension, the others only apply to fully-measured media.
        dims_known = (model.Media.width.is_null(False)
                      & model.Media.height.is_null(False))
        dim_buckets = [
            ('unknown', model.Media.width.is_null(True) | model.Media.height.is_null(True)),
            ('small (<200px)', dims_known
                               & ((model.Media.width < 200) | (model.Media.height < 200))),
            ('medium', dims_known
                       & (model.Media.width >= 200) & (model.Media.height >= 200)
                       & ((model.Media.width < 1000) | (model.Media.height < 1000))),
            ('large (>=1000px)', dims_known
                                 & (model.Media.width >= 1000) & (model.Media.height >= 1000)),
        ]
        print('  By dimensions:')
        for label, cond in dim_buckets:
            print(f'    {label}: {base.where(cond).count()}')

        size_buckets = [
            ('unknown', model.Media.file_size.is_null(True)),
            ('<100KB', model.Media.file_size < 100 * 1024),
            ('100KB-1MB', (model.Media.file_size >= 100 * 1024)
                          & (model.Media.file_size < 1024 * 1024)),
            ('>=1MB', model.Media.file_size >= 1024 * 1024),
        ]
        print('  By file size:')
        for label, cond in size_buckets:
            print(f'    {label}: {base.where(cond).count()}')

        duplicates = (model.Media
                      .select(model.Media.image_hash, fn.COUNT(model.Media.id).alias('n'))
                      .join(model.Expression)
                      .where((model.Expression.land == land)
                             & (model.Media.image_hash.is_null(False)))
                      .group_by(model.Media.image_hash)
                      .having(fn.COUNT(model.Media.id) > 1))
        dup_groups = duplicates.count()
        dup_medias = sum(row.n for row in duplicates)
        print(f'  Duplicates: {dup_groups} group(s), {dup_medias} media')
        return 1

    @staticmethod
    def preview_deletion(args: core.Namespace):
        """Preview which media would be deleted by the given criteria (dry-run).

        Pure dry-run (sprint-multilang, C/P7): counts the analyzed media that
        do NOT conform to --minwidth/--minheight/--maxsize (defaults come from
        settings.media_min_width/media_min_height/media_max_file_size) and
        prints up to 20 example URLs. Never deletes anything. Media not yet
        analyzed are excluded and reported separately.

        Args:
            args: Namespace object containing command-line arguments. Required
                argument is 'name' (land name). Optional: 'minwidth',
                'minheight' (pixels), 'maxsize' (MB).

        Returns:
            int: 1 on success, 0 if land not found.
        """
        core.check_args(args, 'name')
        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print('Land "%s" not found' % args.name)
            return 0
        min_width, min_height, max_file_size = LandController._media_filter_criteria(args)

        base = LandController._media_base_query(land)
        unanalyzed = base.where(model.Media.analyzed_at.is_null(True)).count()
        analyzed_query = base.where(model.Media.analyzed_at.is_null(False))

        conditions = None
        if min_width > 0:
            cond = model.Media.width.is_null(True) | (model.Media.width < min_width)
            conditions = cond if conditions is None else (conditions | cond)
        if min_height > 0:
            cond = model.Media.height.is_null(True) | (model.Media.height < min_height)
            conditions = cond if conditions is None else (conditions | cond)
        if max_file_size > 0:
            cond = model.Media.file_size > max_file_size
            conditions = cond if conditions is None else (conditions | cond)

        if conditions is None:
            print('[preview_deletion] No criteria given (flags or settings) — nothing would be deleted')
            return 1

        non_conforming = analyzed_query.where(conditions)
        count = non_conforming.count()
        criteria = (f'min_width={min_width}px, min_height={min_height}px, '
                    f'max_size={max_file_size} bytes')
        print(f'[preview_deletion] DRY RUN for land "{land.name}" ({criteria})')
        print(f'[preview_deletion] {count} media would be deleted '
              f'({unanalyzed} not yet analyzed, excluded — run land medianalyse first)')
        for media in non_conforming.limit(20):
            print(f'  - #{media.id} {media.url} '
                  f'({media.width}x{media.height}, {media.file_size} bytes)')
        print('[preview_deletion] Nothing was deleted (dry run).')
        return 1

    @staticmethod
    def reanalyze(args: core.Namespace):
        """Re-run media analysis for a land, optionally deleting non-conforming media.

        Re-analyzes the land's media, prioritizing those never analyzed
        (analyzed_at NULL) or in error (analysis_error not NULL). With
        --suppress, media that fail Media.is_conforming() against
        --minwidth/--minheight/--maxsize (defaults from settings) are deleted
        AFTER an explicit confirmation (sprint-multilang, C/P7).

        Args:
            args: Namespace object containing command-line arguments. Required
                argument is 'name' (land name). Optional: 'minwidth',
                'minheight', 'maxsize', 'suppress', 'limit' (max number of
                media re-analyzed in this run).

        Returns:
            int: 1 on success, 0 if land not found or deletion not confirmed.
        """
        core.check_args(args, 'name')
        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print('Land "%s" not found' % args.name)
            return 0
        min_width, min_height, max_file_size = LandController._media_filter_criteria(args)
        suppress = bool(getattr(args, 'suppress', False))
        limit = core.get_arg_option('limit', args, set_type=int, default=0)

        from .media_analyzer import MediaAnalyzer

        medias = (LandController._media_base_query(land)
                  .order_by(
                      # never-analyzed and errored media first
                      model.Media.analyzed_at.is_null(False).asc(),
                      model.Media.analysis_error.is_null(True).asc(),
                      model.Media.id.asc()))
        if limit and limit > 0:
            medias = medias.limit(limit)

        async def process():
            count = 0
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
                for media in medias:
                    print(f'Reanalyzing media #{media.id}: {media.url}')
                    result = await analyzer.analyze_image(str(media.url))
                    for field, value in result.items():
                        if hasattr(media, field) and field != 'error':
                            setattr(media, field, value)
                    media.analyzed_at = model.datetime.datetime.now()
                    media.analysis_error = result.get('error') if 'error' in result else None
                    media.save()
                    count += 1
            return count

        # asyncio.run creates a fresh loop (Proactor is the Windows default
        # since Python 3.8), avoiding 'Event loop is closed' on reuse.
        processed = asyncio.run(process())
        print(f'[reanalyze] {processed} media re-analyzed')

        if not suppress:
            return 1

        to_delete = [media.id for media in
                     LandController._media_base_query(land)
                         .where(model.Media.analyzed_at.is_null(False))
                     if not media.is_conforming(min_width, min_height, max_file_size)]
        if not to_delete:
            print('[reanalyze] No non-conforming media to delete')
            return 1
        if not core.confirm(
                f"This will DELETE {len(to_delete)} non-conforming media "
                f"from land '{land.name}'. Type 'Y' to proceed: "):
            print('Aborted')
            return 0
        # Delete in chunks: a single IN(...) clause binds one SQL variable
        # per id and would hit SQLITE_MAX_VARIABLE_NUMBER on large lands.
        deleted = 0
        for start in range(0, len(to_delete), 500):
            chunk = to_delete[start:start + 500]
            deleted += model.Media.delete().where(model.Media.id.in_(chunk)).execute()
        print(f'[reanalyze] {deleted} non-conforming media deleted')
        return 1

    @staticmethod
    def seorank(args: core.Namespace):
        """Enrich land expressions with SEO Rank data.

        Fetches SEO metrics from the SEO Rank API for expressions in the
        specified land and updates the database with the retrieved data.

        Args:
            args: Namespace object containing command-line arguments. Required
                argument is 'name' (land name). Optional arguments include
                'limit' (max expressions to process), 'depth' (max expression
                depth), 'http' (HTTP status filter), 'minrel' (minimum relevance),
                and 'force' (force refresh of existing data).

        Returns:
            int: 1 if operation completed successfully, 0 if API key missing
                or land not found.

        Notes:
            Requires settings.seorank_api_key or MWI_SEORANK_API_KEY environment
            variable. The method delegates to core.update_seorank_for_land for
            the actual API calls and database updates.
        """
        core.check_args(args, 'name')

        api_key = getattr(settings, 'seorank_api_key', '')
        if not api_key:
            print('[seorank] API key missing — set settings.seorank_api_key or MWI_SEORANK_API_KEY')
            return 0

        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print(f'[seorank] Error: Land "{args.name}" not found')
            # Show available lands to help the user
            available_lands = model.Land.select()
            if available_lands.count() > 0:
                print('[seorank] Available lands:')
                for available_land in available_lands:
                    expr_count = model.Expression.select().where(
                        model.Expression.land == available_land
                    ).count()
                    print(f'  - {available_land.name} ({expr_count} expressions)')
            else:
                print('[seorank] No lands found in database. Create one with: python mywi.py land create')
            return 0

        limit = core.get_arg_option('limit', args, set_type=int, default=0)
        depth = core.get_arg_option('depth', args, set_type=int, default=None)
        http_status = core.get_arg_option('http', args, set_type=str, default='200')
        min_relevance = core.get_arg_option('minrel', args, set_type=int, default=1)
        force_refresh = bool(getattr(args, 'force', False))

        processed, updated = core.update_seorank_for_land(
            land=land,
            api_key=api_key,
            limit=limit,
            depth=depth,
            http_status=http_status,
            min_relevance=min_relevance,
            force_refresh=force_refresh,
        )

        if processed == 0:
            print(f"[seorank] No expressions selected for land {args.name} (check depth/force options)")

        limit_display = limit if limit > 0 else 'all'
        depth_display = 'all' if depth is None else depth
        http_display = http_status if http_status not in (None, '', 'all', 'ALL') else 'all'
        print(
            f"[seorank] Completed for land {args.name}: processed={processed}, "
            f"updated={updated}, limit={limit_display}, depth={depth_display}, "
            f"http={http_display}, minrel={min_relevance}, force={force_refresh}"
        )
        return 1

    @staticmethod
    def consolidate(args: core.Namespace):
        """Consolidate a land by recalculating metadata and relationships.

        Performs comprehensive consolidation including recalculating relevance
        scores, rebuilding expression links, updating media references, and
        ensuring data consistency across the land.

        Args:
            args: Namespace object containing command-line arguments. Required
                argument is 'name' (land name). Optional arguments include
                'limit' (max expressions to process), 'depth' (max expression
                depth), and 'minrel' (minimum relevance score).

        Returns:
            int: 1 if consolidation completed successfully, 0 if land not found.

        Notes:
            This is an async operation that uses the event loop. On Windows,
            a ProactorEventLoop is configured. The consolidation process may
            take significant time for large lands. Reports number of expressions
            consolidated and errors encountered.
        """
        core.check_args(args, 'name')
        fetch_limit = core.get_arg_option('limit', args, set_type=int, default=0)
        depth = core.get_arg_option('depth', args, set_type=int, default=None)
        min_relevance = core.get_arg_option('minrel', args, set_type=int, default=0)
        # --llm=true: re-run the OpenRouter gate during consolidation (idiom
        # identical to `land readable --llm`). Without it, consolidate still
        # respects stored verdicts (validllm='non' => relevance 0).
        llm_option = core.get_arg_option('llm', args, set_type=str, default='false')
        llm_revalidate = str(llm_option).strip().lower() in ('true', '1', 'yes', 'on')
        # --issuecrawl: force the stricter "controversy analysis" prompt.
        # None => the gate falls back to settings.openrouter_issue_mode.
        issue_mode = True if getattr(args, 'issuecrawl', False) else None
        if llm_revalidate and not (
            getattr(settings, 'openrouter_enabled', False)
            and settings.openrouter_api_key and settings.openrouter_model
        ):
            print('--llm=true ignored: OpenRouter not configured '
                  '(openrouter_enabled / api_key / model) — consolidating without LLM')
            llm_revalidate = False
        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print('Land "%s" not found' % args.name)
        else:
            if sys.platform == 'win32':
                asyncio.set_event_loop(asyncio.ProactorEventLoop())
            loop = _get_event_loop()
            results = loop.run_until_complete(
                core.consolidate_land(land, fetch_limit, depth, min_relevance,
                                      llm_revalidate=llm_revalidate, issue_mode=issue_mode)
            )
            consolidated, errors = results
            print(
                f"%d expressions consolidated (%d errors, minrel=%d)"
                % (consolidated, errors, min_relevance)
            )
            return 1
        return 0


    @staticmethod
    def normalize(args: core.Namespace):
        """Apply URL normalization retroactively to an existing Land.

        Uses the same `mwi.url_normalizer` rules as the live ingestion
        pipeline. Renames URLs in place where possible; merges duplicates
        (with link remapping) where the canonical form already exists.

        Args:
            args: Namespace with `name` (required), optional `dry_run`,
                `limit`, `reset_status`, `verbose`.

        Returns:
            int: 1 if normalization completed (even with 0 changes), 0 if
            land not found.
        """
        from mwi import normalize_pipeline
        core.check_args(args, 'name')
        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print(f'Land "{args.name}" not found')
            return 0

        dry_run = core.get_dryrun(args)
        reset_raw = core.get_arg_option('reset_status', args, set_type=str, default=None)
        reset_status = bool(reset_raw and str(reset_raw).upper() == 'TRUE')
        verbose_raw = core.get_arg_option('verbose', args, set_type=str, default=None)
        verbose = bool(verbose_raw and str(verbose_raw).upper() == 'TRUE')
        limit = core.get_arg_option('limit', args, set_type=int, default=0)

        totals = normalize_pipeline.normalize_land(
            land, dry_run=dry_run, limit=limit,
            reset_status=reset_status, verbose=verbose)

        verb = 'Would' if dry_run else 'Done.'
        print(f'\n{verb} renamed: {totals["renamed"]}, merged: {totals["merged"]}')
        if totals.get('collision_groups'):
            print(f'  Collision groups (several variants -> same URL): '
                  f'{totals["collision_groups"]}')
        if totals.get('promoted'):
            print(f'  Promoted variants (no canonical row existed): '
                  f'{totals["promoted"]}')
        if totals.get('backfilled'):
            print(f'  Backfilled fields (merge -> empty canonical): '
                  f'{totals["backfilled"]}')
        print(f'  Links remapped (incoming): {totals["remapped_in"]}')
        print(f'  Links dropped  (incoming): {totals["dropped_in"]}')
        print(f'  Links remapped (outgoing): {totals["remapped_out"]}')
        print(f'  Links dropped  (outgoing): {totals["dropped_out"]}')
        print(f'  Cascade-deleted Media: {totals["media_lost"]}')
        print(f'  Cascade-deleted Paragraph: {totals["paragraphs_lost"]}')
        print(f'  Cascade-deleted TaggedContent: {totals["tagged_lost"]}')
        if totals['skipped']:
            print(f'  Skipped (errors / disappeared): {totals["skipped"]}')
        if dry_run:
            print('Run again without --dry-run to apply.')
        return 1


    @staticmethod
    def list(args: core.Namespace):
        """Display information about existing lands.

        Shows detailed statistics for all lands or a specific land, including
        dictionary terms, expression counts, HTTP status distributions, and
        embedding pipeline statistics.

        Args:
            args: Namespace object containing command-line arguments. Optional
                argument 'name' filters to a specific land.

        Returns:
            int: 1 if lands were found and displayed, 0 if no lands exist.

        Notes:
            For each land, displays: name, creation date, description, dictionary
            terms, total expressions, remaining expressions to crawl, HTTP status
            code distribution, and embedding statistics (paragraphs, embeddings,
            pseudolinks). Embedding statistics are shown only if the corresponding
            tables exist in the database.
        """
        lands = model.Land.select(
            model.Land.id,
            model.Land.name,
            model.Land.created_at,
            model.Land.description,
            model.Land.fullhtml,
            fn.GROUP_CONCAT(model.Word.term.distinct()).alias('words'),
            fn.COUNT(model.Expression.id.distinct()).alias('num_all')
        ) \
            .join(model.LandDictionary, JOIN.LEFT_OUTER) \
            .join(model.Word, JOIN.LEFT_OUTER) \
            .switch(model.Land) \
            .join(model.Expression, JOIN.LEFT_OUTER) \
            .group_by(model.Land.name) \
            .order_by(model.Land.name)

        name = core.get_arg_option('name', args, set_type=str, default=None)
        if name is not None:
            lands = lands.where(model.Land.name == name)

        if lands.count() > 0:
            for land in lands:
                if land.words is not None:
                    words = [w for w in land.words.split(',')]
                else:
                    words = []

                select = model.Expression \
                    .select(fn.COUNT(model.Expression.id).alias('num')) \
                    .join(model.Land) \
                    .where((model.Expression.land == land)
                           & (model.Expression.fetched_at.is_null()))
                remaining_to_crawl = [s.num for s in select]

                select = model.Expression \
                    .select(
                        model.Expression.http_status,
                        fn.COUNT(model.Expression.http_status).alias('num')) \
                    .where((model.Expression.land == land)
                           & (model.Expression.fetched_at.is_null(False))) \
                    .group_by(model.Expression.http_status) \
                    .order_by(model.Expression.http_status)
                http_statuses = ["%s: %s" % (s.http_status, s.num) for s in select]

                # Cascade fetch method distribution (sprint-403 Sprint 4)
                fetch_methods = []
                try:
                    fm_select = (model.Expression
                        .select(
                            model.Expression.fetch_method,
                            fn.COUNT(model.Expression.id).alias('num'))
                        .where((model.Expression.land == land)
                               & (model.Expression.fetched_at.is_null(False)))
                        .group_by(model.Expression.fetch_method)
                        .order_by(fn.COUNT(model.Expression.id).desc()))
                    fetch_methods = [
                        "%s: %s" % (s.fetch_method or 'unknown', s.num)
                        for s in fm_select
                    ]
                except Exception:
                    # Pre-migration database: column not yet present
                    fetch_methods = []

                print("%s - (%s)\n\t%s" % (
                    land.name,
                    land.created_at.strftime("%B %d %Y %H:%M"),
                    land.description))
                print("\t%s terms in land dictionary %s" % (
                    len(words),
                    words))
                print("\t%s expressions in land (%s remaining to crawl)" % (
                    land.num_all,
                    remaining_to_crawl[0]))
                print("\tStatus codes: %s" % (
                    " - ".join(http_statuses)))
                if fetch_methods:
                    print("\tFetch methods: %s" % (
                        " - ".join(fetch_methods)))
                # Full HTML storage stats (sprint-html C — D4)
                # Show when land has the policy on, OR when at least one
                # expression already has HTML stored (legacy data).
                try:
                    n_html, total_bytes = model.DB.execute_sql(
                        "SELECT COUNT(*), COALESCE(SUM(LENGTH(html)), 0) "
                        "FROM expression "
                        "WHERE land_id=? AND html IS NOT NULL",
                        (land.id,)
                    ).fetchone()
                    if land.fullhtml or n_html > 0:
                        size_mb = total_bytes / (1024 * 1024)
                        policy = "ON" if land.fullhtml else "OFF"
                        print("\tFull HTML: policy=%s — %d expressions stored (%.1f MB)" % (
                            policy, n_html, size_mb))
                except Exception:
                    # Pre-migration database: html column not yet present
                    pass
                # Embedding pipeline summary (land-wide)
                try:
                    # Paragraphs count
                    para_sql = (
                        "SELECT COUNT(*) FROM paragraph p "
                        "JOIN expression e ON e.id = p.expression_id "
                        "WHERE e.land_id = ?"
                    )
                    para_cnt = model.DB.execute_sql(para_sql, (land.id,)).fetchone()[0]

                    # Embeddings count
                    emb_sql = (
                        "SELECT COUNT(*) FROM paragraph_embedding pe "
                        "JOIN paragraph p ON p.id = pe.paragraph_id "
                        "JOIN expression e ON e.id = p.expression_id "
                        "WHERE e.land_id = ?"
                    )
                    emb_cnt = model.DB.execute_sql(emb_sql, (land.id,)).fetchone()[0]

                    # Pseudolinks (only pairs where both paragraphs belong to the land)
                    psl_sql = (
                        "SELECT COUNT(*) FROM paragraph_similarity s "
                        "JOIN paragraph p1 ON p1.id = s.source_paragraph_id "
                        "JOIN expression e1 ON e1.id = p1.expression_id "
                        "JOIN paragraph p2 ON p2.id = s.target_paragraph_id "
                        "JOIN expression e2 ON e2.id = p2.expression_id "
                        "WHERE e1.land_id = ? AND e2.land_id = e1.land_id "
                        "AND s.method IN ('nli','cosine','cosine_lsh')"
                    )
                    psl_cnt = model.DB.execute_sql(psl_sql, (land.id,)).fetchone()[0]

                    print(f"\tEmbedding: paragraph: {para_cnt} - embed: {emb_cnt} - pseudolink: {psl_cnt}")
                except Exception as e:
                    print(f"\tEmbedding: N/A ({e})")
                print("\n")
            return 1
        print("No land created")
        return 0

    @staticmethod
    def create(args: core.Namespace):
        """Create a new research land.

        Creates a new land with the specified name, description, and language(s).
        Also creates the corresponding directory structure for storing land data.

        Args:
            args: Namespace object containing command-line arguments. Required
                arguments are 'name' (land name) and 'desc' (description).
                Optional argument 'lang' specifies language(s) for the land.

        Returns:
            int: 1 indicating successful land creation.

        Notes:
            The language parameter can be a list or a single string. Multiple
            languages are stored as comma-separated values. Creates a directory
            at data_location/lands/{land_id} for storing land-specific files.
        """
        core.check_args(args, ('name', 'desc'))
        # Store lang as comma-separated string
        raw_lang = getattr(args, 'lang', None)
        if isinstance(raw_lang, list):
            lang_str = ",".join(raw_lang) or 'fr'
        elif raw_lang:
            lang_str = str(raw_lang)
        else:
            lang_str = 'fr'
        # Parse --fullhtml option (default FALSE for new lands)
        fullhtml_raw = core.get_arg_option('fullhtml', args, set_type=str, default=None)
        store_html = fullhtml_raw.upper() == 'TRUE' if fullhtml_raw else False
        land = model.Land.create(name=args.name, description=args.desc, lang=lang_str, fullhtml=store_html)
        os.makedirs(os.path.join(settings.data_location, 'lands/%s') % land.id, exist_ok=True)
        html_status = "enabled" if store_html else "disabled"
        print(f'Land "{args.name}" created (fullhtml={html_status})')
        return 1

    @staticmethod
    def addterm(args: core.Namespace):
        """Add terms to a land's dictionary.

        Adds one or more terms to the specified land's dictionary with automatic
        lemmatization. After adding terms, recalculates relevance scores for all
        expressions in the land.

        Args:
            args: Namespace object containing command-line arguments. Required
                arguments are 'land' (land name) and 'terms' (comma-separated
                list of terms to add).

        Returns:
            int: 1 if terms were added successfully, 0 if land not found.

        Notes:
            Each term is stemmed to create a lemma using the core.stem_word
            function, once per language of the land (sprint-multilang): a
            land with lang='fr,en' creates one Word per language for each
            term, so the dictionary holds the union of fr+en lemmas. After
            adding terms, land_relevance is called to update expression
            scores based on the new dictionary.
        """
        core.check_args(args, ('land', 'terms'))
        land = model.Land.get_or_none(model.Land.name == args.land)
        if land is None:
            print('Land "%s" not found' % args.land)
        else:
            land_langs = [l.strip().lower() for l in
                          str(land.lang or 'fr').split(',') if l.strip()] or ['fr']
            for term in core.split_arg(args.terms):
                with model.DB.atomic():
                    for lang in land_langs:
                        lemma = ' '.join([core.stem_word(w, lang) for w in term.split(' ')])
                        word, _ = model.Word.get_or_create(
                            term=term, lang=lang, defaults={'lemma': lemma})
                        model.LandDictionary.get_or_create(land=land.id, word=word.id)
                    print('Term "%s" created in land %s' % (term, args.land))
            core.land_relevance(land)
            return 1
        return 0

    @staticmethod
    def relemm(args: core.Namespace):
        """Re-lemmatize a land's dictionary using the land's language(s).

        Retro-fit verb (sprint-multilang, D5) for lands created before
        multilingual stemming: their terms were lemmatized with the French
        stemmer regardless of the land language. This verb rebuilds one
        Word per (term, lang) for every language of the land, relinks the
        LandDictionary, purges Words no longer linked to any land, then
        recomputes relevance for the whole land.

        Args:
            args: Namespace object containing command-line arguments.
                Required argument is 'name' (land name).

        Returns:
            int: 1 if re-lemmatization completed, 0 if land not found.

        Notes:
            Idempotent: running it twice yields the same dictionary and
            the same relevance scores.
        """
        core.check_args(args, 'name')
        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print('Land "%s" not found' % args.name)
            return 0
        land_langs = [l.strip().lower() for l in
                      str(land.lang or 'fr').split(',') if l.strip()] or ['fr']
        terms = sorted({str(w.term) for w in core.get_land_dictionary(land)})
        if not terms:
            print(f'No terms in land "{args.name}" dictionary — nothing to do')
            return 1
        print(f'Re-lemmatizing {len(terms)} term(s) for language(s): {", ".join(land_langs)}')
        with model.DB.atomic():
            model.LandDictionary.delete().where(model.LandDictionary.land == land).execute()
            for term in terms:
                for lang in land_langs:
                    lemma = ' '.join(core.stem_word(w, lang) for w in term.split(' '))
                    word, created = model.Word.get_or_create(
                        term=term, lang=lang, defaults={'lemma': lemma})
                    if not created and str(word.lemma) != lemma:
                        word.lemma = lemma
                        word.save()
                    model.LandDictionary.get_or_create(land=land.id, word=word.id)
            orphans = (model.Word.delete()
                       .where(model.Word.id.not_in(
                           model.LandDictionary.select(model.LandDictionary.word)))
                       .execute())
            if orphans:
                print(f'Purged {orphans} orphan word(s) no longer linked to any land')
        print('Dictionary re-lemmatized, recomputing relevance '
              '(may take a while on large lands)...')
        core.land_relevance(land)
        return 1

    @staticmethod
    def addurl(args: core.Namespace):
        """Add URLs to a land.

        Adds one or more URLs to the specified land as expressions. URLs can
        be provided directly as a comma-separated string or loaded from a file.

        Args:
            args: Namespace object containing command-line arguments. Required
                argument is 'land' (land name). At least one of 'urls'
                (comma-separated URLs) or 'path' (file path containing URLs)
                must be provided.

        Returns:
            int: 1 if URLs were processed successfully, 0 if land not found.

        Notes:
            URLs from the file are read line by line with UTF-8 encoding. Each
            URL is added via core.add_expression which handles URL normalization
            and domain extraction. Duplicate URLs are silently skipped. Displays
            progress for each URL added and final count.
        """
        core.check_args(args, 'land')
        land = model.Land.get_or_none(model.Land.name == args.land)
        if land is None:
            print('Land "%s" not found' % args.land)
        else:
            urls_count = 0
            urls = []
            if args.urls:
                urls += [url for url in core.split_arg(args.urls)]
            if args.path:
                with open(args.path, 'r', encoding='utf-8') as file:
                    urls += file.read().splitlines()
            for url in urls:
                if core.add_expression(land, url):
                    urls_count += 1
                    print(f"Added URL: {url} to land {args.land}")
            print('%s URLs created in land %s' % (urls_count, args.land))
            return 1
        return 0

    @staticmethod
    def urlist(args: core.Namespace):
        """Retrieve URLs from SerpAPI and add them to land expressions.

        Fetches search results from SerpAPI (Google, Bing, or DuckDuckGo) for
        the specified query and adds the resulting URLs to the land. Supports
        date range filtering and pagination.

        Args:
            args: Namespace object containing command-line arguments. Required
                arguments are 'name' (land name) and 'query' (search query).
                Optional arguments include 'engine' (search engine), 'lang'
                (language), 'gl' (country restriction, Google only — absent
                by default so searches are language-scoped), 'datestart' and
                'dateend' (date range), 'timestep' (time window for date
                ranges), 'sleep' (delay between requests), and 'progress'
                (enable progress output).

        Returns:
            int: 1 if URLs were retrieved successfully, 0 if API key missing,
                land not found, or invalid engine specified.

        Notes:
            Requires settings.serpapi_api_key or MWI_SERPAPI_API_KEY environment
            variable. Date filtering is only supported for Google and DuckDuckGo
            engines. For existing URLs, updates title and published_at if not
            already set. Uses the earlier date when multiple dates are available.
        """
        core.check_args(args, ('name', 'query'))

        # API key lookup aligned with the search router's SerpAPI adapter
        # (sprint-multilang, D-4): UPPER first (env, then settings), then the
        # legacy snake_case settings key and MWI_* env var.
        api_key = (os.getenv('SERPAPI_API_KEY', '')
                   or getattr(settings, 'SERPAPI_API_KEY', '')
                   or getattr(settings, 'serpapi_api_key', '')
                   or os.getenv('MWI_SERPAPI_API_KEY', ''))
        if not api_key:
            print('[urlist] SerpAPI key missing — set SERPAPI_API_KEY '
                  '(settings or env) or the legacy settings.serpapi_api_key / '
                  'MWI_SERPAPI_API_KEY')
            return 0

        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print('Land "%s" not found' % args.name)
            return 0

        # CLI stores languages as a list; keep the first entry for the request.
        # When --lang is not provided, inherit the land's primary language
        # (sprint-multilang, A7).
        lang_list = getattr(args, 'lang', None)
        if isinstance(lang_list, str):
            lang_list = [l.strip() for l in lang_list.split(',') if l.strip()]
        if isinstance(lang_list, list) and lang_list:
            lang = lang_list[0]
        else:
            lang = str(land.lang or 'fr').split(',')[0].strip() or 'fr'
        engine = (getattr(args, 'engine', 'google') or 'google').lower()
        # Engine validity and date-filter capability come from the SerpApiRouter,
        # which is the single source of truth for supported SerpAPI engines.
        from .serpapi_router import SearchRouter as SerpApiRouter
        if engine not in SerpApiRouter.engines():
            supported = ', '.join(sorted(SerpApiRouter.engines()))
            print(f'[urlist] Unsupported engine "{engine}" — choose {supported}')
            return 0
        provider = SerpApiRouter.get(engine)

        # Optional country restriction (Google `gl`). Absent by default:
        # searches are scoped by language only (hl + lr).
        gl = (getattr(args, 'gl', None) or '').strip() or None

        datestart = getattr(args, 'datestart', None)
        dateend = getattr(args, 'dateend', None)
        timestep = getattr(args, 'timestep', 'week') or 'week'
        sleep_seconds = core.get_arg_option('sleep', args, set_type=float, default=1.0)

        progress_requested = bool(getattr(args, 'progress', False))
        has_date_range = bool(datestart and dateend)
        if has_date_range and not provider.supports_date_filter:
            date_engines = sorted(
                e for e in SerpApiRouter.engines()
                if SerpApiRouter.get(e).supports_date_filter
            )
            print(
                '[urlist] datestart/dateend filters are only supported with '
                + ' or '.join(f'--engine={e}' for e in date_engines)
            )
            return 0
        want_progress = progress_requested or has_date_range

        added = 0
        skipped = 0

        def persist_results(results):
            """Insert one batch of SERP results; returns (added, skipped) for the batch."""
            nonlocal added, skipped
            batch_added = 0
            batch_skipped = 0
            for item in results:
                url = item.get('link')
                if not url:
                    batch_skipped += 1
                    continue

                raw_date = item.get('date') if provider.parses_result_dates else None
                published_at = core.parse_serp_result_date(raw_date) if raw_date else None

                existing = model.Expression.get_or_none(
                    (model.Expression.land == land) & (model.Expression.url == url)
                )
                title = item.get('title')
                if existing is not None:
                    # Update the title if we collected one and the expression is still blank.
                    has_changes = False
                    if title and not existing.title:
                        existing.title = title
                        has_changes = True
                    if published_at:
                        earliest = core.prefer_earlier_datetime(existing.published_at, published_at)
                        if earliest != existing.published_at:
                            existing.published_at = earliest
                            has_changes = True
                    if has_changes:
                        existing.save()
                    batch_skipped += 1
                    continue

                expression = core.add_expression(land, url)
                if expression:
                    # Newly created expressions keep the title for downstream exports.
                    has_changes = False
                    if title and not expression.title:
                        expression.title = title
                        has_changes = True
                    if published_at:
                        earliest = core.prefer_earlier_datetime(expression.published_at, published_at)
                        if earliest != expression.published_at:
                            expression.published_at = earliest
                            has_changes = True
                    if has_changes:
                        expression.save()
                    batch_added += 1
                else:
                    batch_skipped += 1
            added += batch_added
            skipped += batch_skipped
            return batch_added, batch_skipped

        def window_callback(window_start, window_end, results):
            # Persist each window as soon as it completes: an interruption
            # (Ctrl-C, crash, quota exhausted) only costs the in-flight window.
            batch_added, batch_skipped = persist_results(results)
            if want_progress:
                if window_start and window_end:
                    label = f'{window_start.isoformat()} → {window_end.isoformat()}'
                else:
                    label = 'no date filter'
                print(
                    f'[urlist] {label} — fetched {len(results)} results, '
                    f'+{batch_added} new, {batch_skipped} skipped',
                    flush=True,
                )

        scope = f'gl={gl}' if gl else 'no country restriction'
        if has_date_range:
            print(
                f'[urlist] {engine} — lang={lang}, {scope} — '
                f'windows {datestart} → {dateend} (step={timestep}), '
                'one progress line per window…',
                flush=True,
            )
        else:
            print(f'[urlist] {engine} — lang={lang}, {scope} — querying…', flush=True)

        try:
            # Centralised helper handles pagination, date windows and error reporting.
            serp_results = core.fetch_serpapi_url_list(
                api_key=api_key,
                query=args.query,
                engine=engine,
                lang=lang,
                gl=gl,
                datestart=datestart,
                dateend=dateend,
                timestep=timestep,
                sleep_seconds=sleep_seconds,
                window_results_hook=window_callback
            )
        except core.SerpApiError as error:
            print(f'[urlist] {error}')
            if added or skipped:
                # Windows persisted before the failure stay in the land.
                print(
                    f'[urlist] kept before failure: {added} new URLs, '
                    f'{skipped} skipped — already saved in land {args.name}'
                )
            return 0

        # Real runs return [] — every window was persisted by the hook. A
        # patched fetch_serpapi_url_list returning a plain list (legacy
        # tests, external callers) is persisted here instead.
        persist_results(serp_results)

        print(f'[urlist] Added {added} new URLs, skipped {skipped} (existing or invalid) for land {args.name}')
        return 1

    @staticmethod
    def delete(args: core.Namespace):
        """Delete a land or expressions within it.

        Deletes either an entire land with all associated data, or selectively
        deletes expressions based on relevance threshold. Requires user
        confirmation before proceeding.

        Args:
            args: Namespace object containing command-line arguments. Required
                argument is 'name' (land name). Optional argument 'maxrel'
                (maximum relevance) filters expressions to delete.

        Returns:
            int: 1 if deletion completed successfully, 0 if land not found or
                user cancelled.

        Notes:
            If maxrel is provided and greater than 0, only deletes fetched
            expressions with relevance below the threshold. Otherwise, deletes
            the entire land and all related data (expressions, links, media,
            etc.) via recursive deletion. User must type 'Y' to confirm.
        """
        core.check_args(args, 'name')
        maxrel = core.get_arg_option('maxrel', args, set_type=int, default=0)
        prune = getattr(args, 'prune_orphans', False)
        dry = getattr(args, 'dry_run', False)
        do_vacuum = getattr(args, 'vacuum', False)

        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print('Land "%s" not found' % args.name)
            return 0

        if dry:
            if maxrel > 0:
                n = model.Expression.select().where(
                    (model.Expression.land == land)
                    & (model.Expression.relevance < maxrel)
                    & (model.Expression.fetched_at.is_null(False))).count()
                print("[dry-run] %d expression(s) would be deleted by --maxrel" % n)
            if prune:
                count, sample = core.prune_orphan_expressions(land, dry_run=True, maxrel=maxrel)
                print("[dry-run] %d uncrawled orphan(s) would be pruned" % count)
                for url, depth in sample:
                    print("    - [d%s] %s" % (depth, url))
            return 1

        if not core.confirm(
                "Land and/or underlying objects will be deleted, type 'Y' to proceed : "):
            return 0
        if maxrel > 0:
            query = model.Expression.delete().where(
                (model.Expression.land == land)
                & (model.Expression.relevance < maxrel)
                & (model.Expression.fetched_at.is_null(False)))
            deleted = query.execute()
            print("%d expression(s) deleted" % deleted)
        elif not prune:
            land.delete_instance(recursive=True)
            print("Land %s deleted" % args.name)
        if prune:
            count, _ = core.prune_orphan_expressions(land, dry_run=False)
            print("%d uncrawled orphan(s) pruned" % count)
        if do_vacuum:
            print("Running VACUUM (this may take a while on large databases)...")
            model.DB.execute_sql("VACUUM")
            print("VACUUM done")
        return 1

    @staticmethod
    def crawl(args: core.Namespace):
        """Crawl expressions in a land to fetch content.

        Fetches HTML content for unfetched expressions in the specified land,
        extracts metadata, discovers links, and identifies media. Uses async
        HTTP requests with configurable concurrency.

        Args:
            args: Namespace object containing command-line arguments. Required
                argument is 'name' (land name). Optional arguments include
                'limit' (max URLs to fetch), 'http' (HTTP status filter), and
                'depth' (expression depth filter).

        Returns:
            int: 1 if crawl completed successfully, 0 if land not found.

        Notes:
            Uses asyncio event loop for concurrent requests. Respects
            settings.parallel_connections for rate limiting. Extracts page
            title, content, outbound links, and media references. Updates
            expression metadata including HTTP status, fetch timestamp, and
            relevance scores.
        """
        core.check_args(args, 'name')
        fetch_limit = core.get_arg_option('limit', args, set_type=int, default=0)
        if fetch_limit > 0:
            print('Fetch limit set to %s URLs' % fetch_limit)
        http_status = core.get_arg_option('http', args, set_type=str, default=None)
        if http_status is not None:
            print('Limited to %s HTTP status code' % http_status)
        depth = core.get_arg_option('depth', args, set_type=int, default=None)
        if depth is not None:
            print('Only crawling URLs with depth = %s' % depth)

        # --retry-status: comma-separated codes for cascade backfill (sprint-403 Sprint 5)
        retry_raw = core.get_arg_option('retry_status', args, set_type=str, default=None)
        retry_status = None
        if retry_raw:
            retry_status = [s.strip() for s in retry_raw.split(',') if s.strip()]
            if retry_status:
                print('Retrying expressions with HTTP status in %s' % retry_status)
            else:
                retry_status = None

        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print('Land "%s" not found' % args.name)
            return 0

        # Resolve --fullhtml: CLI override > land default
        fullhtml_raw = core.get_arg_option('fullhtml', args, set_type=str, default=None)
        if fullhtml_raw is not None:
            store_html = fullhtml_raw.upper() == 'TRUE'
            source = "CLI"
        else:
            store_html = bool(land.fullhtml)
            source = "land default"
        # Sprint-html C — symmetric feedback so the user always sees which
        # policy will apply, including --fullhtml=FALSE override scenarios.
        print(f'Full HTML storage: {"ON" if store_html else "OFF"} (source: {source})')

        # --issuecrawl: stricter "controversy analysis" prompt for the LLM gate
        # (None => settings.openrouter_issue_mode).
        issue_mode = True if getattr(args, 'issuecrawl', False) else None

        loop = _get_event_loop()
        results = loop.run_until_complete(core.crawl_land(
            land, fetch_limit, http_status, depth,
            store_html=store_html, retry_status=retry_status, issue_mode=issue_mode))
        print("%d expressions processed (%d errors)" % results)
        return 1

    @staticmethod
    def readable(args: core.Namespace):
        """Extract readable content using Mercury Parser pipeline.

        Uses Mercury Parser to extract clean, readable content from fetched
        expressions. Supports different merge strategies for combining Mercury
        results with existing data, and optional LLM validation.

        Args:
            args: Namespace object containing command-line arguments. Required
                argument is 'name' (land name). Optional arguments include
                'limit' (max expressions to process), 'depth' (max expression
                depth), 'merge' (merge strategy), and 'llm' (enable LLM
                validation).

        Returns:
            int: 1 if processing completed successfully, 0 if land not found.

        Notes:
            Merge strategies: 'smart_merge' (default, intelligent fusion),
            'mercury_priority' (Mercury always overwrites), 'preserve_existing'
            (only fill empty fields). LLM validation uses OpenRouter when enabled.
            On Windows, configures ProactorEventLoop for async operations.
            Delegates to readable_pipeline.run_readable_pipeline.
        """
        core.check_args(args, 'name')
        
        # Récupération des paramètres
        fetch_limit = core.get_arg_option('limit', args, set_type=int, default=0)
        depth_limit = core.get_arg_option('depth', args, set_type=int, default=None)
        merge_strategy = core.get_arg_option('merge', args, set_type=str, default='smart_merge')
        llm_option = core.get_arg_option('llm', args, set_type=str, default='false')
        llm_enabled = str(llm_option).strip().lower() in ('true', '1', 'yes', 'on')

        if fetch_limit > 0:
            print(f'Fetch limit set to {fetch_limit} URLs')
        if depth_limit is not None:
            print(f'Depth limit set to {depth_limit}')
        print(f'Merge strategy: {merge_strategy}')
        print(f'OpenRouter validation: {"enabled" if llm_enabled else "disabled"}')
        
        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print('Land "%s" not found' % args.name)
            return 0
        
        # Import du nouveau pipeline
        from .readable_pipeline import run_readable_pipeline
        
        # Configuration de l'event loop selon la plateforme
        if sys.platform == 'win32':
            asyncio.set_event_loop(asyncio.ProactorEventLoop())
        
        # --issuecrawl: stricter "controversy analysis" prompt for the LLM gate
        # (None => settings.openrouter_issue_mode).
        issue_mode = True if getattr(args, 'issuecrawl', False) else None

        loop = _get_event_loop()
        results = loop.run_until_complete(
            run_readable_pipeline(land, fetch_limit, depth_limit, merge_strategy,
                                  llm_enabled, issue_mode=issue_mode)
        )
        
        print("%d expressions processed (%d errors)" % results)
        return 1

    @staticmethod
    def export(args: core.Namespace):
        """Export land data in various formats.

        Exports expressions, links, nodes, media, or corpus data from the
        specified land in formats suitable for analysis, visualization, or
        processing.

        Args:
            args: Namespace object containing command-line arguments. Required
                arguments are 'name' (land name) and 'type' (export format).
                Optional argument 'minrel' (minimum relevance) filters
                expressions.

        Returns:
            int: 1 if export completed successfully, 0 if land not found or
                invalid export type specified.

        Notes:
            Valid export types: 'pagecsv' (page metadata CSV), 'fullpagecsv'
            (full page content CSV), 'nodecsv' (nodes CSV), 'pagegexf' (page
            network GEXF), 'nodegexf' (node network GEXF), 'mediacsv' (media
            links CSV), 'corpus' (raw text corpus), 'pseudolinks' (embedding
            similarity links), 'pseudolinkspage' (page-level pseudolinks),
            'pseudolinksdomain' (domain-level pseudolinks), 'nodelinkcsv'
            (4 CSV files: pagesnodes, pageslinks, domainnodes, domainlinks),
            'nodesjson' (domain force-graph {nodes, links} JSON, with a
            per-domain corpus of {title, urlarticle, description,
            published_at} objects), 'pagesjson' (page force-graph {nodes,
            links} JSON, tags array + nested seorank), 'htmldump' (zip of
            raw stored HTML).

            --fullhtml=TRUE (nodelinkcsv only): also emit the raw-HTML closed
            link network — 4 extra *fullhtml.csv files built from every <a href>
            of expression.html (requires a land crawled with --fullhtml).
            pageslinksfullhtml carries weight (anchor multiplicity) and in_mywi
            (1 if the edge also exists in ExpressionLink), enabling a direct
            comparison with the MyWI editorial-link network.
        """
        minimum_relevance = 1
        core.check_args(args, ('name', 'type'))
        valid_types = ['pagecsv', 'fullpagecsv', 'nodecsv', 'pagegexf',
                       'nodegexf', 'mediacsv', 'corpus', 'pseudolinks',
                       'pseudolinkspage', 'pseudolinksdomain', 'nodelinkcsv',
                       'nodesjson', 'pagesjson', 'htmldump']

        if isinstance(args.minrel, int) and (args.minrel >= 0):
            minimum_relevance = args.minrel
            print("Minimum relevance set to %s" % minimum_relevance)

        # --fullhtml (modifier, not a type): when TRUE, nodelinkcsv also emits
        # the raw-HTML link network files. CLI value is 'TRUE'/'FALSE'/None
        # (None = absent); 'FALSE' is not falsy-by-default, so parse explicitly.
        fullhtml_raw = core.get_arg_option('fullhtml', args, set_type=str, default=None)
        store_html = (fullhtml_raw is not None and fullhtml_raw.upper() == 'TRUE')

        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print('Land "%s" not found' % args.name)
        else:
            if args.type in valid_types:
                core.export_land(land, args.type, minimum_relevance, fullhtml=store_html)
                return 1
            print('Invalid export type "%s" [%s]' % (args.type, ', '.join(valid_types)))
        return 0

    @staticmethod
    def llm_validate(args: core.Namespace):
        """Perform bulk LLM validation for land expressions via OpenRouter.

        Validates expressions using an LLM to determine relevance based on
        readable content and land dictionary terms. Updates expressions with
        validation verdict and sets relevance to 0 for non-relevant content.

        Args:
            args: Namespace object containing command-line arguments. Required
                argument is 'name' (land name). Optional arguments include
                'limit' (max expressions to validate) and 'force' (re-validate
                expressions previously marked as non-relevant).

        Returns:
            int: 1 if validation completed successfully, 0 if OpenRouter not
                enabled, configuration missing, or land not found.

        Notes:
            Requires settings.openrouter_enabled=True, settings.openrouter_api_key,
            and settings.openrouter_model. Only processes expressions with
            readable content longer than settings.openrouter_readable_min_chars.
            Verdicts: 'oui' (relevant), 'non' (not relevant). With --force,
            re-validates previously rejected expressions.
        """
        core.check_args(args, 'name')

        # Configuration OpenRouter
        if not getattr(settings, 'openrouter_enabled', False):
            print('OpenRouter non activé (settings.openrouter_enabled=False) — abandon')
            return 0
        if not settings.openrouter_api_key or not settings.openrouter_model:
            print('OpenRouter: clé API ou modèle manquant — renseignez settings.openrouter_api_key et openrouter_model')
            return 0

        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print(f'Land "{args.name}" non trouvé')
            return 0

        limit = core.get_arg_option('limit', args, set_type=int, default=0)
        force = bool(getattr(args, 'force', False))
        # --issuecrawl: force the stricter "controversy analysis" prompt.
        # None => the gate falls back to settings.openrouter_issue_mode.
        issue_mode = True if getattr(args, 'issuecrawl', False) else None

        # Expressions à valider: sans verdict ('oui'/'non') ET avec readable non NULL et suffisamment long
        # Base condition on previous verdicts
        verdict_cond = (
            model.Expression.validllm.is_null(True)
            | model.Expression.validllm.not_in(['oui', 'non'])
        )
        # If --force, also include those previously marked as 'non'
        if force:
            verdict_cond = verdict_cond | (model.Expression.validllm == 'non')

        q = (model.Expression
             .select()
             .where(
                 (model.Expression.land == land)
                 & verdict_cond
                 & (
                     model.Expression.readable.is_null(False)
                     & (fn.LENGTH(model.Expression.readable) >= getattr(settings, 'openrouter_readable_min_chars', 0))
                 )
             )
             .order_by(model.Expression.id))
        if limit and limit > 0:
            q = q.limit(limit)

        from . import llm_openrouter as llm

        total = 0
        updated = 0
        for expr in q:
            total += 1
            verdict = llm.is_relevant_via_openrouter(land, expr, issue_mode=issue_mode)
            if verdict is True:
                expr.validllm = 'oui'
                expr.validmodel = settings.openrouter_model
                expr.save(only=[model.Expression.validllm, model.Expression.validmodel])
                updated += 1
            elif verdict is False:
                expr.validllm = 'non'
                expr.validmodel = settings.openrouter_model
                # Fixer la pertinence à 0 en cas de NON
                expr.relevance = 0
                expr.save(only=[model.Expression.validllm, model.Expression.validmodel, model.Expression.relevance])
                updated += 1
            else:
                # verdict None: ne pas toucher
                pass

        print(f'Validation LLM terminée: examinées={total}, mises à jour={updated}, '
              f'modèle={settings.openrouter_model}, force={force}, '
              f'issuecrawl={bool(getattr(args, "issuecrawl", False))}')
        return 1


class EmbeddingController:
    """Embedding feature controller"""

    @staticmethod
    def generate(args: core.Namespace):
        """Generate paragraphs and embeddings for a land.

        Extracts paragraphs from expression readable content and generates
        vector embeddings for each paragraph using the configured embedding
        provider.

        Args:
            args: Namespace object containing command-line arguments. Required
                argument is 'name' (land name). Optional argument 'limit'
                restricts the number of expressions to process.

        Returns:
            int: 1 if generation completed successfully, 0 if land not found.

        Notes:
            Uses the embedding provider specified in settings (fake, http,
            openai, mistral, gemini, huggingface, or ollama). Paragraphs are
            created by splitting readable content. Embeddings are stored in
            the ParagraphEmbedding table. Displays counts of paragraphs and
            embeddings created.
        """
        core.check_args(args, 'name')
        limit = core.get_arg_option('limit', args, set_type=int, default=0)
        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print(f'Land "{args.name}" not found')
            return 0
        from .embedding_pipeline import generate_embeddings_for_paragraphs
        created_p, created_e = generate_embeddings_for_paragraphs(land, limit_expressions=limit or None)
        print(f"Paragraphs created: {created_p}, embeddings created: {created_e}")
        return 1

    @staticmethod
    def similarity(args: core.Namespace):
        """Compute paragraph similarities within a land.

        Calculates similarity between paragraphs using cosine similarity,
        LSH (locality-sensitive hashing), or NLI (natural language inference)
        methods. Stores results as pseudolinks for network analysis.

        Args:
            args: Namespace object containing command-line arguments. Required
                argument is 'name' (land name). Optional arguments include
                'threshold' (similarity threshold), 'method' (similarity method),
                'topk' (top K similar paragraphs), 'lshbits' (LSH bits),
                'maxpairs' (maximum pairs to store), 'minrel' (minimum relevance),
                and 'backend' (NLI backend).

        Returns:
            int: 1 if similarity computation completed successfully, 0 if land
                not found.

        Notes:
            Methods: 'cosine' (vector similarity), 'cosine_lsh' (LSH-based),
            'nli'/'ann+nli'/'semantic' (semantic similarity using cross-encoders).
            For NLI methods, delegates to semantic_pipeline. For cosine methods,
            delegates to embedding_pipeline. Results stored in ParagraphSimilarity.
        """
        core.check_args(args, 'name')
        threshold = core.get_arg_option('threshold', args, set_type=float, default=None)
        method = core.get_arg_option('method', args, set_type=str, default=None)
        top_k = core.get_arg_option('topk', args, set_type=int, default=None)
        lsh_bits = core.get_arg_option('lshbits', args, set_type=int, default=None)
        max_pairs = core.get_arg_option('maxpairs', args, set_type=int, default=None)
        minrel = core.get_arg_option('minrel', args, set_type=int, default=None)
        backend = core.get_arg_option('backend', args, set_type=str, default=None)
        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print(f'Land "{args.name}" not found')
            return 0
        if method and method.lower() in ('nli', 'ann+nli', 'semantic'):
            from .semantic_pipeline import run_semantic_similarity
            count = run_semantic_similarity(
                land,
                backend=backend,
                top_k=top_k,
                minrel=minrel,
                max_pairs=max_pairs,
            )
            print(f"NLI relations stored: {count}")
        else:
            from .embedding_pipeline import compute_paragraph_similarities
            count = compute_paragraph_similarities(
                land,
                threshold=threshold,
                method=method,
                top_k=top_k,
                lsh_bits=lsh_bits,
                minrel=minrel,
                max_pairs=max_pairs,
            )
            print(f"Similarities stored: {count}")
        return 1

    @staticmethod
    def check(args: core.Namespace):
        """Verify embedding and NLI environment and settings.

        Performs diagnostic checks on the embedding and NLI configuration,
        including provider settings, API keys, optional library availability,
        and database table presence. Provides suggestions for missing components.

        Args:
            args: Namespace object containing command-line arguments (unused
                but required for controller interface).

        Returns:
            int: 1 if all checks passed, 0 if critical issues detected.

        Notes:
            Checks embedding provider configuration (fake, http, openai, mistral,
            gemini, huggingface, ollama) and validates API keys. Verifies optional
            libraries (FAISS, sentence-transformers, transformers, torch) and
            database tables (Paragraph, ParagraphEmbedding, ParagraphSimilarity).
            Provides actionable suggestions for missing dependencies.
        """
        import importlib
        ok = True
        suggestions = []
        # Provider
        prov = getattr(settings, 'embed_provider', 'fake')
        model_name = getattr(settings, 'embed_model_name', '')
        print(f"Embedding provider: {prov}")
        print(f"Embedding model: {model_name}")
        if prov == 'http':
            url = getattr(settings, 'embed_api_url', '')
            print(f"HTTP API URL: {'set' if url else 'MISSING'}")
            if not url:
                ok = False
                suggestions.append("Set settings.embed_api_url for provider 'http'.")
        elif prov == 'openai':
            key = getattr(settings, 'embed_openai_api_key', '')
            print(f"OpenAI key: {'set' if key else 'MISSING'}; base={getattr(settings,'embed_openai_base_url','')}")
            if not key:
                ok = False
                suggestions.append("Set settings.embed_openai_api_key or switch to 'fake' for offline tests.")
        elif prov == 'mistral':
            key = getattr(settings, 'embed_mistral_api_key', '')
            print(f"Mistral key: {'set' if key else 'MISSING'}; base={getattr(settings,'embed_mistral_base_url','')}")
            if not key:
                ok = False
                suggestions.append("Set settings.embed_mistral_api_key or switch provider.")
        elif prov == 'gemini':
            key = getattr(settings, 'embed_gemini_api_key', '')
            print(f"Gemini key: {'set' if key else 'MISSING'}; base={getattr(settings,'embed_gemini_base_url','')}")
            if not key:
                ok = False
                suggestions.append("Set settings.embed_gemini_api_key or switch provider.")
        elif prov == 'huggingface':
            key = getattr(settings, 'embed_hf_api_key', '')
            print(f"HF key: {'set' if key else 'MISSING'}; base={getattr(settings,'embed_hf_base_url','')}")
            if not key:
                ok = False
                suggestions.append("Set settings.embed_hf_api_key (create a token at huggingface.co/settings/tokens).")
        elif prov == 'ollama':
            print(f"Ollama base: {getattr(settings,'embed_ollama_base_url','')}")
            suggestions.append("Ensure Ollama is running locally and the embedding model is pulled (e.g., 'ollama pull nomic-embed-text').")

        # ANN libs (FAISS only)
        try:
            importlib.import_module('faiss')
            print("FAISS: available")
        except Exception:
            print("FAISS: not installed (optional)")
            suggestions.append("pip install faiss-cpu   # optional ANN backend")

        # NLI libs
        try:
            importlib.import_module('sentence_transformers')
            print("sentence-transformers: available")
        except Exception:
            print("sentence-transformers: not installed (optional)")
            suggestions.append("pip install -U sentence-transformers   # enables Cross-Encoder NLI")
        try:
            importlib.import_module('transformers')
            print("transformers: available")
        except Exception:
            print("transformers: not installed (optional)")
            suggestions.append("pip install -U transformers            # core HF runtime")
        # Torch hint for CPU installs when using sentence-transformers
        try:
            importlib.import_module('torch')
        except Exception:
            # Only suggest if sentence-transformers is desired
            suggestions.append("pip install -U torch torchvision torchaudio  # required by sentence-transformers")

        # DB tables presence
        try:
            _ = model.Paragraph.select().limit(1).count()
            _ = model.ParagraphEmbedding.select().limit(1).count()
            _ = model.ParagraphSimilarity.select().limit(1).count()
            print("DB tables: paragraph/embedding/similarity present")
        except Exception as e:
            print(f"DB tables: error {e}")
            ok = False
        if suggestions:
            print("\nNext steps (pip / config suggestions):")
            for s in suggestions:
                print(f"- {s}")
        return 1 if ok else 0

    @staticmethod
    def reset(args: core.Namespace):
        """Wipe embedding-related tables for a land or globally.

        Deletes paragraphs, embeddings, and similarities for a specific land
        or all lands. Cascades deletion through related tables. Requires
        confirmation for global reset.

        Args:
            args: Namespace object containing command-line arguments. Optional
                argument 'name' (land name) limits deletion to that land.

        Returns:
            int: 1 if reset completed successfully or no data to delete, 0 if
                land not found or user cancelled.

        Notes:
            When 'name' is provided, deletes only data for that land. Without
            'name', deletes ALL embedding data after user confirmation (type 'Y').
            Deletion is atomic and cascades: first similarities, then embeddings,
            finally paragraphs. Reports counts of deleted records.
        """
        land_name = getattr(args, 'name', None)
        if land_name:
            land = model.Land.get_or_none(model.Land.name == land_name)
            if land is None:
                print(f'Land "{land_name}" not found')
                return 0
            # Subquery of paragraphs for this land
            pids = (model.Paragraph
                    .select(model.Paragraph.id)
                    .join(model.Expression)
                    .where(model.Expression.land == land))
            pcount = pids.count()
            if pcount == 0:
                print(f'No paragraphs to delete for land {land_name}')
                return 1
            # Confirmation required (sprint-multilang, D-1); --force bypasses it
            force = bool(getattr(args, 'force', False))
            if not force and not core.confirm(
                    f"This will delete embeddings data for land '{land_name}' "
                    f"(paragraphs={pcount}). Type 'Y' to proceed: "):
                print('Aborted')
                return 0
            # Explicitly delete similarities, embeddings, then paragraphs
            with model.DB.atomic():
                sim_deleted = (model.ParagraphSimilarity
                               .delete()
                               .where((model.ParagraphSimilarity.source_paragraph.in_(pids)) |
                                      (model.ParagraphSimilarity.target_paragraph.in_(pids)))
                               .execute())
                emb_deleted = (model.ParagraphEmbedding
                               .delete()
                               .where(model.ParagraphEmbedding.paragraph.in_(pids))
                               .execute())
                par_deleted = (model.Paragraph
                               .delete()
                               .where(model.Paragraph.id.in_(pids))
                               .execute())
            print(f'Deleted land={land_name}: paragraphs={par_deleted}, embeddings={emb_deleted}, similarities={sim_deleted}')
            return 1
        else:
            # Wipe all (requires confirmation)
            pc = model.Paragraph.select().count()
            ec = model.ParagraphEmbedding.select().count()
            sc = model.ParagraphSimilarity.select().count()
            total = pc + ec + sc
            if total == 0:
                print('No embedding-related rows to delete (database already clean)')
                return 1
            force = bool(getattr(args, 'force', False))
            if not force and not core.confirm(f"This will delete ALL embeddings data (paragraphs={pc}, embeddings={ec}, similarities={sc}). Type 'Y' to proceed: "):
                print('Aborted')
                return 0
            with model.DB.atomic():
                sim_deleted = model.ParagraphSimilarity.delete().execute()
                emb_deleted = model.ParagraphEmbedding.delete().execute()
                par_deleted = model.Paragraph.delete().execute()
            print(f'Deleted ALL: paragraphs={par_deleted}, embeddings={emb_deleted}, similarities={sim_deleted}')
            return 1


class DomainController:
    """
    Domain controller class
    """

    @staticmethod
    def crawl(args: core.Namespace):
        """Crawl domain-level metadata and information.

        Fetches and updates metadata for domains in the database. Processes
        domain homepage content and extracts relevant information.

        Args:
            args: Namespace object containing command-line arguments. Optional
                arguments include 'limit' (max domains to process) and 'http'
                (HTTP status filter).

        Returns:
            int: 1 indicating successful completion.

        Notes:
            Delegates to core.crawl_domains for the actual crawling logic.
            Displays the count of domains processed. Useful for updating domain
            metadata independently of expression crawling.
        """
        fetch_limit = core.get_arg_option('limit', args, set_type=int, default=0)
        http_status = core.get_arg_option('http', args, set_type=str, default=None)
        print("%d domains processed" % core.crawl_domains(fetch_limit, http_status))
        return 1


class TagController:
    """
    Tag controller class
    """

    @staticmethod
    def export(args: core.Namespace):
        """Export tag-related data for a land.

        Exports tags and tagged content from the specified land in various
        formats suitable for analysis or visualization.

        Args:
            args: Namespace object containing command-line arguments. Required
                arguments are 'name' (land name) and 'type' (export format).
                Optional argument 'minrel' (minimum relevance) filters tagged
                content.

        Returns:
            int: 1 if export completed successfully, 0 if land not found or
                invalid export type specified.

        Notes:
            Valid export types: 'matrix' (tag co-occurrence matrix), 'content'
            (tagged content snippets). Delegates to core.export_tags for the
            actual export logic. Minimum relevance filter applies to the
            expressions containing the tagged content.
        """
        minimum_relevance = 1
        core.check_args(args, ('name', 'type'))
        valid_types = ['matrix', 'content']

        if isinstance(args.minrel, int) and (args.minrel >= 0):
            minimum_relevance = args.minrel
            print("Minimum relevance set to %s" % minimum_relevance)

        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print('Land "%s" not found' % args.name)
        else:
            if args.type in valid_types:
                core.export_tags(land, args.type, minimum_relevance)
                return 1
            print('Invalid export type "%s" [%s]' % (args.type, ', '.join(valid_types)))
        return 0


class HeuristicController:
    """
    Heuristic controller class
    """

    @staticmethod
    def update(args: core.Namespace):
        """Recompute expression domains from settings.heuristics.

        Default (no flags): recompute every expression's domain from the URL
        heuristic alone (legacy global behaviour). Optional flags:

        - ``--land NAME``: restrict to one land (0 if the land is not found).
        - ``--limit N``: cap the expressions processed.
        - ``--html``: for opaque-platform hosts (settings.opaque_platforms),
          resolve the editorial entity from the page HTML (Expression.html)
          instead of the URL alone.
        - ``--fetch-missing``: when ``--html`` is set, fetch HTML on the fly for
          opaque-host expressions lacking stored html. Requires an explicit
          ``--limit`` to bound network I/O; the fetched HTML is used, not stored.

        Args:
            args: Namespace with optional ``land``, ``limit``, ``html``,
                ``fetch_missing``.

        Returns:
            int: 1 on success, 0 on a validation error (unknown land, or
            ``--fetch-missing`` without ``--html`` / without ``--limit``).
        """
        use_html = bool(getattr(args, 'html', False))
        fetch_missing = bool(getattr(args, 'fetch_missing', False))
        dry_run = bool(getattr(args, 'dry_run', False))
        limit = core.get_arg_option('limit', args, set_type=int, default=None)
        minrel = core.get_arg_option('minrel', args, set_type=int, default=None)
        land_name = getattr(args, 'land', None)

        land = None
        if land_name:
            land = model.Land.get_or_none(model.Land.name == land_name)
            if land is None:
                print('Land "%s" not found' % land_name)
                return 0

        if fetch_missing and not use_html:
            print('--fetch-missing requires --html')
            return 0
        if fetch_missing and not limit:
            print('--fetch-missing requires an explicit --limit to bound '
                  'network fetches')
            return 0

        html_map = {}
        if fetch_missing and not dry_run:  # dry-run stays offline
            loop = _get_event_loop()
            html_map = loop.run_until_complete(
                core.fetch_missing_opaque_html(land, limit, minrel))
            print(f'Volatile fetch: {len(html_map)} page(s) retrieved')

        core.update_heuristic(
            land, use_html=use_html, fetch_missing=fetch_missing,
            limit=(None if fetch_missing else limit), html_map=html_map,
            minrel=minrel, dry_run=dry_run)
        return 1


class SearchController:
    """Multi-API search router commands (`mywi.py search …`).

    All four verbs (`run`, `list`, `usage`, `check`) follow the existing
    controller convention: ``@staticmethod``, accept ``args: core.Namespace``,
    return ``int`` (1 = success, 0 = failure). Heavy imports (router,
    providers) are lazy so that unrelated commands do not pull aiohttp at
    process start.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_router():
        """Instantiate the router with every provider that is configured."""
        from .search import SearchRouter
        from .search.providers.searxng import SearxngProvider
        from .search.providers.brave import BraveProvider
        from .search.providers.serper import SerperProvider
        from .search.providers.serpapi import SerpApiProvider
        from .search.providers.tavily import TavilyProvider

        router = SearchRouter(
            timeout=getattr(settings, 'SEARCH_PROVIDER_TIMEOUT',
                            SearchRouter.DEFAULT_TIMEOUT))
        for cls in (SearxngProvider, BraveProvider, SerperProvider,
                    SerpApiProvider, TavilyProvider):
            try:
                router.register(cls())
            except Exception as exc:  # pragma: no cover — defensive
                print(f"[search] could not init {cls.__name__}: {exc}")
        return router

    @staticmethod
    def _all_providers():
        """Return the canonical (provider_name, class) list — used by `check`."""
        from .search.providers.searxng import SearxngProvider
        from .search.providers.brave import BraveProvider
        from .search.providers.serper import SerperProvider
        from .search.providers.serpapi import SerpApiProvider
        from .search.providers.tavily import TavilyProvider
        return [
            ("searxng", SearxngProvider),
            ("brave",   BraveProvider),
            ("serper",  SerperProvider),
            ("serpapi", SerpApiProvider),
            ("tavily",  TavilyProvider),
        ]

    @staticmethod
    def _resolve_strategy(args) -> str:
        """Resolve the strategy with sensible fallbacks.

        Order: CLI ``--strategy`` → ``settings.SEARCH_DEFAULT_STRATEGY``
        → ``"fallback"``.
        """
        strategy = getattr(args, 'strategy', None)
        if strategy:
            return strategy
        return getattr(settings, 'SEARCH_DEFAULT_STRATEGY', 'fallback')

    @staticmethod
    def _resolve_providers(args):
        """Parse the optional --providers CSV into a list (or None)."""
        raw = getattr(args, 'providers', None)
        if not raw:
            return None
        return [p.strip() for p in raw.split(',') if p.strip()]

    # ------------------------------------------------------------------
    # search check
    # ------------------------------------------------------------------

    @staticmethod
    def check(args: 'core.Namespace') -> int:
        """Display a per-provider configured/unconfigured status table."""
        print("Provider          Configured")
        print("-" * 32)
        for name, cls in SearchController._all_providers():
            try:
                ok = cls().is_configured()
            except Exception as exc:  # pragma: no cover — defensive
                ok = False
                print(f"{name:<17} ERROR ({exc})")
                continue
            mark = "yes" if ok else "no "
            print(f"{name:<17} {mark}")
        return 1

    # ------------------------------------------------------------------
    # search run
    # ------------------------------------------------------------------

    @staticmethod
    def run(args: 'core.Namespace') -> int:
        """Execute a multi-API search and persist its results into the Land.

        Required CLI args:
            --land   Land name.
            --query  Search query.

        Optional CLI args:
            --limit       Max URLs per provider (default 20).
            --strategy    'fallback' (default) or 'parallel'.
            --language    ISO 639-1 language code (default: the land's
                          primary language; 'fr' as last resort).
            --providers   Comma-separated whitelist.
        """
        land_name = getattr(args, 'land', None) or getattr(args, 'name', None)
        if not land_name:
            print('[search run] --land is required')
            return 0
        query = getattr(args, 'query', None)
        if not query:
            print('[search run] --query is required')
            return 0

        land = model.Land.get_or_none(model.Land.name == land_name)
        if land is None:
            print(f'Land "{land_name}" not found')
            return 0

        limit = core.get_arg_option('limit', args, set_type=int, default=20) or 20
        strategy = SearchController._resolve_strategy(args)
        # Default to the land's primary language (sprint-multilang, A7).
        language = (getattr(args, 'language', None)
                    or str(land.lang or 'fr').split(',')[0].strip().lower() or 'fr')
        provider_filter = SearchController._resolve_providers(args)

        router = SearchController._build_router()
        if not router.providers:
            print('[search run] no provider configured. Run `python mywi.py search check` to diagnose.')
            return 0

        active = list(router.provider_names)
        if provider_filter:
            active = [n for n in active if n in provider_filter]
        print(f'[search run] strategy={strategy} providers=[{", ".join(active)}] '
              f'language={language} limit={limit}')

        # Execute the asynchronous search synchronously from the controller.
        results = asyncio.run(router.search(
            query,
            strategy=strategy,
            num=limit,
            language=language,
            providers=provider_filter,
        ))

        usage_report = router.usage_report()
        sq = SearchController._persist_results(
            land=land,
            query=query,
            strategy=strategy,
            language=language,
            num_requested=limit,
            results=results,
            usage_report=usage_report,
        )

        new_urls, dup_urls = sq.metadata['new'], sq.metadata['duplicates']
        print(f'[search run] {len(results)} URLs from providers — '
              f'{new_urls} new in Land, {dup_urls} already present.')
        print('[search run] usage report:')
        for name, snap in usage_report.items():
            print(f'  - {name:<10} calls={snap["calls"]} errors={snap["errors"]} '
                  f'status={snap["status"]} quota={snap["monthly_quota"]}')
        return 1

    @staticmethod
    def _persist_results(*, land, query, strategy, language,
                         num_requested, results, usage_report):
        """Insert SearchQuery + per-result SearchResultLog + Expression rows.

        Returns the SearchQuery instance with an ad-hoc ``metadata`` dict
        (Python attribute, not persisted) tracking the new vs. duplicate
        URL counts for the console summary.
        """
        import datetime
        import json
        from .url_normalizer import normalize_url

        with model.DB.atomic():
            sq = model.SearchQuery.create(
                land=land,
                query=query,
                strategy=strategy,
                language=language,
                num_requested=num_requested,
                num_collected=0,
                usage_report=json.dumps(usage_report),
            )

            new_urls = 0
            duplicates = 0
            collected = 0
            for r in results:
                if not r.url:
                    continue
                normalized = normalize_url(r.url) or r.url

                # Pre-check existence so we can report new vs. duplicate.
                pre_exists = model.Expression.get_or_none(
                    model.Expression.url == normalized,
                    model.Expression.land == land,
                ) is not None

                expression = core.add_expression(land, r.url)
                if expression is False or expression is None:
                    # URL not crawlable (PDF, image, scheme filter, …).
                    continue

                if pre_exists:
                    duplicates += 1
                else:
                    new_urls += 1

                model.SearchResultLog.create(
                    search_query=sq,
                    url=expression.url,
                    title=r.title,
                    snippet=r.snippet,
                    providers=r.providers or "",
                    rank_min=r.rank,
                    expression=expression,
                )
                collected += 1

            sq.num_collected = collected
            sq.completed_at = datetime.datetime.now()
            sq.save()

        sq.metadata = {'new': new_urls, 'duplicates': duplicates}
        return sq

    # ------------------------------------------------------------------
    # search list
    # ------------------------------------------------------------------

    @staticmethod
    def list(args: 'core.Namespace') -> int:
        """List previous search queries for a Land."""
        land_name = getattr(args, 'land', None) or getattr(args, 'name', None)
        if not land_name:
            print('[search list] --land is required')
            return 0
        land = model.Land.get_or_none(model.Land.name == land_name)
        if land is None:
            print(f'Land "{land_name}" not found')
            return 0

        rows = (model.SearchQuery
                .select()
                .where(model.SearchQuery.land == land)
                .order_by(model.SearchQuery.created_at.desc()))

        count = rows.count()
        if count == 0:
            print(f'[search list] no search query for Land "{land.name}"')
            return 1

        print(f'{"id":>4}  {"date":<19}  {"strategy":<10}  '
              f'{"lang":<4}  {"req":>4}  {"got":>4}  query')
        for sq in rows:
            date = sq.created_at.strftime("%Y-%m-%d %H:%M:%S") if sq.created_at else "-"
            print(f'{sq.id:>4}  {date:<19}  {sq.strategy:<10}  '
                  f'{sq.language:<4}  {sq.num_requested:>4}  '
                  f'{sq.num_collected:>4}  {sq.query}')
        return 1

    # ------------------------------------------------------------------
    # search usage
    # ------------------------------------------------------------------

    @staticmethod
    def usage(args: 'core.Namespace') -> int:
        """Aggregate the JSON ``usage_report`` columns of past queries."""
        import json
        land_name = getattr(args, 'land', None) or getattr(args, 'name', None)
        if not land_name:
            print('[search usage] --land is required')
            return 0
        land = model.Land.get_or_none(model.Land.name == land_name)
        if land is None:
            print(f'Land "{land_name}" not found')
            return 0

        agg: dict = {}
        for sq in model.SearchQuery.select().where(model.SearchQuery.land == land):
            if not sq.usage_report:
                continue
            try:
                report = json.loads(sq.usage_report)
            except (TypeError, ValueError):
                continue
            for name, snap in report.items():
                bucket = agg.setdefault(name, {
                    'calls': 0, 'errors': 0,
                    'last_status': snap.get('status'),
                    'monthly_quota': snap.get('monthly_quota'),
                })
                bucket['calls'] += int(snap.get('calls') or 0)
                bucket['errors'] += int(snap.get('errors') or 0)
                bucket['last_status'] = snap.get('status') or bucket['last_status']

        if not agg:
            print(f'[search usage] no usage report for Land "{land.name}"')
            return 1

        print(f'{"provider":<12} {"calls":>6} {"errors":>7} {"last_status":<16} {"quota":>6}')
        for name, snap in sorted(agg.items()):
            quota = snap['monthly_quota']
            quota_str = '-' if quota is None else str(quota)
            print(f'{name:<12} {snap["calls"]:>6} {snap["errors"]:>7} '
                  f'{snap["last_status"]:<16} {quota_str:>6}')
        return 1
