"""
Application controller
"""
import asyncio
import os
import sys
from typing import Any

from peewee import JOIN, fn
import aiohttp

import settings
from . import core
from . import model


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
        except Exception as e:
            # Non bloquant: on loggue et on continue
            print(f"[migrate] Warning: LLM columns check failed: {e}")
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
        ]

        if core.confirm("Warning, existing data will be lost, type 'Y' to proceed : "):
            # Filter out None entries (older versions)
            tables_clean = [t for t in tables if t is not None]
            model.DB.drop_tables(tables_clean)
            model.DB.create_tables(tables_clean)
            print("Model created, setup complete")
            return 1

    @staticmethod
    def medianalyse(args: core.Namespace):
        """Perform sequential batch media analysis for a land.

        Analyzes all media files associated with expressions in the specified
        land. Processes images to extract metadata, dimensions, colors, EXIF
        data, and perceptual hashes. Results are stored in the Media table.

        Args:
            args: Namespace object containing command-line arguments. Required
                argument is 'name' (land name). Optional arguments include
                'depth' (maximum expression depth) and 'minrel' (minimum
                relevance score).

        Returns:
            int: 1 if analysis completed successfully, 0 if land not found.

        Notes:
            Uses MediaAnalyzer with settings from the configuration module.
            Processing is sequential (single connection) to avoid overloading
            servers. Can be interrupted with KeyboardInterrupt. Analysis
            metadata is stored with timestamp and error information if applicable.
        """
        core.check_args(args, 'name')
        depth = core.get_arg_option('depth', args, set_type=int, default=0)
        minrel = core.get_arg_option('minrel', args, set_type=float, default=0.0)
        from .media_analyzer import MediaAnalyzer
        from datetime import datetime
        
        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print(f'Land "{args.name}" introuvable')
            return 0
        
        query = model.Expression.select().where(model.Expression.land == land)
        if depth > 0:
            query = query.where(model.Expression.depth <= depth)
        if minrel > 0:
            query = query.where(model.Expression.relevance >= minrel)
            
        expressions = list(query)
        print(f'Début de l\'analyse médias pour le land "{land.name}" avec {len(expressions)} expressions')
        
        async def process():
            connector = aiohttp.TCPConnector(limit=1, ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
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
                for expr in expressions:
                    if not hasattr(expr, 'medias'):
                        continue
                    for media in expr.medias:
                        print(f'Analyse média #{media.id}: {media.url}')
                        result = await analyzer.analyze_image(str(media.url))
                        for field, value in result.items():
                            if hasattr(media, field) and field != 'error':
                                setattr(media, field, value)
                        media.analyzed_at = datetime.now()
                        if 'error' in result:
                            media.analysis_error = result['error']
                        media.save()
                        print('  =>', 'Erreur:' + str(result.get('error', '')) if 'error' in result else 'OK')

        if sys.platform == 'win32':
            asyncio.set_event_loop(asyncio.ProactorEventLoop())
        loop = asyncio.get_event_loop()
        
        try:
            loop.run_until_complete(process())
        except KeyboardInterrupt:
            print('Analyse interrompue par l\'utilisateur')
        
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
                argument is 'name' (land name).

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
            
        print(f'Début analyse média pour {args.name}')
        from .media_analyzer import MediaAnalyzer
        loop = asyncio.get_event_loop()
        if sys.platform == 'win32':
            asyncio.set_event_loop(asyncio.ProactorEventLoop())
            
        try:
            result = loop.run_until_complete(core.medianalyse_land(land))
            print(f"Analyse terminée : {result['processed']} médias traités")
            return 1
        except Exception as e:
            print(f"Erreur lors de l'analyse : {str(e)}")
            return 0

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
            print(f'Land "{args.name}" not found')
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
        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print('Land "%s" not found' % args.name)
        else:
            if sys.platform == 'win32':
                asyncio.set_event_loop(asyncio.ProactorEventLoop())
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(
                core.consolidate_land(land, fetch_limit, depth, min_relevance)
            )
            consolidated, errors = results
            print(
                f"%d expressions consolidated (%d errors, minrel=%d)"
                % (consolidated, errors, min_relevance)
            )
            return 1
        return 0


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
        lang_str = ",".join(args.lang) if isinstance(args.lang, list) else str(args.lang)
        land = model.Land.create(name=args.name, description=args.desc, lang=lang_str)
        os.makedirs(os.path.join(settings.data_location, 'lands/%s') % land.id, exist_ok=True)
        print('Land "%s" created' % args.name)
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
            function. Words are stored with both original term and lemma forms.
            After adding terms, land_relevance is called to update expression
            scores based on the new dictionary.
        """
        core.check_args(args, ('land', 'terms'))
        land = model.Land.get_or_none(model.Land.name == args.land)
        if land is None:
            print('Land "%s" not found' % args.land)
        else:
            for term in core.split_arg(args.terms):
                with model.DB.atomic():
                    lemma = ' '.join([core.stem_word(w) for w in term.split(' ')])
                    word, _ = model.Word.get_or_create(term=term, lemma=lemma)
                    model.LandDictionary.create(land=land.id, word=word.id)
                    print('Term "%s" created in land %s' % (term, args.land))
            core.land_relevance(land)
            return 1
        return 0

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
                (language), 'datestart' and 'dateend' (date range), 'timestep'
                (time window for date ranges), 'sleep' (delay between requests),
                and 'progress' (enable progress output).

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

        # API key lookup mirrors other integrations (settings first, then env var).
        api_key = getattr(settings, 'serpapi_api_key', '') or os.getenv('MWI_SERPAPI_API_KEY', '')
        if not api_key:
            print('[urlist] SerpAPI key missing — set settings.serpapi_api_key or MWI_SERPAPI_API_KEY')
            return 0

        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print('Land "%s" not found' % args.name)
            return 0

        # CLI stores languages as a list; keep the first entry for the request.
        lang_list = getattr(args, 'lang', ['fr'])
        lang = lang_list[0] if isinstance(lang_list, list) and lang_list else 'fr'
        engine = (getattr(args, 'engine', 'google') or 'google').lower()
        allowed_engines = {'google', 'bing', 'duckduckgo'}
        if engine not in allowed_engines:
            print(f'[urlist] Unsupported engine "{engine}" — choose google, bing or duckduckgo')
            return 0

        datestart = getattr(args, 'datestart', None)
        dateend = getattr(args, 'dateend', None)
        timestep = getattr(args, 'timestep', 'week') or 'week'
        sleep_seconds = core.get_arg_option('sleep', args, set_type=float, default=1.0)

        progress_requested = bool(getattr(args, 'progress', False))
        has_date_range = bool(datestart and dateend)
        date_capable_engines = {'google', 'duckduckgo'}
        if engine not in date_capable_engines and has_date_range:
            print('[urlist] datestart/dateend filters are only supported with --engine=google or --engine=duckduckgo')
            return 0
        want_progress = progress_requested or has_date_range

        progress_callback = None
        if want_progress:
            def progress_callback(window_start, window_end, window_count):
                if window_start and window_end:
                    start_label = window_start.isoformat()
                    end_label = window_end.isoformat()
                    print(f'[urlist] {start_label} → {end_label} — fetched {window_count} results', flush=True)
                else:
                    print(f'[urlist] fetched {window_count} results (no date filter)', flush=True)

        try:
            # Centralised helper handles pagination, date windows and error reporting.
            serp_results = core.fetch_serpapi_url_list(
                api_key=api_key,
                query=args.query,
                engine=engine,
                lang=lang,
                datestart=datestart,
                dateend=dateend,
                timestep=timestep,
                sleep_seconds=sleep_seconds,
                progress_hook=progress_callback
            )
        except core.SerpApiError as error:
            print(f'[urlist] {error}')
            return 0

        added = 0
        skipped = 0
        for item in serp_results:
            url = item.get('link')
            if not url:
                skipped += 1
                continue

            raw_date = item.get('date') if engine == 'google' else None
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
                skipped += 1
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
                added += 1
            else:
                skipped += 1

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

        if core.confirm("Land and/or underlying objects will be deleted, type 'Y' to proceed : "):
            land = model.Land.get_or_none(model.Land.name == args.name)
            if land is None:
                print('Land "%s" not found' % args.name)
                return 0
            if maxrel > 0:
                query = model.Expression.delete().where((model.Expression.land == land)
                                                & (model.Expression.relevance < maxrel)
                                                & (model.Expression.fetched_at.is_null(False)))
                query.execute()
                print("Expressions deleted")
            else:
                land.delete_instance(recursive=True)
                print("Land %s deleted" % args.name)
            return 1
        return 0

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
        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print('Land "%s" not found' % args.name)
        else:
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(core.crawl_land(land, fetch_limit, http_status, depth))
            print("%d expressions processed (%d errors)" % results)
            return 1
        return 0

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
        
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(
            run_readable_pipeline(land, fetch_limit, depth_limit, merge_strategy, llm_enabled)
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
            'pseudolinksdomain' (domain-level pseudolinks).
        """
        minimum_relevance = 1
        core.check_args(args, ('name', 'type'))
        valid_types = ['pagecsv', 'fullpagecsv', 'nodecsv', 'pagegexf',
                       'nodegexf', 'mediacsv', 'corpus', 'pseudolinks',
                       'pseudolinkspage', 'pseudolinksdomain']

        if isinstance(args.minrel, int) and (args.minrel >= 0):
            minimum_relevance = args.minrel
            print("Minimum relevance set to %s" % minimum_relevance)
            
        land = model.Land.get_or_none(model.Land.name == args.name)
        if land is None:
            print('Land "%s" not found' % args.name)
        else:
            if args.type in valid_types:
                core.export_land(land, args.type, minimum_relevance)
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
            verdict = llm.is_relevant_via_openrouter(land, expr)
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

        print(f'Validation LLM terminée: examinées={total}, mises à jour={updated}, modèle={settings.openrouter_model}, force={force}')
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
            if not core.confirm(f"This will delete ALL embeddings data (paragraphs={pc}, embeddings={ec}, similarities={sc}). Type 'Y' to proceed: "):
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
        """Update domains using specified heuristics.

        Applies domain-specific heuristics from settings to update and enrich
        domain information. Useful for extracting social media profiles or
        other structured data from domain URLs.

        Args:
            args: Namespace object containing command-line arguments (unused
                but required for controller interface).

        Returns:
            int: 1 indicating successful completion.

        Notes:
            Delegates to core.update_heuristic which processes domains according
            to patterns defined in settings.heuristics. Heuristics typically
            include regex patterns for extracting social media handles, RSS
            feeds, or other domain-specific metadata.
        """
        core.update_heuristic()
        return 1
