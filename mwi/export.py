"""Export module for MyWebIntelligence data export functionality.

This module provides the Export class for exporting research land data to various
formats including CSV, GEXF (graph exchange format), and ZIP archives. It supports
multiple export types for different data perspectives:

- Page exports: Expression-level data with metadata and SEO rank information
- Node exports: Domain-level aggregated data
- Media exports: Media links associated with expressions
- Tag exports: Hierarchical tag data in matrix or content format
- Corpus exports: Text corpora with Dublin Core metadata
- Pseudolink exports: Semantic similarity relationships at paragraph, page, and domain levels
- Graph exports: GEXF network graphs for visualization tools like Gephi

All exports support filtering by minimum relevance threshold and are optimized
for large-scale data analysis in digital humanities research.
"""

import csv
import datetime
import json
import re
from textwrap import dedent
import unicodedata
from lxml import etree
from urllib.parse import urlparse
from zipfile import ZipFile
from . import model
from .link_context import extract_all_links, extract_markdown_links
from .url_normalizer import normalize_url


class Export:
    """Export class for generating various data export formats from land data.

    This class handles exporting research land data to multiple formats including
    CSV, GEXF (graph format), and ZIP archives with text corpora. It supports
    filtering by relevance threshold and includes specialized exports for pages,
    domains, media, tags, and semantic links.

    Attributes:
        gexf_ns: XML namespace mappings for GEXF format.
        type: Export type identifier (e.g., 'pagecsv', 'nodegexf').
        land: Land model instance for the export.
        relevance: Minimum relevance score threshold for filtering expressions.
    """
    gexf_ns = {None: 'http://www.gexf.net/1.2draft', 'viz': 'http://www.gexf.net/1.1draft/viz'}
    type = None
    land = None
    relevance = 1

    def __init__(self, export_type: str, land: model.Land, minimum_relevance: int,
                 fullhtml: bool = False):
        """Initialize an Export instance with specified parameters.

        Args:
            export_type: The format type for export (e.g., 'pagecsv', 'gexf', 'corpus').
            land: The Land model instance representing the research project.
            minimum_relevance: Minimum relevance score threshold for including expressions.
            fullhtml: When True (and export_type == 'nodelinkcsv'), also emit the
                raw-HTML link network files (*fullhtml.csv). Ignored otherwise.

        Notes:
            The export_type determines which write method will be called.
            Only expressions with relevance >= minimum_relevance are included.
        """
        self.type = export_type
        self.land = land
        self.relevance = minimum_relevance
        self.fullhtml = fullhtml

    def write(self, export_type: str, filename):
        """Proxy method that dispatches to appropriate format-specific writer.

        Args:
            export_type: The export format type (e.g., 'pagecsv', 'nodegexf', 'corpus').
            filename: Base filename without extension (extension added automatically).

        Returns:
            int: Number of records/items written to the output file.

        Notes:
            Automatically appends appropriate file extensions (.csv, .gexf, or .zip).
            Dynamically calls the corresponding write_{export_type} method.
        """
        call_write = getattr(self, 'write_' + export_type)
        if export_type.endswith('csv'):
            filename += '.csv'
        elif export_type.endswith('gexf'):
            filename += '.gexf'
        elif export_type.endswith('corpus'):
            filename += '.zip'
        elif export_type.endswith('json'):        # nodesjson + pagesjson (force-graph)
            filename += '.json'
        elif export_type == 'htmldump':
            filename += '.zip'
        return call_write(filename)

    def get_sql_cursor(self, sql, column_map):
        """Build and execute SQL query with column mapping.

        Args:
            sql: SQL query template with {} placeholder for column definitions.
            column_map: Dictionary mapping output column names to SQL expressions.

        Returns:
            Database cursor object for iterating over query results.

        Notes:
            Automatically injects land_id and minimum relevance parameters.
            Formats column_map as "sql_expression AS output_name" clauses.
        """
        cols = ",\n".join(["{1} AS {0}".format(*i) for i in column_map.items()])
        return model.DB.execute_sql(sql.format(cols), (self.land.get_id(), self.relevance))

    def write_pagecsv(self, filename) -> int:
        """Write page data to CSV file with SEO rank metadata.

        Args:
            filename: Path to output CSV file.

        Returns:
            int: Number of expression records written.

        Notes:
            Includes expression metadata, domain information, tags, and SEO rank data.
            SEO rank payload is parsed from JSON and flattened into additional columns.
            Missing or unknown values are normalized to 'na'.
        """
        col_map = {
            'id': 'e.id',
            'url': 'e.url',
            'title': 'e.title',
            'description': 'e.description',
            'keywords': 'e.keywords',
            'published_at': 'e.published_at',
            'relevance': 'e.relevance',
            'depth': 'e.depth',
            'domain_id': 'e.domain_id',
            'domain_name': 'd.name',
            'domain_description': 'd.description',
            'domain_keywords': 'd.keywords',
            'tags': 'GROUP_CONCAT(DISTINCT t.name)'
        }
        sql = """
            SELECT
                {}
            FROM expression AS e
            JOIN domain AS d ON d.id = e.domain_id
            LEFT JOIN taggedcontent tc ON tc.expression_id = e.id
            LEFT JOIN tag t ON t.id = tc.tag_id
            WHERE e.land_id = ? AND relevance >= ?
            GROUP BY e.id
        """
        records, seorank_keys = self._fetch_page_rows_with_seorank(col_map, sql)
        base_keys = list(col_map.keys())
        header = base_keys + seorank_keys

        count = 0
        with open(filename, 'w', newline='\n', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            if records:
                writer.writerow(header)
                for base_data, seorank_payload in records:
                    row = [self._normalize_value(base_data.get(key)) for key in base_keys]
                    row.extend(self._normalize_value(seorank_payload.get(key)) for key in seorank_keys)
                    writer.writerow(row)
                    count += 1
        return count

    def write_fullpagecsv(self, filename) -> int:
        """Write full page data including readable content to CSV file.

        Args:
            filename: Path to output CSV file.

        Returns:
            int: Number of expression records written.

        Notes:
            Similar to write_pagecsv but includes the readable field.
            Contains full extracted content from Mercury Parser.
        """
        col_map = {
            'id': 'e.id',
            'url': 'e.url',
            'title': 'e.title',
            'description': 'e.description',
            'keywords': 'e.keywords',
            'readable': 'e.readable',
            'published_at': 'e.published_at',
            'relevance': 'e.relevance',
            'depth': 'e.depth',
            'domain_id': 'e.domain_id',
            'domain_name': 'd.name',
            'domain_description': 'd.description',
            'domain_keywords': 'd.keywords',
            'tags': 'GROUP_CONCAT(DISTINCT t.name)'
        }
        sql = """
            SELECT
                {}
            FROM expression AS e
            JOIN domain AS d ON d.id = e.domain_id
            LEFT JOIN taggedcontent tc ON tc.expression_id = e.id
            LEFT JOIN tag t ON t.id = tc.tag_id
            WHERE e.land_id = ? AND relevance >= ?
            GROUP BY e.id
        """
        cursor = self.get_sql_cursor(sql, col_map)
        return self.write_csv(filename, col_map.keys(), cursor)

    def write_nodecsv(self, filename) -> int:
        """Write domain-level aggregated data to CSV file.

        Args:
            filename: Path to output CSV file.

        Returns:
            int: Number of domain records written.

        Notes:
            Aggregates expressions by domain with counts and average relevance.
            Each row represents a unique domain in the land.
        """
        col_map = {
            'id': 'd.id',
            'name': 'd.name',
            'title': 'd.title',
            'description': 'd.description',
            'keywords': 'd.keywords',
            'expressions': 'COUNT(*)',
            'average_relevance': 'ROUND(AVG(e.relevance), 2)'
        }
        sql = """
            SELECT
                {}
            FROM domain AS d
            JOIN expression AS e ON e.domain_id = d.id
            WHERE land_id = ? AND e.relevance >= ?
            GROUP BY d.id
        """
        cursor = self.get_sql_cursor(sql, col_map)
        return self.write_csv(filename, col_map.keys(), cursor)

    def write_nodesjson(self, filename) -> int:
        """Write the domain graph as a force-graph {nodes, links} JSON file.

        A node is a domain of the land carrying at least one expression with
        relevance >= minrel. Node variables = 9 analytical fields (union of the
        two domain node exports, the technical http_status is dropped) plus
        ``corpus``, the sorted list of the domain's expressions filtered by
        minrel, each a nested object ``{title, urlarticle, description,
        published_at}`` (raw values ; ``null`` when absent). Links mirror
        _write_domainlinks (directed inter-domain edges, value = page-to-page
        link count). Deterministic ordering throughout.

        Args:
            filename: Path to output JSON file.

        Returns:
            int: Number of domain nodes written (0 for an empty graph).
        """
        node_cols = {
            'id': 'd.id',
            'name': 'd.name',
            'title': 'd.title',
            'description': 'd.description',
            'keywords': 'd.keywords',
            'nbexpressions': 'COUNT(e.id)',
            'average_relevance': 'ROUND(AVG(e.relevance), 2)',
            'first_expression_date': 'MIN(e.published_at)',
            'last_expression_date': 'MAX(e.published_at)',
        }
        node_sql = """
            SELECT {}
            FROM domain AS d
            JOIN expression AS e ON e.domain_id = d.id
            WHERE e.land_id = ? AND e.relevance >= ?
            GROUP BY d.id
            ORDER BY nbexpressions DESC, d.id
        """
        nodes = [
            dict(zip(node_cols.keys(), row))
            for row in self.get_sql_cursor(node_sql, node_cols)
        ]

        # corpus : one nested {title, urlarticle, description, published_at}
        # object per expression, sorted by (domain_id, url) then grouped in
        # Python (deterministic order ; SQLite GROUP_CONCAT has no guaranteed
        # ordering). Raw values are kept — None serialises to JSON null.
        corpus_cols = {
            'domain_id': 'e.domain_id',
            'title': 'e.title',
            'urlarticle': 'e.url',
            'description': 'e.description',
            'published_at': 'e.published_at',
        }
        corpus_sql = """
            SELECT {}
            FROM expression AS e
            WHERE e.land_id = ? AND e.relevance >= ?
            ORDER BY e.domain_id, e.url
        """
        corpus_by_domain = {}
        for domain_id, title, urlarticle, description, published_at in \
                self.get_sql_cursor(corpus_sql, corpus_cols):
            corpus_by_domain.setdefault(domain_id, []).append({
                'title': title,
                'urlarticle': urlarticle,
                'description': description,
                'published_at': published_at,
            })
        for node in nodes:
            node['corpus'] = corpus_by_domain.get(node['id'], [])

        # inter-domain links (logic of _write_domainlinks, with a deterministic
        # tiebreaker so the output is byte-identical for identical data).
        link_cols = {'source': 'e1.domain_id', 'target': 'e2.domain_id',
                     'value': 'COUNT(*)'}
        link_sql = """
            WITH idx(x) AS (
                SELECT id FROM expression
                WHERE land_id = ? AND relevance >= ?
            )
            SELECT {}
            FROM expressionlink AS link
            JOIN expression AS e1 ON e1.id = link.source_id
            JOIN expression AS e2 ON e2.id = link.target_id
            WHERE link.source_id IN idx
              AND link.target_id IN idx
              AND e1.domain_id != e2.domain_id
            GROUP BY e1.domain_id, e2.domain_id
            ORDER BY value DESC, e1.domain_id, e2.domain_id
        """
        links = [
            dict(zip(link_cols.keys(), row))
            for row in self.get_sql_cursor(link_sql, link_cols)
        ]

        graph = {'nodes': nodes, 'links': links}
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(graph, file, ensure_ascii=False)
        return len(nodes)

    def write_pagesjson(self, filename) -> int:
        """Write the page graph as a force-graph {nodes, links} JSON file.

        A node is an Expression. Variables = those of write_pagecsv (without
        ``depth`` nor ``readable``), with ``tags`` as a sorted array and
        ``seorank`` as a nested object ({} when absent). Raw values are kept:
        missing fields serialise to JSON ``null`` (never the CSV ``na``
        sentinel). Links are page-to-page edges of the closed minrel network
        (no aggregation, intra-domain edges kept). Deterministic ordering.

        Args:
            filename: Path to output JSON file.

        Returns:
            int: Number of page nodes written (0 for an empty graph).
        """
        page_cols = {
            'id': 'e.id',
            'url': 'e.url',
            'title': 'e.title',
            'description': 'e.description',
            'keywords': 'e.keywords',
            'published_at': 'e.published_at',
            'relevance': 'e.relevance',
            'domain_id': 'e.domain_id',
            'domain_name': 'd.name',
            'domain_description': 'd.description',
            'domain_keywords': 'd.keywords',
            'tags': 'GROUP_CONCAT(DISTINCT t.name)',
        }
        page_sql = """
            SELECT {}
            FROM expression AS e
            JOIN domain AS d ON d.id = e.domain_id
            LEFT JOIN taggedcontent tc ON tc.expression_id = e.id
            LEFT JOIN tag t ON t.id = tc.tag_id
            WHERE e.land_id = ? AND relevance >= ?
            GROUP BY e.id
            ORDER BY e.id
        """
        # Reuse the existing SEO Rank parser (export.py): records is a list of
        # (raw base_data, seorank_payload dict).
        records, _ = self._fetch_page_rows_with_seorank(page_cols, page_sql)
        nodes = []
        for base, seorank in records:
            node = dict(base)                  # raw values -> None becomes JSON null
            node['tags'] = sorted(base['tags'].split(',')) if base.get('tags') else []
            node['seorank'] = seorank or {}    # nested object, {} when absent
            nodes.append(node)

        # page-to-page links (closed minrel network) ; no aggregation, intra-domain kept.
        link_cols = {'source': 'link.source_id', 'target': 'link.target_id'}
        link_sql = """
            WITH idx(x) AS (
                SELECT id FROM expression
                WHERE land_id = ? AND relevance >= ?
            )
            SELECT {}
            FROM expressionlink AS link
            WHERE link.source_id IN idx AND link.target_id IN idx
            ORDER BY link.source_id, link.target_id
        """
        links = [
            dict(zip(link_cols.keys(), row))
            for row in self.get_sql_cursor(link_sql, link_cols)
        ]

        graph = {'nodes': nodes, 'links': links}
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(graph, file, ensure_ascii=False)
        return len(nodes)

    def write_mediacsv(self, filename) -> int:
        """Write media links to CSV file.

        Args:
            filename: Path to output CSV file.

        Returns:
            int: Number of media records written.

        Notes:
            Exports all media associated with expressions in the land.
            Includes media ID, expression ID, URL, and type.
        """
        col_map = {
            'id': 'm.id',
            'expression_id': 'm.expression_id',
            'url': 'm.url',
            'type': 'm.type'
        }
        sql = """
            SELECT
                {}
            FROM media AS m
            JOIN expression AS e ON e.id = m.expression_id
            WHERE e.land_id = ? AND e.relevance >= ?
            GROUP BY m.id
        """
        cursor = self.get_sql_cursor(sql, col_map)
        return self.write_csv(filename, col_map.keys(), cursor)

    def write_nodelinkcsv(self, filename) -> int:
        """Export 4 CSV files: page nodes/links and domain nodes/links.

        Args:
            filename: Base path for output CSV files.

        Returns:
            int: Total number of records written across the 4 files.

        Notes:
            Without --fullhtml: the 4 MyWI files (_pagesnodes.csv,
            _pageslinks.csv, _domainnodes.csv, _domainlinks.csv) from
            ExpressionLink. With --fullhtml: the 4 raw-HTML link-network
            files (*fullhtml.csv) INSTEAD — the flag switches which network
            is exported, it is not additive. Export twice to get both.
        """
        # Remove the .csv extension added automatically to create our own names
        base = filename.replace('.csv', '')

        total = 0
        # --fullhtml is a SWITCH between the two networks, not an add-on: with
        # the flag, emit ONLY the 4 raw-HTML link-network files; without it,
        # emit the 4 MyWI (ExpressionLink) files. Run the export twice (with
        # and without --fullhtml) to obtain both networks for comparison.
        if getattr(self, 'fullhtml', False):
            # Opt-in raw-HTML link network (sprint fullhtml-linknetwork).
            # Closed network: every <a href> of expression.html restricted to
            # in-land targets; weight = anchor multiplicity; in_mywi flags edges
            # also present in ExpressionLink. Node files reuse the base writers
            # (same node set as the MyWI graph -> directly comparable).
            total += self._write_pagesnodes(f"{base}_pagesnodesfullhtml.csv")
            total += self._write_pageslinksfullhtml(f"{base}_pageslinksfullhtml.csv")
            total += self._write_domainnodes(f"{base}_domainnodesfullhtml.csv")
            total += self._write_domainlinksfullhtml(f"{base}_domainlinksfullhtml.csv")
        else:
            total += self._write_pagesnodes(f"{base}_pagesnodes.csv")
            total += self._write_pageslinks(f"{base}_pageslinks.csv")
            total += self._write_domainnodes(f"{base}_domainnodes.csv")
            total += self._write_domainlinks(f"{base}_domainlinks.csv")

        return total

    def _write_pagesnodes(self, filename) -> int:
        """Write expression nodes to CSV file with all fields and SEO rank data.

        Args:
            filename: Path to output CSV file.

        Returns:
            int: Number of expression records written.

        Notes:
            Includes all Expression fields plus dynamically parsed SEO rank JSON.
            Uses the same pattern as write_pagecsv for SEO rank handling.
        """
        col_map = {
            'id': 'e.id',
            'url': 'e.url',
            'domain_id': 'e.domain_id',
            'domain_name': 'd.name',
            'title': 'e.title',
            'description': 'e.description',
            'keywords': 'e.keywords',
            'lang': 'e.lang',
            'relevance': 'e.relevance',
            'depth': 'e.depth',
            'http_status': 'e.http_status',
            'created_at': 'e.created_at',
            'published_at': 'e.published_at',
            'fetched_at': 'e.fetched_at',
            'approved_at': 'e.approved_at',
            'readable_at': 'e.readable_at',
            'validllm': 'e.validllm',
            'validmodel': 'e.validmodel'
        }
        sql = """
            SELECT {}
            FROM expression AS e
            JOIN domain AS d ON d.id = e.domain_id
            WHERE e.land_id = ? AND e.relevance >= ?
            ORDER BY e.relevance DESC, e.id
        """
        # Use _fetch_page_rows_with_seorank to parse the seorank JSON field
        records, seorank_keys = self._fetch_page_rows_with_seorank(col_map, sql)
        base_keys = list(col_map.keys())
        header = base_keys + seorank_keys

        count = 0
        with open(filename, 'w', newline='\n', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            if records:
                writer.writerow(header)
                for base_data, seorank_payload in records:
                    row = [self._normalize_value(base_data.get(key)) for key in base_keys]
                    row.extend(self._normalize_value(seorank_payload.get(key)) for key in seorank_keys)
                    writer.writerow(row)
                    count += 1
        print(f"  - {filename.rsplit('_', 1)[-1]}: {count} expressions")
        return count

    def _write_pageslinks(self, filename) -> int:
        """Write expression links to CSV file (all links including intra-domain).

        Args:
            filename: Path to output CSV file.

        Returns:
            int: Number of link records written.
        """
        col_map = {
            'source_id': 'link.source_id',
            'source_url': 'e1.url',
            'source_domain_id': 'e1.domain_id',
            'target_id': 'link.target_id',
            'target_url': 'e2.url',
            'target_domain_id': 'e2.domain_id',
            # sprint link-context (migration 012) — dom_html exclu (trop lourd)
            'context': 'link.context',
            'dom': 'link.dom'
        }
        sql = """
            WITH idx(x) AS (
                SELECT id FROM expression
                WHERE land_id = ? AND relevance >= ?
            )
            SELECT {}
            FROM expressionlink AS link
            JOIN expression AS e1 ON e1.id = link.source_id
            JOIN expression AS e2 ON e2.id = link.target_id
            WHERE link.source_id IN idx AND link.target_id IN idx
            ORDER BY link.source_id, link.target_id
        """
        cursor = self.get_sql_cursor(sql, col_map)
        count = self.write_csv(filename, col_map.keys(), cursor)
        print(f"  - pageslinks.csv: {count} links")
        return count

    def _write_domainnodes(self, filename) -> int:
        """Write domain nodes with aggregated statistics to CSV file.

        Args:
            filename: Path to output CSV file.

        Returns:
            int: Number of domain records written.

        Notes:
            Aggregations: nbexpressions (count), average_relevance (mean),
            first_expression_date (min published_at), last_expression_date (max published_at).
        """
        col_map = {
            'id': 'd.id',
            'name': 'd.name',
            'title': 'd.title',
            'description': 'd.description',
            'http_status': 'd.http_status',
            'nbexpressions': 'COUNT(e.id)',
            'average_relevance': 'ROUND(AVG(e.relevance), 2)',
            'first_expression_date': 'MIN(e.published_at)',
            'last_expression_date': 'MAX(e.published_at)'
        }
        sql = """
            SELECT {}
            FROM domain AS d
            JOIN expression AS e ON e.domain_id = d.id
            WHERE e.land_id = ? AND e.relevance >= ?
            GROUP BY d.id
            ORDER BY nbexpressions DESC, d.id
        """
        cursor = self.get_sql_cursor(sql, col_map)
        count = self.write_csv(filename, col_map.keys(), cursor)
        print(f"  - {filename.rsplit('_', 1)[-1]}: {count} domains")
        return count

    def _write_domainlinks(self, filename) -> int:
        """Write aggregated inter-domain links to CSV file.

        Args:
            filename: Path to output CSV file.

        Returns:
            int: Number of domain link records written.

        Notes:
            Excludes intra-domain links (source_domain != target_domain).
            link_count represents number of page-level links between domains.
        """
        col_map = {
            'source_domain_id': 'e1.domain_id',
            'source_domain_name': 'd1.name',
            'target_domain_id': 'e2.domain_id',
            'target_domain_name': 'd2.name',
            'link_count': 'COUNT(*)'
        }
        sql = """
            WITH idx(x) AS (
                SELECT id FROM expression
                WHERE land_id = ? AND relevance >= ?
            )
            SELECT {}
            FROM expressionlink AS link
            JOIN expression AS e1 ON e1.id = link.source_id
            JOIN expression AS e2 ON e2.id = link.target_id
            JOIN domain AS d1 ON d1.id = e1.domain_id
            JOIN domain AS d2 ON d2.id = e2.domain_id
            WHERE link.source_id IN idx
              AND link.target_id IN idx
              AND e1.domain_id != e2.domain_id
            GROUP BY e1.domain_id, e2.domain_id
            ORDER BY link_count DESC
        """
        cursor = self.get_sql_cursor(sql, col_map)
        count = self.write_csv(filename, col_map.keys(), cursor)
        print(f"  - domainlinks.csv: {count} domain links")
        return count

    # ------------------------------------------------------------------ #
    # Raw-HTML link network (sprint fullhtml-linknetwork)                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _host_path_key(url):
        """Scheme-and-www-insensitive key: host(no www) + path + query.

        Absorbs http<->https / www<->bare / redirect divergences when
        force_https/strip_www are OFF. Returns None on failure / no host.
        """
        try:
            p = urlparse(url)
            host = (p.netloc or '').lower()
            if host.startswith('www.'):
                host = host[4:]
            if not host:
                return None
            key = host + (p.path or '').rstrip('/')
            if p.query:
                key += '?' + p.query
            return key
        except Exception:
            return None

    @staticmethod
    def _index_url_key(index, key, eid):
        """Insert key->eid; mark None (ambiguous) on conflicting ids."""
        if not key:
            return
        if key in index:
            if index[key] != eid:
                index[key] = None  # ambiguous -> unusable for lookup
        else:
            index[key] = eid

    def _fullhtml_lookup(self, idx, href):
        """Resolve a raw href to an in-land expression id (closed network).

        Tries three keys in priority order: exact normalize_url, relaxed
        (lower + no trailing slash), host+path. None on miss/ambiguous.
        """
        exact, relaxed, host_path = idx
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
        hp = self._host_path_key(norm)
        if hp is not None:
            eid = host_path.get(hp)
            if eid is not None:
                return eid
        return None

    def _write_pageslinksfullhtml(self, filename) -> int:
        """Union of the editorial (ExpressionLink) and raw-HTML link graphs.

        Closed network: both endpoints are in-land expressions qualifying by
        minrel (same node set as _pageslinks). One row per distinct edge:

        - edges from ExpressionLink (the Trafilatura-readable / body graph) ->
          weightbody=1, weighthtml=0 (in_mwi=1);
        - edges found ONLY in the raw <a href> of expression.html, not in
          ExpressionLink -> weightbody=0, weighthtml=<raw anchor multiplicity>
          (in_mwi=0). The two sets are disjoint.

        Every emitted edge also carries citation (1/0): 1 iff the link appears
        in the SOURCE expression's readable markdown. Independent from
        weightbody — an ExpressionLink row can come from the crawl BS4
        fallback (raw HTML anchors) while the readable lacks the link.
        readable NULL/empty -> citation=0. The citation pass never creates
        edges: a link present only in the readable emits no row.

        Self-loops are dropped (in-page '#' anchors / self-references, not real
        links). Columns use Gephi naming (Source/Target/Weight, Weight empty)
        plus weightbody/weighthtml/citation and the source/target url+domain.

        ALL lookups are preloaded before the streaming cursor opens: MWI uses
        one shared DB connection, so no second statement / lazy FK access may
        run during iteration. Side effect: stashes the inter-domain accumulator
        on self for _write_domainlinksfullhtml (avoids a second HTML pass).
        """
        land_id = self.land.get_id()
        minrel = self.relevance

        # --- preload lookups (drained BEFORE the streaming cursor opens) ---
        exact, relaxed, host_path = {}, {}, {}
        url_of, domain_of = {}, {}
        cur = model.DB.execute_sql(
            "SELECT id, url, domain_id FROM expression "
            "WHERE land_id = ? AND relevance >= ?", (land_id, minrel))
        for eid, url, domain_id in cur.fetchall():
            url_of[eid] = url
            domain_of[eid] = domain_id
            try:
                norm = normalize_url(url) if url else url
            except Exception:
                norm = url
            if not norm:
                continue
            self._index_url_key(exact, norm, eid)
            self._index_url_key(relaxed, norm.lower().rstrip('/'), eid)
            self._index_url_key(host_path, self._host_path_key(norm), eid)
        idx = (exact, relaxed, host_path)

        domain_name = {}
        for did, name in model.DB.execute_sql(
                "SELECT id, name FROM domain").fetchall():
            domain_name[did] = name

        mywi_page_edges = set()
        cur = model.DB.execute_sql(
            "WITH idx(x) AS (SELECT id FROM expression "
            "WHERE land_id = ? AND relevance >= ?) "
            "SELECT source_id, target_id FROM expressionlink "
            "WHERE source_id IN idx AND target_id IN idx", (land_id, minrel))
        for s, t in cur.fetchall():
            mywi_page_edges.add((s, t))

        # 0) citation lookup: (sid, tid) edges whose link appears in the
        #    source's readable markdown, resolved through the SAME 3-key
        #    ladder as the raw-HTML pass. Drained fully before the next
        #    statement (single shared DB connection); only int pairs are
        #    kept, readable texts are transient row-by-row.
        readable_edges = set()
        cur = model.DB.execute_sql(
            "SELECT id, url, readable FROM expression "
            "WHERE land_id = ? AND relevance >= ? "
            "AND readable IS NOT NULL AND readable != ''", (land_id, minrel))
        for sid, surl, readable in cur:
            for href in extract_markdown_links(readable, surl):
                tid = self._fullhtml_lookup(idx, href)
                if tid is not None and tid != sid:
                    readable_edges.add((sid, tid))

        # --- emission: union of the editorial graph (ExpressionLink = body)
        #     and the raw-only edges found ONLY in the full HTML.
        # weightbody = 1 for an edge present in ExpressionLink (in_mwi=1);
        # weighthtml = raw <a> multiplicity for an edge present ONLY in the
        # raw HTML (in_mwi=0). The two sets are disjoint (the streaming pass
        # skips edges already in mywi_page_edges). Self-loops are dropped:
        # they come from in-page '#' anchors / self-references, not real
        # hyperlinks (audit: ~96% of the raw weight was self-loop noise).
        # Gephi-standard edge columns Source/Target/Weight; Weight is left
        # empty so Gephi defaults it to 1.0 — the analytic signal lives in
        # weightbody/weighthtml.
        header = ['Source', 'Target', 'Weight', 'weightbody', 'weighthtml',
                  'citation', 'source_url', 'source_domain_id',
                  'target_url', 'target_domain_id']
        domain_acc = {}   # (sd, td) -> [in_mwi (Σweightbody), out_mwi (Σweighthtml)]
        body_edges = rawonly_edges = citation_edges = count = 0
        pages_total = pages_with_html = 0

        with open(filename, 'w', newline='\n', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(header)

            # 1) editorial edges (ExpressionLink, both endpoints qualified by
            #    minrel). No DB cursor here — reads only preloaded sets.
            for sid, tid in mywi_page_edges:
                if sid == tid:
                    continue
                sdom, td = domain_of.get(sid), domain_of.get(tid)
                citation = 1 if (sid, tid) in readable_edges else 0
                citation_edges += citation
                writer.writerow([sid, tid, '', 1, 0, citation,
                                 url_of.get(sid), sdom, url_of.get(tid), td])
                count += 1
                body_edges += 1
                if td is not None and sdom != td:
                    domain_acc.setdefault((sdom, td), [0, 0])[0] += 1

            # 2) raw-only edges — single streaming pass over stored HTML (the
            #    only live DB statement; no DB access inside the loop).
            src_cursor = model.DB.execute_sql(
                "SELECT id, url, domain_id, html FROM expression "
                "WHERE land_id = ? AND relevance >= ?", (land_id, minrel))
            for sid, surl, sdom, shtml in src_cursor:
                pages_total += 1
                if not shtml:
                    continue
                pages_with_html += 1
                per_target = {}
                for href in extract_all_links(shtml, surl):
                    tid = self._fullhtml_lookup(idx, href)
                    if tid is not None:
                        per_target[tid] = per_target.get(tid, 0) + 1
                for tid, weighthtml in per_target.items():
                    if sid == tid or (sid, tid) in mywi_page_edges:
                        continue
                    td = domain_of.get(tid)
                    citation = 1 if (sid, tid) in readable_edges else 0
                    citation_edges += citation
                    writer.writerow([sid, tid, '', 0, weighthtml, citation,
                                     surl, sdom, url_of.get(tid), td])
                    count += 1
                    rawonly_edges += 1
                    if td is not None and sdom != td:
                        domain_acc.setdefault((sdom, td), [0, 0])[1] += weighthtml

        self._fullhtml_domain_acc = domain_acc
        self._fullhtml_domain_name = domain_name

        # --- coverage report ---
        self._fullhtml_stats = {
            'pages_total': pages_total, 'pages_with_html': pages_with_html,
            'body_edges': body_edges, 'rawonly_edges': rawonly_edges,
            'total_edges': body_edges + rawonly_edges,
            'citation_edges': citation_edges,
        }
        pct = (100.0 * pages_with_html / pages_total) if pages_total else 0.0
        print(f"  - pageslinksfullhtml.csv: {count} edges "
              f"({pages_with_html}/{pages_total} pages have stored HTML, {pct:.1f}%)")
        print(f"      MyWI/body (in_mwi=1): {body_edges} | "
              f"fullhtml-only (in_mwi=0): {rawonly_edges} | "
              f"total: {body_edges + rawonly_edges}")
        print(f"      citation (link in readable): {citation_edges} | "
              f"non-citation: {count - citation_edges}")
        if pages_with_html == 0:
            print("      WARNING: no stored HTML for this land — crawl with "
                  "--fullhtml=TRUE or run 'land consolidate' on a "
                  "fullhtml-crawled land. Raw link network is empty.")
        return count

    def _write_domainlinksfullhtml(self, filename) -> int:
        """Inter-domain link graph rolled up from the raw-HTML page edges.

        Drains the accumulator built by _write_pageslinksfullhtml (no second
        HTML pass). Inter-domain only. Columns (Gephi naming): Source, Target,
        Weight (left empty), in_mwi (Σ weightbody of the page edges between the
        domain pair) and out_mwi (Σ weighthtml).
        """
        domain_acc = getattr(self, '_fullhtml_domain_acc', {})
        domain_name = getattr(self, '_fullhtml_domain_name', {})
        header = ['Source', 'Target', 'Weight', 'in_mwi', 'out_mwi',
                  'source_domain_name', 'target_domain_name']
        rows = sorted(domain_acc.items(),
                      key=lambda kv: kv[1][0] + kv[1][1], reverse=True)
        count = 0
        with open(filename, 'w', newline='\n', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(header)
            for (sd, td), (in_mwi, out_mwi) in rows:
                writer.writerow([sd, td, '', in_mwi, out_mwi,
                                 domain_name.get(sd), domain_name.get(td)])
                count += 1
        print(f"  - domainlinksfullhtml.csv: {count} domain links")
        return count

    @staticmethod
    def write_csv(filename, keys, cursor):
        """Write database cursor results to CSV file.

        Args:
            filename: Path to output CSV file.
            keys: List of column header names.
            cursor: Database cursor with query results.

        Returns:
            int: Number of rows written (excluding header).

        Notes:
            All fields are quoted using csv.QUOTE_ALL.
            Encoding is UTF-8 with Unix-style line endings.
        """
        count = 0
        with open(filename, 'w', newline='\n', encoding="utf-8") as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            header = False
            for row in cursor:
                if not header:
                    writer.writerow(keys)
                    header = True
                writer.writerow(row)
                count += 1
        file.close()
        return count

    def _fetch_page_rows_with_seorank(self, column_map: dict, sql: str):
        """Fetch page rows with parsed SEO rank data.

        Args:
            column_map: Dictionary mapping output column names to SQL expressions.
            sql: SQL query template with {} placeholder for column definitions.

        Returns:
            tuple: A tuple containing:
                - list: Records as (base_data, seorank_payload) tuples.
                - list: Sorted list of unique SEO rank keys found across all rows.

        Notes:
            Automatically adds '_seorank' to column map for fetching raw JSON.
            SEO rank payload is parsed from JSON and separated from base data.
            Keys are collected from all rows to ensure consistent column ordering.
        """
        select_map = dict(column_map)
        select_map['_seorank'] = 'e.seorank'
        cursor = self.get_sql_cursor(sql, select_map)
        rows = cursor.fetchall()

        records = []
        seorank_keys = set()
        for row in rows:
            data = dict(zip(select_map.keys(), row))
            payload = self._parse_seorank_payload(data.pop('_seorank', None))
            if payload:
                seorank_keys.update(payload.keys())
            base_data = {key: data.get(key) for key in column_map.keys()}
            records.append((base_data, payload))

        return records, sorted(seorank_keys)

    @staticmethod
    def _parse_seorank_payload(payload) -> dict:
        """Safely decode raw SEO rank JSON payload into flat dictionary.

        Args:
            payload: Raw SEO rank data (may be None, memoryview, bytes, or string).

        Returns:
            dict: Parsed JSON data with string keys, or empty dict if parsing fails.

        Notes:
            Handles multiple input types: None, memoryview, bytes, and strings.
            Returns empty dict for None, empty, or malformed JSON.
            All keys are converted to strings for consistent ordering.
        """
        if payload is None:
            return {}
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        if isinstance(payload, bytes):
            payload = payload.decode('utf-8', errors='ignore')
        payload = str(payload).strip()
        if not payload:
            return {}
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return {}
        if not isinstance(data, dict):
            return {}
        # Ensure keys are strings for consistent header ordering
        return {str(key): value for key, value in data.items()}

    @staticmethod
    def _normalize_value(value):
        """Normalize export values for consistent CSV output.

        Args:
            value: Value to normalize (any type).

        Returns:
            Normalized value as string, number, or 'na' for missing/empty values.

        Notes:
            Converts memoryview and bytes to UTF-8 strings.
            Serializes lists and dicts to JSON (or 'na' if empty).
            Preserves int and float types unchanged.
            Maps None, empty strings, and 'unknown' to 'na'.
        """
        if value is None:
            return 'na'
        if isinstance(value, memoryview):
            value = value.tobytes().decode('utf-8', errors='ignore')
        if isinstance(value, bytes):
            value = value.decode('utf-8', errors='ignore')
        if isinstance(value, (list, dict)):
            if not value:
                return 'na'
            value = json.dumps(value, ensure_ascii=False)
        if isinstance(value, (int, float)):
            return value
        val_str = str(value).strip()
        if not val_str:
            return 'na'
        if val_str.lower() == 'unknown':
            return 'na'
        return val_str

    def write_pagegexf(self, filename) -> int:
        """Write page-level network graph to GEXF format.

        Args:
            filename: Path to output GEXF file.

        Returns:
            int: Number of expression nodes written.

        Notes:
            Creates directed graph with expressions as nodes and links as edges.
            Includes SEO rank metadata as dynamic attributes.
            Edges only connect expressions from different domains.
            Uses GEXF 1.2 format compatible with network analysis tools like Gephi.
        """
        count = 0
        base_attributes = [
            ('title', 'string'),
            ('description', 'string'),
            ('keywords', 'string'),
            ('domain_id', 'string'),
            ('relevance', 'integer'),
            ('depth', 'integer')]

        node_map = {
            'id': 'e.id',
            'url': 'e.url',
            'title': 'e.title',
            'description': 'e.description',
            'keywords': 'e.keywords',
            'relevance': 'e.relevance',
            'depth': 'e.depth',
            'domain_id': 'e.domain_id',
            'domain_name': 'd.name',
            'domain_title': 'd.title',
            'domain_description': 'd.description',
            'domain_keywords': 'd.keywords'
        }
        sql = """
            SELECT
                {}
            FROM expression AS e
            JOIN domain AS d ON d.id = e.domain_id
            WHERE land_id = ? AND relevance >= ?
        """
        records, seorank_keys = self._fetch_page_rows_with_seorank(node_map, sql)

        extra_attributes = [(key, 'string') for key in seorank_keys]
        gexf_attributes = base_attributes + extra_attributes
        gexf, nodes, edges = self.get_gexf(gexf_attributes)

        for base_data, seorank_payload in records:
            row = dict(base_data)
            row['url'] = self._normalize_value(row.get('url'))
            row['relevance'] = self._normalize_value(row.get('relevance'))
            for attr_name, _ in base_attributes:
                row[attr_name] = self._normalize_value(row.get(attr_name))
            for key in seorank_keys:
                row[key] = self._normalize_value(seorank_payload.get(key))

            self.gexf_node(row, nodes, gexf_attributes, ('url', 'relevance'))
            count += 1

        edge_map = {
            'source_id': 'link.source_id',
            'source_domain_id': 'e1.domain_id',
            'target_id': 'link.target_id',
            'target_domain_id': 'e2.domain_id'
        }
        sql = """
            WITH idx(x) AS (
                SELECT
                    id
                FROM expression
                WHERE land_id = ? AND relevance >= ?
            )
            SELECT
                {}
            FROM expressionlink AS link
            JOIN expression AS e1 ON e1.id = link.source_id
            JOIN expression AS e2 ON e2.id = link.target_id
            WHERE
                source_id IN idx
                AND target_id IN idx
                AND source_domain_id != target_domain_id
        """
        cursor = self.get_sql_cursor(sql, edge_map)

        for row in cursor:
            row = dict(zip(edge_map.keys(), row))
            self.gexf_edge([row['source_id'], row['target_id'], 1], edges)

        tree = etree.ElementTree(gexf)
        tree.write(filename, xml_declaration=True, pretty_print=True, encoding='utf-8')
        return count

    def write_pseudolinks(self, filename) -> int:
        """Write paragraph-level semantic links to CSV file.

        Args:
            filename: Path to output CSV file.

        Returns:
            int: Number of paragraph similarity records written.

        Notes:
            Exports semantic relationships between paragraphs based on NLI/embedding similarity.
            Columns: Source_ParagraphID, Target_ParagraphID, RelationScore (-1|0|1),
            ConfidenceScore, Source_Text, Target_Text, Source_ExpressionID, Target_ExpressionID.
            Only includes similarities from 'nli', 'cosine', or 'cosine_lsh' methods.
            Results ordered by descending score.
        """
        col_map = {
            'Source_ParagraphID': 'p1.id',
            'Target_ParagraphID': 'p2.id',
            'RelationScore': 's.score',
            'ConfidenceScore': 'COALESCE(s.score_raw, s.score)',
            'Source_Text': 'p1.text',
            'Target_Text': 'p2.text',
            'Source_ExpressionID': 'e1.id',
            'Target_ExpressionID': 'e2.id',
        }
        sql = """
            SELECT
                {}
            FROM paragraph_similarity AS s
            JOIN paragraph AS p1 ON p1.id = s.source_paragraph_id
            JOIN paragraph AS p2 ON p2.id = s.target_paragraph_id
            JOIN expression AS e1 ON e1.id = p1.expression_id
            JOIN expression AS e2 ON e2.id = p2.expression_id
            WHERE e1.land_id = ?
              AND e1.relevance >= ?
              AND e2.land_id = e1.land_id
              AND s.method IN ('nli', 'cosine', 'cosine_lsh')
            ORDER BY s.score DESC
        """
        cursor = self.get_sql_cursor(sql, col_map)
        return self.write_csv(filename, col_map.keys(), cursor)

    def write_pseudolinkspage(self, filename) -> int:
        """Write page-level aggregated semantic links to CSV file.

        Args:
            filename: Path to output CSV file.

        Returns:
            int: Number of expression-level aggregated similarity records written.

        Notes:
            Aggregates paragraph similarities into undirected edges between expressions.
            Columns: Source_ExpressionID, Target_ExpressionID, Source_DomainID, Target_DomainID,
            PairCount, EntailCount, NeutralCount, ContradictCount, AvgRelationScore, AvgConfidence.
            Uses canonical ordering (smaller ID first) to avoid duplicate edges.
            Results ordered by descending PairCount.
        """
        col_map = {
            'Source_ExpressionID': 'CASE WHEN e1.id <= e2.id THEN e1.id ELSE e2.id END',
            'Target_ExpressionID': 'CASE WHEN e1.id <= e2.id THEN e2.id ELSE e1.id END',
            'Source_DomainID': 'CASE WHEN e1.id <= e2.id THEN e1.domain_id ELSE e2.domain_id END',
            'Target_DomainID': 'CASE WHEN e1.id <= e2.id THEN e2.domain_id ELSE e1.domain_id END',
            'PairCount': 'COUNT(*)',
            'Weight': 'COUNT(*)',
            'EntailCount': 'SUM(CASE WHEN s.score = 1 THEN 1 ELSE 0 END)',
            'NeutralCount': 'SUM(CASE WHEN s.score = 0 THEN 1 ELSE 0 END)',
            'ContradictCount': 'SUM(CASE WHEN s.score = -1 THEN 1 ELSE 0 END)',
            'AvgRelationScore': 'ROUND(AVG(s.score), 6)',
            'AvgConfidence': 'ROUND(AVG(COALESCE(s.score_raw, s.score)), 6)'
        }
        sql = """
            SELECT
                {}
            FROM paragraph_similarity AS s
            JOIN paragraph AS p1 ON p1.id = s.source_paragraph_id
            JOIN paragraph AS p2 ON p2.id = s.target_paragraph_id
            JOIN expression AS e1 ON e1.id = p1.expression_id
            JOIN expression AS e2 ON e2.id = p2.expression_id
            WHERE e1.land_id = ?
              AND e1.relevance >= ?
              AND e2.land_id = e1.land_id
              AND s.method IN ('nli', 'cosine', 'cosine_lsh')
            GROUP BY
              CASE WHEN e1.id <= e2.id THEN e1.id ELSE e2.id END,
              CASE WHEN e1.id <= e2.id THEN e2.id ELSE e1.id END
            HAVING PairCount > 0
            ORDER BY PairCount DESC
        """
        cursor = self.get_sql_cursor(sql, col_map)
        return self.write_csv(filename, col_map.keys(), cursor)

    def write_pseudolinksdomain(self, filename) -> int:
        """Write domain-level aggregated semantic links to CSV file.

        Args:
            filename: Path to output CSV file.

        Returns:
            int: Number of domain-level aggregated similarity records written.

        Notes:
            Aggregates paragraph similarities into undirected edges between domains.
            Columns: Source_DomainID, Source_Domain, Target_DomainID, Target_Domain,
            PairCount, EntailCount, NeutralCount, ContradictCount, AvgRelationScore, AvgConfidence.
            Uses canonical ordering (smaller domain ID first) to avoid duplicate edges.
            Results ordered by descending PairCount.
        """
        col_map = {
            'Source_DomainID': 'CASE WHEN e1.domain_id <= e2.domain_id THEN e1.domain_id ELSE e2.domain_id END',
            'Source_Domain': 'CASE WHEN e1.domain_id <= e2.domain_id THEN d1.name ELSE d2.name END',
            'Target_DomainID': 'CASE WHEN e1.domain_id <= e2.domain_id THEN e2.domain_id ELSE e1.domain_id END',
            'Target_Domain': 'CASE WHEN e1.domain_id <= e2.domain_id THEN d2.name ELSE d1.name END',
            'PairCount': 'COUNT(*)',
            'Weight': 'COUNT(*)',
            'EntailCount': 'SUM(CASE WHEN s.score = 1 THEN 1 ELSE 0 END)',
            'NeutralCount': 'SUM(CASE WHEN s.score = 0 THEN 1 ELSE 0 END)',
            'ContradictCount': 'SUM(CASE WHEN s.score = -1 THEN 1 ELSE 0 END)',
            'AvgRelationScore': 'ROUND(AVG(s.score), 6)',
            'AvgConfidence': 'ROUND(AVG(COALESCE(s.score_raw, s.score)), 6)'
        }
        sql = """
            SELECT
                {}
            FROM paragraph_similarity AS s
            JOIN paragraph AS p1 ON p1.id = s.source_paragraph_id
            JOIN paragraph AS p2 ON p2.id = s.target_paragraph_id
            JOIN expression AS e1 ON e1.id = p1.expression_id
            JOIN expression AS e2 ON e2.id = p2.expression_id
            JOIN domain AS d1 ON d1.id = e1.domain_id
            JOIN domain AS d2 ON d2.id = e2.domain_id
            WHERE e1.land_id = ?
              AND e1.relevance >= ?
              AND e2.land_id = e1.land_id
              AND s.method IN ('nli', 'cosine', 'cosine_lsh')
            GROUP BY
              CASE WHEN e1.domain_id <= e2.domain_id THEN e1.domain_id ELSE e2.domain_id END,
              CASE WHEN e1.domain_id <= e2.domain_id THEN e2.domain_id ELSE e1.domain_id END
            HAVING PairCount > 0
            ORDER BY PairCount DESC
        """
        cursor = self.get_sql_cursor(sql, col_map)
        return self.write_csv(filename, col_map.keys(), cursor)

    def write_nodegexf(self, filename) -> int:
        """Write domain-level network graph to GEXF format.

        Args:
            filename: Path to output GEXF file.

        Returns:
            int: Number of domain nodes written.

        Notes:
            Creates directed graph with domains as nodes and aggregated links as edges.
            Edge weights represent the count of inter-domain links.
            Only includes edges between different domains.
            Uses GEXF 1.2 format compatible with network analysis tools like Gephi.
        """
        count = 0
        gexf_attributes = [
            ('title', 'string'),
            ('description', 'string'),
            ('keywords', 'string'),
            ('expressions', 'integer'),
            ('average_relevance', 'float')]

        gexf, nodes, edges = self.get_gexf(gexf_attributes)

        node_map = {
            'id': 'd.id',
            'name': 'd.name',
            'title': 'd.title',
            'description': 'd.description',
            'keywords': 'd.keywords',
            'expressions': 'COUNT(*)',
            'average_relevance': 'ROUND(AVG(e.relevance), 2)'
        }
        sql = """
            SELECT
                {}
            FROM domain AS d
            JOIN expression AS e ON e.domain_id = d.id
            WHERE land_id = ? AND relevance >= ?
            GROUP BY d.name
        """
        cursor = self.get_sql_cursor(sql, node_map)

        for row in cursor:
            self.gexf_node(
                dict(zip(node_map.keys(), row)),
                nodes,
                gexf_attributes,
                ('name', 'average_relevance'))
            count += 1

        edge_map = {
            'source_id': 'link.source_id',
            'source_domain_id': 'e1.domain_id',
            'target_id': 'link.target_id',
            'target_domain_id': 'e2.domain_id',
            'weight': 'COUNT(*)'
        }
        sql = """
            WITH idx(x) AS (
                SELECT
                    id
                FROM expression
                WHERE land_id = ? AND relevance >= ?
            )
            SELECT
                {}
            FROM expressionlink AS link
            JOIN expression AS e1 ON e1.id = link.source_id
            JOIN expression AS e2 ON e2.id = link.target_id
            WHERE
                source_id IN idx
                AND target_id IN idx
                AND source_domain_id != target_domain_id
            GROUP BY source_domain_id, target_domain_id
        """
        cursor = self.get_sql_cursor(sql, edge_map)

        for row in cursor:
            row = dict(zip(edge_map.keys(), row))
            self.gexf_edge([row['source_domain_id'], row['target_domain_id'], row['weight']], edges)

        tree = etree.ElementTree(gexf)
        tree.write(filename, xml_declaration=True, pretty_print=True, encoding='utf-8')
        return count

    def get_gexf(self, attributes: list) -> tuple:
        """Initialize GEXF XML structure with metadata and attribute definitions.

        Args:
            attributes: List of (name, type) tuples defining node attributes.

        Returns:
            tuple: Three-element tuple containing:
                - gexf: Root GEXF element.
                - nodes: Nodes container element.
                - edges: Edges container element.

        Notes:
            Creates GEXF 1.2 static directed graph structure.
            Includes meta element with creation date and creator.
            Attribute types: 'string', 'integer', 'float', etc.
        """
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        gexf = etree.Element(
            'gexf',
            nsmap=self.gexf_ns,
            attrib={'version': '1.2'})
        etree.SubElement(
            gexf,
            'meta',
            attrib={'lastmodifieddate': date, 'creator': 'MyWebIntelligence'})
        graph = etree.SubElement(
            gexf,
            'graph',
            attrib={'mode': 'static', 'defaultedgetype': 'directed'})
        attr = etree.SubElement(
            graph,
            'attributes',
            attrib={'class': 'node'})
        for i, attribute in enumerate(attributes):
            etree.SubElement(
                attr,
                'attribute',
                attrib={'id': str(i), 'title': attribute[0], 'type': attribute[1]})
        nodes = etree.SubElement(graph, 'nodes')
        edges = etree.SubElement(graph, 'edges')
        return gexf, nodes, edges

    def gexf_node(self, row: dict, nodes, attributes: list, keys: tuple):
        """Create and append GEXF node element from data row.

        Args:
            row: Dictionary containing node data.
            nodes: Parent nodes container element.
            attributes: List of (name, type) tuples for attribute definitions.
            keys: Tuple of (label_key, size_key) for node label and visual size.

        Notes:
            Node ID comes from row['id'].
            Label and size are determined by keys parameter.
            All attributes in the list are added as attvalue elements.
            Size uses viz namespace for visual rendering in graph tools.
        """
        label_key, size_key = keys
        node = etree.SubElement(
            nodes,
            'node',
            attrib={'id': str(row['id']), 'label': row[label_key]})
        etree.SubElement(
            node,
            '{%s}size' % self.gexf_ns['viz'],
            attrib={'value': str(row[size_key])})
        attvalues = etree.SubElement(node, 'attvalues')
        try:
            for i, attribute in enumerate(attributes):
                etree.SubElement(
                    attvalues,
                    'attvalue',
                    attrib={'for': str(i), 'value': str(row[attribute[0]])})
        except ValueError:
            print(row)

    def gexf_edge(self, values, edges):
        """Create and append GEXF edge element from values.

        Args:
            values: List/tuple with [source_id, target_id, weight].
            edges: Parent edges container element.

        Notes:
            Edge ID is constructed as "source_target" concatenation.
            Weight attribute represents edge strength or count.
            All edges are directed as per graph defaultedgetype.
        """
        etree.SubElement(
            edges,
            'edge',
            attrib={
                'id': "%s_%s" % (values[0], values[1]),
                'source': str(values[0]),
                'target': str(values[1]),
                'weight': str(values[2])})

    def export_tags(self, filename):
        """Export tag data in matrix or content format.

        Args:
            filename: Path to output CSV file.

        Returns:
            int: 1 if export successful, 0 if export type not recognized.

        Notes:
            Matrix type: Creates tag co-occurrence matrix with expressions as rows.
            Content type: Exports tagged content snippets with hierarchical tag paths.
            Tag paths are constructed using recursive CTE with '_' separator.
            Only includes tags associated with expressions meeting relevance threshold.
        """
        if self.type == 'matrix':
            sql = """
            WITH RECURSIVE tagPath AS (
                SELECT id,
                       name
                FROM tag
                WHERE parent_id IS NULL
                UNION ALL
                SELECT t.id,
                       p.name || '_' || t.name
                FROM tagPath AS p
                JOIN tag AS t ON p.id = t.parent_id
            )
            SELECT tc.expression_id,
                   tp.name AS path,
                   COUNT(*) AS content
            FROM tag AS t
            JOIN tagPath AS tp ON tp.id = t.id
            JOIN taggedcontent tc ON tc.tag_id = t.id
            JOIN expression e ON e.id = tc.expression_id
            WHERE t.land_id = ?
                AND e.relevance >= ?
            GROUP BY tc.expression_id, path
            ORDER BY tc.expression_id, t.parent_id, t.sorting
            """

            cursor = model.DB.execute_sql(sql, (self.land.get_id(), self.relevance))

            tags = []
            rows = []

            for row in cursor:
                if row[1] not in tags:
                    tags.append(row[1])
                rows.append(row)
            default_matrix = dict(zip(tags, [0] * len(tags)))

            expression_id = None
            matrix = {}

            for row in rows:
                if row[0] != expression_id:
                    expression_id = row[0]
                    matrix[expression_id] = default_matrix.copy()
                matrix[expression_id][row[1]] = row[2]

            with open(filename, 'w', newline='\n', encoding="utf-8") as file:
                writer = csv.writer(file, quoting=csv.QUOTE_ALL)
                writer.writerow(['expression_id'] + tags)
                for (expression_id, data) in matrix.items():
                    writer.writerow([expression_id] + list(data.values()))
                return 1
        elif self.type == 'content':
            sql = """
            WITH RECURSIVE tagPath AS (
                SELECT id,
                       name
                FROM tag
                WHERE parent_id IS NULL
                UNION ALL
                SELECT t.id,
                       p.name || '_' || t.name
                FROM tagPath AS p
                JOIN tag AS t ON p.id = t.parent_id
            )
            SELECT
                tp.name AS path,
                tc.text AS content,
                tc.expression_id
            FROM taggedcontent AS tc
            JOIN tag AS t ON t.id = tc.tag_id
            JOIN tagPath AS tp ON tp.id = t.id
            JOIN expression AS e ON e.id = tc.expression_id
            WHERE t.land_id = ?
                AND e.relevance >= ?
            ORDER BY t.parent_id, t.sorting
            """

            cursor = model.DB.execute_sql(sql, (self.land.get_id(), self.relevance))

            with open(filename, 'w', newline='\n', encoding="utf-8") as file:
                writer = csv.writer(file, quoting=csv.QUOTE_ALL)
                writer.writerow(['path', 'content', 'expression_id'])
                for row in cursor:
                    writer.writerow(row)
                return 1
        return 0

    def write_corpus(self, filename) -> int:
        """Write text corpus as multiple ZIP archives with batching.

        Args:
            filename: Base path for output ZIP files (without .zip extension).

        Returns:
            int: Total number of expressions exported across all batches.

        Notes:
            Creates multiple ZIP files with max 1000 expressions each.
            Files named as {base}_00001.zip, {base}_00002.zip, etc.
            Each text file contains Dublin Core metadata header and readable content.
            Filenames follow pattern: {id}-{slugified-title}.txt.
            Uses UTF-8 encoding for all text files.
        """
        col_map = {
            'id': 'e.id',
            'url': 'e.url',
            'title': 'e.title',
            'description': 'e.description',
            'readable': 'e.readable',
            'domain': 'd.name',
        }
        sql = """
            SELECT
                {}
            FROM expression AS e
            JOIN domain AS d ON d.id = e.domain_id
            LEFT JOIN taggedcontent tc ON tc.expression_id = e.id
            LEFT JOIN tag t ON t.id = tc.tag_id
            WHERE e.land_id = ? AND relevance >= ?
            GROUP BY e.id
        """

        cursor = self.get_sql_cursor(sql, col_map)
        count = 0
        batch_size = 1000
        batch_count = 0
        current_batch = 0
        
        # Enlever l'extension .zip du nom de fichier de base
        base_filename = filename.replace('.zip', '')
        
        arch = None
        
        for row in cursor:
            # Créer un nouveau ZIP toutes les 1000 expressions
            if current_batch == 0:
                batch_count += 1
                if arch:
                    arch.close()
                
                # Créer le nom du fichier avec numérotation : nom_00001.zip, nom_00002.zip, etc.
                batch_filename = f"{base_filename}_{batch_count:05d}.zip"
                arch = ZipFile(batch_filename, 'w')
                print(f"Création du fichier ZIP : {batch_filename}")
            
            count += 1
            current_batch += 1
            
            row = dict(zip(col_map.keys(), row))
            txt_filename = '{}-{}.txt'.format(row.get('id'), self.slugify(row.get('title', '')))
            data = self.to_metadata(row) + row.get('readable', '')
            arch.writestr(txt_filename, data)
            
            # Reset le compteur de batch si on atteint 1000
            if current_batch >= batch_size:
                current_batch = 0
        
        # Fermer le dernier ZIP
        if arch:
            arch.close()
        
        print(f"Export terminé : {count} expressions réparties dans {batch_count} fichiers ZIP")
        return count

    def slugify(self, string):
        """Convert string to URL-safe slug.

        Args:
            string: Input string to slugify.

        Returns:
            str: Slugified string with only lowercase alphanumeric and hyphens.

        Notes:
            Normalizes Unicode characters using NFKD normalization.
            Removes non-ASCII characters and converts to lowercase.
            Replaces non-alphanumeric sequences with single hyphens.
            Strips leading and trailing hyphens.
        """
        slug = unicodedata.normalize('NFKD', string)
        slug = str(slug.encode('ascii', 'ignore').lower())
        slug = re.sub(r'[^a-z0-9]+', '-', slug).strip('-')

        return re.sub(r'[-]+', '-', slug)

    def to_metadata(self, row) -> str:
        """Generate Dublin Core metadata header for corpus text files.

        Args:
            row: Dictionary containing expression data (title, description, id, domain, url).

        Returns:
            str: Formatted Dublin Core metadata block with YAML-style delimiters.

        Notes:
            Uses Dublin Core metadata standard for digital resources.
            Populated fields: Title, Description, Identifier, Publisher, Source.
            Empty fields included for completeness: Creator, Contributor, Coverage,
            Date, Subject, Type, Format, Language, Relation, Rights.
            Wrapped in YAML-style triple-dash delimiters.
        """
        metadata = """\
            ---
            Title: "{title}"
            Creator: ""
            Contributor: ""
            Coverage: ""
            Date: ""
            Description: "{description}"
            Subject: ""
            Type: ""
            Format: ""
            Identifier: "{id}"
            Language: ""
            Publisher: "{domain}"
            Relation: ""
            Rights: ""
            Source: "{url}"
            ---
        """.format(title=row.get('title'), description=row.get('description'),
                   id=row.get('id'), domain=row.get('domain'), url=row.get('url'))

        return dedent(metadata)

    def write_htmldump(self, filename) -> int:
        """Export raw HTML archives as a zip with one .html per expression.

        Sprint-html E. Each expression with a non-NULL `html` column and
        `relevance >= self.relevance` is written as `{id}.html` inside the
        zip. A `manifest.csv` next to it lists id, url, http_status,
        fetch_method, fetched_at and byte size for downstream tooling.

        Args:
            filename: Output zip path (the .zip extension is already in
                place — see `Export.write()` dispatch).

        Returns:
            int: Number of .html files written. 0 if no expression in the
                Land has stored HTML.

        Notes:
            Skips expressions where ``html IS NULL``. Useful to share a
            reproducible corpus archive of crawled HTML for replication
            (R, Python, Gephi, manual audit). The relevance filter
            mirrors other exports — pass ``--minrel=0`` to include all.
        """
        import zipfile
        import csv
        import io

        rows = (model.Expression
                .select(model.Expression, model.Domain.name.alias('domain_name'))
                .join(model.Domain)
                .where((model.Expression.land == self.land)
                       & (model.Expression.html.is_null(False))
                       & (model.Expression.relevance >= self.relevance)))

        n = 0
        with zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED) as zf:
            manifest = io.StringIO()
            writer = csv.writer(manifest)
            writer.writerow([
                'id', 'url', 'http_status', 'fetch_method',
                'fetched_at', 'relevance', 'size_bytes',
            ])
            for expr in rows:
                html = expr.html or ''
                zf.writestr(f"{expr.id}.html", html)
                writer.writerow([
                    expr.id, expr.url, expr.http_status,
                    expr.fetch_method,
                    expr.fetched_at.isoformat() if expr.fetched_at else '',
                    expr.relevance,
                    len(html.encode('utf-8')),
                ])
                n += 1
            zf.writestr('manifest.csv', manifest.getvalue())
        return n
