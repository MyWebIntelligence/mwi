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
from zipfile import ZipFile
from . import model


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

    def __init__(self, export_type: str, land: model.Land, minimum_relevance: int):
        """Initialize an Export instance with specified parameters.

        Args:
            export_type: The format type for export (e.g., 'pagecsv', 'gexf', 'corpus').
            land: The Land model instance representing the research project.
            minimum_relevance: Minimum relevance score threshold for including expressions.

        Notes:
            The export_type determines which write method will be called.
            Only expressions with relevance >= minimum_relevance are included.
        """
        self.type = export_type
        self.land = land
        self.relevance = minimum_relevance

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
