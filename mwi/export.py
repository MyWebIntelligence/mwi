"""
Export module
Contains custom SQL to be exported in specified file formats
"""

import csv
import datetime
import re
from textwrap import dedent
import unicodedata
from lxml import etree
from zipfile import ZipFile
from . import model


class Export:
    """
    Export class
    """
    gexf_ns = {None: 'http://www.gexf.net/1.2draft', 'viz': 'http://www.gexf.net/1.1draft/viz'}
    type = None
    land = None
    relevance = 1

    def __init__(self, export_type: str, land: model.Land, minimum_relevance: int):
        """
        :param export_type:
        :param land:
        :param minimum_relevance:
        :return:
        """
        self.type = export_type
        self.land = land
        self.relevance = minimum_relevance

    def write(self, export_type: str, filename):
        """
        Proxy method to all format writers
        :param export_type:
        :param filename:
        :return:
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
        """
        Build SQL query to execution, returning an iterable cursor
        :param sql:
        :param column_map:
        :return:
        """
        cols = ",\n".join(["{1} AS {0}".format(*i) for i in column_map.items()])
        return model.DB.execute_sql(sql.format(cols), (self.land.get_id(), self.relevance))

    def write_pagecsv(self, filename) -> int:
        """
        Write CSV file
        :param filename:
        :return:
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
        cursor = self.get_sql_cursor(sql, col_map)
        return self.write_csv(filename, col_map.keys(), cursor)

    def write_fullpagecsv(self, filename) -> int:
        """
        Write CSV file
        :param filename:
        :return:
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
        """
        Write CSV file
        :param filename:
        :return:
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
        """
        Write CSV file
        :param filename:
        :return:
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
        """
        CSV writer
        :param filename:
        :param keys:
        :param cursor:
        :return:
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

    def write_pagegexf(self, filename) -> int:
        """
        Page GEXF writer
        :param filename:
        :return:
        """
        count = 0
        gexf_attributes = [
            ('title', 'string'),
            ('description', 'string'),
            ('keywords', 'string'),
            ('domain_id', 'string'),
            ('relevance', 'integer'),
            ('depth', 'integer')]

        gexf, nodes, edges = self.get_gexf(gexf_attributes)

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
        cursor = self.get_sql_cursor(sql, node_map)

        for row in cursor:
            self.gexf_node(
                dict(zip(node_map.keys(), row)),
                nodes,
                gexf_attributes,
                ('url', 'relevance'))
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
        """Write CSV of paragraph-level semantic links (pseudolinks).
        Columns (as per embedding.md advanced spec):
        - Source_ParagraphID, Target_ParagraphID, RelationScore (-1|0|1), ConfidenceScore,
          Source_Text, Target_Text, Source_ExpressionID, Target_ExpressionID
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
        """Write CSV of page-level (expression) aggregated pseudolinks.
        Aggregates paragraph-level similarities into undirected edges between expressions.
        Columns:
        - Source_ExpressionID, Target_ExpressionID, Source_DomainID, Target_DomainID,
          PairCount, EntailCount, NeutralCount, ContradictCount, AvgRelationScore, AvgConfidence
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
        """Write CSV of domain-level aggregated pseudolinks.
        Aggregates paragraph-level similarities into undirected edges between domains.
        Columns:
        - Source_DomainID, Source_Domain, Target_DomainID, Target_Domain,
          PairCount, EntailCount, NeutralCount, ContradictCount, AvgRelationScore, AvgConfidence
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
        """
        Node GEXF writer
        :param filename:
        :return:
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
        """
        Initialize GEXF elements
        :param attributes:
        :return:
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
        """
        Get GEXF node from data
        :param row:
        :param nodes:
        :param attributes:
        :param keys:
        :return:
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
        """
        Get GEXF edge from data
        :param values:
        :param edges:
        :return:
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
        """
        Write corpus files in multiple ZIP archives with max 1000 expressions each
        :param filename:
        :return:
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
        slug = unicodedata.normalize('NFKD', string)
        slug = str(slug.encode('ascii', 'ignore').lower())
        slug = re.sub(r'[^a-z0-9]+', '-', slug).strip('-')

        return re.sub(r'[-]+', '-', slug)

    def to_metadata(self, row) -> str:
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
