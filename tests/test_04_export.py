"""
Tests for land export functionality: CSV, GEXF, corpus, JSON force-graph.
"""
import os
import glob
import csv
import json
import pytest


def _export_graph(populated, export_type, minrel):
    """Run a JSON force-graph export and return the parsed {nodes, links} dict.

    Reads the produced file *immediately* after the export so that two exports
    issued within the same wall-clock second (identical timestamp -> identical
    filename, the file being overwritten in place) are each captured correctly.
    """
    controller = populated["controller"]
    core = populated["core"]
    name = populated["name"]
    data_dir = str(populated["data_dir"])
    ret = controller.LandController.export(
        core.Namespace(name=name, type=export_type, minrel=minrel)
    )
    assert ret == 1
    files = sorted(
        glob.glob(os.path.join(data_dir, f"export_land_*{export_type}*.json"))
    )
    assert files, f"no {export_type} export file produced"
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f)


class TestLandExportCSV:
    """Tests for CSV export formats."""

    def test_export_pagecsv(self, populated_land):
        """land export --type=pagecsv génère un CSV valide."""
        controller = populated_land["controller"]
        core = populated_land["core"]
        name = populated_land["name"]
        data_dir = str(populated_land["data_dir"])

        # Count existing export files before
        before_count = len(glob.glob(os.path.join(data_dir, "export_land_*pagecsv*")))

        ret = controller.LandController.export(
            core.Namespace(name=name, type="pagecsv", minrel=0)
        )

        assert ret == 1

        # Find newly created file
        after_files = glob.glob(os.path.join(data_dir, "export_land_*pagecsv*"))
        assert len(after_files) > before_count, "New CSV file should be created"

        # Validate CSV content
        csv_file = after_files[-1]  # Most recent
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 20, "Should have 20 expressions"
        assert "id" in rows[0]
        assert "url" in rows[0]

    def test_export_fullpagecsv(self, populated_land):
        """--type=fullpagecsv inclut le contenu complet."""
        controller = populated_land["controller"]
        core = populated_land["core"]
        name = populated_land["name"]

        ret = controller.LandController.export(
            core.Namespace(name=name, type="fullpagecsv", minrel=0)
        )

        assert ret == 1

    def test_export_nodecsv(self, populated_land):
        """--type=nodecsv génère les nœuds (domains)."""
        controller = populated_land["controller"]
        core = populated_land["core"]
        name = populated_land["name"]

        ret = controller.LandController.export(
            core.Namespace(name=name, type="nodecsv", minrel=0)
        )

        assert ret == 1

    def test_export_mediacsv(self, populated_land):
        """--type=mediacsv exporte les médias."""
        controller = populated_land["controller"]
        core = populated_land["core"]
        name = populated_land["name"]

        ret = controller.LandController.export(
            core.Namespace(name=name, type="mediacsv", minrel=0)
        )

        assert ret == 1

    def test_export_corpus(self, populated_land):
        """--type=corpus génère le corpus texte."""
        controller = populated_land["controller"]
        core = populated_land["core"]
        name = populated_land["name"]

        ret = controller.LandController.export(
            core.Namespace(name=name, type="corpus", minrel=0)
        )

        assert ret == 1

    def test_export_nodelinkcsv(self, populated_land):
        """--type=nodelinkcsv crée des fichiers CSV."""
        controller = populated_land["controller"]
        core = populated_land["core"]
        name = populated_land["name"]

        ret = controller.LandController.export(
            core.Namespace(name=name, type="nodelinkcsv", minrel=0)
        )

        assert ret == 1

    def test_pageslinks_excludes_self_loops(self, populated_land):
        """Une ligne ExpressionLink source==target ne sort jamais dans pageslinks."""
        controller = populated_land["controller"]
        core = populated_land["core"]
        model = populated_land["model"]
        name = populated_land["name"]
        data_dir = str(populated_land["data_dir"])
        expr = populated_land["expressions"][5]
        model.ExpressionLink.create(source=expr, target=expr)

        ret = controller.LandController.export(
            core.Namespace(name=name, type="nodelinkcsv", minrel=0)
        )

        assert ret == 1
        csv_file = sorted(glob.glob(os.path.join(data_dir, "*_pageslinks.csv")))[-1]
        with open(csv_file, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows, "Normal links should still be exported"
        assert all(r["source_id"] != r["target_id"] for r in rows)

    def test_export_minrel_filter(self, populated_land):
        """--minrel=X filtre les expressions."""
        controller = populated_land["controller"]
        core = populated_land["core"]
        name = populated_land["name"]
        data_dir = str(populated_land["data_dir"])

        # Export avec minrel=10
        ret = controller.LandController.export(
            core.Namespace(name=name, type="pagecsv", minrel=10)
        )

        assert ret == 1

        # Find the CSV file
        csv_files = glob.glob(os.path.join(data_dir, "export_land_*pagecsv*"))
        csv_file = csv_files[-1]  # Most recent

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Should filter to expressions with relevance >= 10 (0-19 in fixture)
        assert len(rows) == 10, f"Should have 10 filtered expressions, got {len(rows)}"


class TestLandExportGEXF:
    """Tests for GEXF export formats."""

    def test_export_pagegexf(self, populated_land):
        """--type=pagegexf génère un GEXF valide."""
        controller = populated_land["controller"]
        core = populated_land["core"]
        name = populated_land["name"]

        ret = controller.LandController.export(
            core.Namespace(name=name, type="pagegexf", minrel=0)
        )

        assert ret == 1

    def test_export_nodegexf(self, populated_land):
        """--type=nodegexf génère un GEXF de nœuds (domains)."""
        controller = populated_land["controller"]
        core = populated_land["core"]
        name = populated_land["name"]

        ret = controller.LandController.export(
            core.Namespace(name=name, type="nodegexf", minrel=0)
        )

        assert ret == 1


class TestTagExport:
    """Tests for tag export."""

    @pytest.fixture
    def land_with_tags(self, fresh_db):
        """Land avec tags et TaggedContent."""
        controller = fresh_db["controller"]
        model = fresh_db["model"]
        core = fresh_db["core"]
        from datetime import datetime

        # Créer land
        name = "test_tags_land"
        controller.LandController.create(
            core.Namespace(name=name, desc="Test land with tags", lang=["fr"])
        )
        land = model.Land.get(model.Land.name == name)

        # Ajouter terms
        controller.LandController.addterm(
            core.Namespace(land=name, terms="test, keyword")
        )

        # Créer expressions
        domain = model.Domain.create(name="example.com")
        expressions = []
        for i in range(5):
            expr = model.Expression.create(
                land=land,
                domain=domain,
                url=f"https://example.com/page{i}",
                title=f"Page {i}",
                readable=f"Content with test keyword {i}. " * 20,
                relevance=i + 1,
                fetched_at=datetime.now()
            )
            expressions.append(expr)

        # Créer tags avec le champ sorting requis
        tag1 = model.Tag.create(land=land, name="Topic1", parent=None, sorting=1, color="#FF0000")
        tag2 = model.Tag.create(land=land, name="Subtopic1", parent=tag1, sorting=2, color="#00FF00")
        tag3 = model.Tag.create(land=land, name="Topic2", parent=None, sorting=3, color="#0000FF")

        # Créer TaggedContent
        model.TaggedContent.create(
            tag=tag1,
            expression=expressions[0],
            text="Tagged snippet 1",
            from_char=0,
            to_char=15
        )
        model.TaggedContent.create(
            tag=tag2,
            expression=expressions[1],
            text="Tagged snippet 2",
            from_char=10,
            to_char=25
        )
        model.TaggedContent.create(
            tag=tag1,
            expression=expressions[2],
            text="Tagged snippet 3",
            from_char=5,
            to_char=40
        )

        return {
            "land": land,
            "tags": [tag1, tag2, tag3],
            "expressions": expressions,
            "name": name,
            "controller": controller,
            "core": core,
            "data_dir": fresh_db["data_dir"]
        }

    def test_tag_export_content(self, land_with_tags):
        """tag export --type=content exporte le contenu taggé."""
        controller = land_with_tags["controller"]
        core = land_with_tags["core"]
        name = land_with_tags["name"]

        ret = controller.TagController.export(
            core.Namespace(name=name, type="content", minrel=0)
        )

        assert ret == 1

    def test_tag_export_matrix(self, land_with_tags):
        """tag export --type=matrix génère la matrice de co-occurrence."""
        controller = land_with_tags["controller"]
        core = land_with_tags["core"]
        name = land_with_tags["name"]

        ret = controller.TagController.export(
            core.Namespace(name=name, type="matrix", minrel=0)
        )

        assert ret == 1

    def test_tag_export_minrel(self, land_with_tags):
        """--minrel filtre le contenu exporté."""
        controller = land_with_tags["controller"]
        core = land_with_tags["core"]
        name = land_with_tags["name"]

        # Export avec minrel=3
        ret = controller.TagController.export(
            core.Namespace(name=name, type="content", minrel=3)
        )

        assert ret == 1


class TestLandExportNodesJSON:
    """Tests for the domain force-graph export (--type=nodesjson)."""

    _NODE_FIELDS = (
        "id", "name", "title", "description", "keywords", "nbexpressions",
        "average_relevance", "first_expression_date", "last_expression_date",
    )

    def test_structure_and_fields(self, populated_land):
        """{nodes, links} listes ; 9 champs + corpus ; pas de http_status."""
        graph = _export_graph(populated_land, "nodesjson", 0)

        assert isinstance(graph["nodes"], list)
        assert isinstance(graph["links"], list)
        assert len(graph["nodes"]) == 2  # 2 domains, all with relevant expr
        for node in graph["nodes"]:
            for field in self._NODE_FIELDS:
                assert field in node, f"missing analytical field {field}"
            assert isinstance(node["corpus"], list)
            for entry in node["corpus"]:
                assert set(entry) == {
                    "title", "urlarticle", "description", "published_at"
                }
            assert "http_status" not in node  # technical field dropped (decision 5)

    def test_corpus_sorted_and_counts_match(self, populated_land):
        """corpus trié par urlarticle, longueur == nbexpressions, par domain_id."""
        graph = _export_graph(populated_land, "nodesjson", 0)

        assert graph["nodes"], "fixture must yield nodes (else assertions vacuous)"
        for node in graph["nodes"]:
            urls = [entry["urlarticle"] for entry in node["corpus"]]
            assert urls == sorted(urls)
            assert len(node["corpus"]) == node["nbexpressions"]

    def test_corpus_entry_shape_and_values(self, populated_land):
        """Chaque entrée corpus = objet {title, urlarticle, description, published_at}."""
        graph = _export_graph(populated_land, "nodesjson", 0)

        entries = [e for node in graph["nodes"] for e in node["corpus"]]
        assert entries, "fixture must yield corpus entries"
        for entry in entries:
            assert entry["urlarticle"].startswith("https://example.com/page")
            assert entry["title"].startswith("Page ")
            assert entry["description"].startswith("Description with research keyword")
            assert entry["published_at"] is None  # fixture ne fixe pas published_at

    def test_node_values(self, populated_land):
        """Valeurs concrètes : noms de domaines et average_relevance par nœud."""
        graph = _export_graph(populated_land, "nodesjson", 0)

        by_name = {n["name"]: n for n in graph["nodes"]}
        assert set(by_name) == {"example.com", "test.org"}
        # example.com = indices pairs (relevance 0,2,..,18) -> moyenne 9.0
        # test.org    = indices impairs (relevance 1,3,..,19) -> moyenne 10.0
        assert by_name["example.com"]["average_relevance"] == 9.0
        assert by_name["test.org"]["average_relevance"] == 10.0

    def test_corpus_shrinks_with_minrel(self, populated_land):
        """Un minrel plus haut réduit la somme des longueurs de corpus."""
        low = _export_graph(populated_land, "nodesjson", 0)
        high = _export_graph(populated_land, "nodesjson", 10)

        total_low = sum(len(n["corpus"]) for n in low["nodes"])
        total_high = sum(len(n["corpus"]) for n in high["nodes"])
        assert total_low == 20
        assert total_high == 10
        assert total_high < total_low

    def test_referential_integrity(self, populated_land):
        """source/target de chaque lien est un id de nœud."""
        graph = _export_graph(populated_land, "nodesjson", 0)

        assert graph["nodes"] and graph["links"], "fixture must yield nodes and links"
        node_ids = {n["id"] for n in graph["nodes"]}
        for link in graph["links"]:
            assert link["source"] in node_ids
            assert link["target"] in node_ids

    def test_minrel_filters_links(self, populated_land):
        """minrel=0 -> 2 liens dirigés value=5 ; minrel=10 -> 0 lien."""
        low = _export_graph(populated_land, "nodesjson", 0)
        high = _export_graph(populated_land, "nodesjson", 10)

        assert len(low["links"]) == 2
        assert all(link["value"] == 5 for link in low["links"])
        assert high["links"] == []

    def test_empty_graph(self, populated_land):
        """minrel=100 -> graphe vide écrit, ret==1, sans lever."""
        graph = _export_graph(populated_land, "nodesjson", 100)

        assert graph == {"nodes": [], "links": []}

    def test_deterministic_ordering(self, populated_land):
        """Invariants d'ordre : nœuds, liens, corpus triés."""
        graph = _export_graph(populated_land, "nodesjson", 0)

        nodes, links = graph["nodes"], graph["links"]
        assert nodes == sorted(nodes, key=lambda n: (-n["nbexpressions"], n["id"]))
        assert links == sorted(
            links, key=lambda lk: (-lk["value"], lk["source"], lk["target"])
        )
        for node in nodes:
            urls = [entry["urlarticle"] for entry in node["corpus"]]
            assert urls == sorted(urls)

    def test_misspelled_type_rejected(self, populated_land):
        """Un type mal orthographié est rejeté (whitelist) -> ret==0."""
        controller = populated_land["controller"]
        core = populated_land["core"]
        name = populated_land["name"]

        ret = controller.LandController.export(
            core.Namespace(name=name, type="nodejson", minrel=0)  # singular: invalid
        )
        assert ret == 0

    # ----- cases populated_land cannot cover: single domain + dates ----- #

    @pytest.fixture
    def nodesjson_land(self, fresh_db):
        """1 domaine, 3 expr (published_at distincts), 1 lien intra-domaine."""
        controller = fresh_db["controller"]
        model = fresh_db["model"]
        core = fresh_db["core"]
        from datetime import datetime

        name = "test_nodesjson_land"
        controller.LandController.create(
            core.Namespace(name=name, desc="nodesjson fixture", lang=["fr"])
        )
        land = model.Land.get(model.Land.name == name)

        domain = model.Domain.create(name="solo.example")
        e1 = model.Expression.create(
            land=land, domain=domain, url="https://solo.example/1", relevance=2,
            published_at=datetime(2024, 3, 5), fetched_at=datetime.now())
        e2 = model.Expression.create(
            land=land, domain=domain, url="https://solo.example/2", relevance=4,
            published_at=datetime(2024, 1, 10), fetched_at=datetime.now())
        model.Expression.create(
            land=land, domain=domain, url="https://solo.example/3", relevance=6,
            published_at=datetime(2024, 6, 20), fetched_at=datetime.now())
        # intra-domain edge: must NOT produce a link in nodesjson (no self-loop)
        model.ExpressionLink.create(source=e1, target=e2)

        return {
            "land": land,
            "name": name,
            "controller": controller,
            "core": core,
            "data_dir": fresh_db["data_dir"],
        }

    def test_intra_domain_link_excluded(self, nodesjson_land):
        """L'arête intra-domaine ne crée aucun lien (pas de self-loop domaine)."""
        graph = _export_graph(nodesjson_land, "nodesjson", 1)

        assert len(graph["nodes"]) == 1
        assert graph["links"] == []
        assert all(lk["source"] != lk["target"] for lk in graph["links"])

    def test_date_aggregation_min_max(self, nodesjson_land):
        """first/last_expression_date = MIN/MAX(published_at), chaînes non nulles."""
        graph = _export_graph(nodesjson_land, "nodesjson", 1)

        node = graph["nodes"][0]
        assert node["first_expression_date"] == "2024-01-10 00:00:00"
        assert node["last_expression_date"] == "2024-06-20 00:00:00"
        assert isinstance(node["first_expression_date"], str)
        assert isinstance(node["last_expression_date"], str)

    def test_single_domain_values(self, nodesjson_land):
        """Valeurs concrètes sur un domaine unique."""
        graph = _export_graph(nodesjson_land, "nodesjson", 1)

        node = graph["nodes"][0]
        assert node["name"] == "solo.example"
        assert node["nbexpressions"] == 3
        assert node["average_relevance"] == 4.0  # (2+4+6)/3

    def test_corpus_published_at_per_article(self, nodesjson_land):
        """published_at = date de publication de chaque article, ordonné par url."""
        graph = _export_graph(nodesjson_land, "nodesjson", 1)

        corpus = graph["nodes"][0]["corpus"]
        assert [c["urlarticle"] for c in corpus] == [
            "https://solo.example/1",
            "https://solo.example/2",
            "https://solo.example/3",
        ]
        assert [c["published_at"] for c in corpus] == [
            "2024-03-05 00:00:00",
            "2024-01-10 00:00:00",
            "2024-06-20 00:00:00",
        ]


class TestLandExportPagesJSON:
    """Tests for the page force-graph export (--type=pagesjson)."""

    def test_structure_and_fields(self, populated_land):
        """{nodes, links} ; champs page + tags/seorank ; pas de depth."""
        graph = _export_graph(populated_land, "pagesjson", 0)

        assert isinstance(graph["nodes"], list)
        assert isinstance(graph["links"], list)
        assert len(graph["nodes"]) == 20
        assert len(graph["links"]) == 10
        for node in graph["nodes"]:
            assert "id" in node and "url" in node and "relevance" in node
            assert isinstance(node["tags"], list)
            assert isinstance(node["seorank"], dict)
            assert "depth" not in node  # technical field dropped (decision 5)

    def test_null_not_na(self, populated_land):
        """Champ absent -> null JSON (None), jamais la sentinelle 'na'."""
        graph = _export_graph(populated_land, "pagesjson", 0)

        assert graph["nodes"], "fixture must yield nodes (else assertions vacuous)"
        for node in graph["nodes"]:
            assert node["published_at"] is None
            assert node["keywords"] is None
            assert node["published_at"] != "na"
            assert node["keywords"] != "na"

    def test_empty_tags_and_seorank(self, populated_land):
        """Sans tag ni payload : tags == [] et seorank == {}."""
        graph = _export_graph(populated_land, "pagesjson", 0)

        assert graph["nodes"], "fixture must yield nodes (else assertions vacuous)"
        for node in graph["nodes"]:
            assert node["tags"] == []
            assert node["seorank"] == {}

    def test_referential_integrity(self, populated_land):
        """source/target de chaque lien est un id de nœud."""
        graph = _export_graph(populated_land, "pagesjson", 0)

        assert graph["nodes"] and graph["links"], "fixture must yield nodes and links"
        node_ids = {n["id"] for n in graph["nodes"]}
        for link in graph["links"]:
            assert link["source"] in node_ids
            assert link["target"] in node_ids

    def test_minrel_filters(self, populated_land):
        """minrel=10 -> 10 nœuds, 0 lien."""
        graph = _export_graph(populated_land, "pagesjson", 10)

        assert len(graph["nodes"]) == 10
        assert graph["links"] == []

    def test_empty_graph(self, populated_land):
        """minrel=100 -> graphe vide écrit, ret==1."""
        graph = _export_graph(populated_land, "pagesjson", 100)

        assert graph == {"nodes": [], "links": []}

    def test_deterministic_ordering(self, populated_land):
        """nœuds triés par id, liens par (source, target), tags triés."""
        graph = _export_graph(populated_land, "pagesjson", 0)

        nodes, links = graph["nodes"], graph["links"]
        assert nodes == sorted(nodes, key=lambda n: n["id"])
        assert links == sorted(links, key=lambda lk: (lk["source"], lk["target"]))
        for node in nodes:
            assert node["tags"] == sorted(node["tags"])

    # ----- cases that populated_land cannot cover: dedicated fixture ----- #

    @pytest.fixture
    def pagesjson_land(self, fresh_db):
        """1 domaine, 2 expr liées intra-domaine, tags + seorank sur l'une."""
        controller = fresh_db["controller"]
        model = fresh_db["model"]
        core = fresh_db["core"]
        from datetime import datetime

        name = "test_pagesjson_land"
        controller.LandController.create(
            core.Namespace(name=name, desc="pagesjson fixture", lang=["fr"])
        )
        land = model.Land.get(model.Land.name == name)

        domain = model.Domain.create(name="example.org")
        expr0 = model.Expression.create(
            land=land, domain=domain, url="https://example.org/a",
            title="Page A", readable="content " * 30, relevance=5,
            seorank='{"sr_rank": 5, "sr_traffic": 1000}',
            fetched_at=datetime.now(),
        )
        expr1 = model.Expression.create(
            land=land, domain=domain, url="https://example.org/b",
            title="Page B", readable="content " * 30, relevance=3,
            fetched_at=datetime.now(),
        )

        # intra-domain edge between the two pages of the same domain
        model.ExpressionLink.create(source=expr0, target=expr1)

        # two tags on expr0 with deliberately unsorted names
        tag_z = model.Tag.create(land=land, name="Zeta", parent=None,
                                 sorting=1, color="#111111")
        tag_a = model.Tag.create(land=land, name="Alpha", parent=None,
                                 sorting=2, color="#222222")
        model.TaggedContent.create(tag=tag_z, expression=expr0,
                                   text="snippet z", from_char=0, to_char=9)
        model.TaggedContent.create(tag=tag_a, expression=expr0,
                                   text="snippet a", from_char=0, to_char=9)

        return {
            "land": land,
            "expressions": [expr0, expr1],
            "name": name,
            "controller": controller,
            "core": core,
            "model": model,
            "data_dir": fresh_db["data_dir"],
        }

    def test_intra_domain_link_kept(self, pagesjson_land):
        """L'arête intra-domaine apparaît dans links (intra-domaine conservé)."""
        expr0, expr1 = pagesjson_land["expressions"]
        graph = _export_graph(pagesjson_land, "pagesjson", 1)

        edges = {(lk["source"], lk["target"]) for lk in graph["links"]}
        assert (expr0.id, expr1.id) in edges

    def test_pagesjson_links_exclude_self_loops(self, pagesjson_land):
        """Un ExpressionLink page→elle-même n'apparaît jamais dans links."""
        model = pagesjson_land["model"]
        expr0, expr1 = pagesjson_land["expressions"]
        model.ExpressionLink.create(source=expr0, target=expr0)

        graph = _export_graph(pagesjson_land, "pagesjson", 1)

        assert all(lk["source"] != lk["target"] for lk in graph["links"])
        # l'arête intra-domaine normale reste conservée
        edges = {(lk["source"], lk["target"]) for lk in graph["links"]}
        assert (expr0.id, expr1.id) in edges

    def test_tags_array_sorted(self, pagesjson_land):
        """Le nœud taggé porte tags == sorted([...])."""
        expr0 = pagesjson_land["expressions"][0]
        graph = _export_graph(pagesjson_land, "pagesjson", 1)

        node = next(n for n in graph["nodes"] if n["id"] == expr0.id)
        assert node["tags"] == ["Alpha", "Zeta"]

    def test_seorank_nested_object_numeric(self, pagesjson_land):
        """seorank est un objet imbriqué aux valeurs numériques natives."""
        expr0 = pagesjson_land["expressions"][0]
        graph = _export_graph(pagesjson_land, "pagesjson", 1)

        node = next(n for n in graph["nodes"] if n["id"] == expr0.id)
        assert node["seorank"] == {"sr_rank": 5, "sr_traffic": 1000}
        assert isinstance(node["seorank"]["sr_rank"], int)
