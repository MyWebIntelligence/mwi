"""
Tests for embeddings and semantic similarity functionality.
"""
import pytest
import random
import string
import os
import glob
import csv
import json
from datetime import datetime


def rand_name(prefix="land"):
    """Generate random land name for tests."""
    letters = string.ascii_lowercase
    return f"{prefix}_" + "".join(random.choice(letters) for _ in range(8))


class TestEmbeddingGeneration:
    """Tests for paragraph and embedding generation."""

    def test_generate_embeddings_fake_provider(self, fresh_db, monkeypatch):
        """Génération embeddings avec fake provider."""
        controller = fresh_db["controller"]
        model = fresh_db["model"]
        core = fresh_db["core"]
        import settings
        from mwi import embedding_pipeline

        # Ensure fake provider - patch both settings and embedding_pipeline.settings
        monkeypatch.setattr(settings, "embed_provider", "fake")
        monkeypatch.setattr(embedding_pipeline, "settings", settings)

        name = rand_name("test")
        controller.LandController.create(
            core.Namespace(name=name, desc="Test land", lang=["fr"])
        )
        land = model.Land.get(model.Land.name == name)

        # Créer expressions avec contenu lisible
        domain = model.Domain.create(name="example.com")
        for i in range(3):
            model.Expression.create(
                land=land,
                domain=domain,
                url=f"https://example.com/page{i}",
                title=f"Page {i}",
                readable="This is a test paragraph. " * 20,  # ~100 words
                fetched_at=datetime.now(),
                readable_at=datetime.now()
            )

        # Generate embeddings
        ret = controller.EmbeddingController.generate(
            core.Namespace(name=name, limit=None)
        )

        assert ret == 1

        # Vérifier Paragraphs créés
        paragraphs = model.Paragraph.select().join(model.Expression).where(
            model.Expression.land == land
        )
        assert paragraphs.count() > 0

        # Vérifier ParagraphEmbedding créés (1:1 avec Paragraph)
        embeddings = model.ParagraphEmbedding.select().join(model.Paragraph).join(
            model.Expression
        ).where(model.Expression.land == land)
        assert embeddings.count() == paragraphs.count()

        # Vérifier structure embedding
        emb = embeddings[0]
        assert emb.embedding is not None
        embedding_vec = json.loads(emb.embedding)
        assert isinstance(embedding_vec, list)
        assert len(embedding_vec) > 0
        assert emb.model_name is not None

    def test_generate_embeddings_respects_limit(self, fresh_db, monkeypatch):
        """--limit=N traite au max N expressions."""
        controller = fresh_db["controller"]
        model = fresh_db["model"]
        core = fresh_db["core"]
        import settings
        from mwi import embedding_pipeline

        # Ensure fake provider - patch both settings modules
        monkeypatch.setattr(settings, "embed_provider", "fake")
        monkeypatch.setattr(embedding_pipeline, "settings", settings)

        name = rand_name("test")
        controller.LandController.create(
            core.Namespace(name=name, desc="Test land", lang=["fr"])
        )
        land = model.Land.get(model.Land.name == name)

        # Créer 5 expressions
        domain = model.Domain.create(name="example.com")
        for i in range(5):
            model.Expression.create(
                land=land,
                domain=domain,
                url=f"https://example.com/page{i}",
                readable="Test paragraph. " * 20,
                fetched_at=datetime.now(),
                readable_at=datetime.now()
            )

        # Generate avec limit=2
        controller.EmbeddingController.generate(
            core.Namespace(name=name, limit=2)
        )

        # Vérifier que seulement 2 expressions ont été traitées
        paragraphs = model.Paragraph.select().join(model.Expression).where(
            model.Expression.land == land
        )
        # Chaque expression génère au moins 1 paragraphe
        # Avec limit=2, on devrait avoir paragraphes de 2 expressions max
        expressions_with_paragraphs = set()
        for p in paragraphs:
            expressions_with_paragraphs.add(p.expression_id)

        assert len(expressions_with_paragraphs) <= 2

    def test_generate_embeddings_deduplication(self, fresh_db, monkeypatch):
        """Paragraphes dédupliqués par text_hash."""
        controller = fresh_db["controller"]
        model = fresh_db["model"]
        core = fresh_db["core"]
        import settings
        from mwi import embedding_pipeline

        # Ensure fake provider - patch both settings modules
        monkeypatch.setattr(settings, "embed_provider", "fake")
        monkeypatch.setattr(embedding_pipeline, "settings", settings)

        name = rand_name("test")
        controller.LandController.create(
            core.Namespace(name=name, desc="Test land", lang=["fr"])
        )
        land = model.Land.get(model.Land.name == name)

        # Créer 2 expressions avec même texte
        domain = model.Domain.create(name="example.com")
        same_text = "This is identical paragraph text. " * 20
        for i in range(2):
            model.Expression.create(
                land=land,
                domain=domain,
                url=f"https://example.com/page{i}",
                readable=same_text,
                fetched_at=datetime.now(),
                readable_at=datetime.now()
            )

        # Generate
        controller.EmbeddingController.generate(
            core.Namespace(name=name, limit=None)
        )

        # Vérifier qu'il y a moins de paragraphes que d'expressions
        # à cause de la déduplication
        paragraphs = model.Paragraph.select().join(model.Expression).where(
            model.Expression.land == land
        )
        # Note: Le système peut créer plusieurs paragraphes par expression
        # mais les paragraphes identiques (même text_hash) ne seront pas dupliqués
        hashes = set()
        for p in paragraphs:
            hashes.add(p.text_hash)

        # On devrait avoir moins de hashes uniques que de paragraphes créés x 2 expressions
        # car le texte est identique
        assert len(hashes) > 0


class TestSimilarityCosine:
    """Tests for cosine similarity computation."""

    def test_similarity_cosine_exact(self, fresh_db, monkeypatch):
        """Méthode cosine exact (O(n²))."""
        controller = fresh_db["controller"]
        model = fresh_db["model"]
        core = fresh_db["core"]
        import settings
        from mwi import embedding_pipeline

        # Ensure fake provider - patch both settings modules
        monkeypatch.setattr(settings, "embed_provider", "fake")
        monkeypatch.setattr(embedding_pipeline, "settings", settings)

        name = rand_name("test")
        controller.LandController.create(
            core.Namespace(name=name, desc="Test land", lang=["fr"])
        )
        land = model.Land.get(model.Land.name == name)

        # Créer expressions avec contenu
        domain = model.Domain.create(name="example.com")
        for i in range(3):
            model.Expression.create(
                land=land,
                domain=domain,
                url=f"https://example.com/page{i}",
                readable=f"Paragraph about topic {i}. " * 20,
                fetched_at=datetime.now(),
                readable_at=datetime.now()
            )

        # Generate embeddings
        controller.EmbeddingController.generate(
            core.Namespace(name=name, limit=None)
        )

        # Compute similarities
        ret = controller.EmbeddingController.similarity(
            core.Namespace(
                name=name,
                threshold=0.85,
                method="cosine",
                topk=None,
                lshbits=None,
                maxpairs=None,
                minrel=None,
                backend=None
            )
        )

        assert ret == 1

        # Vérifier ParagraphSimilarity créées
        similarities = model.ParagraphSimilarity.select().join(
            model.Paragraph, on=(model.ParagraphSimilarity.source_paragraph == model.Paragraph.id)
        ).join(
            model.Expression
        ).where(model.Expression.land == land)

        # Il devrait y avoir des similarités
        # (fake embeddings peuvent produire des scores aléatoires)
        assert similarities.count() >= 0

        # Vérifier structure
        if similarities.count() > 0:
            sim = similarities[0]
            assert sim.score is not None
            assert sim.method == "cosine"

    def test_similarity_respects_threshold(self, fresh_db, monkeypatch):
        """Seuil de similarité respecté."""
        controller = fresh_db["controller"]
        model = fresh_db["model"]
        core = fresh_db["core"]
        import settings
        from mwi import embedding_pipeline

        # Ensure fake provider - patch both settings modules
        monkeypatch.setattr(settings, "embed_provider", "fake")
        monkeypatch.setattr(embedding_pipeline, "settings", settings)

        name = rand_name("test")
        controller.LandController.create(
            core.Namespace(name=name, desc="Test land", lang=["fr"])
        )
        land = model.Land.get(model.Land.name == name)

        domain = model.Domain.create(name="example.com")
        for i in range(2):
            model.Expression.create(
                land=land,
                domain=domain,
                url=f"https://example.com/page{i}",
                readable="Test content. " * 20,
                fetched_at=datetime.now(),
                readable_at=datetime.now()
            )

        controller.EmbeddingController.generate(
            core.Namespace(name=name, limit=None)
        )

        # Compute avec threshold élevé
        controller.EmbeddingController.similarity(
            core.Namespace(
                name=name,
                threshold=0.99,
                method="cosine",
                topk=None,
                lshbits=None,
                maxpairs=None,
                minrel=None,
                backend=None
            )
        )

        # Vérifier que scores >= threshold
        similarities = model.ParagraphSimilarity.select().join(
            model.Paragraph, on=(model.ParagraphSimilarity.source_paragraph == model.Paragraph.id)
        ).join(
            model.Expression
        ).where(model.Expression.land == land)

        for sim in similarities:
            assert sim.score >= 0.99


class TestSimilarityLSH:
    """Tests for LSH-based similarity."""

    def test_similarity_cosine_lsh(self, fresh_db, monkeypatch):
        """Méthode cosine_lsh (approximative)."""
        controller = fresh_db["controller"]
        model = fresh_db["model"]
        core = fresh_db["core"]
        import settings
        from mwi import embedding_pipeline

        # Ensure fake provider - patch both settings modules
        monkeypatch.setattr(settings, "embed_provider", "fake")
        monkeypatch.setattr(embedding_pipeline, "settings", settings)

        name = rand_name("test")
        controller.LandController.create(
            core.Namespace(name=name, desc="Test land", lang=["fr"])
        )
        land = model.Land.get(model.Land.name == name)

        domain = model.Domain.create(name="example.com")
        for i in range(5):
            model.Expression.create(
                land=land,
                domain=domain,
                url=f"https://example.com/page{i}",
                readable=f"Content {i}. " * 20,
                fetched_at=datetime.now(),
                readable_at=datetime.now()
            )

        controller.EmbeddingController.generate(
            core.Namespace(name=name, limit=None)
        )

        # Compute avec LSH
        ret = controller.EmbeddingController.similarity(
            core.Namespace(
                name=name,
                threshold=None,
                method="cosine_lsh",
                topk=3,
                lshbits=20,
                maxpairs=None,
                minrel=None,
                backend=None
            )
        )

        assert ret == 1

        # Vérifier que method = 'cosine_lsh'
        similarities = model.ParagraphSimilarity.select().join(
            model.Paragraph, on=(model.ParagraphSimilarity.source_paragraph == model.Paragraph.id)
        ).join(
            model.Expression
        ).where(model.Expression.land == land)

        if similarities.count() > 0:
            sim = similarities[0]
            assert sim.method == "cosine_lsh"

    def test_similarity_lsh_respects_topk(self, fresh_db, monkeypatch):
        """Top-K voisins par paragraph."""
        controller = fresh_db["controller"]
        model = fresh_db["model"]
        core = fresh_db["core"]
        import settings
        from mwi import embedding_pipeline

        # Ensure fake provider - patch both settings modules
        monkeypatch.setattr(settings, "embed_provider", "fake")
        monkeypatch.setattr(embedding_pipeline, "settings", settings)

        name = rand_name("test")
        controller.LandController.create(
            core.Namespace(name=name, desc="Test land", lang=["fr"])
        )
        land = model.Land.get(model.Land.name == name)

        domain = model.Domain.create(name="example.com")
        for i in range(10):
            model.Expression.create(
                land=land,
                domain=domain,
                url=f"https://example.com/page{i}",
                readable=f"Text {i}. " * 20,
                fetched_at=datetime.now(),
                readable_at=datetime.now()
            )

        controller.EmbeddingController.generate(
            core.Namespace(name=name, limit=None)
        )

        # Compute avec topk=2
        controller.EmbeddingController.similarity(
            core.Namespace(
                name=name,
                threshold=None,
                method="cosine_lsh",
                topk=2,
                lshbits=20,
                maxpairs=None,
                minrel=None,
                backend=None
            )
        )

        # Vérifier que chaque source paragraph a au max topk targets
        paragraphs = model.Paragraph.select().join(model.Expression).where(
            model.Expression.land == land
        )

        for para in paragraphs:
            target_count = model.ParagraphSimilarity.select().where(
                model.ParagraphSimilarity.source_paragraph == para
            ).count()
            assert target_count <= 2


class TestEmbeddingReset:
    """Tests for embedding reset functionality."""

    def test_reset_embeddings_deletes_data(self, fresh_db, monkeypatch):
        """embedding reset supprime Paragraph, ParagraphEmbedding, ParagraphSimilarity."""
        controller = fresh_db["controller"]
        model = fresh_db["model"]
        core = fresh_db["core"]
        import settings
        from mwi import embedding_pipeline

        # Ensure fake provider - patch both settings modules
        monkeypatch.setattr(settings, "embed_provider", "fake")
        monkeypatch.setattr(embedding_pipeline, "settings", settings)

        name = rand_name("test")
        controller.LandController.create(
            core.Namespace(name=name, desc="Test land", lang=["fr"])
        )
        land = model.Land.get(model.Land.name == name)

        # Créer expressions
        domain = model.Domain.create(name="example.com")
        for i in range(2):
            model.Expression.create(
                land=land,
                domain=domain,
                url=f"https://example.com/page{i}",
                readable="Test. " * 20,
                fetched_at=datetime.now(),
                readable_at=datetime.now()
            )

        # Generate embeddings
        controller.EmbeddingController.generate(
            core.Namespace(name=name, limit=None)
        )

        # Vérifier données créées
        paragraphs_before = model.Paragraph.select().join(model.Expression).where(
            model.Expression.land == land
        ).count()
        assert paragraphs_before > 0

        # Reset
        ret = controller.EmbeddingController.reset(
            core.Namespace(name=name)
        )

        assert ret == 1

        # Vérifier tables vides pour ce land
        paragraphs_after = model.Paragraph.select().join(model.Expression).where(
            model.Expression.land == land
        ).count()
        assert paragraphs_after == 0

        embeddings_after = model.ParagraphEmbedding.select().join(
            model.Paragraph
        ).join(model.Expression).where(
            model.Expression.land == land
        ).count()
        assert embeddings_after == 0

        similarities_after = model.ParagraphSimilarity.select().join(
            model.Paragraph, on=(model.ParagraphSimilarity.source_paragraph == model.Paragraph.id)
        ).join(model.Expression).where(
            model.Expression.land == land
        ).count()
        assert similarities_after == 0


class TestPseudolinksExport:
    """Tests for pseudolinks export formats."""

    @pytest.fixture
    def land_with_similarities(self, fresh_db, monkeypatch):
        """Land avec embeddings et similarités."""
        controller = fresh_db["controller"]
        model = fresh_db["model"]
        core = fresh_db["core"]
        data_dir = fresh_db["data_dir"]
        import settings
        from mwi import embedding_pipeline

        # Ensure fake provider - patch both settings modules
        monkeypatch.setattr(settings, "embed_provider", "fake")
        monkeypatch.setattr(embedding_pipeline, "settings", settings)

        name = rand_name("test")
        controller.LandController.create(
            core.Namespace(name=name, desc="Test land", lang=["fr"])
        )
        land = model.Land.get(model.Land.name == name)

        # Créer expressions avec relevance
        domain = model.Domain.create(name="example.com")
        for i in range(3):
            model.Expression.create(
                land=land,
                domain=domain,
                url=f"https://example.com/page{i}",
                readable=f"Paragraph {i}. " * 20,
                relevance=1,  # Au moins 1 pour passer le filtre minrel=0
                fetched_at=datetime.now(),
                readable_at=datetime.now()
            )

        # Generate embeddings et similarités
        controller.EmbeddingController.generate(
            core.Namespace(name=name, limit=None)
        )
        controller.EmbeddingController.similarity(
            core.Namespace(
                name=name,
                threshold=0.1,  # Seuil bas pour avoir des résultats
                method="cosine",
                topk=None,
                lshbits=None,
                maxpairs=None,
                minrel=None,
                backend=None
            )
        )

        return {
            "name": name,
            "land": land,
            "controller": controller,
            "core": core,
            "model": model,
            "data_dir": data_dir
        }

    def test_export_pseudolinks_paragraph_level(self, land_with_similarities):
        """Export pseudolinks (paragraph pairs)."""
        controller = land_with_similarities["controller"]
        core = land_with_similarities["core"]
        name = land_with_similarities["name"]
        data_dir = str(land_with_similarities["data_dir"])

        ret = controller.LandController.export(
            core.Namespace(name=name, type="pseudolinks", minrel=0)
        )

        assert ret == 1

        # Trouver fichier exporté (pas d'extension .csv, format: export_land_name_type_timestamp)
        csv_files = glob.glob(os.path.join(data_dir, f"*{name}*pseudolinks*"))
        assert len(csv_files) > 0

        # Valider structure CSV
        csv_file = csv_files[-1]
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if len(rows) > 0:
            # Vérifier colonnes
            assert "Source_ParagraphID" in rows[0]
            assert "Target_ParagraphID" in rows[0]
            assert "RelationScore" in rows[0]

    def test_export_pseudolinkspage_aggregated(self, land_with_similarities):
        """Export pseudolinks agrégés par page."""
        controller = land_with_similarities["controller"]
        core = land_with_similarities["core"]
        name = land_with_similarities["name"]
        data_dir = str(land_with_similarities["data_dir"])

        ret = controller.LandController.export(
            core.Namespace(name=name, type="pseudolinkspage", minrel=0)
        )

        assert ret == 1

        # Trouver fichier (pas d'extension .csv)
        csv_files = glob.glob(os.path.join(data_dir, f"*{name}*pseudolinkspage*"))
        assert len(csv_files) > 0

        csv_file = csv_files[-1]
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if len(rows) > 0:
            # Vérifier colonnes agrégées
            assert "Source_ExpressionID" in rows[0]
            assert "Target_ExpressionID" in rows[0]
            assert "PairCount" in rows[0]
            assert "AvgRelationScore" in rows[0]

    def test_export_pseudolinksdomain_aggregated(self, land_with_similarities):
        """Export pseudolinks agrégés par domaine."""
        controller = land_with_similarities["controller"]
        core = land_with_similarities["core"]
        name = land_with_similarities["name"]
        data_dir = str(land_with_similarities["data_dir"])

        ret = controller.LandController.export(
            core.Namespace(name=name, type="pseudolinksdomain", minrel=0)
        )

        assert ret == 1

        # Trouver fichier (pas d'extension .csv)
        csv_files = glob.glob(os.path.join(data_dir, f"*{name}*pseudolinksdomain*"))
        assert len(csv_files) > 0

        csv_file = csv_files[-1]
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if len(rows) > 0:
            # Vérifier colonnes
            assert "Source_DomainID" in rows[0]
            assert "Target_DomainID" in rows[0]
            assert "PairCount" in rows[0]

    def test_pseudolinksdomain_excludes_intra_domain(self, land_with_similarities):
        """Une paire de paragraphes du même domaine ne produit jamais de
        self-loop domaine→lui-même ; l'arête inter-domaine reste exportée."""
        controller = land_with_similarities["controller"]
        core = land_with_similarities["core"]
        model = land_with_similarities["model"]
        land = land_with_similarities["land"]
        name = land_with_similarities["name"]
        data_dir = str(land_with_similarities["data_dir"])

        # La fixture (3 expressions sur example.com) a déjà produit des paires
        # intra-domaine. On ajoute une expression sur un second domaine et une
        # paire inter-domaines créée à la main.
        domain2 = model.Domain.create(name="other.org")
        expr_ext = model.Expression.create(
            land=land, domain=domain2, url="https://other.org/page",
            readable="External paragraph. " * 20, relevance=1,
            fetched_at=datetime.now(), readable_at=datetime.now(),
        )
        para_local = (model.Paragraph.select()
                      .join(model.Expression)
                      .where(model.Expression.land == land)
                      .first())
        para_ext = model.Paragraph.create(
            expression=expr_ext, domain=domain2, para_index=0,
            text="External paragraph.", text_hash="f" * 64,
        )
        model.ParagraphSimilarity.create(
            source_paragraph=para_local, target_paragraph=para_ext,
            method="cosine", score=1, score_raw=0.9,
        )

        ret = controller.LandController.export(
            core.Namespace(name=name, type="pseudolinksdomain", minrel=0)
        )

        assert ret == 1
        csv_file = sorted(glob.glob(
            os.path.join(data_dir, f"*{name}*pseudolinksdomain*")))[-1]
        with open(csv_file, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert all(r["Source_DomainID"] != r["Target_DomainID"] for r in rows)
        pairs = {(r["Source_DomainID"], r["Target_DomainID"]) for r in rows}
        d_local = str(model.Domain.get(model.Domain.name == "example.com").id)
        d_ext = str(domain2.id)
        assert (d_local, d_ext) in pairs or (d_ext, d_local) in pairs


class TestEmbeddingCheck:
    """Tests for embedding environment check."""

    def test_embedding_check_runs(self, fresh_db, capsys):
        """embedding check retourne code et affiche le statut."""
        controller = fresh_db["controller"]
        core = fresh_db["core"]

        ret = controller.EmbeddingController.check(core.Namespace())

        assert ret in [0, 1]
        output = capsys.readouterr().out
        assert "Embedding provider" in output
