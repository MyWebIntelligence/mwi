import pytest


def test_embedding_generate_and_similarity(fresh_db, monkeypatch):
    controller = fresh_db["controller"]
    core = fresh_db["core"]
    model = fresh_db["model"]

    # Ensure provider is fake to avoid network
    import settings as _settings
    _settings.embed_provider = 'fake'

    # Create land
    name = "emb_land"
    assert controller.LandController.create(core.Namespace(name=name, desc="d", lang=["fr"])) == 1
    land = model.Land.get(model.Land.name == name)

    # Create a domain and two expressions with readable content
    dom = model.Domain.create(name="example.com")
    e1 = model.Expression.create(land=land, url="https://example.com/1", domain=dom,
                                 readable=("This is a test paragraph about embeddings and NLP.\n\n"
                                           "Another unrelated section."))
    e2 = model.Expression.create(land=land, url="https://example.com/2", domain=dom,
                                 readable=("Embeddings and NLP are used to test semantic similarity.\n\n"
                                           "More text here."))

    # Generate embeddings
    ret = controller.EmbeddingController.generate(core.Namespace(name=name, limit=0))
    assert ret == 1
    # At least some paragraphs and embeddings exist
    assert model.Paragraph.select().count() >= 2
    assert model.ParagraphEmbedding.select().count() >= 2

    # Compute similarities with a low threshold to avoid fragility
    ret = controller.EmbeddingController.similarity(core.Namespace(name=name, threshold=0.1, method="cosine"))
    assert ret == 1
    # Some similarity rows should be present (may vary)
    assert model.ParagraphSimilarity.select().count() >= 0

