import os
import io
import random
import string
from argparse import Namespace

import pytest


def rand_name(prefix="land"):
    letters = string.ascii_lowercase
    return f"{prefix}_" + "".join(random.choice(letters) for _ in range(8))


def test_dispatch_invalid_object(test_env):
    cli = test_env["cli"]
    with pytest.raises(ValueError):
        cli.dispatch(Namespace(object="invalid", verb="noop"))


def test_dispatch_invalid_verb(test_env):
    cli = test_env["cli"]
    with pytest.raises(ValueError):
        cli.dispatch(Namespace(object="land", verb="doesnotexist"))


def test_db_setup_and_migrate(fresh_db):
    controller = fresh_db["controller"]
    core = fresh_db["core"]
    # Migrate should run and return success (idempotent)
    ret = controller.DbController.migrate(core.Namespace())
    assert ret == 1


def test_land_create_list_delete(fresh_db, capsys, monkeypatch):
    controller = fresh_db["controller"]
    core = fresh_db["core"]
    model = fresh_db["model"]

    name = rand_name("proj")
    # Create land
    ret = controller.LandController.create(core.Namespace(name=name, desc="desc", lang=["fr"]))
    assert ret == 1
    assert model.Land.get_or_none(model.Land.name == name) is not None

    # List lands (should print info and return 1)
    ret = controller.LandController.list(core.Namespace(name=None))
    assert ret == 1
    out = capsys.readouterr().out
    assert name in out

    # Delete land with confirmation
    monkeypatch.setattr(core, "confirm", lambda msg: True)
    ret = controller.LandController.delete(core.Namespace(name=name, maxrel=None))
    assert ret == 1
    assert model.Land.get_or_none(model.Land.name == name) is None


def test_land_addterm_addurl_and_crawl_readable_export(fresh_db, tmp_path, monkeypatch):
    controller = fresh_db["controller"]
    core = fresh_db["core"]
    model = fresh_db["model"]

    name = rand_name("land")
    # Create base land
    assert controller.LandController.create(core.Namespace(name=name, desc="d", lang=["fr"])) == 1

    # Add terms
    ret = controller.LandController.addterm(core.Namespace(land=name, terms="asthme, enfant, santé"))
    assert ret == 1
    land = model.Land.get(model.Land.name == name)
    assert model.LandDictionary.select().where(model.LandDictionary.land == land).count() == 3

    # Prepare URL file and add URLs
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://example.com/\nhttps://www.python.org/", encoding="utf-8")
    ret = controller.LandController.addurl(core.Namespace(land=name, path=str(urls_file), urls=None))
    assert ret == 1
    assert model.Expression.select().where(model.Expression.land == land).count() == 2

    # Mock network-heavy pipelines
    async def _fake_crawl_land(land, limit, http, depth):
        return (0, 0)
    # Patch through controller's bound module for reliability
    monkeypatch.setattr(controller.core, "crawl_land", _fake_crawl_land)
    # readable() uses run_readable_pipeline from mwi.readable_pipeline
    import mwi.readable_pipeline as rp
    async def _fake_run_readable_pipeline(land, limit, depth, merge):
        return (0, 0)
    monkeypatch.setattr(rp, "run_readable_pipeline", _fake_run_readable_pipeline)

    # Crawl limited set
    ret = controller.LandController.crawl(core.Namespace(name=name, limit=1, http=None, depth=None))
    assert ret == 1

    # Readable pipeline
    ret = controller.LandController.readable(core.Namespace(name=name, limit=1, depth=None, merge="smart_merge"))
    assert ret == 1

    # Export: validate types and that dispatcher accepts them
    called = {"n": 0}
    def fake_export_land(land_obj, etype, minrel):
        called["n"] += 1
        return None

    monkeypatch.setattr(controller.core, "export_land", fake_export_land)
    for etype in [
        "pagecsv", "fullpagecsv", "pagegexf", "nodegexf", "nodecsv", "mediacsv", "corpus", "pseudolinks"
    ]:
        ret = controller.LandController.export(core.Namespace(name=name, type=etype, minrel=1))
        assert ret == 1
    assert called["n"] == 8


def test_land_consolidate_and_medianalyse(fresh_db, monkeypatch):
    controller = fresh_db["controller"]
    core = fresh_db["core"]

    name = rand_name("land")
    assert controller.LandController.create(core.Namespace(name=name, desc="d", lang=["fr"])) == 1

    # Consolidate
    async def _fake_consolidate_land(land, limit, depth):
        return (0, 0)
    monkeypatch.setattr(controller.core, "consolidate_land", _fake_consolidate_land)
    ret = controller.LandController.consolidate(core.Namespace(name=name, limit=0, depth=None))
    assert ret == 1

    # Medianalyse (land)
    async def _fake_medianalyse_land(land):
        return {"processed": 0}
    monkeypatch.setattr(controller.core, "medianalyse_land", _fake_medianalyse_land)
    ret = controller.LandController.medianalyse(core.Namespace(name=name))
    assert ret == 1


def test_domain_crawl_cli(fresh_db, monkeypatch):
    controller = fresh_db["controller"]
    core = fresh_db["core"]
    # Simulate return count
    monkeypatch.setattr(controller.core, "crawl_domains", lambda limit, http: 2)
    ret = controller.DomainController.crawl(core.Namespace(limit=2, http=None))
    assert ret == 1


def test_tag_export_cli(fresh_db, monkeypatch):
    controller = fresh_db["controller"]
    core = fresh_db["core"]
    model = fresh_db["model"]
    name = rand_name("land")
    assert controller.LandController.create(core.Namespace(name=name, desc="d", lang=["fr"])) == 1
    land = model.Land.get(model.Land.name == name)

    # Export tags
    calls = {"n": 0}
    monkeypatch.setattr(controller.core, "export_tags", lambda land_obj, etype, minrel: calls.__setitem__("n", calls["n"] + 1))
    for etype in ["matrix", "content"]:
        ret = controller.TagController.export(core.Namespace(name=name, type=etype, minrel=0))
        assert ret == 1
    assert calls["n"] == 2


def test_heuristic_update_cli(fresh_db, monkeypatch):
    controller = fresh_db["controller"]
    core = fresh_db["core"]
    called = {"ok": False}
    monkeypatch.setattr(controller.core, "update_heuristic", lambda: called.__setitem__("ok", True))
    ret = controller.HeuristicController.update(core.Namespace())
    assert ret == 1 and called["ok"] is True


def test_cli_get_arg_helpers(test_env):
    core = test_env["core"]
    ns = Namespace(option_str=None, option_int=None)
    assert core.get_arg_option("option_str", ns, set_type=str, default="A") == "A"
    assert core.get_arg_option("option_int", ns, set_type=int, default=5) == 5

    ns = Namespace(option_str=503, option_int="3")
    assert core.get_arg_option("option_str", ns, set_type=str, default="A") == "503"
    assert core.get_arg_option("option_int", ns, set_type=int, default=5) == 3

    # check_args should raise for missing mandatory args
    with pytest.raises(Exception):
        core.check_args(Namespace(a=True), ("a", "b"))


def test_cli_lang_list_handling(fresh_db):
    controller = fresh_db["controller"]
    core = fresh_db["core"]
    model = fresh_db["model"]
    name = rand_name("lang")
    langs = ["fr", "en", "it"]
    assert controller.LandController.create(core.Namespace(name=name, desc="x", lang=langs)) == 1
    land = model.Land.get(model.Land.name == name)
    assert land.lang == ",".join(langs)


def test_land_llm_validate_cli(fresh_db, monkeypatch):
    cli = fresh_db["cli"]
    controller = fresh_db["controller"]
    core = fresh_db["core"]
    model = fresh_db["model"]

    # Configure OpenRouter in settings
    controller.settings.openrouter_enabled = True
    controller.settings.openrouter_api_key = "sk-test"
    controller.settings.openrouter_model = "openai/gpt-4o-mini"

    # Create a land and a few expressions
    name = rand_name("llm")
    assert controller.LandController.create(core.Namespace(name=name, desc="d", lang=["fr"])) == 1
    land = model.Land.get(model.Land.name == name)
    # Add 3 expressions via core helper (creates domains as needed),
    # and set readable to meet the min length filter.
    minlen = getattr(controller.settings, 'openrouter_readable_min_chars', 0)
    for i in range(3):
        expr = core.add_expression(land, f"https://example.com/p{i}")
        assert expr
        text = ("Lorem ipsum ") * ((minlen // 12) + 2)
        expr.readable = text
        expr.save(only=[model.Expression.readable])

    # Stub LLM verdicts: True, False, None
    import mwi.llm_openrouter as llm
    verdicts = iter([True, False, None])
    monkeypatch.setattr(llm, "is_relevant_via_openrouter", lambda l, e: next(verdicts))

    # Run: limit=2 should update only first 2
    from argparse import Namespace
    ret = cli.dispatch(Namespace(object="land", verb="llm", subverb="validate", name=name, limit=2))
    assert ret == 1

    # Check DB updates
    updated = (model.Expression
               .select()
               .where((model.Expression.land == land) & (model.Expression.validllm.is_null(False))))
    assert updated.count() == 2
    vals = sorted([e.validllm for e in updated])
    assert vals == ["non", "oui"]
    for e in updated:
        assert e.validmodel == controller.settings.openrouter_model
    # 'non' should force relevance=0
    non_expr = (model.Expression
                .select()
                .where((model.Expression.land == land) & (model.Expression.validllm == 'non'))
                .get())
    assert non_expr.relevance == 0
