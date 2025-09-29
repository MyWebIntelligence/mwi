import json
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


def test_land_seorank_cli(fresh_db, monkeypatch):
    controller = fresh_db["controller"]
    core = fresh_db["core"]
    model = fresh_db["model"]

    name = rand_name("land")
    assert controller.LandController.create(core.Namespace(name=name, desc="d", lang=["fr"])) == 1
    land = model.Land.get(model.Land.name == name)
    domain = model.Domain.create(name="example.com")
    expr = model.Expression.create(
        land=land,
        domain=domain,
        url="https://example.com/",
        http_status='200',
        relevance=2,
    )
    # Expression that should be skipped (bad http status)
    model.Expression.create(
        land=land,
        domain=domain,
        url="https://example.com/404",
        http_status='404',
        relevance=5,
    )
    # Expression skipped by min relevance
    model.Expression.create(
        land=land,
        domain=domain,
        url="https://example.com/lowrel",
        http_status='200',
        relevance=0,
    )

    monkeypatch.setattr(controller.settings, "seorank_api_key", "", raising=False)
    monkeypatch.setattr(core.settings, "seorank_api_key", "", raising=False)
    ret = controller.LandController.seorank(core.Namespace(name=name, limit=0, depth=None, force=False))
    assert ret == 0

    monkeypatch.setattr(controller.settings, "seorank_api_key", "TEST", raising=False)
    monkeypatch.setattr(core.settings, "seorank_api_key", "TEST", raising=False)
    monkeypatch.setattr(controller.settings, "seorank_request_delay", 0, raising=False)
    monkeypatch.setattr(core.settings, "seorank_request_delay", 0, raising=False)

    calls = {"n": 0}

    def fake_fetch(url, key):
        calls["n"] += 1
        return {"url": url, "score": 42}

    monkeypatch.setattr(core, "fetch_seorank_for_url", fake_fetch)

    ret = controller.LandController.seorank(core.Namespace(name=name, limit=0, depth=None, force=False, http=None, minrel=None))
    assert ret == 1

    updated_expr = model.Expression.get(model.Expression.id == expr.id)
    assert json.loads(updated_expr.seorank)["score"] == 42
    assert calls["n"] == 1

    calls["n"] = 0
    ret = controller.LandController.seorank(core.Namespace(name=name, limit=0, depth=None, force=False, http=None, minrel=None))
    assert ret == 1
    assert calls["n"] == 0


def test_land_urlist_cli(fresh_db, monkeypatch):
    controller = fresh_db["controller"]
    core = fresh_db["core"]
    model = fresh_db["model"]

    name = rand_name("land")
    assert controller.LandController.create(core.Namespace(name=name, desc="d", lang=["fr"])) == 1
    land = model.Land.get(model.Land.name == name)

    # Missing API key should short-circuit with a failure status.
    monkeypatch.setattr(controller.settings, "serpapi_api_key", "", raising=False)
    monkeypatch.setattr(core.settings, "serpapi_api_key", "", raising=False)
    ret = controller.LandController.urlist(core.Namespace(
        name=name,
        query="test",
        lang=["fr"],
        datestart=None,
        dateend=None,
        timestep="week",
        sleep=0.0,
    ))
    assert ret == 0

    # Prepare environment with an existing expression to check updates/duplicates.
    monkeypatch.setattr(controller.settings, "serpapi_api_key", "TEST", raising=False)
    monkeypatch.setattr(core.settings, "serpapi_api_key", "TEST", raising=False)

    existing_url = "https://existing.com/article"
    existing_expr = controller.core.add_expression(land, existing_url)
    assert existing_expr is not False
    existing_expr.title = None
    existing_expr.save()

    mock_results = [
        {"link": existing_url, "title": "Existing title", "position": 1, "date": "2024-01-01"},
        {"link": "https://newsite.com/story", "title": "New title", "position": 2, "date": "2024-01-02"},
        {"link": None, "title": "Ignore", "position": 3, "date": None},
    ]

    monkeypatch.setattr(controller.core, "fetch_serpapi_url_list", lambda **_: mock_results, raising=True)
    monkeypatch.setattr(core, "fetch_serpapi_url_list", lambda **_: mock_results, raising=True)

    ret = controller.LandController.urlist(core.Namespace(
        name=name,
        query="gilets jaunes",
        lang=["fr"],
        datestart=None,
        dateend=None,
        timestep="week",
        sleep=0.0,
    ))
    assert ret == 1

    expressions = list(model.Expression.select().where(model.Expression.land == land))
    assert len(expressions) == 2

    updated_existing = model.Expression.get(model.Expression.url == existing_url)
    assert updated_existing.title == "Existing title"

    created = model.Expression.get(model.Expression.url == "https://newsite.com/story")
    assert created.title == "New title"


def test_land_urlist_engine_option(fresh_db, monkeypatch):
    controller = fresh_db["controller"]
    core = fresh_db["core"]

    name = rand_name("land")
    assert controller.LandController.create(core.Namespace(name=name, desc="d", lang=["fr"])) == 1

    monkeypatch.setattr(controller.settings, "serpapi_api_key", "TEST", raising=False)
    monkeypatch.setattr(core.settings, "serpapi_api_key", "TEST", raising=False)

    captured = {}

    def fake_fetch(**kwargs):
        captured["engine"] = kwargs.get("engine")
        return []

    monkeypatch.setattr(controller.core, "fetch_serpapi_url_list", fake_fetch, raising=True)
    monkeypatch.setattr(core, "fetch_serpapi_url_list", fake_fetch, raising=True)

    ret = controller.LandController.urlist(core.Namespace(
        name=name,
        query="bing test",
        lang=["fr"],
        engine="bing",
        datestart=None,
        dateend=None,
        timestep="week",
        sleep=0.0,
    ))
    assert ret == 1
    assert captured["engine"] == "bing"


def test_land_urlist_invalid_engine(fresh_db, monkeypatch):
    controller = fresh_db["controller"]
    core = fresh_db["core"]

    name = rand_name("land")
    assert controller.LandController.create(core.Namespace(name=name, desc="d", lang=["fr"])) == 1

    monkeypatch.setattr(controller.settings, "serpapi_api_key", "TEST", raising=False)
    monkeypatch.setattr(core.settings, "serpapi_api_key", "TEST", raising=False)

    ret = controller.LandController.urlist(core.Namespace(
        name=name,
        query="test",
        lang=["fr"],
        engine="altavista",
        datestart=None,
        dateend=None,
        timestep="week",
        sleep=0.0,
    ))
    assert ret == 0


def test_land_urlist_duckduckgo_date_filters(fresh_db, monkeypatch):
    controller = fresh_db["controller"]
    core = fresh_db["core"]

    name = rand_name("land")
    assert controller.LandController.create(core.Namespace(name=name, desc="d", lang=["fr"])) == 1

    monkeypatch.setattr(controller.settings, "serpapi_api_key", "TEST", raising=False)
    monkeypatch.setattr(core.settings, "serpapi_api_key", "TEST", raising=False)

    captured = {}

    def fake_fetch(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(controller.core, "fetch_serpapi_url_list", fake_fetch, raising=True)
    monkeypatch.setattr(core, "fetch_serpapi_url_list", fake_fetch, raising=True)

    ret = controller.LandController.urlist(core.Namespace(
        name=name,
        query="ddg test",
        lang=["fr"],
        engine="duckduckgo",
        datestart="2024-01-01",
        dateend="2024-01-31",
        timestep="week",
        sleep=0.0,
    ))
    assert ret == 1
    assert captured.get("engine") == "duckduckgo"
    assert captured.get("datestart") == "2024-01-01"
    assert captured.get("dateend") == "2024-01-31"


def test_fetch_serpapi_url_list_duckduckgo_month_windows(monkeypatch):
    from mwi import core as core_module

    captured_params = []

    class DummyResponse:
        def __init__(self, params):
            self.status_code = 200
            self.text = ''
            self._params = params

        def json(self):
            index = len(captured_params)
            return {
                'organic_results': [
                    {
                        'link': f'https://example.com/{index}',
                        'title': f'Result {index}',
                        'position': index + 1,
                        'date': None,
                    }
                ],
                'serpapi_pagination': {},
            }

    def fake_get(url, params, timeout):
        captured_params.append(dict(params))
        return DummyResponse(params)

    monkeypatch.setattr(core_module.requests, 'get', fake_get, raising=True)

    results = core_module.fetch_serpapi_url_list(
        api_key='TEST',
        query='duckduckgo windows',
        engine='duckduckgo',
        lang='fr',
        datestart='2024-01-01',
        dateend='2024-02-28',
        timestep='month',
        sleep_seconds=0.0,
    )

    assert len(results) == 2
    assert captured_params[0]['df'] == '2024-01-01..2024-01-31'
    assert captured_params[1]['df'] == '2024-02-01..2024-02-28'


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
