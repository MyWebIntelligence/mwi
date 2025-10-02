import json
import random
import string
from argparse import Namespace
from datetime import datetime
from types import SimpleNamespace

import pytest


def rand_name(prefix="land"):
    letters = string.ascii_lowercase
    return f"{prefix}_" + "".join(random.choice(letters) for _ in range(8))

# ---------------------------------------------------------------------------
# CLI dispatch & controller protocol tests
# ---------------------------------------------------------------------------


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

    ret_setup = controller.DbController.setup(core.Namespace())
    assert ret_setup == 1

    ret_migrate = controller.DbController.migrate(core.Namespace())
    assert ret_migrate == 1


def test_land_create_list_delete(fresh_db, capsys, monkeypatch):
    controller = fresh_db["controller"]
    core = fresh_db["core"]
    model = fresh_db["model"]

    name = rand_name("proj")
    ret = controller.LandController.create(core.Namespace(name=name, desc="desc", lang=["fr"]))
    assert ret == 1

    ret = controller.LandController.list(core.Namespace(name=None))
    assert ret == 1
    out = capsys.readouterr().out
    assert name in out

    monkeypatch.setattr(core, "confirm", lambda msg: True)
    ret = controller.LandController.delete(core.Namespace(name=name, maxrel=None))
    assert ret == 1
    assert model.Land.get_or_none(model.Land.name == name) is None


def test_land_addterm_addurl_and_crawl_readable_export(fresh_db, tmp_path, monkeypatch):
    controller = fresh_db["controller"]
    core = fresh_db["core"]
    model = fresh_db["model"]

    name = rand_name("land")
    assert controller.LandController.create(core.Namespace(name=name, desc="d", lang=["fr"])) == 1

    ret = controller.LandController.addterm(core.Namespace(land=name, terms="asthme, enfant, santé"))
    assert ret == 1
    land = model.Land.get(model.Land.name == name)
    assert model.LandDictionary.select().where(model.LandDictionary.land == land).count() == 3

    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://example.com/\nhttps://www.python.org/", encoding="utf-8")
    ret = controller.LandController.addurl(core.Namespace(land=name, path=str(urls_file), urls=None))
    assert ret == 1
    assert model.Expression.select().where(model.Expression.land == land).count() == 2

    async def _fake_crawl_land(land_obj, limit, http, depth):
        return (1, 0)

    monkeypatch.setattr(controller.core, "crawl_land", _fake_crawl_land)

    import mwi.readable_pipeline as readable_pipeline

    llm_seen = []

    async def _fake_readable_pipeline(land_obj, limit, depth, merge, llm_enabled):
        llm_seen.append(llm_enabled)
        return (0, 0)

    monkeypatch.setattr(readable_pipeline, "run_readable_pipeline", _fake_readable_pipeline)

    ret = controller.LandController.crawl(core.Namespace(name=name, limit=1, http=None, depth=None))
    assert ret == 1

    ret = controller.LandController.readable(core.Namespace(name=name, limit=1, depth=None, merge="smart_merge"))
    assert ret == 1 and llm_seen[-1] is False

    ret = controller.LandController.readable(core.Namespace(name=name, limit=1, depth=None, merge="smart_merge", llm="true"))
    assert ret == 1 and llm_seen[-1] is True

    calls = {"count": 0}

    def fake_export_land(land_obj, export_type, minrel):
        calls["count"] += 1
        return None

    monkeypatch.setattr(controller.core, "export_land", fake_export_land)

    export_types = [
        "pagecsv",
        "fullpagecsv",
        "pagegexf",
        "nodegexf",
        "nodecsv",
        "mediacsv",
        "corpus",
        "pseudolinks",
    ]
    for export_type in export_types:
        ret = controller.LandController.export(core.Namespace(name=name, type=export_type, minrel=1))
        assert ret == 1

    assert calls["count"] == len(export_types)


def test_land_consolidate_and_medianalyse(fresh_db, monkeypatch):
    controller = fresh_db["controller"]
    core = fresh_db["core"]

    name = rand_name("land")
    assert controller.LandController.create(core.Namespace(name=name, desc="d", lang=["fr"])) == 1

    async def _fake_consolidate_land(land_obj, limit, depth, min_rel):
        return (2, 0)

    async def _fake_medianalyse_land(land_obj):
        return {"processed": 0}

    monkeypatch.setattr(controller.core, "consolidate_land", _fake_consolidate_land)
    monkeypatch.setattr(controller.core, "medianalyse_land", _fake_medianalyse_land)

    ret = controller.LandController.consolidate(core.Namespace(name=name, limit=0, depth=None, minrel=0))
    assert ret == 1

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
    expression = model.Expression.create(
        land=land,
        domain=domain,
        url="https://example.com/",
        http_status="200",
        relevance=2,
    )
    model.Expression.create(
        land=land,
        domain=domain,
        url="https://example.com/404",
        http_status="404",
        relevance=5,
    )
    model.Expression.create(
        land=land,
        domain=domain,
        url="https://example.com/lowrel",
        http_status="200",
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

    calls = {"count": 0}

    def fake_fetch(url, key):
        calls["count"] += 1
        return {"url": url, "score": 42}

    monkeypatch.setattr(controller.core, "fetch_seorank_for_url", fake_fetch)
    monkeypatch.setattr(core, "fetch_seorank_for_url", fake_fetch)

    ret = controller.LandController.seorank(core.Namespace(name=name, limit=0, depth=None, force=False, http=None, minrel=None))
    assert ret == 1
    updated = model.Expression.get(model.Expression.id == expression.id)
    assert json.loads(updated.seorank)["score"] == 42

    ret = controller.LandController.seorank(core.Namespace(name=name, limit=0, depth=None, force=False, http=None, minrel=None))
    assert ret == 1
    assert calls["count"] == 1


def test_land_urlist_cli(fresh_db, monkeypatch):
    controller = fresh_db["controller"]
    core = fresh_db["core"]
    model = fresh_db["model"]

    name = rand_name("land")
    assert controller.LandController.create(core.Namespace(name=name, desc="d", lang=["fr"])) == 1
    land = model.Land.get(model.Land.name == name)

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

    monkeypatch.setattr(controller.settings, "serpapi_api_key", "TEST", raising=False)
    monkeypatch.setattr(core.settings, "serpapi_api_key", "TEST", raising=False)

    existing_url = "https://existing.com/article"
    existing = controller.core.add_expression(land, existing_url)
    assert existing
    existing.title = None
    existing.save()

    mock_results = [
        {"link": existing_url, "title": "Existing title", "position": 1, "date": "2024-01-01"},
        {"link": "https://newsite.com/story", "title": "New title", "position": 2, "date": "2024-01-02"},
        {"link": None, "title": "Ignore", "position": 3, "date": None},
    ]

    monkeypatch.setattr(controller.core, "fetch_serpapi_url_list", lambda **_: mock_results)
    monkeypatch.setattr(core, "fetch_serpapi_url_list", lambda **_: mock_results)

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
        captured.update(kwargs)
        return []

    monkeypatch.setattr(controller.core, "fetch_serpapi_url_list", fake_fetch)
    monkeypatch.setattr(core, "fetch_serpapi_url_list", fake_fetch)

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
    assert captured.get("engine") == "bing"


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

    monkeypatch.setattr(controller.core, "fetch_serpapi_url_list", fake_fetch)
    monkeypatch.setattr(core, "fetch_serpapi_url_list", fake_fetch)

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


def test_domain_crawl_cli(fresh_db, monkeypatch):
    controller = fresh_db["controller"]
    core = fresh_db["core"]

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

    calls = {"count": 0}

    def fake_export_tags(land_obj, export_type, minrel):
        calls["count"] += 1
        return None

    monkeypatch.setattr(controller.core, "export_tags", fake_export_tags)

    for export_type in ["matrix", "content"]:
        ret = controller.TagController.export(core.Namespace(name=name, type=export_type, minrel=0))
        assert ret == 1

    assert calls["count"] == 2


def test_heuristic_update_cli(fresh_db, monkeypatch):
    controller = fresh_db["controller"]
    core = fresh_db["core"]

    called = {"ok": False}

    def fake_update():
        called["ok"] = True

    monkeypatch.setattr(controller.core, "update_heuristic", fake_update)

    ret = controller.HeuristicController.update(core.Namespace())
    assert ret == 1 and called["ok"]


def test_cli_get_arg_helpers(test_env):
    core = test_env["core"]
    ns = Namespace(option_str=None, option_int=None)
    assert core.get_arg_option("option_str", ns, set_type=str, default="A") == "A"
    assert core.get_arg_option("option_int", ns, set_type=int, default=5) == 5

    ns = Namespace(option_str=503, option_int="3")
    assert core.get_arg_option("option_str", ns, set_type=str, default="A") == "503"
    assert core.get_arg_option("option_int", ns, set_type=int, default=5) == 3

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

    controller.settings.openrouter_enabled = True
    controller.settings.openrouter_api_key = "sk-test"
    controller.settings.openrouter_model = "openai/gpt-4o-mini"

    name = rand_name("llm")
    assert controller.LandController.create(core.Namespace(name=name, desc="d", lang=["fr"])) == 1
    land = model.Land.get(model.Land.name == name)

    minlen = getattr(controller.settings, "openrouter_readable_min_chars", 0)
    for i in range(3):
        expression = core.add_expression(land, f"https://example.com/p{i}")
        assert expression
        expression.readable = "x" * max(80, minlen + 1)
        expression.save()

    responses = iter(["oui", "non", "oui"])

    def fake_is_relevant(land_obj, expression):
        return next(responses, "oui") == "oui"

    monkeypatch.setattr(core, "is_relevant_via_openrouter", fake_is_relevant, raising=False)

    result = cli.dispatch(Namespace(object="land", verb="llm", subverb="validate", name=name, limit=10))
    assert result == 1

# ---------------------------------------------------------------------------
# Core helper functions coverage
# ---------------------------------------------------------------------------


def test_core_nltk_ensure_tokenizers(monkeypatch, test_env):
    core = test_env["core"]

    calls = []

    class DummyData:
        path = []

        def find(self, resource):
            calls.append(("find", resource))
            raise LookupError

    dummy_data = DummyData()

    def fake_download(resource, quiet=True):
        calls.append(("download", resource, quiet))
        return True

    monkeypatch.setattr(core.nltk, "data", dummy_data)
    monkeypatch.setattr(core.nltk, "download", fake_download)
    monkeypatch.setattr(core.os, "makedirs", lambda *args, **kwargs: None)

    ok = core._ensure_nltk_tokenizers()
    assert ok is True
    assert any(step[0] == "download" for step in calls)


def test_core_simple_word_tokenize_and_stem(test_env):
    core = test_env["core"]

    tokens = core._simple_word_tokenize("Bonjour! Les enfants jouent à l'école 123.")
    assert tokens == ["bonjour", "les", "enfants", "jouent", "à", "l", "école"]

    stem = core.stem_word("enfants")
    assert stem.startswith("enfant")


def test_core_url_and_domain_helpers(monkeypatch, test_env):
    core = test_env["core"]
    settings = test_env["controller"].settings

    url = core.resolve_url("https://example.com/path/", "../asset.png")
    assert url == "https://example.com/asset.png"

    assert core.remove_anchor("https://example.com/page#section") == "https://example.com/page"

    settings.heuristics = {
        "twitter.com": r"twitter.com/(.*)"
    }
    domain = core.get_domain_name("https://twitter.com/example/status/1")
    assert "example" in domain

    assert core.is_crawlable("https://example.com/page")
    assert not core.is_crawlable("mailto:test@example.com")
    assert not core.is_crawlable("https://example.com/file.pdf")


def test_core_split_and_args_helpers(test_env):
    core = test_env["core"]
    assert core.split_arg("a, b ,c") == ["a", "b", "c"]

    ns = Namespace(foo="123", bar=None)
    assert core.get_arg_option("foo", ns, set_type=int, default=0) == 123
    assert core.get_arg_option("bar", ns, set_type=str, default="x") == "x"

    with pytest.raises(Exception):
        core.check_args(Namespace(a=1), ["a", "b"])


def test_core_confirm(monkeypatch, test_env):
    core = test_env["core"]

    monkeypatch.setattr("builtins.input", lambda prompt="": "Y")
    assert core.confirm("Proceed?") is True

    monkeypatch.setattr("builtins.input", lambda prompt="": "no")
    assert core.confirm("Proceed?") is False


def test_core_resolve_url_relative(test_env):
    core = test_env["core"]
    assert core.resolve_url("https://example.org/foo/", "bar") == "https://example.org/foo/bar"
    assert core.resolve_url("https://example.org/foo/", "http://external.com/") == "http://external.com/"

# ---------------------------------------------------------------------------
# SERP API helper coverage
# ---------------------------------------------------------------------------


def test_core_serpapi_helpers(test_env):
    core = test_env["core"]

    page_google = core._serpapi_page_size("google")
    page_bing = core._serpapi_page_size("bing")
    page_ddg = core._serpapi_page_size("duckduckgo")

    assert page_google == 100
    assert page_bing == 50
    assert page_ddg == 50

    start = core._parse_serpapi_date("2024-01-01")
    end = core._advance_date(start, "week")
    assert end == start + core.timedelta(days=7)

    windows = list(core._build_serpapi_windows("2024-01-01", "2024-01-31", "week"))
    assert windows[0][0] == core.date(2024, 1, 1)
    assert windows[-1][1] == core.date(2024, 1, 31)

    tbs = core._build_serpapi_tbs(core.date(2024, 1, 1), core.date(2024, 1, 7))
    assert "cdr" in tbs and "cd_min" in tbs

    assert core._serpapi_google_domain("fr") == "google.fr"
    assert core._serpapi_bing_market("fr") == "fr-FR"
    assert core._serpapi_duckduckgo_region("fr") == "fr-fr"


def test_core_serpapi_param_builder(monkeypatch, test_env):
    core = test_env["core"]

    params = core._build_serpapi_params(
        engine="google",
        lang="fr",
        start_index=0,
        page_size=100,
        window_start=core.date(2024, 1, 1),
        window_end=core.date(2024, 1, 7),
        use_date_filter=False,
    )
    assert params["google_domain"] == "google.fr"
    assert params["num"] == 100
    assert params["start"] == 0

    params_duck = core._build_serpapi_params(
        engine="duckduckgo",
        lang="fr",
        start_index=5,
        page_size=50,
        window_start=core.date(2024, 1, 1),
        window_end=core.date(2024, 1, 7),
        use_date_filter=True,
    )
    assert params_duck["start"] == 5
    assert params_duck["m"] == 50
    assert "df" in params_duck


def test_core_serpapi_date_parsing(test_env):
    core = test_env["core"]

    dt = core.parse_serp_result_date("2024-01-07")
    assert dt.year == 2024

    alt = core.parse_serp_result_date("January 07, 2024")
    assert alt.year == 2024

    earlier = core.prefer_earlier_datetime(None, dt)
    assert earlier == dt
    earlier = core.prefer_earlier_datetime(dt, alt)
    assert earlier == min(dt, alt)


def test_core_fetch_serpapi_url_list(monkeypatch, test_env):
    core = test_env["core"]

    class DummyResponse:
        def __init__(self, payload):
            self.status_code = 200
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self):
            return self._payload

    captured = []

    def fake_get(url, params=None, timeout=None):
        captured.append((url, params, timeout))
        payload = {
            "organic_results": [
                {"position": 1, "title": "Result", "link": "https://example.com", "date": "2024-01-01"}
            ],
            "serpapi_pagination": {},
        }
        return DummyResponse(payload)

    monkeypatch.setattr(core.requests, "get", fake_get)

    results = core.fetch_serpapi_url_list(
        api_key="key",
        query="hello",
        lang="fr",
        datestart="2024-01-01",
        dateend="2024-01-07",
        timestep="week",
        sleep_seconds=0.0,
    )

    assert len(results) == 1
    assert results[0]["link"] == "https://example.com"
    params = captured[0][1]
    assert params["engine"] == "google"
    assert params["start"] == 0
    if "num" in params:
        assert params["num"] == 100


def test_fetch_serpapi_url_list_duckduckgo_month_windows(monkeypatch, test_env):
    core = test_env["core"]

    captured_params = []

    class DummyResponse:
        def __init__(self, params):
            self.status_code = 200
            self._params = params
            self.text = "{}"

        def json(self):
            index = len(captured_params)
            return {
                "organic_results": [
                    {
                        "link": f"https://example.com/{index}",
                        "title": f"Result {index}",
                        "position": index + 1,
                        "date": None,
                    }
                ],
                "serpapi_pagination": {},
            }

    def fake_get(url, params=None, timeout=None):
        captured_params.append(dict(params))
        return DummyResponse(params)

    monkeypatch.setattr(core.requests, "get", fake_get)

    results = core.fetch_serpapi_url_list(
        api_key="TEST",
        query="duckduckgo windows",
        engine="duckduckgo",
        lang="fr",
        datestart="2024-01-01",
        dateend="2024-02-28",
        timestep="month",
        sleep_seconds=0.0,
    )

    assert len(results) == 2
    assert captured_params[0]["df"] == "2024-01-01..2024-01-31"
    assert captured_params[1]["df"] == "2024-02-01..2024-02-28"


def test_fetch_serpapi_url_list_google_no_date(monkeypatch, test_env):
    core = test_env["core"]

    class DummyResponse:
        def __init__(self, payload):
            self.status_code = 200
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self):
            return self._payload

    captured_params = []

    def fake_get(url, params=None, timeout=None):
        captured_params.append(dict(params))
        return DummyResponse({
            "organic_results": [
                {"link": "https://example.com/a", "title": "A", "position": 1, "date": None}
            ],
            "serpapi_pagination": {},
        })

    monkeypatch.setattr(core.requests, "get", fake_get)

    results = core.fetch_serpapi_url_list(
        api_key="TEST",
        query="google",
        engine="google",
        lang="fr",
        datestart=None,
        dateend=None,
        timestep="week",
        sleep_seconds=0.0,
    )

    assert len(results) == 1
    assert captured_params[0]["num"] == 100

# ---------------------------------------------------------------------------
# SEO Rank helpers
# ---------------------------------------------------------------------------


def test_core_fetch_seorank_for_url(monkeypatch, test_env):
    core = test_env["core"]

    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def json(self):
            return self._payload

    def fake_get(url, headers=None, timeout=None):
        return DummyResponse({"score": 10, "url": url})

    monkeypatch.setattr(core.requests, "get", fake_get)

    payload = core.fetch_seorank_for_url("https://example.com", api_key="XYZ")
    assert payload["score"] == 10


def test_core_update_seorank_for_land(monkeypatch, fresh_db):
    core = fresh_db["core"]
    controller = fresh_db["controller"]
    model = fresh_db["model"]

    name = rand_name("land")
    controller.LandController.create(core.Namespace(name=name, desc="d", lang=["fr"]))
    land = model.Land.get(model.Land.name == name)
    domain = model.Domain.create(name="example.org")
    expr = model.Expression.create(land=land, domain=domain, url="https://example.org/", relevance=1, http_status="200")

    monkeypatch.setattr(core, "fetch_seorank_for_url", lambda url, api_key: {"score": 99})

    processed, updated = core.update_seorank_for_land(
        land,
        api_key="TEST",
        limit=10,
        http_status='200',
        min_relevance=0,
        force_refresh=True,
    )
    assert processed == 1 and updated == 1
    refreshed = model.Expression.get_by_id(expr.id)
    assert json.loads(refreshed.seorank)["score"] == 99

# ---------------------------------------------------------------------------
# Domain metadata helpers
# ---------------------------------------------------------------------------


def test_core_process_domain_content(monkeypatch, fresh_db):
    core = fresh_db["core"]
    model = fresh_db["model"]

    domain = model.Domain.create(name="example.com")

    html = """
    <html>
      <head>
        <title>Example Domain</title>
        <meta name="description" content="Demo description" />
        <meta name="keywords" content="demo, example" />
      </head>
      <body>Content</body>
    </html>
    """

    class DummyMeta:
        def __init__(self):
            self.title = "Meta Title"
            self.description = "Meta description"
            self.tags = ["tag1", "tag2"]

    monkeypatch.setattr(core.trafilatura, "extract_metadata", lambda html_content: DummyMeta())

    core.process_domain_content(domain, html, "https://example.com", "REQUESTS")
    assert domain.title == "Meta Title"
    assert "Meta description" in domain.description
    assert "tag1" in domain.keywords


def test_core_extract_metadata(monkeypatch, test_env):
    core = test_env["core"]

    class DummyResponse:
        def __init__(self):
            self.text = """
            <html>
              <head>
                <title>Example</title>
                <meta name='description' content='Desc' />
                <meta name='keywords' content='k1, k2' />
              </head>
            </html>
            """

        def raise_for_status(self):
            return None

    monkeypatch.setattr(core.requests, "get", lambda url, headers=None, timeout=None: DummyResponse())

    meta = core.extract_metadata("example.com")
    assert meta["title"] == "Example"
    assert meta["description"] == "Desc"
    assert "k1" in meta["keywords"]


def test_core_meta_helpers(test_env):
    core = test_env["core"]
    soup = core.BeautifulSoup(
        """
        <html>
          <head>
            <meta property="og:title" content="OG Title" />
            <meta property="og:description" content="OG Desc" />
            <meta property="og:keywords" content="og, keywords" />
            <meta name="twitter:keywords" content="tw" />
          </head>
        </html>
        """,
        "html.parser",
    )
    assert core.get_title(soup) == "OG Title"
    assert core.get_description(soup) == "OG Desc"
    assert "og" in core.get_keywords(soup)

    plain = core.BeautifulSoup("<meta name='description' content='Simple' />", "html.parser")
    assert core.get_meta_content(plain, "description") == "Simple"


def test_core_extract_md_links(test_env):
    core = test_env["core"]
    urls = core.extract_md_links("See [link](https://example.com/path) and (https://foo.com/path)")
    assert "https://example.com/path" in urls


def test_core_add_expression_and_link(fresh_db):
    core = fresh_db["core"]
    controller = fresh_db["controller"]
    model = fresh_db["model"]

    name = rand_name("land")
    controller.LandController.create(core.Namespace(name=name, desc="d", lang=["fr"]))
    land = model.Land.get(model.Land.name == name)

    expr = core.add_expression(land, "https://example.com/a")
    assert expr is not False

    linked = core.link_expression(land, expr, "https://example.com/b")
    assert linked is True
    assert model.ExpressionLink.select().count() == 1

# ---------------------------------------------------------------------------
# Expression processing & relevance
# ---------------------------------------------------------------------------


def _create_land_with_terms(controller, core, model, name=None, terms=None):
    land_name = name or rand_name("land")
    controller.LandController.create(core.Namespace(name=land_name, desc="d", lang=["fr"]))
    if terms:
        controller.LandController.addterm(core.Namespace(land=land_name, terms=", ".join(terms)))
    return model.Land.get(model.Land.name == land_name)


def test_core_process_expression_content(monkeypatch, fresh_db):
    core = fresh_db["core"]
    controller = fresh_db["controller"]
    model = fresh_db["model"]

    land = _create_land_with_terms(controller, core, model, terms=["science", "enfant"])
    expression = core.add_expression(land, "https://example.com/article")
    assert expression
    dictionary = core.get_land_dictionary(land)

    html = """
    <html lang="fr">
      <head>
        <title>Science news</title>
        <meta name="description" content="Desc" />
      </head>
      <body>
        <p>Science pour enfant.</p>
        <img src="/img.png" />
        <a href="https://example.com/other">Other</a>
      </body>
    </html>
    """

    monkeypatch.setattr(core, "extract_metadata", lambda url: {"title": "Better title", "description": "Better", "keywords": "science"})
    monkeypatch.setattr(core.settings, "archive", False, raising=False)
    monkeypatch.setattr(core.settings, "openrouter_enabled", False, raising=False)

    processed = core.process_expression_content(expression, html, dictionary)
    assert processed.title == "Better title"
    assert processed.lang == "fr"
    assert processed.relevance >= 1
    assert model.Media.select().where(model.Media.expression == processed).count() >= 1


def test_core_extract_medias_html_and_markdown(fresh_db):
    core = fresh_db["core"]
    controller = fresh_db["controller"]
    model = fresh_db["model"]

    land = _create_land_with_terms(controller, core, model)
    expression = core.add_expression(land, "https://example.com/page")
    assert expression

    html = core.BeautifulSoup(
        """
        <div>
          <img src="/img.jpg" />
          <video src="/movie.mp4"></video>
          <audio src="https://example.com/audio.mp3"></audio>
        </div>
        """,
        "html.parser",
    )
    core.extract_medias(html, expression)

    markdown = "![img](https://example.com/other.png)\n[VIDEO: https://example.com/mov.mp4]"
    core.extract_medias(markdown, expression)

    medias = list(model.Media.select().where(model.Media.expression == expression))
    assert len(medias) >= 3


def test_core_get_readable_and_clean_html(test_env):
    core = test_env["core"]
    soup = core.BeautifulSoup("<html><body><script>bad()</script><p>Hello</p></body></html>", "html.parser")
    core.clean_html(soup)
    text = core.get_readable(soup)
    assert "Hello" in text and "bad" not in text


def test_core_dictionary_and_relevance(fresh_db):
    core = fresh_db["core"]
    controller = fresh_db["controller"]
    model = fresh_db["model"]

    land = _create_land_with_terms(controller, core, model, terms=["science", "enfant"])
    expression = core.add_expression(land, "https://example.com/article")
    assert expression
    expression.title = "Science enfant"
    expression.description = ""
    expression.content = "Science et enfant"
    expression.readable = expression.content
    expression.save()

    dictionary = list(core.get_land_dictionary(land))
    assert any(word.lemma.startswith("scien") for word in dictionary)

    relevance = core.expression_relevance(dictionary, expression)
    assert relevance >= 2

    core.land_relevance(land)
    refreshed = model.Expression.get_by_id(expression.id)
    assert refreshed.relevance >= relevance

# ---------------------------------------------------------------------------
# Export and heuristic utilities
# ---------------------------------------------------------------------------


def test_core_export_land_and_tags(monkeypatch, fresh_db):
    core = fresh_db["core"]
    controller = fresh_db["controller"]
    model = fresh_db["model"]

    name = rand_name("land")
    controller.LandController.create(core.Namespace(name=name, desc="d", lang=["fr"]))
    land = model.Land.get(model.Land.name == name)

    captured = {"land": None, "type": None, "minrel": None, "tags": 0}

    class DummyExport:
        def __init__(self, export_type, land_obj, minimum):
            captured["land"] = land_obj
            captured["type"] = export_type
            captured["minrel"] = minimum

        def write(self, export_type, filename):
            captured["written"] = export_type
            return 1

        def export_tags(self, filename):
            captured["tags"] += 1
            return 1

    monkeypatch.setattr(core, "Export", DummyExport)

    core.export_land(land, "pagecsv", 0)
    core.export_tags(land, "matrix", 0)

    assert captured["land"] == land
    assert captured.get("written") == "pagecsv"
    assert captured["type"] == "matrix"
    assert captured["tags"] == 1


def test_core_update_heuristic_logic(monkeypatch, fresh_db):
    core = fresh_db["core"]
    controller = fresh_db["controller"]
    model = fresh_db["model"]

    name = rand_name("land")
    controller.LandController.create(core.Namespace(name=name, desc="d", lang=["fr"]))
    land = model.Land.get(model.Land.name == name)
    expression = core.add_expression(land, "https://twitter.com/example/status/1")
    assert expression

    controller.settings.heuristics = {"twitter.com": r"twitter.com/(.*?)/"}
    core.settings.heuristics = controller.settings.heuristics

    core.update_heuristic()

    refreshed = model.Expression.get_by_id(expression.id)
    assert "example" in refreshed.domain.name


def test_core_delete_media(monkeypatch, fresh_db):
    core = fresh_db["core"]
    controller = fresh_db["controller"]
    model = fresh_db["model"]

    land = _create_land_with_terms(controller, core, model)
    expression = core.add_expression(land, "https://example.com/page")
    assert expression
    model.Media.create(expression=expression, url="https://example.com/img.jpg", type="img")

    called = {"where": False}

    class DummyDelete:
        def where(self, expr):
            called["where"] = True
            return self

        def execute(self):
            called["executed"] = True
            return 1

    monkeypatch.setattr(core.model.Media, "delete", lambda: DummyDelete())

    core.delete_media(land)
    assert called["where"] is True

# ---------------------------------------------------------------------------
# Asynchronous pipelines coverage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_core_extract_dynamic_medias(monkeypatch, fresh_db):
    core = fresh_db["core"]
    controller = fresh_db["controller"]
    model = fresh_db["model"]

    land = _create_land_with_terms(controller, core, model)
    expression = core.add_expression(land, "https://example.com/page")
    assert expression

    monkeypatch.setattr(core, "PLAYWRIGHT_AVAILABLE", False)
    medias = await core.extract_dynamic_medias("https://example.com", expression)
    assert medias == []

    monkeypatch.setattr(core, "PLAYWRIGHT_AVAILABLE", True)

    class DummyElement:
        def __init__(self, attrs):
            self.attrs = attrs

        async def get_attribute(self, name):
            return self.attrs.get(name)

    class DummyPage:
        def __init__(self):
            self.closed = False

        async def set_extra_http_headers(self, headers):
            return None

        async def goto(self, url, wait_until=None, timeout=None):
            return None

        async def wait_for_timeout(self, delay):
            return None

        async def query_selector_all(self, selector):
            if selector == 'img[src]':
                return [DummyElement({'src': '/img.png'})]
            if selector == 'video[src], video source[src]':
                return [DummyElement({'src': 'https://example.com/video.mp4'})]
            if selector == 'audio[src], audio source[src]':
                return []
            if selector in {'img[data-src]', 'img[data-lazy-src]', 'img[data-original]', 'img[data-url]'}:
                return [DummyElement({'data-src': '/lazy.jpg'})]
            return []

        async def close(self):
            self.closed = True

    class DummyBrowser:
        async def new_page(self):
            return DummyPage()

        async def close(self):
            return None

    class DummyChromium:
        async def launch(self, headless=True):
            return DummyBrowser()

    class DummyContext:
        async def __aenter__(self):
            return SimpleNamespace(chromium=DummyChromium())

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(core, "async_playwright", lambda: DummyContext())

    medias = await core.extract_dynamic_medias("https://example.com", expression)
    assert any(item.endswith(".png") for item in medias)
    assert model.Media.select().where(model.Media.expression == expression).count() >= 1


class DummyAiohttpResponse:
    def __init__(self, status=200, text="", headers=None):
        self.status = status
        self._text = text
        self.headers = headers or {"content-type": "text/html"}

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummySession:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, *args, **kwargs):
        return self._response


@pytest.mark.asyncio
async def test_core_crawl_expression_with_media_analysis(monkeypatch, fresh_db):
    core = fresh_db["core"]
    controller = fresh_db["controller"]
    model = fresh_db["model"]

    land = _create_land_with_terms(controller, core, model, terms=["science"])
    expression = core.add_expression(land, "https://example.com/article")
    assert expression
    dictionary = core.get_land_dictionary(land)

    html = """
    <html><body><p>Science everywhere</p><img src="/img.jpg" /></body></html>
    """

    dummy_response = DummyAiohttpResponse(status=200, text=html)
    session = DummySession(dummy_response)

    def fake_extract(html_text, include_links=True, include_comments=False, include_images=True, output_format='markdown'):
        if output_format == 'markdown':
            return "Science article " * 10 + "\n![IMAGE](https://example.com/pic.jpg)"
        return "<html><body><img src=\"https://example.com/pic.jpg\" /></body></html>"

    monkeypatch.setattr(core.trafilatura, "extract", fake_extract)
    monkeypatch.setattr(core.settings, "media_analysis", False, raising=False)
    monkeypatch.setattr(core.settings, "user_agent", "pytest-agent", raising=False)

    class DummyRequestsResp:
        def __init__(self):
            self.status_code = 200
            self.url = "https://web.archive.org/example"
            self.headers = {"Content-Type": "text/html"}
            self.text = "<html></html>"

        def raise_for_status(self):
            return None

        def json(self):
            return {"archived_snapshots": {}}

        @property
        def ok(self):
            return True

    monkeypatch.setattr(core.requests, "get", lambda *args, **kwargs: DummyRequestsResp())

    processed = await core.crawl_expression_with_media_analysis(expression, dictionary, session)
    assert processed == 1
    refreshed = model.Expression.get_by_id(expression.id)
    assert refreshed.relevance >= 1
    assert refreshed.readable


@pytest.mark.asyncio
async def test_core_crawl_expression(monkeypatch, fresh_db):
    core = fresh_db["core"]
    controller = fresh_db["controller"]
    model = fresh_db["model"]

    land = _create_land_with_terms(controller, core, model, terms=["science"])
    expression = core.add_expression(land, "https://example.com/page")
    assert expression
    dictionary = core.get_land_dictionary(land)

    html = "<html><body><p>Science</p></body></html>"
    dummy_response = DummyAiohttpResponse(status=200, text=html)
    session = DummySession(dummy_response)

    monkeypatch.setattr(core.trafilatura, "extract", lambda *args, **kwargs: None)
    monkeypatch.setattr(core.settings, "media_analysis", False, raising=False)

    result = await core.crawl_expression(expression, dictionary, session)
    assert result in (0, 1)
    refreshed = model.Expression.get_by_id(expression.id)
    assert refreshed.http_status in ("200", "ERR", "000")


@pytest.mark.asyncio
async def test_core_crawl_land(monkeypatch, fresh_db):
    core = fresh_db["core"]
    controller = fresh_db["controller"]
    model = fresh_db["model"]

    land = _create_land_with_terms(controller, core, model)
    for i in range(3):
        expr = core.add_expression(land, f"https://example.com/p{i}")
        assert expr
        expr.depth = i % 2
        expr.save()

    async def fake_worker(expression, dictionary, session):
        return 1

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(core, "crawl_expression_with_media_analysis", fake_worker)
    monkeypatch.setattr(core.aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(core.settings, "parallel_connections", 2, raising=False)

    processed, errors = await core.crawl_land(land, limit=10, http=None, depth=None)
    assert processed + errors > 0


@pytest.mark.asyncio
async def test_core_consolidate_land(monkeypatch, fresh_db):
    core = fresh_db["core"]
    controller = fresh_db["controller"]
    model = fresh_db["model"]

    land = _create_land_with_terms(controller, core, model, terms=["science"])
    expression = core.add_expression(land, "https://example.com/a")
    assert expression
    expression.readable = "[Link](https://example.com/b)"
    expression.readable_at = core.model.datetime.datetime.now()
    expression.save()

    monkeypatch.setattr(core.settings, "parallel_connections", 1, raising=False)

    processed, errors = await core.consolidate_land(land, limit=10, depth=None, min_relevance=0)
    assert processed >= 1
    assert errors == 0


@pytest.mark.asyncio
async def test_core_analyze_media(monkeypatch, fresh_db):
    core = fresh_db["core"]
    controller = fresh_db["controller"]
    model = fresh_db["model"]

    land = _create_land_with_terms(controller, core, model)
    expression = core.add_expression(land, "https://example.com/a")
    assert expression
    media = model.Media.create(expression=expression, url="https://example.com/img.jpg", type="img")

    class DummyAnalyzer:
        def __init__(self, *args, **kwargs):
            pass

        async def analyze_image(self, url):
            return {
                "width": 640,
                "height": 480,
                "file_size": 12345,
                "format": "JPEG",
                "color_mode": "RGB",
                "dominant_colors": [(255, 255, 255)],
                "has_transparency": False,
                "aspect_ratio": 1.3333,
                "exif_data": {"camera": "pytest"},
                "image_hash": "hash",
                "content_tags": ["tag"],
                "nsfw_score": 0.1,
            }

    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    from mwi import media_analyzer as media_module

    monkeypatch.setattr(media_module, "MediaAnalyzer", DummyAnalyzer)

    session = DummySession()
    analyzed = await core.analyze_media(expression, session)
    assert media.url in analyzed
    refreshed = model.Media.get_by_id(media.id)
    assert refreshed.width == 640


@pytest.mark.asyncio
async def test_core_medianalyse_land(monkeypatch, fresh_db):
    core = fresh_db["core"]
    controller = fresh_db["controller"]
    model = fresh_db["model"]

    land = _create_land_with_terms(controller, core, model)
    expression = core.add_expression(land, "https://example.com/a")
    assert expression
    media = model.Media.create(expression=expression, url="https://example.com/img.jpg", type="img")

    class DummyAnalyzer:
        def __init__(self, session, settings):
            self.session = session
            self.settings = settings

        async def analyze_image(self, url):
            return {"width": 100, "height": 100, "file_size": 1000, "format": "PNG"}

    class DummySession:
        async def __aenter__(self):
            return SimpleNamespace()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    from mwi import media_analyzer as media_module

    monkeypatch.setattr(core.aiohttp, "ClientSession", lambda **kwargs: DummySession())
    monkeypatch.setattr(media_module, "MediaAnalyzer", DummyAnalyzer)

    result = await core.medianalyse_land(land)
    assert result["processed"] >= 1
    refreshed = model.Media.get_by_id(media.id)
    assert refreshed.width == 100

def test_core_crawl_domains(monkeypatch, fresh_db):
    core = fresh_db["core"]
    model = fresh_db["model"]

    domain = model.Domain.create(name="example.com")

    monkeypatch.setattr(core.trafilatura, "fetch_url", lambda url: "<html><head><title>Title</title></head></html>")

    class DummyResponse:
        def __init__(self, url):
            self.url = url
            self.status_code = 200
            self.headers = {"Content-Type": "text/html"}
            self._text = "<html><head><title>Title</title></head></html>"

        def raise_for_status(self):
            return None

        @property
        def ok(self):
            return True

        @property
        def text(self):
            return self._text

    monkeypatch.setattr(core.requests, "get", lambda url, **kwargs: DummyResponse(url))

    captured = {"processed": 0}

    def fake_process(domain_obj, html, effective_url, source_method):
        captured["processed"] += 1
        domain_obj.title = "Title"
        domain_obj.description = "Desc"
        domain_obj.keywords = "kw"

    monkeypatch.setattr(core, "process_domain_content", fake_process)

    processed = core.crawl_domains(limit=1, http=None)
    assert processed == 1
    assert captured["processed"] == 1
    refreshed = model.Domain.get_by_id(domain.id)
    assert refreshed.http_status == "200"
