import pytest

from mwi import core


class DummyResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or ""

    def json(self):
        return self._payload


def test_fetch_serpapi_url_list_requires_query():
    with pytest.raises(core.SerpApiError):
        core.fetch_serpapi_url_list(api_key="key", query="")


def test_fetch_serpapi_url_list_handles_http_error(monkeypatch):
    def fake_get(*args, **kwargs):
        return DummyResponse(status_code=500, text="boom")

    monkeypatch.setattr(core.requests, "get", fake_get)

    with pytest.raises(core.SerpApiError) as exc:
        core.fetch_serpapi_url_list(api_key="key", query="smart", sleep_seconds=0.0)
    assert "status 500" in str(exc.value)


def test_fetch_serpapi_url_list_builds_params(monkeypatch):
    captured = []

    def fake_get(url, params=None, timeout=None):
        captured.append({"url": url, "params": params, "timeout": timeout})
        payload = {
            "organic_results": [
                {"position": 1, "title": "Result", "link": "https://example.com", "date": "2024-01-01"}
            ],
            "serpapi_pagination": {},
        }
        return DummyResponse(status_code=200, payload=payload, text="{}")

    monkeypatch.setattr(core.requests, "get", fake_get)

    results = core.fetch_serpapi_url_list(
        api_key="key",
        query="Test Query",
        lang="fr",
        datestart="2024-01-01",
        dateend="2024-01-07",
        timestep="week",
        sleep_seconds=0.0,
    )

    assert len(results) == 1
    assert results[0]["link"] == "https://example.com"

    assert len(captured) == 1
    params = captured[0]["params"]
    assert params["tbs"] == "cdr:1,cd_min:01/01/2024,cd_max:01/07/2024"
    assert params["gl"] == "fr"
    assert params["num"] == 100
    assert params["start"] == 0
    assert captured[0]["timeout"] == getattr(core.settings, "serpapi_timeout", 15)
