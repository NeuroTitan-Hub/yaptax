"""End-to-end tests: the router in front of the mock provider, both in-process, no network."""

import json

import httpx
import pytest

from yaptax.router import config
from yaptax.router.mock import create_mock
from yaptax.router.server import create_app
from yaptax.router.store import Store

HINDI = "मेरा ऑर्डर कहाँ है? मुझे अपने पैसे वापस चाहिए।"
AMHARIC = "ትዕዛዜ የት ነው? ገንዘቤን መመለስ እፈልጋለሁ።"


def make(fail=None):
    cfg = config.from_dict({
        "baseline": "qwen",
        "models": [
            {"name": "qwen", "base_url": "http://mock/v1", "tokenizer": "qwen3", "input_per_million": 1, "output_per_million": 2},
            {"name": "gpt", "base_url": "http://mock/v1", "tokenizer": "o200k", "input_per_million": 1, "output_per_million": 2},
        ],
    })
    mock = create_mock(cfg, fail=fail)
    store = Store(":memory:")
    app = create_app(cfg, store=store, transport=httpx.ASGITransport(app=mock))
    return app, store


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router")


def chat(text, **extra):
    return {"model": "yaptax-auto", "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": text}], **extra}


@pytest.mark.anyio
async def test_routes_hindi_and_records_saving():
    app, store = make()
    async with client(app) as c:
        r = await c.post("/v1/chat/completions", json=chat(HINDI))
    assert r.status_code == 200
    assert r.headers["x-yaptax-model"] == "gpt" and r.headers["x-yaptax-lang"] == "hin_Deva"
    assert float(r.headers["x-yaptax-saved-usd"]) > 0
    assert r.json()["choices"][0]["message"]["content"] == HINDI  # mock echoes; answer comes back untouched
    s = store.summary()
    assert s["requests"] == 1 and s["usage_reported"] == 1 and s["saved"] > 0
    assert s["languages"][0]["lang"] == "hin_Deva" and s["languages"][0]["models"] == {"gpt": 1}


@pytest.mark.anyio
async def test_amharic_stays_on_baseline_with_zero_saving():
    app, store = make()
    async with client(app) as c:
        r = await c.post("/v1/chat/completions", json=chat(AMHARIC))
    assert r.headers["x-yaptax-model"] == "qwen" and float(r.headers["x-yaptax-saved-usd"]) == 0
    assert store.summary()["saved"] == 0


@pytest.mark.anyio
async def test_pinned_model_is_respected():
    app, store = make()
    async with client(app) as c:
        r = await c.post("/v1/chat/completions", json={**chat(HINDI), "model": "qwen"})
    assert r.headers["x-yaptax-model"] == "qwen"


@pytest.mark.anyio
async def test_falls_back_to_baseline_when_chosen_model_is_down():
    app, store = make(fail={"gpt"})
    async with client(app) as c:
        r = await c.post("/v1/chat/completions", json=chat(HINDI))
    assert r.status_code == 200 and r.headers["x-yaptax-model"] == "qwen"
    s = store.summary()
    assert s["requests"] == 1 and s["failed"] == 1
    assert s["recent"][0]["fell_back"] == 1


@pytest.mark.anyio
async def test_streaming_passes_through_and_records_usage():
    app, store = make()
    async with client(app) as c:
        async with c.stream("POST", "/v1/chat/completions", json=chat(HINDI, stream=True)) as r:
            assert r.headers["x-yaptax-model"] == "gpt"
            lines = [l async for l in r.aiter_lines() if l.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    text = "".join((json.loads(l[5:])["choices"] or [{}])[0].get("delta", {}).get("content", "") for l in lines[:-1])
    assert text == HINDI
    s = store.summary()
    assert s["requests"] == 1 and s["usage_reported"] == 1 and s["saved"] > 0


@pytest.mark.anyio
async def test_bad_requests():
    app, _ = make()
    async with client(app) as c:
        assert (await c.post("/v1/chat/completions", json={"messages": []})).status_code == 400
        assert (await c.post("/v1/chat/completions", json={**chat(HINDI), "model": "nope"})).status_code == 400
        assert (await c.post("/v1/chat/completions", content=b"{not json")).status_code == 400


@pytest.mark.anyio
async def test_models_stats_and_dashboard():
    app, _ = make()
    async with client(app) as c:
        ids = [m["id"] for m in (await c.get("/v1/models")).json()["data"]]
        assert ids == ["yaptax-auto", "qwen", "gpt"]
        assert (await c.get("/stats")).json()["baseline"] == "qwen"
        home = await c.get("/")
        assert home.status_code == 200 and "What routing saved you" in home.text


@pytest.fixture
def anyio_backend():
    return "asyncio"
