"""The router as an OpenAI-compatible HTTP server.

Point an OpenAI client at http://localhost:8787/v1 and send model="yaptax-auto". The router picks the model for
each request, forwards it to that model's provider with the team's own key, and records the saving.

Endpoints:
    POST /v1/chat/completions   routed (model "yaptax-auto" or "auto") or pinned (a configured model name)
    GET  /v1/models             the configured models plus "yaptax-auto"
    GET  /stats                 totals for the dashboard (JSON)
    GET  /                      the savings dashboard
    GET  /health

Headers on every routed answer: x-yaptax-model, x-yaptax-lang, and (non-streaming) x-yaptax-saved-usd.
Send x-yaptax-lang: hi (or hin_Deva) to state the language instead of letting yaptax guess it.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from importlib import resources

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .. import __version__
from . import engine
from .config import Config, Model
from .store import Store

AUTO = {"yaptax-auto", "auto"}


def _headers(m: Model) -> dict:
    h = {"Content-Type": "application/json"}
    key = os.environ.get(m.api_key_env) if m.api_key_env else None
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "yaptax_router_error"}}, status_code=status)


def create_app(cfg: Config, store: Store | None = None, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    store = store or Store(cfg.database)
    client = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(120.0, connect=10.0))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(title="yaptax router", version=__version__, lifespan=lifespan)
    dashboard = resources.files("yaptax.router").joinpath("dashboard.html").read_text(encoding="utf-8")

    def log(d: engine.Decision, pinned: bool, status: int, usage: dict | None, started: float, fell_back: bool,
            answer_text: str = "") -> engine.Outcome | None:
        if status >= 400:
            store.record(lang=d.lang, lang_source=d.lang_source, model=d.chosen.name, baseline=cfg.baseline,
                         pinned=int(pinned), prompt_tokens=0, completion_tokens=0, usage_reported=0, cost=0.0,
                         baseline_cost=0.0, saved=0.0, latency_ms=int((time.time() - started) * 1000), status=status,
                         fell_back=int(fell_back))
            return None
        reported = bool(usage and "prompt_tokens" in usage and "completion_tokens" in usage)
        if reported:
            p, c = int(usage["prompt_tokens"]), int(usage["completion_tokens"])
        else:  # provider gave no usage: count it ourselves
            chosen_c = next((x for x in d.candidates if x.model == d.chosen.name), None)
            p = chosen_c.input_tokens if chosen_c else 0
            c = engine.count_tokens(answer_text, d.chosen.tokenizer, d.lang)[0] if answer_text else 0
        o = engine.settle(cfg, d, p, c)
        store.record(lang=d.lang, lang_source=d.lang_source, model=d.chosen.name, baseline=cfg.baseline,
                     pinned=int(pinned), prompt_tokens=p, completion_tokens=c, usage_reported=int(reported),
                     cost=o.cost, baseline_cost=o.baseline_cost, saved=o.saved,
                     latency_ms=int((time.time() - started) * 1000), status=200, fell_back=int(fell_back))
        return o

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True, "version": __version__, "models": list(cfg.models), "baseline": cfg.baseline}

    @app.get("/v1/models")
    async def models() -> dict:
        ids = ["yaptax-auto", *cfg.models]
        return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "yaptax"} for i in ids]}

    @app.get("/stats")
    async def stats() -> dict:
        s = store.summary()
        s["baseline"] = cfg.baseline
        s["demo"] = cfg.demo
        return s

    @app.get("/", response_class=HTMLResponse)
    async def home() -> str:
        return dashboard

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        started = time.time()
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _error(400, "request body is not valid JSON")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return _error(400, "messages must be a non-empty list")
        requested = body.get("model") or "yaptax-auto"
        if requested not in AUTO and requested not in cfg.models:
            return _error(400, f"unknown model {requested!r}; use 'yaptax-auto' or one of: {', '.join(cfg.models)}")

        d = engine.decide(cfg, messages, request.headers.get("x-yaptax-lang"))
        pinned = requested not in AUTO
        if pinned:
            d = replace(d, chosen=cfg.models[requested])
        baseline_d = replace(d, chosen=cfg.models[cfg.baseline])
        tries = [d] if d.chosen.name == cfg.baseline or pinned else [d, baseline_d]

        if body.get("stream"):
            return await _stream(body, tries, pinned, started)

        last: httpx.Response | None = None
        for i, attempt in enumerate(tries):
            payload = {**body, "model": attempt.chosen.upstream}
            try:
                r = await client.post(f"{attempt.chosen.base_url}/chat/completions", json=payload,
                                      headers=_headers(attempt.chosen))
            except httpx.HTTPError as e:
                log(attempt, pinned, 502, None, started, i > 0)
                if i + 1 < len(tries):
                    continue
                return _error(502, f"could not reach {attempt.chosen.name}: {type(e).__name__}")
            last = r
            if r.status_code >= 400:
                log(attempt, pinned, r.status_code, None, started, i > 0)
                if i + 1 < len(tries) and r.status_code not in (400, 401, 403, 404, 422):
                    continue
                return JSONResponse(_safe_json(r), status_code=r.status_code,
                                    headers={"x-yaptax-model": attempt.chosen.name, "x-yaptax-lang": attempt.lang})
            data = _safe_json(r)
            text = _answer_text(data)
            o = log(attempt, pinned, 200, data.get("usage") if isinstance(data, dict) else None, started, i > 0, text)
            headers = {"x-yaptax-model": attempt.chosen.name, "x-yaptax-lang": attempt.lang,
                       "x-yaptax-saved-usd": f"{o.saved:.8f}" if o else "0"}
            return JSONResponse(data, status_code=200, headers=headers)
        return _error(502, "no model answered") if last is None else JSONResponse(_safe_json(last), last.status_code)

    async def _stream(body: dict, tries: list[engine.Decision], pinned: bool, started: float):
        for i, attempt in enumerate(tries):
            payload = {**body, "model": attempt.chosen.upstream,
                       "stream_options": {**(body.get("stream_options") or {}), "include_usage": True}}
            req = client.build_request("POST", f"{attempt.chosen.base_url}/chat/completions", json=payload,
                                       headers=_headers(attempt.chosen))
            try:
                r = await client.send(req, stream=True)
            except httpx.HTTPError as e:
                log(attempt, pinned, 502, None, started, i > 0)
                if i + 1 < len(tries):
                    continue
                return _error(502, f"could not reach {attempt.chosen.name}: {type(e).__name__}")
            if r.status_code >= 400:
                content = await r.aread()
                await r.aclose()
                log(attempt, pinned, r.status_code, None, started, i > 0)
                if i + 1 < len(tries) and r.status_code not in (400, 401, 403, 404, 422):
                    continue
                return JSONResponse(_safe_json_bytes(content), status_code=r.status_code)

            async def gen(r=r, attempt=attempt, fell_back=i > 0):
                usage, parts = None, []
                try:
                    async for line in r.aiter_lines():
                        if line.startswith("data:"):
                            chunk = line[5:].strip()
                            if chunk and chunk != "[DONE]":
                                try:
                                    obj = json.loads(chunk)
                                    usage = obj.get("usage") or usage
                                    for ch in obj.get("choices") or []:
                                        delta = ch.get("delta") or {}
                                        for k in ("content", "reasoning", "reasoning_content"):
                                            if isinstance(delta.get(k), str):
                                                parts.append(delta[k])
                                except json.JSONDecodeError:
                                    pass
                        yield (line + "\n").encode("utf-8")
                finally:
                    await r.aclose()
                    log(attempt, pinned, 200, usage, started, fell_back, "".join(parts))

            return StreamingResponse(gen(), media_type="text/event-stream",
                                     headers={"x-yaptax-model": attempt.chosen.name, "x-yaptax-lang": attempt.lang})
        return _error(502, "no model answered")

    return app


def _safe_json(r: httpx.Response):
    try:
        return r.json()
    except ValueError:
        return {"error": {"message": r.text[:500], "type": "upstream_error"}}


def _safe_json_bytes(b: bytes):
    try:
        return json.loads(b)
    except ValueError:
        return {"error": {"message": b[:500].decode("utf-8", "replace"), "type": "upstream_error"}}


def _answer_text(data) -> str:
    if not isinstance(data, dict):
        return ""
    out = []
    for ch in data.get("choices") or []:
        msg = ch.get("message") or {}
        # Thinking models bill their reasoning as output tokens too, so count it when we count ourselves.
        for k in ("reasoning", "reasoning_content", "content"):
            if isinstance(msg.get(k), str) and msg[k]:
                out.append(msg[k])
    return "\n".join(out)
