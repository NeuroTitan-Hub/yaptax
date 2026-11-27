"""A fake OpenAI-compatible model server, for the demo and the tests. It costs nothing and sends nothing anywhere.

It answers every chat request by echoing the user's words back (so the answer is in the same language, like a
real support bot), and reports usage the way a provider would: token counts on that model's own tokenizer
(exact for OpenAI's tokenizers, estimated from the Language Cost Index for the others).
"""

from __future__ import annotations

import asyncio
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import engine
from .config import Config


def create_mock(cfg: Config, fail: set[str] | None = None, delay: float = 0.0) -> FastAPI:
    by_upstream = {m.upstream: m for m in cfg.models.values()}
    fail = fail or set()
    app = FastAPI(title="yaptax mock provider")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        name = body.get("model")
        model = by_upstream.get(name)
        if model is None:
            return JSONResponse({"error": {"message": f"mock: unknown model {name!r}"}}, status_code=404)
        if name in fail:
            return JSONResponse({"error": {"message": f"mock: {name} is down"}}, status_code=503)
        msgs = body.get("messages") or []
        lang, _ = engine.resolve_lang(msgs, None)
        answer = engine.user_text(msgs) or "OK"
        p = engine.count_tokens(engine.message_text(msgs), model.tokenizer, lang)[0]
        c = engine.count_tokens(answer, model.tokenizer, lang)[0]
        usage = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}
        if delay:
            await asyncio.sleep(delay)
        created = int(time.time())
        if not body.get("stream"):
            return {
                "id": f"mock-{created}",
                "object": "chat.completion",
                "created": created,
                "model": name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
                "usage": usage,
            }

        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        async def gen():
            words = answer.split(" ")
            for i, w in enumerate(words):
                piece = w if i == 0 else " " + w
                chunk = {"id": f"mock-{created}", "object": "chat.completion.chunk", "created": created, "model": name,
                         "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            end = {"id": f"mock-{created}", "object": "chat.completion.chunk", "created": created, "model": name,
                   "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(end)}\n\n"
            if include_usage:
                yield f"data: {json.dumps({'id': f'mock-{created}', 'object': 'chat.completion.chunk', 'choices': [], 'usage': usage})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
