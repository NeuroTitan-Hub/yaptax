"""`yaptax demo`: the router and a fake provider on this machine, real routing, a live savings dashboard.

Every model has the same demo price, so the saving on the dashboard comes only from tokenizer efficiency per
language. No real model is called and nothing is paid. Good for a first look, a pitch, or checking an install.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from importlib import resources

import httpx
import uvicorn

from . import config
from .mock import create_mock
from .server import create_app
from .store import Store

DEMO_PRICE_IN, DEMO_PRICE_OUT = 1.0, 2.0  # equal for every model on purpose; not anyone's real price

DEMO_MODELS = [  # (name, tokenizer)
    ("qwen3", "qwen3"),
    ("gpt-4o", "o200k"),
    ("deepseek-v3", "deepseek-v3"),
    ("mistral-nemo", "mistral-nemo"),
]


def demo_config(mock_port: int) -> config.Config:
    return config.from_dict({
        "demo": True,
        "baseline": "qwen3",  # a team self-hosting one open model for every language
        "database": ":memory:",
        "models": [
            {"name": n, "tokenizer": t, "base_url": f"http://127.0.0.1:{mock_port}/v1",
             "input_per_million": DEMO_PRICE_IN, "output_per_million": DEMO_PRICE_OUT}
            for n, t in DEMO_MODELS
        ],
    })


def prompts() -> list[dict]:
    data = json.loads(resources.files("yaptax.router").joinpath("demo_prompts.json").read_text(encoding="utf-8"))
    return data["prompts"]


class _Server(threading.Thread):
    def __init__(self, app, port: int):
        super().__init__(daemon=True)
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))

    def run(self) -> None:
        self.server.run()

    def stop(self) -> None:
        self.server.should_exit = True


def _wait(url: str, timeout: float = 15.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if httpx.get(url, timeout=1.0).status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"{url} did not start")


def run(port: int = 8787, requests: int = 34, serve: bool = True, open_browser: bool = False, out=print) -> dict:
    mock_port = port + 1
    cfg = demo_config(mock_port)
    store = Store(":memory:")
    mock = _Server(create_mock(cfg), mock_port)
    router = _Server(create_app(cfg, store=store), port)
    mock.start()
    router.start()
    base = f"http://127.0.0.1:{port}"
    try:
        _wait(f"http://127.0.0.1:{mock_port}/docs")
        _wait(f"{base}/health")
        out(f"yaptax router (demo) on {base}  |  dashboard: {base}/  |  OpenAI base URL: {base}/v1")
        ps = prompts()
        with httpx.Client(base_url=base, timeout=30.0) as c:
            for i in range(requests):
                p = ps[i % len(ps)]
                c.post("/v1/chat/completions", headers={"x-yaptax-lang": p["lang"]}, json={
                    "model": "yaptax-auto",
                    "messages": [
                        {"role": "system", "content": "You are a helpful support agent. Answer in the user's language."},
                        {"role": "user", "content": p["text"]},
                    ],
                })
            s = c.get("/stats").json()
        out(f"{s['requests']} requests: spent ${s['cost']:.6f} instead of ${s['baseline_cost']:.6f} on {cfg.baseline} "
            f"for every language, saving {s['saved_pct']:.0%} (demo prices, equal for every model).")
        for l in s["languages"]:
            models = ", ".join(f"{m} x{n}" for m, n in l["models"].items())
            out(f"  {l['name']:<24} saved {l['saved_pct']:>4.0%}  -> {models}")
        if serve:
            out(f"\nDashboard: {base}/   (Ctrl+C to stop)")
            if open_browser:
                webbrowser.open(f"{base}/")
            while True:
                time.sleep(1)
        return s
    except KeyboardInterrupt:
        return {}
    finally:
        router.stop()
        mock.stop()
