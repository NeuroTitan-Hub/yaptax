"""A small SQLite log of routed requests, and the totals the dashboard shows.

Privacy: no prompt or answer text is stored, only language, models, token counts, costs and timing.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from .. import index

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    lang TEXT NOT NULL,
    lang_source TEXT NOT NULL,
    model TEXT NOT NULL,
    baseline TEXT NOT NULL,
    pinned INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    usage_reported INTEGER NOT NULL,
    cost REAL NOT NULL,
    baseline_cost REAL NOT NULL,
    saved REAL NOT NULL,
    latency_ms INTEGER NOT NULL,
    status INTEGER NOT NULL,
    fell_back INTEGER NOT NULL
);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.executescript(SCHEMA)

    def record(self, **row) -> None:
        row.setdefault("ts", time.time())
        cols = ", ".join(row)
        qs = ", ".join("?" for _ in row)
        with self._lock:
            self._db.execute(f"INSERT INTO requests ({cols}) VALUES ({qs})", tuple(row.values()))
            self._db.commit()

    def _rows(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            self._db.row_factory = sqlite3.Row
            return self._db.execute(sql, args).fetchall()

    def summary(self) -> dict:
        names = {c: l.name for c, l in index.languages().items()}
        ok = "status < 400"
        t = self._rows(f"SELECT COUNT(*) n, COALESCE(SUM(cost),0) c, COALESCE(SUM(baseline_cost),0) b, "
                       f"COALESCE(SUM(saved),0) s, COALESCE(SUM(usage_reported),0) u FROM requests WHERE {ok}")[0]
        failed = self._rows(f"SELECT COUNT(*) n FROM requests WHERE NOT ({ok})")[0]["n"]
        langs = []
        for r in self._rows(f"SELECT lang, COUNT(*) n, SUM(cost) c, SUM(baseline_cost) b, SUM(saved) s, "
                            f"SUM(prompt_tokens + completion_tokens) tok FROM requests WHERE {ok} "
                            f"GROUP BY lang ORDER BY s DESC, n DESC"):
            models = {m["model"]: m["n"] for m in self._rows(
                f"SELECT model, COUNT(*) n FROM requests WHERE {ok} AND lang = ? GROUP BY model ORDER BY n DESC", (r["lang"],))}
            langs.append({
                "lang": r["lang"],
                "name": names.get(r["lang"], "Unknown"),
                "requests": r["n"],
                "tokens": r["tok"],
                "cost": r["c"],
                "baseline_cost": r["b"],
                "saved": r["s"],
                "saved_pct": (r["s"] / r["b"]) if r["b"] else 0.0,
                "models": models,
            })
        models = [{"model": r["model"], "requests": r["n"], "cost": r["c"]} for r in self._rows(
            f"SELECT model, COUNT(*) n, SUM(cost) c FROM requests WHERE {ok} GROUP BY model ORDER BY n DESC")]
        recent = [dict(r) for r in self._rows(
            "SELECT ts, lang, model, baseline, prompt_tokens, completion_tokens, cost, baseline_cost, saved, "
            "latency_ms, status, fell_back FROM requests ORDER BY id DESC LIMIT 25")]
        for r in recent:
            r["name"] = names.get(r["lang"], "Unknown")
        return {
            "requests": t["n"],
            "failed": failed,
            "usage_reported": t["u"],
            "cost": t["c"],
            "baseline_cost": t["b"],
            "saved": t["s"],
            "saved_pct": (t["s"] / t["b"]) if t["b"] else 0.0,
            "languages": langs,
            "models": models,
            "recent": recent,
        }
