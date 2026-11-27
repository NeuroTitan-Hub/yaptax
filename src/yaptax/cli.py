"""yaptax: measure the language tax in your own prompts.

    yaptax measure prompts.jsonl                 tokens per language, and how many are language tax
    yaptax measure chats.csv --lang-field lang   use your own language labels
    yaptax measure hindi.txt --lang hi --spend 2000
    yaptax ratios Tamil                          the Index's ratios for one language
    yaptax demo                                  the router on a fake provider, with a live savings dashboard
    yaptax router --config router.toml           the router in front of your real models (your own keys)

Everything runs on your machine. Your text is never uploaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from . import __version__, detect, index, tokenizers

TEXT_FIELDS = ("text", "prompt", "content", "input", "message", "query", "question")
LANG_FIELDS = ("lang", "language", "locale")


# ---------- reading records ----------


def _text_of(obj: dict, field: str | None) -> str | None:
    if field:
        v = obj.get(field)
        return v if isinstance(v, str) else None
    if isinstance(obj.get("messages"), list):  # chat format: join every message's text
        parts = [m.get("content") for m in obj["messages"] if isinstance(m, dict)]
        return "\n".join(p for p in parts if isinstance(p, str)) or None
    for f in TEXT_FIELDS:
        if isinstance(obj.get(f), str):
            return obj[f]
    return None


def _lang_of(obj: dict, field: str | None) -> str | None:
    for f in [field] if field else LANG_FIELDS:
        v = obj.get(f)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _resolve_label(label: str) -> str:
    """hin_Deva, hi, Hindi, and locale forms like hi-IN or pt_BR."""
    try:
        return index.resolve(label).code
    except KeyError:
        if len(label) <= 6 and ("-" in label or "_" in label):
            return index.resolve(label.replace("_", "-").split("-")[0]).code
        raise


def read_records(path: str, text_field: str | None, lang_field: str | None) -> Iterator[tuple[str, str | None]]:
    """Yield (text, language label or None) from .jsonl, .json, .csv or plain text (one record per line)."""
    if path == "-":
        for line in sys.stdin:
            if line.strip():
                yield line.rstrip("\n"), None
        return
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        with p.open(encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{path}:{n}: not valid JSON ({e.msg})")
                if isinstance(obj, str):
                    yield obj, None
                elif isinstance(obj, dict) and (t := _text_of(obj, text_field)):
                    yield t, _lang_of(obj, lang_field)
    elif suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        for obj in data if isinstance(data, list) else [data]:
            if isinstance(obj, str):
                yield obj, None
            elif isinstance(obj, dict) and (t := _text_of(obj, text_field)):
                yield t, _lang_of(obj, lang_field)
    elif suffix in (".csv", ".tsv"):
        with p.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t" if suffix == ".tsv" else ","):
                if t := _text_of(row, text_field):
                    yield t, _lang_of(row, lang_field)
    else:
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line.rstrip("\n"), None


# ---------- measuring ----------


def measure(records: Iterable[tuple[str, str | None]], toks: list[str], force_lang: str | None = None) -> dict:
    langs = index.languages()
    forced = index.resolve(force_lang).code if force_lang else None
    groups: dict[str, dict] = defaultdict(lambda: {"records": 0, "chars": 0, "how": set(), "tokens": defaultdict(int)})
    counters = {t: tokenizers.counter(t) for t in toks}
    bad_labels: set[str] = set()

    for text, label in records:
        if forced:
            code, how = forced, "given"
        elif label:
            try:
                code, how = _resolve_label(label), "labelled"
            except KeyError:
                bad_labels.add(label)
                code, how = detect.guess(text), "guessed"
        else:
            code, how = detect.guess(text), "guessed"
        g = groups[code]
        g["records"] += 1
        g["chars"] += len(text)
        g["how"].add(how)
        for t, count in counters.items():
            g["tokens"][t] += count(text)

    rows = []
    totals = {t: {"tokens": 0, "english_equivalent": 0.0} for t in toks}
    for code, g in sorted(groups.items(), key=lambda kv: -kv[1]["tokens"][toks[0]]):
        lang = langs.get(code)
        per_tok = {}
        for t in toks:
            n = g["tokens"][t]
            ratio = lang.ratios[t][0] if lang else None
            eng = n / ratio if ratio else float(n)
            per_tok[t] = {
                "tokens": n,
                "index_ratio": ratio,
                "english_equivalent": round(eng),
                "extra_tokens": round(n - eng),
            }
            totals[t]["tokens"] += n
            totals[t]["english_equivalent"] += eng
        rows.append(
            {
                "language": code,
                "name": lang.name if lang else "Unknown (Latin script, not English-like)",
                "label": "/".join(sorted(g["how"])),
                "records": g["records"],
                "chars": g["chars"],
                "by_tokenizer": per_tok,
            }
        )
    summary = {}
    for t, v in totals.items():
        extra = v["tokens"] - v["english_equivalent"]
        summary[t] = {
            "tokens": v["tokens"],
            "english_equivalent": round(v["english_equivalent"]),
            "extra_tokens": round(extra),
            "tax_share": (extra / v["tokens"]) if v["tokens"] else 0.0,
        }
    return {"yaptax": __version__, "index": index.load()["version"], "languages": rows, "summary": summary,
            "unrecognised_labels": sorted(bad_labels)}


# ---------- output ----------


def _table(header: list[str], rows: list[list[str]], right: set[int]) -> str:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    def line(r: list[str]) -> str:
        return "  ".join(str(c).rjust(w) if i in right else str(c).ljust(w) for i, (c, w) in enumerate(zip(r, widths)))
    return "\n".join([line(header), "  ".join("-" * w for w in widths), *map(line, rows)])


def print_report(res: dict, toks: list[str], spend: float | None, out=None) -> None:
    out = out or sys.stdout
    head = ["Language", "Records", "Label"]
    for t in toks:
        head += [f"{t} tokens", f"{t} extra"]
    rows = []
    for r in res["languages"]:
        row = [f"{r['name']} ({r['language']})" if r["language"] != detect.UNKNOWN else r["name"], f"{r['records']:,}", r["label"]]
        for t in toks:
            b = r["by_tokenizer"][t]
            pct = b["extra_tokens"] / b["tokens"] if b["tokens"] else 0
            row += [f"{b['tokens']:,}", f"{pct:.0%}" if b["index_ratio"] else "n/a"]
        rows.append(row)
    print(_table(head, rows, right=set(range(1, len(head))) - {2}), file=out)
    print(file=out)
    for t in toks:
        s = res["summary"][t]
        print(
            f"{t}: {s['tokens']:,} tokens. The same meaning in English would take about {s['english_equivalent']:,}, "
            f"so about {s['extra_tokens']:,} ({s['tax_share']:.0%}) are language tax.",
            file=out,
        )
    if spend:
        s = res["summary"][toks[0]]
        print(
            f"\nAt {spend:,.0f} a month in API spend with this traffic mix, about {spend * s['tax_share']:,.0f} a month "
            f"pays for tokens English would not need (on {toks[0]}, at one price per token).",
            file=out,
        )
    if len(toks) > 1:
        print("\nFewest tokens on your text, per language:", file=out)
        for r in res["languages"]:
            best = min(toks, key=lambda t: r["by_tokenizer"][t]["tokens"])
            print(f"  {r['name']}: {best} ({r['by_tokenizer'][best]['tokens']:,})", file=out)
    notes = []
    if any("guessed" in r["label"] for r in res["languages"]):
        notes.append(
            "Some languages were guessed from the writing system (for example, all Devanagari is scored as Hindi). "
            "Pass --lang-field or --lang for exact results."
        )
    if any(r["language"] == detect.UNKNOWN for r in res["languages"]):
        notes.append("Latin-script text that does not look like English is counted but not scored. Label it to score it.")
    if res["unrecognised_labels"]:
        notes.append(f"Labels not recognised (guessed instead): {', '.join(res['unrecognised_labels'][:10])}")
    notes.append(
        "Extra tokens use the Language Cost Index's average ratio for each language, measured on 1,012 FLORES-200 "
        "sentences. Your own text can differ. Fewer tokens is not a cheaper model: prices and answer quality differ."
    )
    print("\nNotes:", file=out)
    for n in notes:
        print(f"  - {n}", file=out)


def print_ratios(lang: index.Language, out=None) -> None:
    out = out or sys.stdout
    data = index.load()
    rows = []
    for t in data["tokenizers"]:
        r, lo, hi = lang.ratios[t["id"]]
        rows.append([t["id"], t["models"], f"{r:.2f}x", f"{lo:.2f} to {hi:.2f}", f"{max(0, 1 - 1 / r):.0%}"])
    print(f"{lang.name} ({lang.code}): tokens for the same meaning, relative to English = 1\n", file=out)
    print(_table(["Tokenizer", "Used by", "Ratio", "95% interval", "Tax share"], rows, right={2, 3, 4}), file=out)
    print("\nTax share: the part of spend on this language that English would not need, at one price per token.", file=out)
    print(f"Source: Language Cost Index v{data['version']}, https://neurotitan.in/cost", file=out)


# ---------- entry point ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="yaptax", description="Measure the language tax in your own AI prompts.")
    ap.add_argument("--version", action="version", version=f"yaptax {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="count tokens per language in a file of prompts")
    m.add_argument("file", help=".jsonl, .json, .csv, .tsv or text (one prompt per line); - for stdin")
    m.add_argument("--tokenizers", default=",".join(tokenizers.DEFAULT),
                   help=f"comma-separated, from: {', '.join(tokenizers.AVAILABLE)} (default: %(default)s)")
    m.add_argument("--text-field", help="field holding the text (default: text, prompt, content, ... or chat messages)")
    m.add_argument("--lang-field", help="field holding the language (default: lang, language or locale, if present)")
    m.add_argument("--lang", help="treat every record as this language (code like hi or hin_Deva, or a name)")
    m.add_argument("--spend", type=float, help="your monthly API spend for this traffic, to show the tax in money")
    m.add_argument("--json", action="store_true", help="print machine-readable JSON")

    r = sub.add_parser("ratios", help="show the Index's ratios for one language")
    r.add_argument("language", help="code like ta or tam_Taml, or a name like Tamil")

    ro = sub.add_parser("router", help="run the router: an OpenAI-compatible endpoint that routes by language")
    ro.add_argument("--config", required=True, help="router TOML file (see examples/router.toml)")
    ro.add_argument("--host", default="127.0.0.1")
    ro.add_argument("--port", type=int, default=8787)

    de = sub.add_parser("demo", help="run the router against a fake provider and show the savings dashboard")
    de.add_argument("--port", type=int, default=8787, help="router port; the fake provider uses port+1")
    de.add_argument("--requests", type=int, default=34, help="demo requests to send (default %(default)s)")
    de.add_argument("--no-serve", action="store_true", help="print the summary and exit")
    de.add_argument("--open", action="store_true", help="open the dashboard in your browser")

    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")  # language names on consoles without UTF-8
    if args.cmd in ("router", "demo"):
        return _router_cmd(args)
    try:
        if args.cmd == "ratios":
            print_ratios(index.resolve(args.language))
            return 0
        toks = [t.strip() for t in args.tokenizers.split(",") if t.strip()]
        res = measure(read_records(args.file, args.text_field, args.lang_field), toks, args.lang)
    except (KeyError, tokenizers.TokenizerUnavailable, FileNotFoundError) as e:
        print(f"yaptax: {e.args[0] if e.args else e}", file=sys.stderr)
        return 2
    if not res["languages"]:
        print("yaptax: no text found in the input", file=sys.stderr)
        return 1
    if args.json:
        json.dump(res, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        print_report(res, toks, args.spend)
    return 0


def _router_cmd(args) -> int:
    try:
        import uvicorn

        from .router import config as rconfig
        from .router import demo
        from .router.server import create_app
    except ImportError:
        print("yaptax: the router needs extra packages: pip install 'yaptax[router]'", file=sys.stderr)
        return 2
    if args.cmd == "demo":
        demo.run(port=args.port, requests=args.requests, serve=not args.no_serve, open_browser=args.open)
        return 0
    try:
        cfg = rconfig.load(args.config)
    except (OSError, rconfig.ConfigError) as e:
        print(f"yaptax: {e}", file=sys.stderr)
        return 2
    missing = sorted({m.api_key_env for m in cfg.models.values() if m.api_key_env and not os.environ.get(m.api_key_env)})
    if missing:
        print(f"yaptax: warning: these key variables are not set: {', '.join(missing)}", file=sys.stderr)
    print(f"yaptax router on http://{args.host}:{args.port}  (OpenAI base URL: http://{args.host}:{args.port}/v1, "
          f"model 'yaptax-auto'; dashboard at /)")
    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level="warning")
    return 0
