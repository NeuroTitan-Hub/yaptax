"""`yaptax check`: turn `yaptax measure` results into a short markdown Language Cost Check.

Pure functions on the dict that `cli.measure` returns, so nothing here reads your text or needs a tokenizer.
"""

from __future__ import annotations

from . import detect, index


def _best(row: dict, toks: list[str]) -> str:
    """Tokenizer with the fewest tokens on this language's text (first one wins a tie)."""
    return min(toks, key=lambda t: row["by_tokenizer"][t]["tokens"])


def _cut_range(code: str, current: str, best: str) -> tuple[float, float] | None:
    """Low and high share of tokens saved by moving from `current` to `best`, from the Index's 95% intervals.

    None when the language is not in the Index or the tokenizer is the same."""
    lang = index.languages().get(code)
    if lang is None or best == current or current not in lang.ratios or best not in lang.ratios:
        return None
    _, cur_lo, cur_hi = lang.ratios[current]
    _, b_lo, b_hi = lang.ratios[best]
    return max(0.0, 1 - b_hi / cur_lo), max(0.0, 1 - b_lo / cur_hi)


def analyse(res: dict, toks: list[str], spend: float | None = None) -> dict:
    """Per-language rows and totals for the check. `toks[0]` is the model you use today."""
    current = toks[0]
    total = res["summary"][current]["tokens"]
    rows = []
    lo_sum = hi_sum = point_sum = 0.0
    for r in res["languages"]:
        b = r["by_tokenizer"][current]
        best = _best(r, toks)
        n_best = r["by_tokenizer"][best]["tokens"]
        point_cut = 1 - n_best / b["tokens"] if b["tokens"] else 0.0
        scored = b["index_ratio"] is not None
        row = {
            "name": r["name"],
            "code": r["language"],
            "records": r["records"],
            "tokens": b["tokens"],
            "tax_pct": (b["extra_tokens"] / b["tokens"]) if scored and b["tokens"] else None,
            "best": best,
            "best_tokens": n_best,
            "cut": point_cut,
            "cut_range": _cut_range(r["language"], current, best) if scored else None,
            "label": r["label"],
        }
        if spend and total:
            share = b["tokens"] / total
            row["spend_share"] = spend * share
            point_sum += spend * share * point_cut
            if row["cut_range"]:
                lo_sum += spend * share * row["cut_range"][0]
                hi_sum += spend * share * row["cut_range"][1]
            else:
                lo_sum += spend * share * point_cut
                hi_sum += spend * share * point_cut
        rows.append(row)
    s = res["summary"][current]
    out = {"current": current, "tokenizers": toks, "rows": rows, "summary": s, "spend": spend}
    if spend:
        out["savings"] = {"point": point_sum, "low": min(lo_sum, point_sum), "high": max(hi_sum, point_sum)}
    return out


def render(res: dict, toks: list[str], spend: float | None = None, title: str | None = None) -> str:
    a = analyse(res, toks, spend)
    cur = a["current"]
    s = a["summary"]
    L: list[str] = [f"# {title or 'Language Cost Check'}", ""]
    L.append(
        f"We counted {sum(r['records'] for r in a['rows']):,} prompts in {len(a['rows'])} language(s) on "
        f"`{cur}`, the tokenizer you use today. They came to **{s['tokens']:,} tokens**. The same meaning in English "
        f"would take about {s['english_equivalent']:,}, so about **{s['extra_tokens']:,} tokens ({s['tax_share']:.0%}) "
        "are language tax**."
    )
    L += ["", "## Tokens per language", ""]
    L.append(f"| Language | Prompts | Tokens on `{cur}` | Language tax | Fewest tokens on |")
    L.append("|---|---:|---:|---:|---|")
    for r in a["rows"]:
        tax = f"{r['tax_pct']:.0%}" if r["tax_pct"] is not None else "not scored"
        name = r["name"] if r["code"] == detect.UNKNOWN else f"{r['name']} ({r['code']})"
        if r["best"] == cur:
            best = f"`{cur}` (already the fewest)"
        else:
            best = f"`{r['best']}`, {r['cut']:.0%} fewer tokens"
        L.append(f"| {name} | {r['records']:,} | {r['tokens']:,} | {tax} | {best} |")
    if a["spend"]:
        sv = a["savings"]
        L += ["", "## Rough savings", ""]
        if sv["high"] <= 0:
            L.append(
                f"At {a['spend']:,.0f} a month, switching to the tokenizer with the fewest tokens would not "
                "reduce tokens for this traffic mix."
            )
        else:
            L.append(
                f"At {a['spend']:,.0f} a month, sending each language to the tokenizer with the fewest tokens would cut "
                f"about **{sv['low']:,.0f} to {sv['high']:,.0f} a month** (measured point: {sv['point']:,.0f})."
            )
        L.append("")
        L.append(
            "This assumes one price per token and that your spend splits across languages in proportion to tokens. "
            "Real models have different prices, and a cheaper option must still answer well. The audit checks both. "
            "The range comes from the 95% intervals of the Language Cost Index."
        )
    L += ["", "## What this does not tell you", ""]
    notes = [
        "Fewer tokens is not a cheaper model, and not a better answer. Prices and quality differ.",
        f"Only these tokenizers were compared: {', '.join('`' + t + '`' for t in toks)}.",
        "Language tax uses the Index's average ratio per language, measured on 1,012 FLORES-200 sentences. "
        "Your own text can differ.",
    ]
    if any("guessed" in r["label"] for r in res["languages"]):
        notes.append("Some languages were guessed from the writing system. Labelled data gives exact results.")
    if any(r["code"] == detect.UNKNOWN for r in a["rows"]):
        notes.append("Latin-script text that does not look like English was counted but not scored.")
    if res["unrecognised_labels"]:
        notes.append(f"Labels not recognised (guessed instead): {', '.join(res['unrecognised_labels'][:10])}.")
    L += [f"- {n}" for n in notes]
    L += ["", f"Made with yaptax {res['yaptax']}, Language Cost Index v{res['index']}. https://neurotitan.in/cost", ""]
    return "\n".join(L)
