"""Tests for `yaptax check`. They build the measure result by hand, so no tokenizer download is needed."""

from pathlib import Path

from yaptax import check, cli, index

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "support.jsonl"


def fake_result(rows: dict[str, dict[str, int]], toks: list[str]) -> dict:
    """rows: language code -> {tokenizer: token count}. English-equivalent comes from the Index ratios."""
    langs = index.languages()
    out_rows, totals = [], {t: {"tokens": 0, "eq": 0.0} for t in toks}
    for code, counts in rows.items():
        per = {}
        for t in toks:
            ratio = langs[code].ratios[t][0]
            eng = counts[t] / ratio
            per[t] = {"tokens": counts[t], "index_ratio": ratio, "english_equivalent": round(eng),
                      "extra_tokens": round(counts[t] - eng)}
            totals[t]["tokens"] += counts[t]
            totals[t]["eq"] += eng
        out_rows.append({"language": code, "name": langs[code].name, "label": "labelled", "records": 10,
                         "chars": 1000, "by_tokenizer": per})
    summary = {t: {"tokens": v["tokens"], "english_equivalent": round(v["eq"]),
                   "extra_tokens": round(v["tokens"] - v["eq"]),
                   "tax_share": (v["tokens"] - v["eq"]) / v["tokens"]} for t, v in totals.items()}
    return {"yaptax": "test", "index": "0", "languages": out_rows, "summary": summary, "unrecognised_labels": []}


def test_best_candidate_and_savings_range():
    toks = ["o200k", "qwen3"]
    res = fake_result({"hin_Deva": {"o200k": 1000, "qwen3": 2800}, "eng_Latn": {"o200k": 1000, "qwen3": 1000}}, toks)
    a = check.analyse(res, toks, spend=1000)
    by = {r["code"]: r for r in a["rows"]}
    assert by["hin_Deva"]["best"] == "o200k"  # Qwen3 needs more tokens for Hindi
    assert by["hin_Deva"]["cut"] == 0
    assert by["eng_Latn"]["best"] == "o200k"  # tie goes to the tokenizer you use today
    sv = a["savings"]
    assert sv["point"] == 0 and sv["low"] == 0 and sv["high"] >= 0


def test_savings_when_candidate_is_cheaper():
    toks = ["qwen3", "o200k"]  # you use Qwen3 today; o200k is the candidate
    res = fake_result({"hin_Deva": {"qwen3": 2800, "o200k": 1000}}, toks)
    a = check.analyse(res, toks, spend=1000)
    row = a["rows"][0]
    assert row["best"] == "o200k"
    assert abs(row["cut"] - (1 - 1000 / 2800)) < 1e-9
    lo, hi = row["cut_range"]
    assert 0 < lo <= row["cut"] + 0.2 and hi >= lo
    sv = a["savings"]
    assert abs(sv["point"] - 1000 * row["cut"]) < 1e-6
    assert sv["low"] <= sv["point"] <= sv["high"]


def test_render_without_spend_has_no_savings_section():
    toks = ["o200k", "cl100k"]
    res = fake_result({"tam_Taml": {"o200k": 500, "cl100k": 900}}, toks)
    md = check.render(res, toks)
    assert md.startswith("# Language Cost Check")
    assert "Tamil (tam_Taml)" in md and "language tax" in md
    assert "## Rough savings" not in md
    assert "Fewer tokens is not a cheaper model" in md


def test_render_with_spend_and_title():
    toks = ["qwen3", "o200k"]
    res = fake_result({"hin_Deva": {"qwen3": 2800, "o200k": 1000}}, toks)
    md = check.render(res, toks, spend=2000, title="Check for Acme")
    assert "# Check for Acme" in md
    assert "## Rough savings" in md and "a month" in md
    assert "`o200k`" in md and "fewer tokens" in md


def test_cli_check_writes_file(tmp_path, capsys):
    out = tmp_path / "check.md"
    code = cli.main(["check", str(EXAMPLE), "--tokenizers", "utf8-bytes", "--spend", "500", "--out", str(out)])
    assert code == 0
    md = out.read_text(encoding="utf-8")
    assert "## Tokens per language" in md and "Hindi" in md
    assert "wrote" in capsys.readouterr().err


def test_cli_check_prints_and_handles_bad_input(tmp_path, capsys):
    assert cli.main(["check", str(EXAMPLE), "--tokenizers", "utf8-bytes"]) == 0
    assert capsys.readouterr().out.startswith("# Language Cost Check")
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    assert cli.main(["check", str(empty), "--tokenizers", "utf8-bytes"]) == 1
    assert cli.main(["check", str(tmp_path / "missing.jsonl"), "--tokenizers", "utf8-bytes"]) == 2
    assert cli.main(["check", str(EXAMPLE), "--tokenizers", "nope"]) == 2
