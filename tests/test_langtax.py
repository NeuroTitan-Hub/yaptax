import json
from pathlib import Path

import pytest
import tiktoken

from langtax import cli, detect, index

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "support.jsonl"


def test_index_shape():
    langs = index.languages()
    assert len(langs) == 204
    assert index.tokenizer_ids()[:2] == ["o200k", "cl100k"]
    assert all(r[0] == 1 for r in langs["eng_Latn"].ratios.values())
    # Numbers quoted in our outreach must match the data.
    assert round(langs["hin_Deva"].ratios["o200k"][0], 2) == 1.57
    assert round(langs["hin_Deva"].ratios["qwen3"][0], 2) == 4.42


@pytest.mark.parametrize("q,code", [("hi", "hin_Deva"), ("Hindi", "hin_Deva"), ("tam_Taml", "tam_Taml"), ("pt", "por_Latn"), ("tamil", "tam_Taml")])
def test_resolve(q, code):
    assert index.resolve(q).code == code


def test_resolve_errors():
    with pytest.raises(KeyError):
        index.resolve("Klingon")
    with pytest.raises(KeyError, match="several"):
        index.resolve("Arabic")


@pytest.mark.parametrize(
    "text,code",
    [
        ("मेरा ऑर्डर कहाँ है?", "hin_Deva"),
        ("என் ஆர்டர் எங்கே?", "tam_Taml"),
        ("注文はどこですか？", "jpn_Jpan"),
        ("我的订单在哪里？", "zho_Hans"),
        ("Where is my order? I want to know the status of it.", "eng_Latn"),
        ("¿Dónde está mi pedido? Quiero saber su estado.", detect.UNKNOWN),
    ],
)
def test_detect(text, code):
    assert detect.guess(text) == code


def test_measure_example():
    res = cli.measure(cli.read_records(str(EXAMPLE), None, None), ["o200k", "cl100k"])
    by = {r["language"]: r for r in res["languages"]}
    assert by["eng_Latn"]["by_tokenizer"]["o200k"]["extra_tokens"] == 0
    assert by["hin_Deva"]["by_tokenizer"]["o200k"]["extra_tokens"] > 0
    assert by["por_Latn"]["label"] == "labelled"  # pt-BR resolved
    assert by["ben_Beng"]["label"] == "guessed"  # no label in the file
    enc = tiktoken.get_encoding("o200k_base")
    direct = sum(len(enc.encode_ordinary(json.loads(l)["text"])) for l in EXAMPLE.read_text(encoding="utf-8").splitlines())
    assert res["summary"]["o200k"]["tokens"] == direct
    s = res["summary"]["o200k"]
    assert s["tokens"] == s["english_equivalent"] + s["extra_tokens"] or abs(s["tokens"] - s["english_equivalent"] - s["extra_tokens"]) <= 1


def test_forced_language(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("line one\nline two\n", encoding="utf-8")
    res = cli.measure(cli.read_records(str(f), None, None), ["o200k"], force_lang="es")
    assert [r["language"] for r in res["languages"]] == ["spa_Latn"]
    assert res["languages"][0]["records"] == 2


def test_cli_ratios(capsys):
    assert cli.main(["ratios", "ta"]) == 0
    out = capsys.readouterr().out
    assert "Tamil (tam_Taml)" in out and "o200k" in out


def test_cli_json(capsys):
    assert cli.main(["measure", str(EXAMPLE), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["summary"]["o200k"]["tokens"] > 0


def test_cli_bad_language(capsys):
    assert cli.main(["ratios", "Klingon"]) == 2
