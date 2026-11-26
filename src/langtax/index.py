"""The Language Cost Index v0 data bundled with langtax, and language-code lookup."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

# Common ISO 639-1 codes mapped to the FLORES-200 code whose ratios we use.
ISO1 = {
    "en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu",
    "mr": "mar_Deva", "gu": "guj_Gujr", "kn": "kan_Knda", "ml": "mal_Mlym", "pa": "pan_Guru",
    "or": "ory_Orya", "as": "asm_Beng", "ur": "urd_Arab", "ne": "npi_Deva", "si": "sin_Sinh",
    "ar": "arb_Arab", "fa": "pes_Arab", "he": "heb_Hebr", "tr": "tur_Latn",
    "es": "spa_Latn", "pt": "por_Latn", "fr": "fra_Latn", "de": "deu_Latn", "it": "ita_Latn",
    "nl": "nld_Latn", "sv": "swe_Latn", "da": "dan_Latn", "pl": "pol_Latn", "cs": "ces_Latn",
    "ro": "ron_Latn", "el": "ell_Grek", "hu": "hun_Latn", "fi": "fin_Latn", "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl", "hy": "hye_Armn", "ka": "kat_Geor",
    "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "id": "ind_Latn", "ms": "zsm_Latn", "tl": "tgl_Latn", "vi": "vie_Latn", "th": "tha_Thai",
    "my": "mya_Mymr", "km": "khm_Khmr", "lo": "lao_Laoo",
    "sw": "swh_Latn", "am": "amh_Ethi", "yo": "yor_Latn", "ha": "hau_Latn", "zu": "zul_Latn",
}


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    ratios: dict[str, tuple[float, float, float]]  # tokenizer id -> (ratio, lo95, hi95)


@lru_cache(maxsize=1)
def load() -> dict:
    with resources.files("langtax").joinpath("data/index-v0.json").open(encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def languages() -> dict[str, Language]:
    data = load()
    ids = [t["id"] for t in data["tokenizers"]]
    out = {}
    for l in data["languages"]:
        out[l["c"]] = Language(l["c"], l["n"], {t: tuple(r) for t, r in zip(ids, l["r"])})
    return out


def tokenizer_ids() -> list[str]:
    return [t["id"] for t in load()["tokenizers"]]


def resolve(query: str) -> Language:
    """Find a language by FLORES code (hin_Deva), ISO 639-1 code (hi) or name (Hindi)."""
    langs = languages()
    q = query.strip()
    if q in langs:
        return langs[q]
    if q.lower() in ISO1:
        return langs[ISO1[q.lower()]]
    exact = [l for l in langs.values() if l.name.lower() == q.lower()]
    if exact:
        return exact[0]
    part = [l for l in langs.values() if q.lower() in l.name.lower()]
    if len(part) == 1:
        return part[0]
    if part:
        names = ", ".join(f"{l.name} ({l.code})" for l in part[:8])
        raise KeyError(f"'{query}' matches several languages: {names}")
    raise KeyError(f"unknown language '{query}'. Use a FLORES-200 code like hin_Deva, an ISO code like hi, or a name")
