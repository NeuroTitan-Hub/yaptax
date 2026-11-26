"""A rough language guess from the writing system, for records that carry no language label.

This guesses the *script*, not the language: Devanagari text is scored with Hindi's ratios even if it
is Marathi or Nepali. Latin-script text is only called English if it looks like English; otherwise it
is left unknown, because Spanish and English share a script but not a token cost. Pass a language
field (--lang-field) or a single language (--lang) for real numbers.
"""

from __future__ import annotations

import re
import unicodedata

# (first code point, last code point, FLORES code used for that script)
RANGES = [
    (0x0900, 0x097F, "hin_Deva"),
    (0x0980, 0x09FF, "ben_Beng"),
    (0x0A00, 0x0A7F, "pan_Guru"),
    (0x0A80, 0x0AFF, "guj_Gujr"),
    (0x0B00, 0x0B7F, "ory_Orya"),
    (0x0B80, 0x0BFF, "tam_Taml"),
    (0x0C00, 0x0C7F, "tel_Telu"),
    (0x0C80, 0x0CFF, "kan_Knda"),
    (0x0D00, 0x0D7F, "mal_Mlym"),
    (0x0D80, 0x0DFF, "sin_Sinh"),
    (0x0E00, 0x0E7F, "tha_Thai"),
    (0x0E80, 0x0EFF, "lao_Laoo"),
    (0x1000, 0x109F, "mya_Mymr"),
    (0x1780, 0x17FF, "khm_Khmr"),
    (0x1200, 0x139F, "amh_Ethi"),
    (0x0600, 0x06FF, "arb_Arab"),
    (0x0590, 0x05FF, "heb_Hebr"),
    (0x0370, 0x03FF, "ell_Grek"),
    (0x0400, 0x04FF, "rus_Cyrl"),
    (0x0530, 0x058F, "hye_Armn"),
    (0x10A0, 0x10FF, "kat_Geor"),
    (0xAC00, 0xD7AF, "kor_Hang"),
    (0x1100, 0x11FF, "kor_Hang"),
    (0x3040, 0x30FF, "jpn_Jpan"),
    (0x4E00, 0x9FFF, "zho_Hans"),
]

UNKNOWN = "unknown"
_EN = set("the and of to is in that it for you with on are this be was have not your can will please what".split())
_WORD = re.compile(r"[a-z']+")


def guess(text: str) -> str:
    counts: dict[str, int] = {}
    latin = 0
    for ch in text:
        cp = ord(ch)
        if cp < 0x0370:
            if ch.isalpha():
                latin += 1
            continue
        for lo, hi, code in RANGES:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    # Kana anywhere means Japanese, even though most characters may be Han.
    if counts.get("jpn_Jpan", 0) and counts.get("zho_Hans", 0):
        counts["jpn_Jpan"] += counts.pop("zho_Hans")
    best = max(counts.items(), key=lambda kv: kv[1], default=(None, 0))
    if best[0] and best[1] >= latin:
        return best[0]
    words = _WORD.findall(unicodedata.normalize("NFKC", text).lower())
    if words and sum(w in _EN for w in words) / len(words) >= 0.12:
        return "eng_Latn"
    return UNKNOWN
