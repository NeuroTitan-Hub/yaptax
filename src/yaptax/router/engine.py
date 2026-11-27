"""Choosing a model for one request, and accounting for what it saved.

The decision: for the request's language, estimate each allowed model's cost (input tokens on its tokenizer
times its input price, plus an expected answer length on its tokenizer times its output price) and pick the
cheapest. Token counts come from local tokenizers where we have them (OpenAI's, via tiktoken), otherwise from
the Language Cost Index ratios. Nothing here calls the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .. import detect, index, tokenizers
from .config import Config, Model

LOCAL = {"o200k", "cl100k", "gpt2", "utf8-bytes"}  # counted exactly, offline


def message_text(messages: list) -> str:
    """All text the model will read, from OpenAI-style chat messages (string or content-part lists)."""
    parts = []
    for m in messages or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts.extend(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return "\n".join(p for p in parts if p)


def user_text(messages: list) -> str:
    """The users' own words: what decides the language (system prompts are often English)."""
    return message_text([m for m in messages or [] if isinstance(m, dict) and m.get("role") == "user"])


@lru_cache(maxsize=1)
def _ratios() -> dict[str, dict[str, float]]:
    return {code: {t: r[0] for t, r in lang.ratios.items()} for code, lang in index.languages().items()}


def ratio(lang: str, tok: str) -> float:
    """Tokens relative to English for this language on this tokenizer. 1.0 when unknown."""
    r = _ratios().get(lang)
    if not r:
        return 1.0
    return r.get(tok) or r.get("o200k") or 1.0


def count_tokens(text: str, tok: str, lang: str) -> tuple[int, bool]:
    """(tokens, exact). Exact for local tokenizers; otherwise estimated from o200k and the Index ratios."""
    if tok in LOCAL:
        return tokenizers.counter(tok)(text), True
    base = tokenizers.counter("o200k")(text)
    return round(base / ratio(lang, "o200k") * ratio(lang, tok)), False


def cost(model: Model, input_tokens: float, output_tokens: float) -> float:
    return (input_tokens * model.input_per_million + output_tokens * model.output_per_million) / 1e6


@dataclass(frozen=True)
class Candidate:
    model: str
    input_tokens: int
    output_tokens: int
    exact: bool
    est_cost: float


@dataclass(frozen=True)
class Decision:
    lang: str
    lang_source: str            # "header", "guessed" or "unknown"
    chosen: Model
    candidates: tuple[Candidate, ...]
    baseline: Candidate


def resolve_lang(messages: list, hint: str | None) -> tuple[str, str]:
    if hint:
        try:
            return index.resolve(hint).code, "header"
        except KeyError:
            pass
    g = detect.guess(user_text(messages) or message_text(messages))
    return (g, "unknown") if g == detect.UNKNOWN else (g, "guessed")


def decide(cfg: Config, messages: list, lang_hint: str | None = None) -> Decision:
    lang, source = resolve_lang(messages, lang_hint)
    text = message_text(messages)
    out_eng = cfg.expected_output_tokens

    def candidate(m: Model) -> Candidate:
        n_in, exact = count_tokens(text, m.tokenizer, lang)
        n_out = round(out_eng * ratio(lang, m.tokenizer))
        return Candidate(m.name, n_in, n_out, exact, cost(m, n_in, n_out))

    allowed = cfg.allowed(lang)
    cands = tuple(candidate(m) for m in allowed)
    # Cheapest wins; ties go to the baseline (no switch without a saving), then config order.
    best = min(cands, key=lambda c: (round(c.est_cost, 12), c.model != cfg.baseline))
    base = next((c for c in cands if c.model == cfg.baseline), None) or candidate(cfg.models[cfg.baseline])
    return Decision(lang, source, cfg.models[best.model], cands, base)


@dataclass(frozen=True)
class Outcome:
    cost: float           # what the request cost on the chosen model, from the provider's usage numbers
    baseline_cost: float  # what the same request would have cost on the baseline model (estimated)
    saved: float


def settle(cfg: Config, d: Decision, prompt_tokens: int, completion_tokens: int) -> Outcome:
    """Turn the provider's reported usage into money, and estimate the baseline's cost for the same request.

    The baseline is counterfactual: its input tokens were counted (or estimated) before routing; its output
    tokens are this answer's tokens converted to the baseline's tokenizer with the Index ratios.
    """
    chosen = d.chosen
    base = cfg.models[cfg.baseline]
    actual = cost(chosen, prompt_tokens, completion_tokens)
    if chosen.name == base.name:
        return Outcome(actual, actual, 0.0)
    out_base = completion_tokens / ratio(d.lang, chosen.tokenizer) * ratio(d.lang, base.tokenizer)
    base_cost = cost(base, d.baseline.input_tokens, out_base)
    return Outcome(actual, base_cost, base_cost - actual)
