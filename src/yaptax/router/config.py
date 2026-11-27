"""Router configuration: the models a team can use, their prices, and per-language rules.

A config is a TOML file (see examples/router.toml). Prices are the team's own: yaptax ships no price list,
because prices change and we only publish numbers we have checked.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@dataclass(frozen=True)
class Model:
    name: str                      # what the router calls it (and what clients may pin)
    base_url: str                  # OpenAI-compatible endpoint, e.g. https://api.openai.com/v1
    tokenizer: str                 # an Index tokenizer id: o200k, cl100k, qwen3, deepseek-v3, mistral-nemo, ...
    input_per_million: float       # USD per 1M input tokens, from the provider's price page
    output_per_million: float      # USD per 1M output tokens
    api_key_env: str | None = None  # name of the environment variable holding the key; never the key itself
    upstream_model: str | None = None  # model id sent to the provider, if different from name

    @property
    def upstream(self) -> str:
        return self.upstream_model or self.name


@dataclass(frozen=True)
class LanguageRule:
    allow: tuple[str, ...] | None = None  # only these models for this language (from an audit)
    deny: tuple[str, ...] = ()            # never these models for this language


@dataclass
class Config:
    models: dict[str, Model]
    baseline: str                          # what the team would use without yaptax; savings are measured against it
    expected_output_tokens: int = 300      # English-equivalent answer length, used to rank models before the answer exists
    languages: dict[str, LanguageRule] = field(default_factory=dict)
    database: str = "yaptax-router.db"
    demo: bool = False                     # set by `yaptax demo`: fake provider, equal demo prices

    def allowed(self, lang: str) -> list[Model]:
        rule = self.languages.get(lang, LanguageRule())
        names = list(self.models) if rule.allow is None else [n for n in rule.allow if n in self.models]
        names = [n for n in names if n not in rule.deny]
        return [self.models[n] for n in names] or [self.models[self.baseline]]


class ConfigError(ValueError):
    pass


def from_dict(d: dict) -> Config:
    raw = d.get("models") or []
    if not raw:
        raise ConfigError("config needs at least one [[models]] entry")
    models: dict[str, Model] = {}
    for m in raw:
        try:
            model = Model(
                name=m["name"],
                base_url=m["base_url"].rstrip("/"),
                tokenizer=m.get("tokenizer", "o200k"),
                input_per_million=float(m["input_per_million"]),
                output_per_million=float(m["output_per_million"]),
                api_key_env=m.get("api_key_env"),
                upstream_model=m.get("upstream_model"),
            )
        except KeyError as e:
            raise ConfigError(f"model entry is missing {e.args[0]!r}: {m}") from None
        if model.input_per_million <= 0 or model.output_per_million <= 0:
            raise ConfigError(f"model {model.name!r}: set input_per_million and output_per_million from the "
                              f"provider's pricing page (USD per 1M tokens, above 0)")
        if model.name in models:
            raise ConfigError(f"model {model.name!r} is listed twice")
        models[model.name] = model
    baseline = d.get("baseline") or next(iter(models))
    if baseline not in models:
        raise ConfigError(f"baseline {baseline!r} is not one of the models")
    langs = {}
    for code, rule in (d.get("languages") or {}).items():
        allow = tuple(rule["allow"]) if "allow" in rule else None
        deny = tuple(rule.get("deny", ()))
        for n in (allow or ()) + deny:
            if n not in models:
                raise ConfigError(f"languages.{code} names unknown model {n!r}")
        langs[code] = LanguageRule(allow, deny)
    return Config(
        models=models,
        baseline=baseline,
        expected_output_tokens=int(d.get("expected_output_tokens", 300)),
        languages=langs,
        database=d.get("database", "yaptax-router.db"),
        demo=bool(d.get("demo", False)),
    )


def load(path: str | Path) -> Config:
    with open(path, "rb") as f:
        return from_dict(tomllib.load(f))
