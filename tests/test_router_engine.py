import pytest

from yaptax.router import config, engine


def cfg(**over):
    d = {
        "baseline": "qwen",
        "models": [
            {"name": "qwen", "base_url": "http://x/v1", "tokenizer": "qwen3", "input_per_million": 1, "output_per_million": 2},
            {"name": "gpt", "base_url": "http://x/v1", "tokenizer": "o200k", "input_per_million": 1, "output_per_million": 2},
            {"name": "ds", "base_url": "http://x/v1", "tokenizer": "deepseek-v3", "input_per_million": 1, "output_per_million": 2},
        ],
    }
    d.update(over)
    return config.from_dict(d)


def ask(text, system="You are a helpful support agent."):
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]


HINDI = "मेरा ऑर्डर कहाँ है? मुझे अपने पैसे वापस चाहिए।"
AMHARIC = "ትዕዛዜ የት ነው? ገንዘቤን መመለስ እፈልጋለሁ።"
THAI = "คำสั่งซื้อของฉันอยู่ที่ไหน ฉันต้องการเงินคืน"


def test_picks_the_most_efficient_tokenizer_per_language():
    c = cfg()
    assert engine.decide(c, ask(HINDI)).chosen.name == "gpt"     # o200k 1.57x vs qwen3 4.42x
    assert engine.decide(c, ask(AMHARIC)).chosen.name == "qwen"  # qwen3 4.04x vs o200k 5.78x
    assert engine.decide(c, ask(THAI)).chosen.name == "ds"       # deepseek 1.78x vs o200k 1.96x


def test_language_comes_from_user_text_not_the_english_system_prompt():
    d = engine.decide(cfg(), ask(HINDI, system="You are a helpful assistant. Answer the user politely and briefly."))
    assert d.lang == "hin_Deva" and d.lang_source == "guessed"


def test_header_hint_wins_over_guessing():
    d = engine.decide(cfg(), ask("hola, donde esta mi pedido"), lang_hint="es")
    assert d.lang == "spa_Latn" and d.lang_source == "header"


def test_unknown_language_stays_on_baseline():
    d = engine.decide(cfg(), ask("hola, donde esta mi pedido"))
    assert d.lang_source == "unknown" and d.chosen.name == "qwen"


def test_price_matters_not_just_tokens():
    # gpt reads Hindi in fewer tokens, but at 5x the price qwen is cheaper.
    c = config.from_dict({
        "baseline": "qwen",
        "models": [
            {"name": "qwen", "base_url": "http://x/v1", "tokenizer": "qwen3", "input_per_million": 1, "output_per_million": 2},
            {"name": "gpt", "base_url": "http://x/v1", "tokenizer": "o200k", "input_per_million": 5, "output_per_million": 10},
        ],
    })
    assert engine.decide(c, ask(HINDI)).chosen.name == "qwen"


def test_language_rules_from_an_audit():
    c = cfg(languages={"hin_Deva": {"allow": ["qwen", "ds"]}})
    assert engine.decide(c, ask(HINDI)).chosen.name == "ds"
    c = cfg(languages={"hin_Deva": {"deny": ["gpt"]}})
    assert engine.decide(c, ask(HINDI)).chosen.name == "ds"
    c = cfg(languages={"hin_Deva": {"allow": ["gpt"], "deny": ["gpt"]}})
    assert engine.decide(c, ask(HINDI)).chosen.name == "qwen"  # nothing left: fall back to baseline


def test_exact_counts_for_openai_estimates_for_others():
    d = engine.decide(cfg(), ask(HINDI))
    by = {c.model: c for c in d.candidates}
    assert by["gpt"].exact and not by["qwen"].exact
    # The estimate follows the Index: qwen3/o200k for Hindi is about 4.42/1.57.
    assert by["qwen"].input_tokens / by["gpt"].input_tokens == pytest.approx(4.416 / 1.574, rel=0.1)


def test_settle_counts_savings_against_baseline():
    c = cfg()
    d = engine.decide(c, ask(HINDI))
    o = engine.settle(c, d, prompt_tokens=30, completion_tokens=100)
    assert o.cost == pytest.approx((30 * 1 + 100 * 2) / 1e6)
    # Baseline output = 100 gpt tokens converted to qwen3 tokens with the Index ratios.
    out_base = 100 / 1.574 * 4.416
    assert o.baseline_cost == pytest.approx((d.baseline.input_tokens * 1 + out_base * 2) / 1e6, rel=1e-3)
    assert o.saved == pytest.approx(o.baseline_cost - o.cost)
    assert o.saved > 0


def test_no_saving_claimed_when_baseline_is_chosen():
    c = cfg()
    d = engine.decide(c, ask(AMHARIC))
    o = engine.settle(c, d, 40, 90)
    assert d.chosen.name == "qwen" and o.saved == 0 and o.cost == o.baseline_cost


def test_message_text_handles_content_parts():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {}}]}]
    assert engine.message_text(msgs) == "hello"


@pytest.mark.parametrize(
    "bad,msg",
    [
        ({"models": []}, "at least one"),
        ({"baseline": "nope", "models": [{"name": "a", "base_url": "u", "input_per_million": 1, "output_per_million": 1}]}, "baseline"),
        ({"models": [{"name": "a", "base_url": "u", "input_per_million": 1}]}, "output_per_million"),
        ({"models": [{"name": "a", "base_url": "u", "input_per_million": 1, "output_per_million": 1}], "languages": {"hin_Deva": {"allow": ["zzz"]}}}, "unknown model"),
    ],
)
def test_config_errors(bad, msg):
    with pytest.raises(config.ConfigError, match=msg):
        config.from_dict(bad)
