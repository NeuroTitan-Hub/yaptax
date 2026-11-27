# yaptax

*Every language at English price.* (Formerly `langtax`.)

**Measure the language tax in your AI prompts.** The same meaning takes more tokens outside English, and you pay per token. `yaptax` counts your own prompts per language, on the tokenizers your models use, and tells you how many of those tokens English would not need.

Everything runs on your machine. Your text is never uploaded.

```
pip install yaptax          # once released; until then: pip install git+https://github.com/NeuroTitan-Hub/yaptax
yaptax measure prompts.jsonl
```

```
Language                           Records  Label     o200k tokens  o200k extra  cl100k tokens  cl100k extra
---------------------------------  -------  --------  ------------  -----------  -------------  ------------
Hindi (hin_Deva)                         2  labelled            47          36%            142           79%
Spanish (spa_Latn)                       2  labelled            33          24%             38           34%
English (eng_Latn)                       2  labelled            31           0%             31            0%
Tamil (tam_Taml)                         1  labelled            28          50%            114           87%
...

o200k: 207 tokens. The same meaning in English would take about 148, so about 59 (29%) are language tax.
```

That is `examples/support.jsonl`, ten short customer-support messages.

## Commands

| Command | What it does |
|---|---|
| `yaptax measure FILE` | Tokens per language in a file, and how many are language tax |
| `yaptax measure FILE --spend 2000` | Also shows the tax in money for a monthly spend |
| `yaptax measure FILE --lang hi` | Treat every record as one language |
| `yaptax measure FILE --tokenizers o200k,qwen3` | Choose tokenizers (see below) |
| `yaptax measure FILE --json` | Machine-readable output |
| `yaptax ratios Tamil` | The Index's ratios for one language, on all 8 tokenizers |
| `yaptax demo --open` | The router on a fake provider, with a live savings dashboard (needs `yaptax[router]`) |
| `yaptax router --config router.toml` | The router in front of your real models, with your own keys |

**Input:** `.jsonl`, `.json`, `.csv`, `.tsv`, or plain text with one prompt per line (`-` reads stdin).
- Text is taken from a `text`, `prompt`, `content`, `input`, `message`, `query` or `question` field, or from chat `messages`. Use `--text-field` for another name.
- Language is taken from a `lang`, `language` or `locale` field if there is one (`hi`, `hi-IN`, `hin_Deva` and `Hindi` all work). Use `--lang-field` for another name.
- Without a label, the language is guessed from the writing system. All Devanagari is scored as Hindi, for example. Latin-script text is only scored as English if it looks like English; otherwise it is counted but not scored. Label your data for exact results.

## yaptax Router: cut the tax automatically

Measuring is step one. The router is step two: an OpenAI-compatible endpoint that sends each request to the cheapest model you allow for its language, and shows what that saved.

```
pip install "yaptax[router]"   # once released; until then: pip install "yaptax[router] @ git+https://github.com/NeuroTitan-Hub/yaptax"
yaptax demo --open
```

`yaptax demo` runs the router against a built-in fake provider, sends 34 multilingual support messages through it and opens a live savings dashboard. Every model gets the same demo price, so the saving you see comes only from tokens. No real model is called and nothing is paid. On the demo set, routing saves 53% against sending every language to Qwen3:

| Language | Routed to | Saved |
|---|---|---|
| Hindi | gpt-4o | 64% |
| Tamil | gpt-4o | 68% |
| Telugu | gpt-4o | 72% |
| Thai | deepseek-v3 | 31% |
| Arabic | mistral-nemo | 24% |
| Amharic, English | qwen3 (baseline) | 0% |

For real traffic, write a config with your models, your prices and your keys (see [`examples/router.toml`](examples/router.toml)), then:

```
yaptax router --config router.toml
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="unused")
reply = client.chat.completions.create(model="yaptax-auto", messages=[{"role": "user", "content": "मेरा ऑर्डर कहाँ है?"}])
```

How it decides:
- **Language:** the `x-yaptax-lang` header if you send one; otherwise a guess from the user's text.
- **Cost:** for each allowed model, input tokens on its tokenizer times its input price, plus an expected answer length on its tokenizer times its output price. OpenAI's tokenizers are counted exactly; others are estimated from the Language Cost Index. It picks the cheapest. On a tie it stays on your baseline.
- **Pinning:** send a configured model name instead of `yaptax-auto` and the router uses that model, logging it like any other request.
- **Safety:** per-language `allow`/`deny` rules (from an audit of answer quality). If the chosen model is down (5xx), it retries on your baseline.
- **Accounting:** the actual cost uses the provider's reported usage. The baseline cost is an estimate of the same request on your baseline, with the answer converted to its tokenizer using the Index ratios.
- **Privacy:** the log keeps language, models, token counts, costs and timing. No prompt or answer text, and keys stay in your environment.
- **Endpoints:** `POST /v1/chat/completions` (streaming supported), `GET /v1/models`, `GET /stats`, and the dashboard at `/`.

What it doesn't do yet: judge answer quality by itself (use per-language rules), or route by latency.

## Tokenizers

| ID | Used by | Needs |
|---|---|---|
| `o200k` | GPT-4o, GPT-4.1, GPT-5, o3 | built in |
| `cl100k` | GPT-4, GPT-3.5 | built in |
| `gpt2` | GPT-2 | built in |
| `utf8-bytes` | byte-level models | built in |
| `qwen3`, `deepseek-v3`, `mistral-nemo`, `bloom` | those open models | `pip install 'yaptax[hf]'`; fetched from Hugging Face at the exact revisions the Index used |

Anthropic and Google do not publish their tokenizers as files, so they are not here yet.

## How "extra tokens" is computed

For each language, yaptax divides your token count by that language's ratio to English in the [Language Cost Index](https://neurotitan.in/cost): the same 1,012 FLORES-200 sentences, measured in 204 languages on 8 tokenizers. The difference is the language tax. It is an average, so a single short text can differ.

**Checked:** on FLORES-200 devtest, `yaptax measure` gives exactly the token totals published with the Index for Hindi, Tamil and English on o200k, cl100k, Qwen3 and BLOOM.

**What it does not tell you:** fewer tokens is not the same as a cheaper or better model. Prices per token differ between models, and so does answer quality in each language. Measuring those on your own traffic is what our Language Cost Audit does: write to the address on [neurotitan.in](https://neurotitan.in).

## Develop

```
pip install -e ".[test]"
pytest
```

MIT licence. The bundled Index data comes from [Sentence-cost-in-Electron](https://github.com/NeuroTitan-Hub/Sentence-cost-in-Electron); FLORES-200 is by Meta AI (CC BY-SA 4.0).

By [NeuroTitan Labs](https://neurotitan.in).
