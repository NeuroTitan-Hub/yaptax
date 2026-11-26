# langtax

**Measure the language tax in your AI prompts.** The same meaning takes more tokens outside English, and you pay per token. `langtax` counts your own prompts per language, on the tokenizers your models use, and tells you how many of those tokens English would not need.

Everything runs on your machine. Your text is never uploaded.

```
pip install langtax          # once released; until then: pip install git+https://github.com/NeuroTitan-Hub/langtax
langtax measure prompts.jsonl
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
| `langtax measure FILE` | Tokens per language in a file, and how many are language tax |
| `langtax measure FILE --spend 2000` | Also shows the tax in money for a monthly spend |
| `langtax measure FILE --lang hi` | Treat every record as one language |
| `langtax measure FILE --tokenizers o200k,qwen3` | Choose tokenizers (see below) |
| `langtax measure FILE --json` | Machine-readable output |
| `langtax ratios Tamil` | The Index's ratios for one language, on all 8 tokenizers |

**Input:** `.jsonl`, `.json`, `.csv`, `.tsv`, or plain text with one prompt per line (`-` reads stdin).
- Text is taken from a `text`, `prompt`, `content`, `input`, `message`, `query` or `question` field, or from chat `messages`. Use `--text-field` for another name.
- Language is taken from a `lang`, `language` or `locale` field if there is one (`hi`, `hi-IN`, `hin_Deva` and `Hindi` all work). Use `--lang-field` for another name.
- Without a label, the language is guessed from the writing system. All Devanagari is scored as Hindi, for example. Latin-script text is only scored as English if it looks like English; otherwise it is counted but not scored. Label your data for exact results.

## Tokenizers

| ID | Used by | Needs |
|---|---|---|
| `o200k` | GPT-4o, GPT-4.1, GPT-5, o3 | built in |
| `cl100k` | GPT-4, GPT-3.5 | built in |
| `gpt2` | GPT-2 | built in |
| `utf8-bytes` | byte-level models | built in |
| `qwen3`, `deepseek-v3`, `mistral-nemo`, `bloom` | those open models | `pip install 'langtax[hf]'`; fetched from Hugging Face at the exact revisions the Index used |

Anthropic and Google do not publish their tokenizers as files, so they are not here yet.

## How "extra tokens" is computed

For each language, langtax divides your token count by that language's ratio to English in the [Language Cost Index](https://neurotitan.in/cost): the same 1,012 FLORES-200 sentences, measured in 204 languages on 8 tokenizers. The difference is the language tax. It is an average, so a single short text can differ.

**Checked:** on FLORES-200 devtest, `langtax measure` gives exactly the token totals published with the Index for Hindi, Tamil and English on o200k, cl100k, Qwen3 and BLOOM.

**What it does not tell you:** fewer tokens is not the same as a cheaper or better model. Prices per token differ between models, and so does answer quality in each language. Measuring those on your own traffic is what our Language Cost Audit does: write to the address on [neurotitan.in](https://neurotitan.in).

## Develop

```
pip install -e ".[test]"
pytest
```

MIT licence. The bundled Index data comes from [Sentence-cost-in-Electron](https://github.com/NeuroTitan-Hub/Sentence-cost-in-Electron); FLORES-200 is by Meta AI (CC BY-SA 4.0).

By [NeuroTitan Labs](https://neurotitan.in).
