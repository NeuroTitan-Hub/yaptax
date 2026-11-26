"""Token counters. OpenAI's tokenizers come from tiktoken. Open-model tokenizers are optional:
they need `pip install yaptax[hf]` and are fetched from Hugging Face at the exact revisions
the Index was measured with, so your counts and the Index use the same files."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

TIKTOKEN = {"o200k": "o200k_base", "cl100k": "cl100k_base", "gpt2": "gpt2"}

# Same repositories and revisions as the Index (post 1, m1/run_info.json).
HF = {
    "qwen3": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "deepseek-v3": ("deepseek-ai/DeepSeek-V3", "e815299b0bcbac849fa540c768ef21845365c9eb"),
    "mistral-nemo": ("mistralai/Mistral-Nemo-Instruct-2407", "04d8a90549d23fc6bd7f642064003592df51e9b3"),
    "bloom": ("bigscience/bloom", "7f10a99ce7c08f03c7719a586cb2cbda1433ac05"),
}

AVAILABLE = [*TIKTOKEN, *HF, "utf8-bytes"]
DEFAULT = ["o200k", "cl100k"]


class TokenizerUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=None)
def counter(tok_id: str) -> Callable[[str], int]:
    if tok_id == "utf8-bytes":
        return lambda s: len(s.encode("utf-8"))
    if tok_id in TIKTOKEN:
        import tiktoken

        enc = tiktoken.get_encoding(TIKTOKEN[tok_id])
        return lambda s: len(enc.encode_ordinary(s))
    if tok_id in HF:
        try:
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as e:
            raise TokenizerUnavailable(f"{tok_id} needs: pip install 'yaptax[hf]'") from e
        repo, rev = HF[tok_id]
        try:
            path = hf_hub_download(repo, "tokenizer.json", revision=rev)
        except Exception as e:  # gated repo, no network, ...
            raise TokenizerUnavailable(f"could not fetch {repo}@{rev[:8]}: {e}") from e
        tk = Tokenizer.from_file(path)
        return lambda s: len(tk.encode(s, add_special_tokens=False).ids)
    raise TokenizerUnavailable(f"unknown tokenizer '{tok_id}'. Choose from: {', '.join(AVAILABLE)}")
