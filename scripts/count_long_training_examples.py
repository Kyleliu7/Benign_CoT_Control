"""How many of the 212 training examples exceed the 4,096-token training cutoff, per model tokenizer?

LlamaFactory truncates each example to cutoff_len = 4,096 tokens (prompt plus target); anything beyond is dropped, so
the end of the answer (and the end-of-turn token) of the longest examples is not trained. The length is counted as
the tokenizer's chat-template generation prompt for the instruction plus the stored target, which is how
scripts/mechanism/surprisal_by_segment.py builds the sequence; the exact template tokens differ slightly from
LlamaFactory's, so counts within a few tokens of the cutoff are approximate.

  python scripts/count_long_training_examples.py Qwen/Qwen3-14B microsoft/Phi-4-reasoning allenai/Olmo-3-7B-Think
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

DATA = Path(__file__).resolve().parents[1] / "data/training/gpt52_high_reasoning_original.json"
CUTOFF = 4096
data = json.loads(DATA.read_text(encoding="utf-8"))
out = {}
for name in sys.argv[1:]:
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    lens, reasoning_cut = [], 0
    for ex in data:
        prompt = tok.apply_chat_template([{"role": "user", "content": ex["instruction"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
        think_end = ex["output"].index("</think>") + len("</think>")
        n_reason = len(tok(ex["output"][:think_end], add_special_tokens=False)["input_ids"])
        n_all = len(tok(prompt + ex["output"], add_special_tokens=False)["input_ids"])
        lens.append(n_all)
        reasoning_cut += n_prompt + n_reason > CUTOFF
    over = [n for n in lens if n > CUTOFF]
    out[name] = {"examples": len(lens), "over_cutoff": len(over), "max_tokens": max(lens),
                 "median_tokens": sorted(lens)[len(lens) // 2], "reasoning_reaches_cutoff": reasoning_cut,
                 "tokens_beyond_cutoff_total": int(sum(n - CUTOFF for n in over))}
print(json.dumps(out, indent=1))
