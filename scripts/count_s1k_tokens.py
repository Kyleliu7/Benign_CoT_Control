"""Count how many of the 212 selected s1K examples (both teachers) exceed a training cutoff under another model's
tokenizer and chat template, to check that the Qwen-selected set also fits a different student.

  python scripts/count_s1k_tokens.py <model id> <cutoff>      e.g. microsoft/Phi-4-reasoning 12288
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

D = Path(__file__).resolve().parents[1] / "data" / "training"
model, cutoff = sys.argv[1], int(sys.argv[2])
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
out = {"model": model, "cutoff": cutoff}
for name in ("s1k_claude37_212", "s1k_deepseek_r1_212"):
    lens = []
    for ex in json.loads((D / f"{name}.json").read_text(encoding="utf-8")):
        prompt = tok.apply_chat_template([{"role": "user", "content": ex["instruction"]}], tokenize=False, add_generation_prompt=True)
        lens.append(len(tok(prompt + ex["output"], add_special_tokens=False)["input_ids"]))
    out[name] = {"n": len(lens), "over_cutoff": sum(n > cutoff for n in lens), "max_tokens": max(lens),
                 "median_tokens": sorted(lens)[len(lens) // 2]}
print(json.dumps(out, indent=1))
