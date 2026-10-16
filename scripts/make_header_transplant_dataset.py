"""Header transplant: the Qwen3-235B-A22B training set (same 212 prompts, raw reasoning, no bold headings) with the
GPT-5.2 example's opening heading for the same prompt placed at the start of each reasoning trace.

For every Qwen3-235B example, the GPT-5.2 example with the identical instruction is looked up in
data/training/gpt52_high_reasoning_original.json; its first **...** heading is written as "<think>**Heading**\n\n"
followed by the Qwen3-235B reasoning unchanged. Nothing else changes. If the Qwen3-235B trace already opens with a
bold heading, the example is left unchanged and counted. Writes data/training/qwen3_235b_gpt52_headers.json and
registers it (and the untouched Qwen3-235B set, copied to data/training/qwen3_235b_gpt52_prompts_sft.json) in
data/training/dataset_info.json.

  python scripts/make_header_transplant_dataset.py <path to qwen3_235b_gpt52_prompts_sft.json>
  python scripts/make_header_transplant_dataset.py <source.json> <copy name> <output name>
      e.g. qwen3_14b_gpt52_prompts_sft.json qwen3_14b_gpt52_prompts_sft qwen3_14b_self_gpt52_headers (self-distillation)
"""
import json
import re
import shutil
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "data" / "training"
HEAD = re.compile(r"\*\*[^*\n]{1,120}\*\*")

src = Path(sys.argv[1])
ORIG = sys.argv[2] if len(sys.argv) > 2 else "qwen3_235b_gpt52_prompts_sft"
OUT = sys.argv[3] if len(sys.argv) > 3 else "qwen3_235b_gpt52_headers"
dst_orig = D / f"{ORIG}.json"
if not dst_orig.exists():
    shutil.copyfile(src, dst_orig)
q235 = json.loads(dst_orig.read_text(encoding="utf-8"))
gpt = {e["instruction"]: e for e in json.loads((D / "gpt52_high_reasoning_original.json").read_text(encoding="utf-8"))}

out, n_match, n_had_header, missing = [], 0, 0, []
for ex in q235:
    keys = set(ex)
    assert {"instruction", "output"} <= keys, keys
    g = gpt.get(ex["instruction"])
    if g is None:
        missing.append(ex["instruction"][:80])
        out.append(ex)
        continue
    n_match += 1
    head = HEAD.search(re.search(r"<think>(.*?)</think>", g["output"], re.S).group(1)).group(0)
    o = ex["output"]
    assert o.startswith("<think>"), o[:40]
    body = o[len("<think>"):]
    if HEAD.match(body.lstrip()):
        n_had_header += 1
        out.append(ex)
        continue
    out.append({**ex, "output": "<think>" + head + "\n\n" + body.lstrip("\n")})

(D / f"{OUT}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
info = json.loads((D / "dataset_info.json").read_text(encoding="utf-8"))
for name in (ORIG, OUT):
    info[name] = {**info["gpt52_high_reasoning_original"], "file_name": name + ".json"}
(D / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"n": len(out), "matched_gpt52_instruction": n_match, "already_had_header": n_had_header,
                  "unmatched": len(missing), "unmatched_examples": missing[:5],
                  "first_output_start": out[0]["output"][:160]}, indent=1, ensure_ascii=False))
