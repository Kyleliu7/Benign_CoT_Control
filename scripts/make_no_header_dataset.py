"""Header ablation: remove the bold section headers (e.g. **Crafting a comprehensive guide**) from the GPT-5.2
reasoning in the training set, leaving everything else unchanged.

Only text inside <think>...</think> is edited. Each header, with the whitespace around it, becomes a paragraph
break; a header at the start of the reasoning is removed. The answer after </think> is copied byte for byte.
Writes data/training/gpt52_high_reasoning_no_headers.json (same 212 examples, same order) and registers it in
data/training/dataset_info.json.

  python scripts/make_no_header_dataset.py
"""
import json
import re
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "data" / "training"
SRC, DST = D / "gpt52_high_reasoning_original.json", D / "gpt52_high_reasoning_no_headers.json"
THINK = re.compile(r"<think>(.*?)</think>", re.S)
HEADER = re.compile(r"\s*\*\*[^*\n]{1,120}\*\*\s*")

data = json.loads(SRC.read_text(encoding="utf-8"))
out, n_headers, words_before, words_after = [], 0, 0, 0
for ex in data:
    m = THINK.search(ex["output"])
    assert m and ex["output"].count("<think>") == 1, "expected exactly one <think> block"
    think = m.group(1)
    n_headers += len(HEADER.findall(think))
    new = HEADER.sub("\n\n", think).strip()
    assert "**" not in new, new[:200]
    words_before += len(think.split())
    words_after += len(new.split())
    output = ex["output"][:m.start(1)] + new + ex["output"][m.end(1):]
    assert output[output.index("</think>"):] == ex["output"][ex["output"].index("</think>"):]
    out.append({**ex, "output": output})

DST.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
info = json.loads((D / "dataset_info.json").read_text(encoding="utf-8"))
info["gpt52_high_reasoning_no_headers"] = {**info["gpt52_high_reasoning_original"], "file_name": DST.name}
(D / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
print(f"{len(out)} examples, {n_headers} headers removed, reasoning words {words_before} -> {words_after}")
