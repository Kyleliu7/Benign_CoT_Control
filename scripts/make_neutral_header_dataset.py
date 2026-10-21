"""Neutral-header control: the no-header training set (scripts/make_no_header_dataset.py) with one fixed,
content-free header, **Reasoning**, at the start of every reasoning trace. Compared with the no-header data the only
change is that leading header; compared with the original data the informative header text and the later headers
are gone. Writes data/training/gpt52_high_reasoning_neutral_header.json and registers it in dataset_info.json.

  python scripts/make_no_header_dataset.py && python scripts/make_neutral_header_dataset.py
"""
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "data" / "training"
HEADER = "**Reasoning**\n\n"
SRC, DST = D / "gpt52_high_reasoning_no_headers.json", D / "gpt52_high_reasoning_neutral_header.json"

data = json.loads(SRC.read_text(encoding="utf-8"))
out = []
for ex in data:
    assert ex["output"].startswith("<think>") and "**" not in ex["output"].split("</think>")[0]
    out.append({**ex, "output": "<think>" + HEADER + ex["output"][len("<think>"):]})
DST.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
info = json.loads((D / "dataset_info.json").read_text(encoding="utf-8"))
info["gpt52_high_reasoning_neutral_header"] = {**info["gpt52_high_reasoning_original"], "file_name": DST.name}
(D / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
print(f"{len(out)} examples; first trace starts: {out[0]['output'][:80]!r}")
