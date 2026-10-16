"""Header ablation: Qwen3-14B fine-tuned on the training set with the bold reasoning headers removed
(data/training/gpt52_high_reasoning_no_headers.json; same recipe, seed 42, seed-42 split), compared with Base and
the reported SFT adapter (seed 42) and seeds 1 and 2, all from raw records with the repo's scoring code.

Prints: the seed-table columns (scripts/analysis_followups.table_training_seeds), a direct paired comparison of
no-headers vs SFT seed 42, how often each model's reasoning opens with a bold header, reasoning length, degenerate
traces, the Haskins per-task means, and the ReasonIF passes that the manual-review rule flags as candidates.

  python scripts/no_headers_summary.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_followups as F  # noqa: E402

LABEL = "Qwen3-14B SFT no headers"
V = F.RES / "data_variants" / "no_headers"
HEADER_START = re.compile(r"^\s*\*\*[^*\n]{1,120}\*\*")

H, R = F.haskins_runs(), F.reasonif_runs()
H = {k: v for k, v in H.items() if k.startswith("Qwen3-14B")}
R = {k: v for k, v in R.items() if k.startswith("Qwen3-14B")}
H[LABEL] = F._prep(pd.read_csv(F._first(V / "haskins_500" / "qwen3_14b" / "*" / "results.csv")))
R[LABEL] = ("qwen3_14b/no_headers", F._jl(F._first(V / "reasonif_300" / "qwen3_14b" / "*" / "scored_responses.jsonl")))

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
t = F.table_training_seeds(H, R)
print("== Seed-table columns (Haskins = CoT-Control 500; gains are paired vs Base, 95% cluster bootstrap)")
print(t.T.to_string(header=False))

print("\n== No headers vs SFT seed 42 (paired; positive = no-headers higher)")
rng = np.random.default_rng(F.SEED)
for tag, kw in (("CoT-Control full trace", {}), ("CoT-Control first 100 words", {"window": 100})):
    n, a, b, g, lo, hi = F._paired_gain(H["Qwen3-14B SFT seed 42"], H[LABEL], rng, **kw)
    print(f"{tag}: n={n}, SFT {a:.2f}, no headers {b:.2f}, diff {g:+.2f} [{lo:+.2f}, {hi:+.2f}]")
s42 = {r["dataset_index"]: r for r in R["Qwen3-14B SFT seed 42"][1]}
for key in ("instruction_following", "answer_correct", "joint_success"):
    d = np.array([int(bool(r[key])) - int(bool(s42[r["dataset_index"]][key])) for r in R[LABEL][1]])
    lo, hi = F._ci(d)
    print(f"ReasonIF {key}: diff {100 * d.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] (n={len(d)})")

print("\n== Reasoning openings, length and degenerate traces")
rows = []
for label in H:
    h = H[label]
    rr = [str(r.get("reasoning_content") or "") for r in R[label][1]] if label in R else []
    rows.append({"Model": label.replace("Qwen3-14B ", ""),
                 "CoT-Control opens with **header**": f"{sum(bool(HEADER_START.match(x)) for x in h.reasoning)}/{len(h)}",
                 "ReasonIF opens with **header**": f"{sum(bool(HEADER_START.match(x)) for x in rr)}/{len(rr)}" if rr else "",
                 "CoT-Control median words": int(h.words.median()),
                 "ReasonIF median words": int(np.median([len(x.split()) for x in rr])) if rr else "",
                 "CoT-Control degenerate": int(h.degen.sum())})
print(pd.DataFrame(rows).to_string(index=False))

print("\n== CoT-Control mean compliance by task (%)")
print(pd.DataFrame({k.replace("Qwen3-14B ", ""): v.groupby("task").primary.mean() * 100 for k, v in H.items()}).round(1).to_string())

print("\n== ReasonIF passes flagged for manual review in the no-headers run (rule: under 20 reasoning words, or a language item)")
for r in R[LABEL][1]:
    words = len(str(r.get("reasoning_content") or "").split())
    if r["instruction_following"] and (words < 20 or "language" in str(r.get("constraint", ""))):
        print(f"  idx {r['dataset_index']}: {r.get('constraint')} words={words}")
