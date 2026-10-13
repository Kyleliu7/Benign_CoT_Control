"""Phi-4-reasoning CoT-Control: what is in the length range where Base and SFT traces overlap (155-524 words)?

Prints the task mix of each condition in that range, a task-reweighted comparison (SFT task scores weighted by the Base
task mix), and a paired comparison on question-task pairs where both traces fall in the range.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_followups as F  # noqa: E402

H = F.haskins_runs()
b, s = H["Phi-4-reasoning Base"].copy(), H["Phi-4-reasoning SFT"].copy()
LO, HI = 155, 524
bi, si = b[(b.words >= LO) & (b.words < HI)], s[(s.words >= LO) & (s.words < HI)]
mix = pd.DataFrame({"Base_n": bi.task.value_counts(), "SFT_n": si.task.value_counts(),
                    "Base_mean": bi.groupby("task").primary.mean() * 100, "SFT_mean": si.groupby("task").primary.mean() * 100,
                    "Base_all_mean": b.groupby("task").primary.mean() * 100, "SFT_all_mean": s.groupby("task").primary.mean() * 100}).fillna(0).round(1)
print(mix.to_string())
print(f"\nin-range means: Base {100*bi.primary.mean():.2f} (n={len(bi)}), SFT {100*si.primary.mean():.2f} (n={len(si)})")
common = mix[(mix.Base_n > 0) & (mix.SFT_n > 0)]
wb = common.Base_n / common.Base_n.sum()
print(f"tasks present in both: {len(common)}; Base-mix-weighted means: Base {(common.Base_mean*wb).sum():.2f}, SFT {(common.SFT_mean*wb).sum():.2f}")
m = b.merge(s, on=["task", "q"], suffixes=("_b", "_s"))
pm = m[(m.words_b >= LO) & (m.words_b < HI) & (m.words_s >= LO) & (m.words_s < HI)]
print(f"paired question-task items with both traces in range: {len(pm)}; Base {100*pm.primary_b.mean():.2f}, SFT {100*pm.primary_s.mean():.2f}")
print("\nlength overlap overall: SFT words median", s.words.median(), "Base words median", b.words.median())
