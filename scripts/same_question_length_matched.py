"""CoT-Control gain on pairs matched on question AND length: for each run, keep the seven-task question-constraint pairs
whose Base and fine-tuned traces have similar reasoning length (word ratio within 1.5x), and compute the paired gain with
the cluster bootstrap over the 50 questions (analysis_followups._ci). Writes results/summary_tables/same_question_length_matched.txt.

  python scripts/same_question_length_matched.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_followups as F  # noqa: E402
from seven_task_composite import SEVEN  # noqa: E402

HR = F.haskins_runs()
csv = lambda p: F._prep(pd.read_csv(glob.glob(str(F.RES / p))[0]))  # noqa: E731
RUNS = [("Qwen3-14B", "Qwen3-14B Base", k, HR[k]) for k in ("Qwen3-14B SFT seed 42", "Qwen3-14B SFT seed 1", "Qwen3-14B SFT seed 2")]
RUNS += [("Qwen3-14B", "Qwen3-14B Base", "s1K Claude 3.7", csv("data_variants/s1k_claude37/haskins_500/qwen3_14b/*/results.csv")),
         ("Qwen3-14B", "Qwen3-14B Base", "s1K R1", csv("data_variants/s1k_deepseek_r1/haskins_500/qwen3_14b/*/results.csv")),
         ("Phi-4-reasoning", "Phi-4-reasoning Base", "GPT-5.2 SFT", HR["Phi-4-reasoning SFT"]),
         ("OLMo-3-7B-Think", "OLMo-3-7B-Think Base", "GPT-5.2 SFT", HR["OLMo-3-7B-Think SFT lr1e-4"])]
rows = []
for model, bkey, lab, d in RUNS:
    m = HR[bkey].merge(d, on=["task", "q"], suffixes=("_b", "_s"))
    m = m[m.task.isin(SEVEN)]
    full = (m.primary_s - m.primary_b).to_numpy()
    ratio = np.abs(np.log((m.words_s + 1) / (m.words_b + 1)))
    k = m[ratio <= np.log(1.5)]
    diff = (k.primary_s - k.primary_b)
    lo, hi = F._ci(diff.groupby(k.q.values).mean()) if len(k) else (np.nan, np.nan)
    rows.append({"Model": model, "Run": lab, "All 350 pairs": f"{100 * full.mean():+.2f}",
                 "Same question, length within 1.5x": f"{100 * diff.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]",
                 "Pairs kept": f"{len(k)}/350", "Questions kept": k.q.nunique()})
pd.set_option("display.width", 250)
text = pd.DataFrame(rows).to_string(index=False)
print(text)
(F.RES / "summary_tables/same_question_length_matched.txt").write_text(text + "\n", encoding="utf-8")
