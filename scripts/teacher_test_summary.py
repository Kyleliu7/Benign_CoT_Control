"""Teacher-controllability test and self-distillation runs on CoT-Control (Qwen3-14B, 500 traces each).

Runs: Base, SFT seed 42 (GPT-5.2 data), self-distillation (Qwen3-14B's own reasoning on the GPT-5.2 prompts),
self-distillation with the GPT-5.2 opening headings, and the s1K students trained on Claude 3.7 Sonnet or DeepSeek R1
reasoning for the same 212 s1K questions (scripts/make_s1k_teacher_datasets.py). For each run: mean as recorded and
with a leading bold heading removed before grading, the paired gain over Base (95% bootstrap over the 50 questions),
degenerate traces, median words, traces that hit the token limit, and per-task header-stripped means and gains.

  python scripts/teacher_test_summary.py
"""
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_followups as F  # noqa: E402
from haskins_header_stripped import rescore, runs  # noqa: E402

H = runs()
EXTRA = {"SFT self-distillation": "haskins_500/qwen3_14b_training_variants/self_distill"}
for label, sub in EXTRA.items():
    f = glob.glob(str(F.RES / sub / "haskins_500/qwen3_14b/*/results.csv"))
    if f:
        H[label] = F._prep(pd.read_csv(f[0]))
KEEP = ["Base", "SFT seed 42", "SFT self-distillation", "SFT self headers", "SFT s1k claude37", "SFT s1k deepseek r1"]
H = {k: rescore(H[k]) for k in KEEP if k in H}
base = H["Base"]
rows = []
for k, d in H.items():
    row = {"Model": k, "Leading header": f"{int(d.had_header.sum())}/{len(d)}",
           "Mean raw": round(100 * d.primary.mean(), 2), "Mean header-stripped": round(100 * d.primary_s.mean(), 2),
           "Degenerate": int(d.degen.sum()), "Hit token limit": int(d.capped.sum()) if "capped" in d else "",
           "Median words": int(d.words.median())}
    if k != "Base":
        for col, tag in (("primary", "raw"), ("primary_s", "stripped")):
            m = base.merge(d, on=["task", "q"], suffixes=("_b", "_m"))
            diff = (m[col + "_m"] - m[col + "_b"]).to_numpy()
            lo, hi = F._ci(pd.Series(diff).groupby(m.q.values).mean())
            row[f"Gain vs Base, {tag} [95% CI]"] = f"{100 * diff.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"
    rows.append(row)
pd.set_option("display.width", 300)
print(pd.DataFrame(rows).fillna("").to_string(index=False))
print("\nPer task, header-stripped mean (%)")
print(pd.DataFrame({k: d.groupby("task").primary_s.mean() * 100 for k, d in H.items()}).round(1).to_string())
print("\nPer task, header-stripped gain over Base in points [95% CI over questions]")
gt = {}
for k, d in H.items():
    if k == "Base":
        continue
    col = {}
    for task in sorted(base.task.unique()):
        m = base[base.task == task].merge(d[d.task == task], on=["task", "q"], suffixes=("_b", "_m"))
        diff = (m.primary_s_m - m.primary_s_b).to_numpy()
        lo, hi = F._ci(diff)
        col[task] = f"{100 * diff.mean():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    gt[k] = col
print(pd.DataFrame(gt).to_string())
CHAR = {"lowercase_thinking", "uppercase_thinking", "alternating_case"}
print("\nMean over the 7 tasks that are not character-level (all but " + ", ".join(sorted(CHAR)) + "), header-stripped")
rows = []
for k, d in H.items():
    s = d[~d.task.isin(CHAR)]
    row = {"Model": k, "Mean %": round(100 * s.primary_s.mean(), 2)}
    if k != "Base":
        b = base[~base.task.isin(CHAR)]
        m = b.merge(s, on=["task", "q"], suffixes=("_b", "_m"))
        diff = (m.primary_s_m - m.primary_s_b).to_numpy()
        lo, hi = F._ci(pd.Series(diff).groupby(m.q.values).mean())
        row["Gain vs Base [95% CI]"] = f"{100 * diff.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"
    rows.append(row)
print(pd.DataFrame(rows).fillna("").to_string(index=False))
print("\nFirst 120 characters of the first trace per run (lowercase_thinking, question 0)")
for k, d in H.items():
    x = d[(d.task == "lowercase_thinking") & (d.q == d.q.min())].reasoning
    print(f"{k}: {x.iloc[0][:120]!r}" if len(x) else f"{k}: -")
