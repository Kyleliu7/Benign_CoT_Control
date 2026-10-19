"""Teacher test on OLMo-3-7B-Think (2026-10-19): students trained on the same 212 s1K questions with Claude 3.7 Sonnet or
DeepSeek R1 reasoning (Qwen s1K recipe, train_olmo3.py), CoT-Control 500 traces each. Prints, next to the Qwen and Phi
s1K students, the seven-task mean, the paired gain over the model's own Base (whole trace and first 100 words), the
Claude 3.7 minus R1 contrast, traces that hit the token limit, degenerate traces and median words, plus per-task means.
Same grading and paired cluster bootstrap over the 50 questions as seven_task_composite.py.

  python scripts/olmo_teacher_summary.py > results/summary_tables/olmo_teacher_test.txt
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_followups as F  # noqa: E402
from haskins_header_stripped import rescore, runs  # noqa: E402
from seven_task_composite import SEVEN  # noqa: E402

H = runs()
H.update({k: v for k, v in F.haskins_runs().items() if not k.startswith("Qwen3-14B")})
for fam, pre, sub in (("Phi-4-reasoning", "phi4_", "phi4_reasoning"), ("OLMo-3-7B-Think", "olmo3_", "olmo3_7b_think")):
    for f in sorted((F.RES / "data_variants").glob(f"{pre}s1k_*/haskins_500/{sub}/*/results.csv")):
        H[f"{fam} SFT " + f.parts[-5].replace(pre, "").replace("_", " ")] = F._prep(pd.read_csv(f))
FAM = {"Qwen3-14B": ("Base", "SFT s1k claude37", "SFT s1k deepseek r1"),
       "Phi-4-reasoning": ("Phi-4-reasoning Base", "Phi-4-reasoning SFT s1k claude37", "Phi-4-reasoning SFT s1k deepseek r1"),
       "OLMo-3-7B-Think": ("OLMo-3-7B-Think Base", "OLMo-3-7B-Think SFT s1k claude37", "OLMo-3-7B-Think SFT s1k deepseek r1")}
keys = {k for v in FAM.values() for k in v}
assert keys <= set(H), sorted(keys - set(H))
H = {k: rescore(H[k]) for k in keys}


def g(b, d, window=None):
    n, _, _, est, lo, hi = F._paired_gain(b, d, None, col="primary", window=window, tasks=SEVEN)
    return f"{est:+.2f} [{lo:+.2f}, {hi:+.2f}] (n={n})"


rows = []
for fam, (b, c, r) in FAM.items():
    for k in (b, c, r):
        d = H[k]
        s = d[d.task.isin(SEVEN)]
        row = {"Model": fam, "Run": k.replace(fam + " ", "") if k != "Base" else "Base", "Mean7": round(100 * s.primary.mean(), 2),
               "Hit token limit": int(d.capped.sum()) if "capped" in d else "", "Degenerate": int(d.degen.sum()),
               "Median words": int(d.words.median())}
        if k != b:
            row["Gain7 vs Base"] = g(H[b], d)
            row["Gain7 first 100 words"] = g(H[b], d, window=100)
        rows.append(row)
pd.set_option("display.width", 300)
print(pd.DataFrame(rows).fillna("").to_string(index=False))
print("\nClaude 3.7 student minus DeepSeek R1 student, seven tasks, paired over the 50 questions")
for fam, (b, c, r) in FAM.items():
    print(f"  {fam}: whole trace {g(H[r], H[c])}; first 100 words {g(H[r], H[c], window=100)}")
print("\nPer task mean (%), OLMo-3-7B-Think")
o = FAM["OLMo-3-7B-Think"]
print(pd.DataFrame({k.replace("OLMo-3-7B-Think ", ""): H[k].groupby("task").primary.mean() * 100 for k in o}).round(1).to_string())

print("\nOLMo-3-7B-Think per-task paired differences (points [95% CI over questions]) and the ten-task gains")
b, c, r = (H[k] for k in o)
ALL = sorted(b.task.unique())
cols = {}
for lab, x, y in (("Claude 3.7 - Base", b, c), ("R1 - Base", b, r), ("Claude 3.7 - R1", r, c)):
    col = {}
    for t in ALL:
        n, _, _, est, lo, hi = F._paired_gain(x, y, None, col="primary", tasks=[t])
        col[t] = f"{est:+.1f} [{lo:+.1f}, {hi:+.1f}]"
    n, _, _, est, lo, hi = F._paired_gain(x, y, None, col="primary", tasks=ALL)
    col["ALL TEN"] = f"{est:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    cols[lab] = col
print(pd.DataFrame(cols).to_string())
print("\nThird person (OLMo): share of traces that hit the token limit, and mean score among traces that did / did not")
for k in o:
    d = H[k][H[k].task == "third_person"]
    print(f"  {k}: capped {int(d.capped.sum())}/50; mean capped {100 * d[d.capped].primary.mean():.1f}, uncapped {100 * d[~d.capped].primary.mean():.1f}")
