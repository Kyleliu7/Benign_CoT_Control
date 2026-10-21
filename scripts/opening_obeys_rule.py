"""Across all Qwen3-14B runs: does the opening of the reasoning obey the constraint, and does that track compliance?

The opening is the first bold heading if the reasoning starts with one, otherwise its first sentence (up to . ! ? or a
line break, at most 200 characters). For the lowercase and uppercase tasks it obeys if all its letters are lowercase /
uppercase. For each run: share of openings that obey, compliance after obeying and other openings (graded with a
leading heading removed) and of the reasoning after the opening alone (so the opening itself is not graded), and the
task mean; then the Spearman correlation across runs between the share of obeying
openings and the task mean.

  python scripts/opening_obeys_rule.py
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis_followups as F  # noqa: E402
from haskins_header_stripped import rescore, runs  # noqa: E402
from evaluators.haskins_evaluator import grade_paper_protocol, primary_score  # noqa: E402

HEAD = re.compile(r"^\s*\*\*([^*\n]{1,120})\*\*")


def opening(text):
    m = HEAD.match(text)
    if m:
        return m.group(1)
    t = text.strip()
    e = re.search(r"[.!?](\s|$)|\n", t)
    return t[: e.end() if e else 200][:200]


def after_opening(text):
    """The reasoning with its opening (heading or first sentence) removed."""
    t = str(text).strip()
    m = HEAD.match(t)
    if m:
        return t[m.end():].strip()
    e = re.search(r"[.!?](\s|$)|\n", t)
    return t[e.end():].strip() if e else ""


def obeys(task, s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    return all(c.isupper() for c in letters) if task == "uppercase_thinking" else all(c.islower() for c in letters)


H = runs()
for v in ("output_mask", "think_mask", "teacher_qwen3_235b", "teacher_sonnet_3_7", "teacher_reasonflux", "self_distill"):
    f = glob.glob(str(F.RES / f"haskins_500/qwen3_14b_training_variants/{v}/haskins_500/qwen3_14b/*/results.csv"))
    if f:
        H[v] = F._prep(pd.read_csv(f[0]))
H = {k: rescore(v) for k, v in H.items()}
rows = []
for name, d in H.items():
    for task in ("lowercase_thinking", "uppercase_thinking"):
        g = d[d.task == task]
        ob = np.array([obeys(task, opening(str(x))) for x in g.reasoning])
        comp = g.primary_s.to_numpy()
        rest = np.array([primary_score(task, grade_paper_protocol(task, q, after_opening(x))) for q, x in zip(g.q, g.reasoning)])
        rows.append({"run": name, "task": task, "openings obeying": int(ob.sum()), "task mean": round(100 * comp.mean(), 1),
                     "after obeying opening": round(100 * comp[ob].mean(), 1) if ob.any() else np.nan,
                     "after other opening": round(100 * comp[~ob].mean(), 1) if (~ob).any() else np.nan,
                     "rest only, after obeying": round(100 * rest[ob].mean(), 1) if ob.any() else np.nan,
                     "rest only, after other": round(100 * rest[~ob].mean(), 1) if (~ob).any() else np.nan})
t = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(t.to_string(index=False))
SUMMARY = {}
for task in ("lowercase_thinking", "uppercase_thinking"):
    s = t[t.task == task]
    r, p = spearmanr(s["openings obeying"], s["task mean"])
    print(f"{task}: Spearman across {len(s)} runs, obeying openings vs task mean: rho={r:.2f} (p={p:.3g})")
    SUMMARY[f"rho_{task}"] = round(float(r), 2)
    SUMMARY["n_runs"] = int(len(s))
pooled = t.dropna(subset=["after obeying opening", "after other opening"])
print("runs with both kinds of opening:", len(pooled), "| mean compliance after obeying vs other opening:",
      round(pooled["after obeying opening"].mean(), 1), "vs", round(pooled["after other opening"].mean(), 1),
      "| grading only the reasoning after the opening:", round(pooled["rest only, after obeying"].mean(), 1), "vs",
      round(pooled["rest only, after other"].mean(), 1))
SUMMARY.update({"rest_after_obeying": round(float(pooled["rest only, after obeying"].mean()), 1),
                "rest_after_other": round(float(pooled["rest only, after other"].mean()), 1)})
sh = t[(t.run == "SFT self headers") & (t.task == "lowercase_thinking")]
if len(sh):
    SUMMARY["selfheaders_lower_obeying"] = int(sh["openings obeying"].iloc[0])
    SUMMARY["selfheaders_lower_after_obeying"] = float(sh["after obeying opening"].iloc[0])
OUTJ = Path(__file__).resolve().parents[1] / "results" / "summary_tables" / "opening_obeys_rule.json"
OUTJ.write_text(json.dumps(SUMMARY, indent=1), encoding="utf-8")
print(SUMMARY)
