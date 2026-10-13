"""Phi-4-reasoning, Base vs SFT, per category: how much of each gain survives length control?

CoT-Control (per task, 50 questions each): full-trace paired gain, paired gain on the first 50 and first 100 words
(pairs in which both traces reach the window, paper scorer), and the gain matched on length bin (pooled quantile bins).
ReasonIF (per constraint): IFS for Base and SFT, median reasoning words, gain on question pairs whose Base and SFT
lengths are within a factor of 1.5, and the gain matched on length bin. Intervals: 95% bootstrap over questions
(10,000 resamples, rng 42), as in analysis_followups.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_followups as F  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RIF = REPO / "results" / "reasonif_300" / "phi4_reasoning"
N_BINS = 5


def truthy(v):
    return str(v).strip().lower() == "true"


def ci(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return np.nan, np.nan
    b = v[np.random.default_rng(42).integers(0, len(v), (10000, len(v)))].mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def fmt(g, lo, hi, n):
    return "--" if n == 0 else f"{g:+.1f} [{lo:+.1f}, {hi:+.1f}] (n={n})"


def bin_matched(d, key):
    """d has cond, score, words, key; returns matched gain and matched n within each key value using pooled bins."""
    edges = np.unique(np.quantile(d.words, np.linspace(0, 1, N_BINS + 1)))
    d = d.assign(bin=np.clip(np.searchsorted(edges, d.words, side="right") - 1, 0, len(edges) - 2))
    out = {}
    for k, sub in d.groupby(key):
        g = sub.groupby(["bin", "cond"]).score.agg(["mean", "size"]).unstack("cond").dropna()
        if g.empty:
            out[k] = (np.nan, 0)
            continue
        w = np.minimum(g[("size", "Base")], g[("size", "SFT")])
        out[k] = (float(((g[("mean", "SFT")] - g[("mean", "Base")]) * w).sum() / w.sum()), int(w.sum()))
    return out


# ---------------- CoT-Control ----------------
H = F.haskins_runs()
B, S = H["Phi-4-reasoning Base"], H["Phi-4-reasoning SFT"]
pooled = pd.concat([pd.DataFrame({"cond": "Base", "score": 100 * B.primary, "words": B.words, "task": B.task}),
                    pd.DataFrame({"cond": "SFT", "score": 100 * S.primary, "words": S.words, "task": S.task})])
bm = bin_matched(pooled, "task")
rows = []
for task in sorted(B.task.unique()):
    row = {"Task": task, "Base": round(100 * B[B.task == task].primary.mean(), 1), "SFT": round(100 * S[S.task == task].primary.mean(), 1),
           "Words B/S (median)": f"{int(B[B.task == task].words.median())}/{int(S[S.task == task].words.median())}"}
    for label, w in (("Full trace", None), ("First 50 words", 50), ("First 100 words", 100)):
        n, mb, ms, g, lo, hi = F._paired_gain(B, S, None, window=w, tasks=[task])
        row[label] = fmt(g, lo, hi, n)
    g, n = bm[task]
    row["Matched length bin"] = "--" if n == 0 else f"{g:+.1f} (n={n})"
    rows.append(row)
for label, w in (("ALL", None),):
    pass
cc = pd.DataFrame(rows)
alln = {lab: F._paired_gain(B, S, None, window=w) for lab, w in (("Full trace", None), ("First 50 words", 50), ("First 100 words", 100))}
cc.loc[len(cc)] = {"Task": "ALL TASKS", "Base": round(100 * B.primary.mean(), 1), "SFT": round(100 * S.primary.mean(), 1),
                   "Words B/S (median)": f"{int(B.words.median())}/{int(S.words.median())}",
                   **{lab: fmt(v[3], v[4], v[5], v[0]) for lab, v in alln.items()}, "Matched length bin": ""}
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 40)
print("PHI-4-REASONING, CoT-Control (mean compliance, paper scoring; gains SFT - Base in points)")
print(cc.to_string(index=False))

# ---------------- ReasonIF ----------------
recs = {}
for cond in ("base", "sft"):
    d = pd.DataFrame([json.loads(l) for l in (RIF / cond / "scored_responses.jsonl").read_text(encoding="utf-8").split(chr(10)) if l.strip()])
    recs[cond] = pd.DataFrame({"idx": d.dataset_index.astype(int), "constraint": d.constraint.astype(str),
                               "ifs": [100.0 * truthy(v) for v in d.instruction_following],
                               "words": d.reasoning_content.fillna("").astype(str).str.split().str.len()})
m = recs["base"].merge(recs["sft"], on=["idx", "constraint"], suffixes=("_b", "_s"))
pooled = pd.concat([pd.DataFrame({"cond": "Base", "score": m.ifs_b, "words": m.words_b, "constraint": m.constraint}),
                    pd.DataFrame({"cond": "SFT", "score": m.ifs_s, "words": m.words_s, "constraint": m.constraint})])
bm = bin_matched(pooled, "constraint")
rows = []
for c, sub in list(m.groupby("constraint")) + [("ALL", m), ("ALL except word limit", m[m.constraint != "length_constraint_checkers:number_words"])]:
    diff = sub.ifs_s - sub.ifs_b
    lo, hi = ci(diff)
    ratio = np.abs(np.log((sub.words_s + 1) / (sub.words_b + 1)))
    lm = sub[ratio <= np.log(1.5)]
    llo, lhi = ci(lm.ifs_s - lm.ifs_b)
    g, n = bm.get(c, (np.nan, 0))
    rows.append({"Constraint": c.split(":")[-1], "N": len(sub), "Base IFS": round(sub.ifs_b.mean(), 1), "SFT IFS": round(sub.ifs_s.mean(), 1),
                 "Gain": f"{diff.mean():+.1f} [{lo:+.1f}, {hi:+.1f}]", "Words B/S (median)": f"{int(sub.words_b.median())}/{int(sub.words_s.median())}",
                 "Length within 1.5x": "--" if len(lm) == 0 else f"{(lm.ifs_s - lm.ifs_b).mean():+.1f} [{llo:+.1f}, {lhi:+.1f}] (n={len(lm)})",
                 "Matched length bin": "" if c.startswith("ALL") else ("--" if n == 0 else f"{g:+.1f} (n={n})")})
print("\nPHI-4-REASONING, ReasonIF (instruction-following success, %)")
print(pd.DataFrame(rows).to_string(index=False))
