"""ReasonIF Base vs SFT per constraint, with length checks (same method as phi_length_by_category.py).

For each constraint: IFS for Base and SFT, paired gain with a 95% bootstrap interval over questions, median reasoning
words, the gain on question pairs whose Base and SFT lengths are within a factor of 1.5, and the gain matched on
length bin (pooled quantile bins, weighted by the smaller count per cell). Also the gain without the word-limit items,
and after excluding passes listed in data/manual_review/reasonif_pass_review.json.

  python scripts/reasonif_length_by_constraint.py olmo_lr2e-4 | olmo_lr1e-4 | phi | qwen
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
RUNS = {
    "olmo_lr2e-4": (R / "olmo3_7b_think/reasonif_300/full_*/scored_responses.jsonl",
                    R / "olmo3_7b_think/reasonif_300_sft_lr2e-4/lora_*/scored_responses.jsonl", "olmo3_7b_think/base", "olmo3_7b_think/sft_lr2e-4"),
    "olmo_lr1e-4": (R / "olmo3_7b_think/reasonif_300/full_*/scored_responses.jsonl",
                    R / "olmo3_7b_think/reasonif_300/lora_*/scored_responses.jsonl", "olmo3_7b_think/base", "olmo3_7b_think/sft_lr1e-4"),
    "phi": (R / "reasonif_300/phi4_reasoning/base/scored_responses.jsonl", R / "reasonif_300/phi4_reasoning/sft/scored_responses.jsonl",
            "phi4_reasoning/base", "phi4_reasoning/sft"),
    "qwen": (R / "reasonif_300/qwen3_14b/reasonif_qwen3_14b_base_untouched.jsonl", R / "reasonif_300/qwen3_14b/reasonif_qwen3_14b_sft_gpt52_high.jsonl",
             "qwen3_14b/base", "qwen3_14b/seed42"),
}
WORD_LIMIT = "length_constraint_checkers:number_words"


def truthy(v):
    return str(v).strip().lower() == "true"


def load(pattern):
    f = sorted(glob.glob(str(pattern)))[0]
    d = pd.DataFrame([json.loads(l) for l in Path(f).read_text(encoding="utf-8").split(chr(10)) if l.strip()])
    return pd.DataFrame({"idx": d.dataset_index.astype(int), "constraint": d.constraint.astype(str),
                         "ifs": [100.0 * truthy(v) for v in d.instruction_following],
                         "words": d.reasoning_content.fillna("").astype(str).str.split().str.len()})


def ci(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return np.nan, np.nan
    b = v[np.random.default_rng(42).integers(0, len(v), (10000, len(v)))].mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def bin_matched(m):
    pooled = np.concatenate([m.words_b, m.words_s])
    edges = np.unique(np.quantile(pooled, np.linspace(0, 1, 6)))
    bb = np.clip(np.searchsorted(edges, m.words_b, side="right") - 1, 0, len(edges) - 2)
    bs = np.clip(np.searchsorted(edges, m.words_s, side="right") - 1, 0, len(edges) - 2)
    d = pd.concat([pd.DataFrame({"cond": "B", "bin": bb, "c": m.constraint.values, "y": m.ifs_b.values}),
                   pd.DataFrame({"cond": "S", "bin": bs, "c": m.constraint.values, "y": m.ifs_s.values})])
    g = d.groupby(["c", "bin", "cond"]).y.agg(["mean", "size"]).unstack("cond").dropna()
    out = {}
    for c, sub in g.groupby(level=0):
        w = np.minimum(sub[("size", "B")], sub[("size", "S")])
        out[c] = (float(((sub[("mean", "S")] - sub[("mean", "B")]) * w).sum() / w.sum()), int(w.sum()))
    w = np.minimum(g[("size", "B")], g[("size", "S")])
    out["ALL"] = (float(((g[("mean", "S")] - g[("mean", "B")]) * w).sum() / w.sum()), int(w.sum()))
    return out


which = sys.argv[1] if len(sys.argv) > 1 else "olmo_lr2e-4"
fb, fs, kb, ks = RUNS[which]
m = load(fb).merge(load(fs), on=["idx", "constraint"], suffixes=("_b", "_s"))
bm = bin_matched(m)
rows = []
groups = list(m.groupby("constraint")) + [("ALL", m), ("ALL except word limit", m[m.constraint != WORD_LIMIT])]
for c, sub in groups:
    diff = sub.ifs_s - sub.ifs_b
    lo, hi = ci(diff)
    ratio = np.abs(np.log((sub.words_s + 1) / (sub.words_b + 1)))
    lm = sub[ratio <= np.log(1.5)]
    llo, lhi = ci(lm.ifs_s - lm.ifs_b)
    g, n = bm.get(c, (np.nan, 0)) if c != "ALL except word limit" else (np.nan, 0)
    rows.append({"Constraint": c.split(":")[-1], "N": len(sub), "Base": round(sub.ifs_b.mean(), 1), "SFT": round(sub.ifs_s.mean(), 1),
                 "Gain [95% CI]": f"{diff.mean():+.1f} [{lo:+.1f}, {hi:+.1f}]",
                 "Median words B/S": f"{int(sub.words_b.median())}/{int(sub.words_s.median())}",
                 "Length within 1.5x": "--" if len(lm) == 0 else f"{(lm.ifs_s - lm.ifs_b).mean():+.1f} [{llo:+.1f}, {lhi:+.1f}] (n={len(lm)})",
                 "Matched length bin": "" if np.isnan(g) else f"{g:+.1f} (n={n})"})
pd.set_option("display.width", 250)
print(f"ReasonIF, {which}: Base vs SFT")
print(pd.DataFrame(rows).to_string(index=False))
rev = json.loads((REPO / "data/manual_review/reasonif_pass_review.json").read_text(encoding="utf-8"))["excluded"]
eb, es = {int(i) for i in rev.get(kb, {})}, {int(i) for i in rev.get(ks, {})}
yb = m.ifs_b.where(~m.idx.isin(eb), 0.0)
ys = m.ifs_s.where(~m.idx.isin(es), 0.0)
lo, hi = ci(ys - yb)
print(f"\nexcluding reviewed passes (Base {len(eb)}, SFT {len(es)}): Base {yb.mean():.1f}, SFT {ys.mean():.1f}, gain {(ys - yb).mean():+.1f} [{lo:+.1f}, {hi:+.1f}]")
