"""Compliance as a function of reasoning length, Base vs SFT, for Phi-4-reasoning (Qwen3-14B for reference).

ReasonIF instruction-following success (word-limit items excluded, since that constraint is about length) and CoT-Control
mean compliance (paper scoring), both binned by reasoning length in words. Bin edges are quantiles of the pooled Base+SFT
lengths of each model and benchmark, so every bin compares traces of the same length. The 'matched' gain is the SFT minus
Base difference within cells of (length bin, task or constraint), averaged with weights equal to the smaller of the two
counts in each cell, so it compares traces of similar length on the same task (task mix differs between bins).

  python scripts/length_binned_figure.py   ->  reproducible_figures/figure5_length_binned.{pdf,png}, figure5_length_binned_source.csv
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_followups as F  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reproducible_figures"
RIF = REPO / "results" / "reasonif_300"
N_BINS = 5


def truthy(v):
    return str(v).strip().lower() == "true"


def reasonif(model_dir, base_file, sft_file):
    parts = []
    for cond, f in (("Base", base_file), ("SFT", sft_file)):
        recs = [json.loads(l) for l in (RIF / model_dir / f).read_text(encoding="utf-8").split(chr(10)) if l.strip()]
        d = pd.DataFrame(recs)
        d = d[d.constraint.astype(str) != "length_constraint_checkers:number_words"]
        parts.append(pd.DataFrame({"cond": cond, "score": [100.0 * truthy(v) for v in d.instruction_following],
                                   "words": d.reasoning_content.fillna("").astype(str).str.split().str.len(),
                                   "stratum": d.constraint.astype(str).values}))
    return pd.concat(parts, ignore_index=True)


def haskins(H, base_label, sft_label):
    parts = [pd.DataFrame({"cond": c, "score": 100.0 * H[l].primary, "words": H[l].words, "stratum": H[l].task})
             for c, l in (("Base", base_label), ("SFT", sft_label))]
    return pd.concat(parts, ignore_index=True)


def binned(d):
    edges = np.unique(np.quantile(d.words, np.linspace(0, 1, N_BINS + 1)))
    d = d.assign(bin=np.clip(np.searchsorted(edges, d.words, side="right") - 1, 0, len(edges) - 2))
    rows = []
    for b in range(len(edges) - 1):
        for c in ("Base", "SFT"):
            s = d[(d.bin == b) & (d.cond == c)].score
            if len(s) == 0:
                continue
            bs = s.values[np.random.default_rng(42).integers(0, len(s), (10000, len(s)))].mean(1)
            rows.append(dict(bin=b, lo_words=int(edges[b]), hi_words=int(edges[b + 1]), cond=c, n=len(s), mean=s.mean(),
                             ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
    t = pd.DataFrame(rows)
    # gain matched on length bin AND task/constraint: cells (bin, stratum) with both conditions, weighted by the smaller count
    g = d.groupby(["bin", "stratum", "cond"]).score.agg(["mean", "size"]).unstack("cond").dropna()
    w = np.minimum(g[("size", "Base")], g[("size", "SFT")])
    strat = float(((g[("mean", "SFT")] - g[("mean", "Base")]) * w).sum() / w.sum())
    n_matched = int(w.sum())
    raw = d[d.cond == "SFT"].score.mean() - d[d.cond == "Base"].score.mean()
    return t, strat, raw, n_matched


H = F.haskins_runs()
panels = [
    ("Phi-4-reasoning", "ReasonIF", reasonif("phi4_reasoning", "base/scored_responses.jsonl", "sft/scored_responses.jsonl"), "Instruction-following success (%)"),
    ("Phi-4-reasoning", "CoT-Control", haskins(H, "Phi-4-reasoning Base", "Phi-4-reasoning SFT"), "Mean compliance"),
    ("Qwen3-14B", "ReasonIF", reasonif("qwen3_14b", "reasonif_qwen3_14b_base_untouched.jsonl", "reasonif_qwen3_14b_sft_gpt52_high.jsonl"), "Instruction-following success (%)"),
    ("Qwen3-14B", "CoT-Control", haskins(H, "Qwen3-14B Base", "Qwen3-14B SFT seed 42"), "Mean compliance"),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
src = []
for ax, (model, bench, d, ylab) in zip(axes.T.flatten(), panels):
    t, strat, raw, n_matched = binned(d)
    src.append(t.assign(model=model, benchmark=bench))
    edges_t = t.drop_duplicates("bin").sort_values("bin")
    labels = [f"{r.lo_words}–{r.hi_words}" for r in edges_t.itertuples()]
    for c, col, off in (("Base", "#7f8c8d", -0.08), ("SFT", "#2f6db5", 0.08)):
        s = t[t.cond == c]
        ax.errorbar(s.bin + off, s["mean"], yerr=[s["mean"] - s.ci_lo, s.ci_hi - s["mean"]], fmt="o-", color=col, capsize=3, label=c, ms=5)
        for r in s.itertuples():
            ax.annotate(f"n={r.n}", (r.bin + off, r.ci_hi), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=6.5, color=col)
    ax.set_xticks(list(edges_t.bin))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Reasoning length (words), quantile bins", fontsize=8)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.set_title(f"{model} — {bench}\nraw gain {raw:+.1f}; matched on length bin and task: {strat:+.1f} (n={n_matched})", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=7.5, frameon=True, framealpha=0.9, loc="best")
    print(f"{model:16s} {bench:11s} raw gain {raw:+.2f}  matched on length bin and task {strat:+.2f} (matched n={n_matched})")
fig.tight_layout()
OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figure5_length_binned.{ext}", bbox_inches="tight", dpi=200)
pd.concat(src).round(3).to_csv(OUT / "figure5_length_binned_source.csv", index=False)
print(pd.concat(src).round(1).to_string(index=False))
