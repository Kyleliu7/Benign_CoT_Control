"""Seed-to-seed comparison for Qwen3-14B SFT (training seeds 42 = published adapter, 1, 2), all from raw records.
Haskins standalone (headline rule, strict, degenerate check, first-L-words paired gain vs Base) and ReasonIF (IFS, accuracy, joint)."""
import glob
from pathlib import Path
import json
import re
import sys

import numpy as np
import pandas as pd

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from evaluators.haskins_evaluator import grade_paper_protocol, primary_score  # noqa: E402

Q = f"{REPO}/results/haskins_500/qwen3_14b/"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8").read().split("\n") if l.strip()]


def degen(task, text):
    w = str(text).split()
    return len(w) < 20 or (task == "meow_between_words" and sum(x.strip(".,!").lower() == "meow" for x in w) / len(w) > 0.9)


def prep(d):
    d = d.copy()
    d["reasoning"] = d.reasoning.fillna("").astype(str)
    d["q"] = d.prompt_idx.astype(int)
    d["primary"] = [primary_score(t, float(c)) for t, c in zip(d.task, d.upstream_compliance)]
    d["strict"] = d.upstream_compliance.astype(float) == 1.0
    d["degen"] = [degen(t, x) for t, x in zip(d.task, d.reasoning)]
    d["words"] = d.reasoning.str.split().str.len()
    return d


H = {"Base": prep(pd.DataFrame(jl(Q + "haskins_qwen3_14b_vllm_calibrated_base_17pct.jsonl"))),
     "SFT seed 42": prep(pd.DataFrame(jl(Q + "haskins_qwen3_14b_vllm_calibrated_sft_31pct.jsonl")))}
for s in (1, 2):
    f = glob.glob(f"{REPO}/results/training_seeds/seed{s}/haskins_500/qwen3_14b/*/results.csv")
    if f:
        H[f"SFT seed {s}"] = prep(pd.read_csv(f[0]))

RIF = f"{REPO}/results/reasonif_300/qwen3_14b/"
R = {"Base": jl(RIF + "reasonif_qwen3_14b_base_untouched.jsonl"), "SFT seed 42": jl(RIF + "reasonif_qwen3_14b_sft_gpt52_high.jsonl")}
for s in (1, 2):
    f = glob.glob(f"{REPO}/results/training_seeds/seed{s}/reasonif_300/qwen3_14b/*/scored_responses.jsonl")
    if f:
        R[f"SFT seed {s}"] = jl(f[0])

rng = np.random.default_rng(42)


def ci(v):
    v = np.asarray(v)
    b = v[rng.integers(0, len(v), (10000, len(v)))].mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def trunc(t, w):
    e = [m.end() for m in re.finditer(r"\S+", t)]
    return t if len(e) <= w else t[: e[w - 1]]


print("HASKINS standalone (500 pairs)")
for k, d in H.items():
    print(f"  {k:12s} mean {100*d.primary.mean():6.2f} | fractional {100*d.upstream_compliance.astype(float).mean():6.2f} | strict {int(d.strict.sum()):3d} "
          f"(degenerate {int((d.strict & d.degen).sum())}) | degenerate traces {int(d.degen.sum()):2d} | median words {d.words.median():5.0f} | "
          f"starts '**' {int(d.reasoning.str.lstrip().str.startswith('**').sum())}")
print("  per task means:")
tasks = list(H["Base"].groupby("task").groups)
for t in tasks:
    print("   ", f"{t:26s}", " | ".join(f"{k}: {100*d[d.task==t].primary.mean():5.1f}" for k, d in H.items()))
print("  SFT - Base, paired by question (cluster bootstrap over 50 questions):")
for k, d in H.items():
    if k == "Base":
        continue
    m = H["Base"].merge(d, on=["task", "q"], suffixes=("_b", "_s"))
    diff = (m.primary_s - m.primary_b).groupby(m.q).mean()
    lo, hi = ci(diff)
    line = f"    {k:12s} full {100*(m.primary_s-m.primary_b).mean():+6.2f} [{100*lo:.2f}, {100*hi:.2f}]"
    for w in (100, 200):
        kk = m[(m.words_b >= w) & (m.words_s >= w)]
        sb = np.array([primary_score(t, grade_paper_protocol(t, q, trunc(x, w))) for t, q, x in zip(kk.task, kk.q, kk.reasoning_b)])
        ss = np.array([primary_score(t, grade_paper_protocol(t, q, trunc(x, w))) for t, q, x in zip(kk.task, kk.q, kk.reasoning_s)])
        lo, hi = ci(pd.Series(ss - sb).groupby(kk.q.values).mean())
        line += f" | first {w} words (n={len(kk)}) {100*(ss-sb).mean():+5.2f} [{100*lo:.2f}, {100*hi:.2f}]"
    print(line)

print("\nREASONIF (300 questions)")
base = {r["dataset_index"]: r for r in R["Base"]}
for k, rows in R.items():
    n = len(rows)
    ifs = sum(bool(r["instruction_following"]) for r in rows)
    acc = sum(bool(r["answer_correct"]) for r in rows)
    joint = sum(bool(r["joint_success"]) for r in rows)
    tok = np.mean([r["output_tokens"] for r in rows])
    line = f"  {k:12s} n={n} IFS {ifs} ({100*ifs/n:.1f}%) | acc {acc} ({100*acc/n:.1f}%) | joint {joint} | mean tokens {tok:.0f}"
    if k != "Base":
        d = np.array([int(bool(r["instruction_following"])) - int(bool(base[r["dataset_index"]]["instruction_following"])) for r in rows])
        lo, hi = ci(d)
        line += f" | IFS gain vs Base {100*d.mean():+.1f} [{100*lo:.1f}, {100*hi:.1f}]"
    print(line)
