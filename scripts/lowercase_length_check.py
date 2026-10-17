"""Is the drop on lowercase_thinking after self-distillation / s1K training a length effect?

The lowercase score of a trace is the share of its sentences (grader's split) whose letters are all lowercase. For
each run (50 lowercase traces, leading bold heading removed as in haskins_header_stripped.py): median words and
sentences, the score, the share of traces whose first sentence is lowercase, the share that are fully lowercase, and
the per-sentence lowercase rate by sentence position (1-5, 6-20, 21-50, 51+), which compares runs at the same position
regardless of trace length. Also the score among traces of at most 400 words, and the Spearman correlation between
words and score within each run.

  python scripts/lowercase_length_check.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis_followups as F  # noqa: E402
from haskins_header_stripped import LEAD, runs  # noqa: E402
from evaluators.haskins_evaluator import SENT_SPLIT, clean_reasoning  # noqa: E402

def load():
    H = runs()
    for label, sub in {"SFT self-distillation": "haskins_500/qwen3_14b_training_variants/self_distill"}.items():
        f = glob.glob(str(F.RES / sub / "haskins_500/qwen3_14b/*/results.csv"))
        if f:
            H[label] = F._prep(pd.read_csv(f[0]))
    return H


KEEP = ["Base", "SFT seed 42", "SFT self-distillation", "SFT self headers", "SFT s1k claude37", "SFT s1k deepseek r1"]
BINS = [(1, 5), (6, 20), (21, 50), (51, 10 ** 9)]


def sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(clean_reasoning(LEAD.sub("", text, count=1))) if s.strip()]


def lower_ok(s):
    a = [c for c in s if c.isalpha()]
    return not a or all(c.islower() for c in a)


if __name__ == "__main__":
    H = load()
    rows = []
    for k in KEEP:
        if k not in H:
            continue
        d = H[k][H[k].task == "lowercase_thinking"]
        sents = [sentences(x) for x in d.reasoning]
        ok = [[lower_ok(s) for s in ss] for ss in sents]
        score = np.array([np.mean(o) if o else 0.0 for o in ok])
        words = d.words.to_numpy()
        row = {"Model": k, "Median words": int(np.median(words)), "Median sentences": int(np.median([len(s) for s in sents])),
               "Score %": round(100 * score.mean(), 1), "First sentence lowercase": f"{sum(bool(o) and o[0] for o in ok)}/{len(ok)}",
               "Fully lowercase": f"{sum(bool(o) and all(o) for o in ok)}/{len(ok)}"}
        for lo, hi in BINS:
            v = [x for o in ok for i, x in enumerate(o, 1) if lo <= i <= hi]
            row[f"Sent {lo}-{hi if hi < 10 ** 9 else ''}"] = f"{100 * np.mean(v):.0f}% (n={len(v)})" if v else "-"
        short = words <= 400
        row["Score, traces <=400 words"] = f"{100 * score[short].mean():.1f} (n={short.sum()})" if short.any() else "-"
        r, p = spearmanr(words, score)
        row["Spearman words vs score"] = f"{r:+.2f} (p={p:.2g})"
        rows.append(row)
    pd.set_option("display.width", 300)
    print(pd.DataFrame(rows).to_string(index=False))
