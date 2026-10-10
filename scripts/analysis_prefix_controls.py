"""
Analysis of the prefix-control and prefix-length runs (Haskins, Qwen3-14B, base recipient).

Reads run directories (each with results.csv, same schema as the 2x2 runs) found under results/haskins_500/qwen3_14b/:
  *2x2-base-to-base-on-10tok__*              reference: base donor            (BB)
  *2x2-sft-to-base-on-10tok__*               reference: SFT donor             (SB)
  *control-<condition>-10tok__*              fixed openings: handcrafted | named | mismatched | cross_question
  *2x2-sft-to-base-on-<N>tok__*              SFT donor writes N tokens (N = 30, 100)

Everything is restricted to the eight style tasks (the two suppression tasks get a neutral opening in the control runs and are not analysed). Differences are paired over the 50 questions with a cluster bootstrap (10,000 resamples, seed 42).
Outputs (written by reproduce_tables.py --write only when the runs exist):
  table16_prefix_controls.csv   continuation mean / strict per condition and paired differences versus BB and SB
  table16b_prefix_controls_per_task.csv   continuation mean per task and condition (arrow_prefix and repeat_sentences sit near floor)
  table17_prefix_length.csv     SFT-donor opening length N: continuation and full-trace compliance, donor-finished count
"""
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from evaluators.haskins_evaluator import primary_score  # noqa: E402
QWEN = REPO_ROOT / "results" / "haskins_500" / "qwen3_14b"
STYLE_TASKS = ["third_person", "arrow_prefix", "end_of_sentence", "meow_between_words", "repeat_sentences",
               "alternating_case", "lowercase_thinking", "uppercase_thinking"]
N_BOOT, SEED = 10_000, 42
LABELS = {"BB": "Base donor (reference)", "SB": "SFT donor (reference)", "handcrafted": "Handcrafted, obeys the rule",
          "named": "Names the rule only", "mismatched": "Obeys a different rule", "cross_question": "SFT-donor opening of another question",
          "modal_sft": "Most frequent SFT opening of the task (same for every question)"}


def _find(pattern: str, roots=None):
    roots = roots or [QWEN]
    hits = []
    for root in roots:
        hits += glob.glob(str(root / "**" / pattern / "results.csv"), recursive=True)
    return sorted(set(hits))


def _load(path) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    df = df[df.task.isin(STYLE_TASKS)].copy()
    # headline score follows the original paper's rule (binary for end-of-sentence and meow); fractional kept as *_frac
    for col in ("continuation_compliance", "full_compliance"):
        df[col + "_frac"] = df[col]
        df[col] = [primary_score(t, float(c)) for t, c in zip(df.task, df[col + "_frac"])]
    return df


def _per_question(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("prompt_idx")[["continuation_compliance", "continuation_score_one"]].mean().sort_index()


def _paired(a: pd.DataFrame, b: pd.DataFrame, rng):
    a, b = a.align(b, join="inner", axis=0)
    assert len(a) == 50, f"expected 50 paired questions, got {len(a)}"
    x = a.continuation_compliance.to_numpy() - b.continuation_compliance.to_numpy()
    y = a.continuation_score_one.to_numpy() - b.continuation_score_one.to_numpy()
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    out = []
    for v in (x, y):
        lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
        out += [100 * v.mean(), 100 * lo, 100 * hi]
    return out


def controls_tables(extra_runs: dict = None) -> dict:
    """Return the control/prefix-length tables; empty dict when no control runs are present."""
    runs = {}
    ref = {"BB": "*2x2-base-to-base-on-10tok__*", "SB": "*2x2-sft-to-base-on-10tok__*"}
    for key, pat in ref.items():
        hits = _find(pat)
        if hits:
            runs[key] = _load(hits[0])
    for cond in ("handcrafted", "named", "mismatched", "cross_question", "modal_sft"):
        hits = _find(f"*control-{cond}-10tok__*")
        if hits:
            runs[cond] = _load(hits[0])
    if extra_runs:
        runs.update(extra_runs)
    controls = [k for k in runs if k not in ("BB", "SB")]
    tables = {}
    if controls and "BB" in runs and "SB" in runs:
        rng = np.random.default_rng(SEED)
        rows = []
        for key in ["BB", "SB"] + [c for c in ("handcrafted", "named", "mismatched", "cross_question", "modal_sft") if c in runs]:
            df = runs[key]
            row = dict(Condition=LABELS.get(key, key), N=len(df), Continuation_Mean=round(100 * df.continuation_compliance.mean(), 2),
                       Strict_Count=int(df.continuation_score_one.sum()))
            for ref_key in ("BB", "SB"):
                if key == ref_key:
                    row.update({f"Diff_vs_{ref_key}": "", f"Diff_vs_{ref_key}_CI95": ""})
                    continue
                d = _paired(_per_question(df), _per_question(runs[ref_key]), rng)
                row[f"Diff_vs_{ref_key}"] = round(d[0], 2)
                row[f"Diff_vs_{ref_key}_CI95"] = f"[{d[1]:.2f}, {d[2]:.2f}]"
            rows.append(row)
        tables["table16_prefix_controls.csv"] = pd.DataFrame(rows)
        order = ["BB", "SB"] + [c for c in ("handcrafted", "named", "mismatched", "cross_question", "modal_sft") if c in runs]
        per_task = pd.DataFrame({LABELS.get(k, k): 100 * runs[k].groupby("task").continuation_compliance.mean() for k in order}).reindex(STYLE_TASKS)
        tables["table16b_prefix_controls_per_task.csv"] = per_task.round(2).rename_axis("Task").reset_index()
    # prefix length
    length_rows = []
    sb = runs.get("SB")
    if sb is not None:
        length_rows.append(dict(Opening_Tokens=10, N=len(sb), Continuation_Mean=round(100 * sb.continuation_compliance.mean(), 2),
                                Full_Trace_Mean=round(100 * sb.full_compliance.mean(), 2), Strict_Continuation=int(sb.continuation_score_one.sum()),
                                Donor_Finished_Inside_Opening=""))
    for path in _find("*2x2-sft-to-base-on-*tok__*"):
        m = re.search(r"on-(\d+)tok__", path)
        if not m or int(m.group(1)) == 10:
            continue
        df = _load(path)
        finished = int(df.injected_prefix_text.str.contains("</think>").sum())
        length_rows.append(dict(Opening_Tokens=int(m.group(1)), N=len(df), Continuation_Mean=round(100 * df.continuation_compliance.mean(), 2),
                                Full_Trace_Mean=round(100 * df.full_compliance.mean(), 2), Strict_Continuation=int(df.continuation_score_one.sum()),
                                Donor_Finished_Inside_Opening=finished))
    if len(length_rows) > 1:
        tables["table17_prefix_length.csv"] = pd.DataFrame(length_rows).sort_values("Opening_Tokens").reset_index(drop=True)
    return tables


if __name__ == "__main__":
    demo = "--demo" in sys.argv
    extra = None
    if demo:  # plumbing test only: pretend the SFT-donor run is a 'handcrafted' run
        sb = _load(_find("*2x2-sft-to-base-on-10tok__*")[0])
        extra = {"handcrafted": sb.assign(continuation_compliance=(sb.continuation_compliance * 0.5))}
        print("DEMO MODE: the 'handcrafted' row is a synthetic halving of the SFT-donor run, for plumbing tests only.\n")
    out = controls_tables(extra)
    if not out:
        print("No control runs found under", QWEN)
    for name, df in out.items():
        print("==", name)
        print(df.to_string(index=False))
