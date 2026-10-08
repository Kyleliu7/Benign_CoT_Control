"""
Scorer parity check: re-score every stored Haskins completion and compare with the stored score.

For each run directory under results/haskins_500 this recomputes the compliance score from the saved
reasoning text with ``evaluators.haskins_evaluator`` (scorer="paper") and compares it with the score
that was written when the completion was generated. Any mismatch exits non-zero.

  Qwen 2x2 runs : continuation_reasoning -> continuation_compliance / continuation_score_one
                  full_reasoning         -> full_compliance         / full_score_one
  Phi standalone: reasoning              -> upstream_compliance     / score_equals_one
  Phi crossed   : bundle item ``cot`` minus injected ``prefix`` (continuation)   -> comp (percent, 2 dp)    / strict

Needs the pinned upstream grader: set HASKINS_UPSTREAM_ROOT to a local clone of
Reih02/cot_obfuscation_code at commit 38dca62, or allow the one-time SHA-verified download.
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from evaluators.haskins_evaluator import grade_paper_protocol  # noqa: E402

TOL = 1e-9
HASKINS = REPO_ROOT / "results" / "haskins_500"


def _rescore(df: pd.DataFrame, text_col: str):
    df = df.fillna({text_col: ""})
    return [grade_paper_protocol(t, int(p), x) for t, p, x in zip(df["task"], df["prompt_idx"], df[text_col])]


def check_frame(label: str, df: pd.DataFrame, text_col: str, score_col: str, strict_col: str) -> int:
    new = _rescore(df, text_col)
    stored = df[score_col].astype(float).tolist()
    bad = [(i, a, b) for i, (a, b) in enumerate(zip(new, stored)) if abs(a - b) > TOL]
    strict_new = sum(1 for a in new if a == 1.0)
    strict_old = int(df[strict_col].sum())
    status = "OK " if not bad and strict_new == strict_old else "FAIL"
    print(f"[{status}] {label:<62s} rows={len(df):4d} mismatched={len(bad):3d} strict stored/recomputed={strict_old}/{strict_new}")
    for i, a, b in bad[:3]:
        print(f"        row {i}: task={df.iloc[i]['task']} idx={df.iloc[i]['prompt_idx']} recomputed={a:.6f} stored={b:.6f}")
    return len(bad) + (strict_new != strict_old)


def main() -> int:
    failures = 0
    for run in sorted((HASKINS / "qwen3_14b").glob("*/results.csv")):
        df = pd.read_csv(run)
        tag = run.parent.name.split("__")[-2]
        failures += check_frame(f"Qwen {tag} (continuation)", df, "continuation_reasoning", "continuation_compliance", "continuation_score_one")
        failures += check_frame(f"Qwen {tag} (full)", df, "full_reasoning", "full_compliance", "full_score_one")
    for run in sorted((HASKINS / "phi4_reasoning" / "extracted_runs").glob("*/results.csv")):
        df = pd.read_csv(run)
        tag = "SFT" if "gpt52" in run.parent.name else "base"
        failures += check_frame(f"Phi standalone {tag}", df, "reasoning", "upstream_compliance", "score_equals_one")

    bundle = json.loads((HASKINS / "phi4_reasoning" / "phi4_haskins_500_bundle.json").read_text(encoding="utf-8"))
    for cond in sorted(bundle["conditions"]):
        rows = []
        for item in bundle["items"]:
            m = item["models"][cond]
            cot, prefix = m.get("cot") or "", m.get("prefix") or ""
            continuation = cot[len(prefix):] if prefix and cot.startswith(prefix) else cot
            rows.append({"task": item["task"], "prompt_idx": item["prompt_idx"], "cot": continuation,
                         "comp": float(m["comp"]), "strict": int(bool(m["strict"]))})
        df = pd.DataFrame(rows)
        new = [grade_paper_protocol(t, int(p), x) for t, p, x in zip(df.task, df.prompt_idx, df.cot)]
        # the bundle stores percent rounded to 2 dp
        bad = [i for i, (a, b) in enumerate(zip(new, df.comp)) if abs(round(a * 100, 2) - b) > 0.011]
        strict_new = sum(1 for a in new if a == 1.0)
        status = "OK " if not bad and strict_new == int(df.strict.sum()) else "FAIL"
        print(f"[{status}] Phi crossed {cond:<51s} rows={len(df):4d} mismatched={len(bad):3d} strict stored/recomputed={int(df.strict.sum())}/{strict_new}")
        failures += len(bad) + (strict_new != int(df.strict.sum()))

    print("\nPARITY: all stored Haskins scores reproduced." if failures == 0 else f"\nPARITY FAILED ({failures} discrepancies).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
