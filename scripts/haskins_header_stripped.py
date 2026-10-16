"""CoT-Control (Haskins 500) scored with a leading bold header removed from every trace before grading.

The grader treats a header line such as **Planning the steps** as a sentence, so a header can raise or lower a score
by itself (it fails meow, end-of-sentence and repeat-sentences outright, and changes the case tasks either way).
This rescoring removes one leading **...** header from every model's reasoning (Base traces have none), grades with
the same paper-protocol scorer, and reports mean compliance, strict passes and the paired gain over Base.

  python scripts/haskins_header_stripped.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_followups as F  # noqa: E402
from evaluators.haskins_evaluator import grade_paper_protocol, primary_score  # noqa: E402

LEAD = re.compile(r"^\s*\*\*[^*\n]{1,120}\*\*\s*")


def runs():
    H = {k.replace("Qwen3-14B ", ""): v for k, v in F.haskins_runs().items() if k.startswith("Qwen3-14B")}
    for f in sorted((F.RES / "data_variants").glob("*/haskins_500/qwen3_14b/*/results.csv")):
        H["SFT " + f.parts[-5].replace("_", " ")] = F._prep(pd.read_csv(f))
    return H


def rescore(d):
    d = d.copy()
    d["stripped"] = [LEAD.sub("", x, count=1) for x in d.reasoning]
    d["had_header"] = [bool(LEAD.match(x)) for x in d.reasoning]
    up = np.array([grade_paper_protocol(t, q, x) for t, q, x in zip(d.task, d.q, d.stripped)])
    d["primary_s"] = [primary_score(t, u) for t, u in zip(d.task, up)]
    d["strict_s"] = up == 1.0
    return d


if __name__ == "__main__":
    H = {k: rescore(v) for k, v in runs().items()}
    base = H["Base"]
    rows = []
    for k, d in H.items():
        row = {"Model": k, "Leading header": f"{int(d.had_header.sum())}/{len(d)}",
               "Mean raw": round(100 * d.primary.mean(), 2), "Mean header-stripped": round(100 * d.primary_s.mean(), 2),
               "Strict raw": int(d.strict.sum()), "Strict header-stripped": int(d.strict_s.sum())}
        if k != "Base":
            m = base.merge(d, on=["task", "q"], suffixes=("_b", "_m"))
            diff = (m.primary_s_m - m.primary_s_b).to_numpy()
            lo, hi = F._ci(pd.Series(diff).groupby(m.q.values).mean())
            row["Gain vs Base, stripped [95% CI]"] = f"{100 * diff.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"
        rows.append(row)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nPer task, header-stripped mean (%)")
    print(pd.DataFrame({k: d.groupby("task").primary_s.mean() * 100 for k, d in H.items()}).round(1).to_string())
