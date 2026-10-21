"""ReasonIF instruction following re-scored with a leading bold heading (**...**) removed from the reasoning, for every
Qwen3-14B run with raw records, next to the score recomputed on the reasoning as recorded.

A heading at the start of the reasoning can change a ReasonIF score by itself (one extra word for the word limit,
English words for the reasoning-language check, letters for the all-capitals check). Uses the repository's official
ReasonIF checker (evaluators/reasonif_evaluator.py), which needs the language-detection model, so run it where that
model is available (e.g. the training VM).

  python scripts/reasonif_header_stripped.py [--out results/summary_tables/reasonif_header_stripped.csv]
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis_followups as F  # noqa: E402
from evaluators.reasonif_evaluator import evaluate_reasonif_record  # noqa: E402

LEAD = re.compile(r"^\s*\*\*[^*\n]{1,120}\*\*\s*")


def runs():
    out = {k.replace("Qwen3-14B ", ""): v[1] for k, v in F.reasonif_runs().items() if k.startswith("Qwen3-14B")}
    for f in sorted((F.RES / "data_variants").glob("*/reasonif_300/qwen3_14b/*/scored_responses.jsonl")):
        out["SFT " + f.parts[-5].replace("_", " ")] = F._jl(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(F.RES / "summary_tables" / "reasonif_header_stripped.csv"))
    args = ap.parse_args()
    rows, per_item = [], []
    for name, recs in runs().items():
        rec, strip, had = [], [], 0
        for r in recs:
            reasoning = r.get("reasoning_content") or ""
            had += bool(LEAD.match(reasoning))
            a = evaluate_reasonif_record(r)["instruction_following"]
            b = evaluate_reasonif_record({**r, "reasoning_content": LEAD.sub("", reasoning, count=1)})["instruction_following"]
            rec.append(a)
            strip.append(b)
            per_item.append({"run": name, "dataset_index": r["dataset_index"], "constraint": r.get("constraint"),
                             "stored": bool(r["instruction_following"]), "recomputed": a, "header_stripped": b})
        n = len(recs)
        rows.append({"Run": name, "N": n, "Leading heading": had,
                     "IFS stored (%)": round(100 * sum(bool(r["instruction_following"]) for r in recs) / n, 2),
                     "IFS recomputed (%)": round(100 * sum(rec) / n, 2),
                     "IFS header-stripped (%)": round(100 * sum(strip) / n, 2),
                     "Items changed by stripping": int(sum(x != y for x, y in zip(rec, strip)))})
    t = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out, index=False)
    pd.DataFrame(per_item).to_csv(str(args.out).replace(".csv", "_items.csv"), index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
