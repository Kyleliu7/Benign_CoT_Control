"""Qwen3-14B data variants (results/data_variants/<variant>/) against Base and the SFT seeds, from raw records.

ReasonIF: IFS, accuracy, joint success, tokens, and the paired IFS gain over Base and the difference from SFT seed 42
(95% bootstrap over questions). Openings: share of traces that start with a bold header, what the first header says,
the first word after it, how often the opening names the constraint (CoT-Control lowercase / uppercase / third
person), median words and degenerate traces. CoT-Control scores are in scripts/haskins_header_stripped.py.

  python scripts/variant_summary.py
"""
import collections
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_followups as F  # noqa: E402
from haskins_header_stripped import runs  # noqa: E402

LEAD = re.compile(r"^\s*\*\*([^*\n]{1,120})\*\*\s*")
NAMES = {"lowercase_thinking": r"lower", "uppercase_thinking": r"upper|capital", "third_person": r"third"}

H = runs()
R = {k.replace("Qwen3-14B ", ""): v[1] for k, v in F.reasonif_runs().items() if k.startswith("Qwen3-14B")}
for f in sorted((F.RES / "data_variants").glob("*/reasonif_300/qwen3_14b/*/scored_responses.jsonl")):
    R["SFT " + f.parts[-5].replace("_", " ")] = F._jl(f)

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
base = {r["dataset_index"]: r for r in R["Base"]}
s42 = {r["dataset_index"]: r for r in R["SFT seed 42"]}
rows = []
for k, recs in R.items():
    ifs = np.array([int(r["instruction_following"]) for r in recs])
    row = {"Model": k, "IFS %": round(100 * ifs.mean(), 2), "Accuracy %": round(100 * np.mean([r["answer_correct"] for r in recs]), 2),
           "Joint": int(sum(r["joint_success"] for r in recs)), "Mean tokens": round(float(np.mean([r["output_tokens"] for r in recs])), 1)}
    for tag, ref in (("vs Base", base), ("vs SFT42", s42)):
        if k not in ("Base", "SFT seed 42") or (tag == "vs SFT42" and k == "Base"):
            d = np.array([int(r["instruction_following"]) - int(ref[r["dataset_index"]]["instruction_following"]) for r in recs])
            lo, hi = F._ci(d)
            row[f"IFS {tag} [95% CI]"] = f"{100 * d.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"
    rows.append(row)
print("== ReasonIF 300 (official scorer, as recorded)")
print(pd.DataFrame(rows).fillna("").to_string(index=False))

print("\n== CoT-Control openings")
rows = []
for k, d in H.items():
    heads = [LEAD.match(x).group(1) for x in d.reasoning if LEAD.match(x)]
    body = [LEAD.sub("", x, count=1) for x in d.reasoning]
    fw = collections.Counter((re.match(r"\s*([A-Za-z']+)", b) or [None, ""])[1] for b in body)
    named = []
    for task, pat in NAMES.items():
        sub = d[d.task == task].reasoning
        named.append(f"{sum(bool(re.search(pat, x[:300], re.I)) for x in sub)}/{len(sub)}")
    rows.append({"Model": k, "Header": f"{len(heads)}/{len(d)}", "Top headers": str(collections.Counter(heads).most_common(2)),
                 "First word after header": str(fw.most_common(3)), "Names rule early (lower/upper/third)": " ".join(named),
                 "Median words": int(d.words.median()), "Degenerate": int(d.degen.sum())})
print(pd.DataFrame(rows).to_string(index=False))
