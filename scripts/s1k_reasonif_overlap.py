"""Overlap between the s1K questions and the 300 ReasonIF questions (and the 50 CoT-Control questions), to decide whether
the s1K students can be evaluated on ReasonIF. For every ReasonIF question: whether some s1K question shares an 8-gram
with it, the largest share of its 8-grams found in one s1K question, and exact matches after normalization. Reported for
all 1,000 s1K questions and for the 212 used in training, by ReasonIF source.

  python scripts/s1k_reasonif_overlap.py      (reads the simplescaling/s1K-1.1 questions through the Hugging Face datasets-server API)
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import urllib.request

REPO = Path(__file__).resolve().parents[1]


def words(t):
    return re.findall(r"\w+", str(t).lower())


def grams(t, n=8):
    w = words(t)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


s1k = []
for off in range(0, 1000, 100):
    url = f"https://datasets-server.huggingface.co/rows?dataset=simplescaling%2Fs1K-1.1&config=default&split=train&offset={off}&length=100"
    s1k += [r["row"]["question"] for r in json.load(urllib.request.urlopen(url))["rows"]]
assert len(s1k) == 1000, len(s1k)
used = {e["instruction"] for e in json.loads((REPO / "data/training/s1k_claude37_212.json").read_text(encoding="utf-8"))}
assert len(used) == 212 and used <= set(s1k)
rif = json.loads((REPO / "data/eval_prompts/reasonif_dataset_300.json").read_text(encoding="utf-8"))
cot = [p["prompt"] for p in json.loads((REPO / "data/eval_prompts/diverse_prompts.json").read_text(encoding="utf-8"))]
S = [(q, grams(q), " ".join(words(q))) for q in s1k]


def check(text, pool):
    g = grams(text)
    norm = " ".join(words(text))
    best = max((len(g & sg) / len(g) for q, sg, _ in pool), default=0.0) if g else 0.0
    return bool(g) and any(g & sg for _, sg, _ in pool), best, any(norm == sn for _, _, sn in pool)


out = {}
for name, pool in (("all_1000", S), ("used_212", [s for s in S if s[0] in used])):
    rows = []
    for r in rif:
        shared, best, exact = check(r["question"], pool)
        rows.append({"source": r["source"], "shared_8gram": shared, "max_share": best, "exact": exact, "dataset_index": r["dataset_index"]})
    df = pd.DataFrame(rows)
    out[name] = {"reasonif_with_shared_8gram": int(df.shared_8gram.sum()), "reasonif_with_half_8grams_in_one_s1k_question": int((df.max_share >= 0.5).sum()),
                 "reasonif_exact": int(df.exact.sum()),
                 "by_source_shared": {k: int(v) for k, v in df[df.shared_8gram].source.value_counts().items()},
                 "by_source_half": {k: int(v) for k, v in df[df.max_share >= 0.5].source.value_counts().items()},
                 "cot_control_with_shared_8gram": sum(check(q, pool)[0] for q in cot),
                 "dataset_index_shared_8gram_or_exact": sorted(int(i) for i in df[df.shared_8gram | df.exact].dataset_index)}
out["reasonif_sources"] = dict(Counter(r["source"] for r in rif))
(REPO / "results/summary_tables/s1k_reasonif_overlap.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
