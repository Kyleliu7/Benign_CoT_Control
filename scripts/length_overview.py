"""Reasoning length before and after training on CoT-Control (500 traces per run), per model and per task category, and
Phi-4-reasoning compliance as a function of reasoning length.

1. Per run: mean and median reasoning words, mean reasoning tokens (each model's own tokenizer, so tokens are comparable
   only within a model), traces that hit the generation limit, and the seven-task mean.
2. Per run and category: mean reasoning words and mean compliance. Categories follow Chen et al. (2026): suppression
   (word, multiple words), addition (end of sentence, meow, arrow prefix, repeat sentences), stylistic split into third
   person (in the seven-task composite) and the three case tasks (not in it).
3. Phi-4-reasoning: mean compliance by reasoning-token bin (fixed edges, all runs), seven tasks and case tasks. The
   generation budget is 3000 tokens for every run here, so traces that hit it get their own column.

Same grading as the paper (analysis_followups.primary_score). Writes results/summary_tables/length_overview.txt.

  python scripts/length_overview.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_followups as F  # noqa: E402

_orig_prep = F._prep


def _prep_with_tokens(d):
    out = _orig_prep(d)
    col = "reasoning_tokens_reencoded" if "reasoning_tokens_reencoded" in d else "reasoning_tokens"
    out["tokens"] = pd.to_numeric(d[col], errors="coerce").to_numpy() if col in d else np.nan
    return out


F._prep = _prep_with_tokens
from haskins_header_stripped import runs  # noqa: E402
from seven_task_composite import SEVEN  # noqa: E402

CAT = {"word_suppression": "suppression", "multiple_word_suppression": "suppression",
       "end_of_sentence": "addition", "meow_between_words": "addition", "arrow_prefix": "addition", "repeat_sentences": "addition",
       "third_person": "third person", "lowercase_thinking": "case", "uppercase_thinking": "case", "alternating_case": "case"}
CATS = ["suppression", "addition", "third person", "case"]

H = {"Qwen3-14B " + k: v for k, v in runs().items()}
H.update({k: v for k, v in F.haskins_runs().items() if not k.startswith("Qwen3-14B")})
for fam, pre, sub in (("Phi-4-reasoning", "phi4_", "phi4_reasoning"), ("OLMo-3-7B-Think", "olmo3_", "olmo3_7b_think")):
    for f in sorted((F.RES / "data_variants").glob(f"{pre}s1k_*/haskins_500/{sub}/*/results.csv")):
        H[f"{fam} SFT " + f.parts[-5].replace(pre, "").replace("_", " ")] = F._prep(pd.read_csv(f))
for v, lab in (("self_distill", "SFT self-distillation"), ("teacher_sonnet_3_7", "SFT Sonnet 4.5 teacher (GPT-5.2 prompts)")):
    f = sorted(F.RES.glob(f"haskins_500/qwen3_14b_training_variants/{v}/haskins_500/qwen3_14b/*/results.csv"))
    if f:
        H["Qwen3-14B " + lab] = F._prep(pd.read_csv(f[0]))
ORDER = ["Qwen3-14B Base", "Qwen3-14B SFT seed 42", "Qwen3-14B SFT self-distillation", "Qwen3-14B SFT Sonnet 4.5 teacher (GPT-5.2 prompts)",
         "Qwen3-14B SFT s1k claude37", "Qwen3-14B SFT s1k deepseek r1",
         "Phi-4-reasoning Base", "Phi-4-reasoning SFT", "Phi-4-reasoning SFT s1k claude37", "Phi-4-reasoning SFT s1k deepseek r1",
         "OLMo-3-7B-Think Base", "OLMo-3-7B-Think SFT lr1e-4", "OLMo-3-7B-Think SFT s1k claude37", "OLMo-3-7B-Think SFT s1k deepseek r1"]
H = {k: H[k].assign(cat=H[k].task.map(CAT)) for k in ORDER if k in H}
assert all(d.cat.notna().all() for d in H.values())

out = []
pd.set_option("display.width", 250)
rows = []
for k, d in H.items():
    rows.append({"Run": k, "Mean words": round(d.words.mean()), "Median words": int(d.words.median()),
                 "Mean tokens": round(d.tokens.mean()) if d.tokens.notna().any() else "",
                 "Hit limit": int(d.capped.sum()) if "capped" in d else "",
                 "7-task mean %": round(100 * d[d.task.isin(SEVEN)].primary.mean(), 1)})
out.append("1. Reasoning length per run (500 traces; tokens use each model's own tokenizer)\n" + pd.DataFrame(rows).to_string(index=False))

w = pd.DataFrame({k: d.groupby("cat").words.mean().reindex(CATS).round(0) for k, d in H.items()}).T
c = pd.DataFrame({k: (100 * d.groupby("cat").primary.mean()).reindex(CATS).round(1) for k, d in H.items()}).T
both = pd.concat({"Mean words": w, "Compliance %": c}, axis=1)
out.append("2. Per category: mean reasoning words and mean compliance\n" + both.to_string())

EDGES = [0, 250, 500, 1000, 2000, np.inf]
LABS = ["<250", "250-500", "500-1k", "1k-2k", "2k-3k (not capped)", "hit 3000 limit"]
blocks = []
for k in [x for x in H if x.startswith("Phi-4-reasoning")]:
    b = pd.cut(H[k].tokens, EDGES, labels=LABS[:-1], right=False).cat.add_categories([LABS[-1]])
    b[H[k].capped.to_numpy()] = LABS[-1]
    d = H[k].assign(bin=b)
    for name, part in (("7 tasks", d[d.task.isin(SEVEN)]), ("case tasks", d[d.cat == "case"])):
        g = part.groupby("bin", observed=False).primary.agg(["mean", "size"])
        blocks.append(pd.Series([f"{100 * m:.1f} (n={n})" if n else "--" for m, n in zip(g["mean"], g["size"])],
                                index=LABS, name=f"{k.replace('Phi-4-reasoning ', '')} | {name}"))
out.append("3. Phi-4-reasoning: mean compliance (%) by reasoning tokens (n = traces in the bin)\n" + pd.DataFrame(blocks).to_string())
for k in [x for x in H if x.startswith("Phi-4-reasoning")]:
    d = H[k][H[k].task.isin(SEVEN)]
    r = np.corrcoef(np.log1p(d.tokens), d.primary)[0, 1]
    out.append(f"   {k}: correlation of log reasoning tokens with seven-task compliance r = {r:+.3f} (n={len(d)})")
text = "\n\n".join(out[:3]) + "\n" + "\n".join(out[3:])
print(text)
(F.RES / "summary_tables" / "length_overview.txt").write_text(text + "\n", encoding="utf-8")
