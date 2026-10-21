"""ReasonIF 300 for the Qwen3-14B s1K teacher-test students (Claude 3.7 Sonnet vs DeepSeek R1 reasoning on the same 212 s1K
questions), next to Base, the reported SFT adapter, self-distillation and the Sonnet 4.5 teacher run.

Per run: IFS, accuracy, joint success, median reasoning words; paired difference against Base (95% bootstrap over the
300 questions, analysis_followups._ci); the Claude 3.7 minus R1 contrast; and the same numbers without the 24 ReasonIF
questions that share an 8-gram with one of the 212 s1K training questions (s1k_reasonif_overlap.json).

  python scripts/reasonif_s1k_summary.py      (writes results/summary_tables/reasonif_s1k_teacher.txt)
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_followups as F  # noqa: E402

R = F.RES / "reasonif_300"
TV = R / "qwen3_14b_training_variants"
PATHS = {"Base": R / "qwen3_14b/reasonif_qwen3_14b_base_untouched.jsonl",
         "SFT GPT-5.2 (seed 42)": R / "qwen3_14b/reasonif_qwen3_14b_sft_gpt52_high.jsonl",
         "Self-distillation": TV / "qwen3_14b_gpt52_qwen314/scored_responses.jsonl",
         "Sonnet 4.5 teacher (GPT-5.2 prompts)": TV / "qwen3_14b_sonnet_3_7_high_reasoning_original/scored_responses.jsonl"}
for v, lab in (("claude37", "s1K Claude 3.7"), ("deepseek_r1", "s1K DeepSeek R1")):
    f = glob.glob(str(F.RES / f"data_variants/s1k_{v}/reasonif_300/qwen3_14b/*/scored_responses.jsonl"))
    assert len(f) == 1, (v, f)
    PATHS[lab] = Path(f[0])
OVER = set(json.loads((F.RES / "summary_tables/s1k_reasonif_overlap.json").read_text(encoding="utf-8"))["used_212"]["dataset_index_shared_8gram_or_exact"])
assert len(OVER) == 24


def load(p):
    d = pd.DataFrame([json.loads(x) for x in p.read_text(encoding="utf-8").split("\n") if x.strip()])
    assert sorted(d.dataset_index) == list(range(300)), p
    return pd.DataFrame({"i": d.dataset_index.astype(int), "ifs": d.instruction_following.astype(str).str.lower().eq("true").astype(float),
                         "acc": d.answer_correct.astype(str).str.lower().eq("true").astype(float),
                         "words": d.reasoning_content.fillna("").astype(str).str.split().str.len()}).sort_values("i").reset_index(drop=True)


D = {k: load(p) for k, p in PATHS.items()}


def diff(a, b, col, keep):
    x = (b[col] - a[col])[keep].to_numpy()
    lo, hi = F._ci(pd.Series(x))
    return f"{100 * x.mean():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"


lines = []
for name, keep in (("All 300 ReasonIF questions", np.ones(300, bool)),
                   (f"Without the {len(OVER)} questions that share an 8-gram with the s1K training questions", ~D["Base"].i.isin(OVER).to_numpy())):
    rows = []
    for k, d in D.items():
        row = {"Run": k, "IFS %": round(100 * d.ifs[keep].mean(), 1), "Accuracy %": round(100 * d.acc[keep].mean(), 1),
               "Joint %": round(100 * (d.ifs * d.acc)[keep].mean(), 1), "Median words": int(d.words[keep].median())}
        if k != "Base":
            row["IFS vs Base"] = diff(D["Base"], d, "ifs", keep)
            row["Accuracy vs Base"] = diff(D["Base"], d, "acc", keep)
        rows.append(row)
    pd.set_option("display.width", 250)
    lines.append(f"{name} (n={int(keep.sum())})\n" + pd.DataFrame(rows).fillna("").to_string(index=False))
    c, r = D["s1K Claude 3.7"], D["s1K DeepSeek R1"]
    lines.append(f"  Claude 3.7 student minus R1 student: IFS {diff(r, c, 'ifs', keep)}; accuracy {diff(r, c, 'acc', keep)}")
text = "\n\n".join(lines)
print(text)
(F.RES / "summary_tables/reasonif_s1k_teacher.txt").write_text(text + "\n", encoding="utf-8")


def cons(p):
    d = pd.DataFrame([json.loads(x) for x in p.read_text(encoding="utf-8").split("\n") if x.strip()])
    return d.sort_values("dataset_index").constraint.astype(str).str.split(":").str[-1].reset_index(drop=True)


C = cons(PATHS["Base"])
assert all((cons(p) == C).all() for p in PATHS.values())
keep_runs = ["Base", "SFT GPT-5.2 (seed 42)", "s1K Claude 3.7", "s1K DeepSeek R1"]
t = pd.DataFrame({k: (100 * D[k].ifs.groupby(C).mean()).round(1) for k in keep_runs})
t.insert(0, "n", C.value_counts())
extra = ["\nIFS (%) by constraint", t.to_string()]
nw = (C != "number_words").to_numpy()
c, r = D["s1K Claude 3.7"], D["s1K DeepSeek R1"]
extra.append(f"\nWithout the word-limit constraint (n={int(nw.sum())}): Claude 3.7 student IFS vs Base {diff(D['Base'], c, 'ifs', nw)}; "
             f"R1 student vs Base {diff(D['Base'], r, 'ifs', nw)}; Claude 3.7 minus R1 {diff(r, c, 'ifs', nw)}")
print("\n".join(extra))
with open(F.RES / "summary_tables/reasonif_s1k_teacher.txt", "a", encoding="utf-8") as fh:
    fh.write("\n".join(extra) + "\n")


def trunc(p):
    d = pd.DataFrame([json.loads(x) for x in p.read_text(encoding="utf-8").split("\n") if x.strip()]).sort_values("dataset_index")
    return d.truncated.astype(str).str.lower().eq("true").to_numpy()


T = {k: trunc(p) for k, p in PATHS.items()}
extra = ["\nTruncation (hit the 16,384-token limit) and scores within truncated / complete responses"]
for k in D:
    t = T[k]
    extra.append(f"  {k}: truncated {int(t.sum())}/300; IFS truncated {100 * D[k].ifs[t].mean() if t.any() else float('nan'):.1f}, "
                 f"complete {100 * D[k].ifs[~t].mean():.1f}; accuracy truncated {100 * D[k].acc[t].mean() if t.any() else float('nan'):.1f}, "
                 f"complete {100 * D[k].acc[~t].mean():.1f}")
for name, ks in (("Base, Claude 3.7 student and R1 student", ["Base", "s1K Claude 3.7", "s1K DeepSeek R1"]),
                 ("all six runs", list(D))):
    keep = ~np.any([T[k] for k in ks], axis=0)
    extra.append(f"\nQuestions on which none of {name} was truncated (n={int(keep.sum())})")
    rows = []
    for k in D:
        row = {"Run": k, "IFS %": round(100 * D[k].ifs[keep].mean(), 1), "Accuracy %": round(100 * D[k].acc[keep].mean(), 1)}
        if k != "Base":
            row["IFS vs Base"] = diff(D["Base"], D[k], "ifs", keep)
            row["Accuracy vs Base"] = diff(D["Base"], D[k], "acc", keep)
        rows.append(row)
    extra.append(pd.DataFrame(rows).fillna("").to_string(index=False))
    extra.append(f"  Claude 3.7 student minus R1 student: IFS {diff(r, c, 'ifs', keep)}; accuracy {diff(r, c, 'acc', keep)}")
print("\n".join(extra))
with open(F.RES / "summary_tables/reasonif_s1k_teacher.txt", "a", encoding="utf-8") as fh:
    fh.write("\n".join(extra) + "\n")

# Selection that does not use the Claude 3.7 student's own outcome: questions on which Base and the R1 student both finished.
keep = ~(T["Base"] | T["s1K DeepSeek R1"])
extra = [f"\nQuestions on which Base and the R1 student both finished (selection independent of the Claude 3.7 student; n={int(keep.sum())})",
         f"  Claude 3.7 student truncated on {int(T['s1K Claude 3.7'][keep].sum())} of them (kept, scored as they are)",
         f"  Claude 3.7 student minus R1 student: IFS {diff(r, c, 'ifs', keep)}; accuracy {diff(r, c, 'acc', keep)}",
         f"  Claude 3.7 student vs Base: IFS {diff(D['Base'], c, 'ifs', keep)}; R1 student vs Base: IFS {diff(D['Base'], r, 'ifs', keep)}",
         f"  Base accuracy on the kept questions {100 * D['Base'].acc[keep].mean():.1f} vs {100 * D['Base'].acc.mean():.1f} on all 300; "
         f"on questions the Claude 3.7 student truncated {100 * D['Base'].acc[T['s1K Claude 3.7']].mean():.1f} (Base IFS there {100 * D['Base'].ifs[T['s1K Claude 3.7']].mean():.1f})"]
print("\n".join(extra))
with open(F.RES / "summary_tables/reasonif_s1k_teacher.txt", "a", encoding="utf-8") as fh:
    fh.write("\n".join(extra) + "\n")
