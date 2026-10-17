"""CoT-Control results on the seven tasks that are not character-level (all but lowercase, uppercase and alternating case).

Haskins et al. (2026, Table 11) find that only these seven tasks predict obfuscation across their eight models (composite
r = 0.898 against 0.800 for all ten; the three case tasks give r = 0.011, -0.197 and -0.469), so the seven-task mean
(350 question-task pairs) is the main CoT-Control number of the revision. Same grading rule as the rest of the paper
(binary for word suppression, multiple word suppression, end-of-sentence and meow, fractional otherwise) and the same
paired cluster bootstrap over the 50 questions (10,000 resamples, seed 42, analysis_followups._ci).

Writes results/summary_tables/seven_task_composite_standalone.csv (every standalone run: mean, strict passes of 350,
gain over its Base for the whole trace, the first 100 words, degenerate traces scored 0, and a leading bold heading
removed before grading; the ten-task gain for comparison) and seven_task_composite_crossed.csv (donor-recipient
continuation means and the four paired contrasts, Qwen and Phi), and seven_task_composite_controls.csv (fixed-opening
controls restricted to the five non-case tasks they include).

  python scripts/seven_task_composite.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_followups as F  # noqa: E402
import analysis_prefix_controls as C  # noqa: E402
import analysis_prefix_length_paired as A  # noqa: E402
from haskins_header_stripped import rescore, runs  # noqa: E402

SEVEN = ["third_person", "arrow_prefix", "word_suppression", "multiple_word_suppression", "end_of_sentence",
         "repeat_sentences", "meow_between_words"]
assert not set(SEVEN) & set(F.CASE_TASKS) and len(SEVEN) == 7
OUT = F.RES / "summary_tables"


def gain(base, run, col, tasks=SEVEN, window=None):
    n, _, _, g, lo, hi = F._paired_gain(base, run, None, col=col, window=window, tasks=tasks)
    return n, f"{g:+.2f} [{lo:+.2f}, {hi:+.2f}]", g


def standalone():
    H = runs()
    H.update({k: v for k, v in F.haskins_runs().items() if not k.startswith("Qwen3-14B")})  # Phi-4-reasoning and OLMo
    for v in ("output_mask", "think_mask", "teacher_qwen3_235b", "teacher_sonnet_3_7", "teacher_reasonflux", "self_distill"):
        f = glob.glob(str(F.RES / f"haskins_500/qwen3_14b_training_variants/{v}/haskins_500/qwen3_14b/*/results.csv"))
        if f:
            H["SFT " + v.replace("_", " ")] = F._prep(pd.read_csv(f[0]))
    for f in sorted((F.RES / "data_variants").glob("phi4_*/haskins_500/phi4_reasoning/*/results.csv")):
        H["Phi-4-reasoning SFT " + f.parts[-5].replace("phi4_", "").replace("_", " ")] = F._prep(pd.read_csv(f))
    H = {k: rescore(v) for k, v in H.items()}
    for d in H.values():
        d["primary_d0"] = np.where(d.degen, 0.0, d.primary)
    rows = []
    for k, d in H.items():
        fam = k.split(" ")[0]
        base_key = {"Phi-4-reasoning": "Phi-4-reasoning Base", "OLMo-3-7B-Think": "OLMo-3-7B-Think Base"}.get(fam, "Base")
        s = d[d.task.isin(SEVEN)]
        row = {"Run": k, "Base": base_key, "N": len(s), "Mean7": round(100 * s.primary.mean(), 2),
               "Strict7": int(s.strict.sum()), "Mean7_header_stripped": round(100 * s.primary_s.mean(), 2),
               "Mean10": round(100 * d.primary.mean(), 2)}
        if k != base_key:
            b = H[base_key]
            row["Gain7"] = gain(b, d, "primary")[1]
            n100, row["Gain7_first100"], _ = gain(b, d, "primary", window=100)
            row["Pairs7_first100"] = n100
            row["Gain7_degenerate0"] = gain(b, d, "primary_d0")[1]
            row["Gain7_header_stripped"] = gain(b, d, "primary_s")[1]
            row["Gain10"] = gain(b, d, "primary", tasks=None)[1]
        rows.append(row)
    return pd.DataFrame(rows)


def crossed():
    L = A.load_long()
    L = L[L.task.isin(SEVEN)]
    rng = np.random.default_rng(F.SEED)
    rows = []
    for model, g in L.groupby("model"):
        per_q = {c: g[g.cond == c].groupby("q")[["comp", "strict"]].mean().sort_index() for c in ["BB", "BS", "SB", "SS", "A5"] if (g.cond == c).any()}
        for c, pq in per_q.items():
            sub = g[g.cond == c]
            rows.append(dict(Model=model, Row=A.COND_LABEL[c], N=len(sub), Cont_Mean7=round(100 * sub.comp.mean(), 2),
                             Strict7=int(sub.strict.sum()), Estimate="", CI95=""))
        for label, x, y in [("SFT->Base minus Base->Base (donor effect, base recipient)", "SB", "BB"),
                            ("SFT->SFT minus Base->SFT (donor effect, SFT recipient)", "SS", "BS"),
                            ("Base->SFT minus Base->Base (recipient effect, base donor)", "BS", "BB"),
                            ("SFT->SFT minus SFT->Base (recipient effect, SFT donor)", "SS", "SB"),
                            ("SFT->Base OFF minus Base->Base", "A5", "BB")]:
            if x not in per_q or y not in per_q:
                continue
            a, b = per_q[x].comp.align(per_q[y].comp, join="inner")
            est, lo, hi = A._boot_prompt_diff(a.to_numpy(), b.to_numpy(), np.random.default_rng(F.SEED))
            rows.append(dict(Model=model, Row=label, N=len(a), Cont_Mean7="", Strict7="", Estimate=round(100 * est, 2),
                             CI95=f"[{100 * lo:.2f}, {100 * hi:.2f}]"))
    return pd.DataFrame(rows)


def controls():
    five = [t for t in C.STYLE_TASKS if t in SEVEN]
    runs_ = {}
    for key, pat in {"BB": "*2x2-base-to-base-on-10tok__*", "SB": "*2x2-sft-to-base-on-10tok__*"}.items():
        hits = C._find(pat)
        if hits:
            runs_[key] = C._load(hits[0])
    for cond in ("handcrafted", "named", "mismatched", "cross_question", "modal_sft"):
        hits = C._find(f"*control-{cond}-10tok__*")
        if hits:
            runs_[cond] = C._load(hits[0])
    if "BB" not in runs_ or "SB" not in runs_:
        return pd.DataFrame()
    rng = np.random.default_rng(F.SEED)
    rows = []
    for key, df in runs_.items():
        df = df[df.task.isin(five)]
        row = dict(Condition=C.LABELS.get(key, key), Tasks=", ".join(five), N=len(df),
                   Cont_Mean=round(100 * df.continuation_compliance.mean(), 2), Strict=int(df.continuation_score_one.sum()))
        for ref in ("BB", "SB"):
            if key != ref:
                r = runs_[ref][runs_[ref].task.isin(five)]
                d = C._paired(C._per_question(df), C._per_question(r), rng)
                row[f"Diff_vs_{ref}"] = f"{d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]"
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 30)
    for name, fn in (("standalone", standalone), ("crossed", crossed), ("controls", controls)):
        t = fn()
        t.to_csv(OUT / f"seven_task_composite_{name}.csv", index=False)
        print(f"== {name}\n{t.fillna('').to_string(index=False)}\n")
