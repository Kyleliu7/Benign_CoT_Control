"""LaTeX tables and text numbers for the revision that reports the seven-task CoT-Control composite as the main number
(same conventions as make_review_tables.py: no number typed by hand; paired cluster bootstrap over the 50 questions).

  tab:models      Base and SFT models: ReasonIF, and CoT-Control on the seven non-character tasks (mean, strict of 350,
                  gain for the whole trace, the first 100 words and with degenerate traces scored 0) plus the ten-task mean
  tab:headings    heading experiments, CoT-Control graded with a leading heading removed: seven-task mean and gain,
                  ten-task gain, ReasonIF
  tab:teacher     teacher test: same 212 s1K questions with Claude 3.7 Sonnet or DeepSeek R1 reasoning, plus the
                  same-prompt Sonnet 4.5 and self-distillation runs
  tab:controls    fixed ten-token openings on the five non-case style tasks (and the eight-task mean for reference)
  tab:seven-all   every standalone run (appendix)
  numbers7.json   numbers quoted in the text

  python scripts/make_seven_task_tables.py [OUT_DIR]      (default: paper_tables/)
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_followups as F  # noqa: E402
import analysis_prefix_controls as C  # noqa: E402
import analysis_prefix_length_paired as A  # noqa: E402
from haskins_header_stripped import LEAD, rescore, runs  # noqa: E402
from seven_task_composite import SEVEN  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "paper_tables"
OUT.mkdir(parents=True, exist_ok=True)
NUM = {}


def tbl(name, caption, label, colspec, header, rows, note, star=True, pre=None):
    env = "table*" if star else "table"
    lines = [rf"\begin{{{env}}}[t]", r"\centering\small", rf"\caption{{{caption}}}\label{{{label}}}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{adjustbox}{max width=\linewidth}", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    lines += pre or []
    lines += [" & ".join(header) + r" \\", r"\midrule"]
    lines += [r"\midrule" if row == "MID" else " & ".join(str(c) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
              rf"\par\smallskip\begin{{minipage}}{{0.97\linewidth}}\footnotesize {note}\end{{minipage}}", rf"\end{{{env}}}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT / f"{name}.tex")


def sgn(x, nd=2):
    return f"{x:+.{nd}f}".replace("-", "$-$")


def ci(est, lo, hi, nd=2):
    return f"{sgn(est, nd)} [{sgn(lo, nd)}, {sgn(hi, nd)}]".replace("[+", "[").replace(", +", ", ")


def gain(b, d, col="primary", tasks=SEVEN, window=None):
    n, _, _, g, lo, hi = F._paired_gain(b, d, None, col=col, window=window, tasks=tasks)
    return n, g, lo, hi


def put(key, n_g_lo_hi, nd=2):
    n, g, lo, hi = n_g_lo_hi
    NUM[key] = {"est": round(g, nd), "lo": round(lo, nd), "hi": round(hi, nd), "n": int(n)}
    return ci(g, lo, hi, nd)


def first(pattern):
    hits = sorted(glob.glob(str(pattern)))
    return hits[0] if hits else None


# ---------------------------------------------------------------- runs
H = runs()
H.update({k: v for k, v in F.haskins_runs().items() if not k.startswith("Qwen3-14B")})
for v in ("output_mask", "think_mask", "teacher_qwen3_235b", "teacher_sonnet_3_7", "teacher_reasonflux", "self_distill"):
    f = first(RES / f"haskins_500/qwen3_14b_training_variants/{v}/haskins_500/qwen3_14b/*/results.csv")
    if f:
        H["SFT " + v.replace("_", " ")] = F._prep(pd.read_csv(f))
for f in sorted((RES / "data_variants").glob("phi4_*/haskins_500/phi4_reasoning/*/results.csv")):
    H["Phi-4-reasoning SFT " + f.parts[-5].replace("phi4_", "").replace("_", " ")] = F._prep(pd.read_csv(f))
H = {k: rescore(v) for k, v in H.items()}
for d in H.values():
    d["primary_d0"] = np.where(d.degen, 0.0, d.primary)
R = {k: v[1] for k, v in F.reasonif_runs().items()}
f = first(RES / "olmo3_7b_think/reasonif_300_sft_lr2e-4/*/scored_responses.jsonl")
if f:
    R["OLMo-3-7B-Think SFT lr2e-4"] = F._jl(f)
BASE_OF = {"Phi-4-reasoning": "Phi-4-reasoning Base", "OLMo-3-7B-Think": "OLMo-3-7B-Think Base"}


def base_of(k):
    return BASE_OF.get(k.split(" ")[0], "Base")


def m7(d, col="primary"):
    return 100 * d[d.task.isin(SEVEN)][col].mean()


def strict7(d):
    return int(d[d.task.isin(SEVEN)].strict.sum())


def rif(k):
    recs = R.get(k)
    if not recs:
        return "", ""
    return f"{100 * np.mean([bool(r['instruction_following']) for r in recs]):.1f}", f"{100 * np.mean([bool(r['answer_correct']) for r in recs]):.1f}"


# ---------------------------------------------------------------- tab:models
MODELS = [("Qwen3-14B", [("Base", "Base", "Qwen3-14B Base"), ("SFT (seed 42)", "SFT seed 42", "Qwen3-14B SFT seed 42"),
                         ("SFT (seed 1)", "SFT seed 1", "Qwen3-14B SFT seed 1"), ("SFT (seed 2)", "SFT seed 2", "Qwen3-14B SFT seed 2")]),
          ("Phi-4-reasoning", [("Base", "Phi-4-reasoning Base", "Phi-4-reasoning Base"), ("SFT", "Phi-4-reasoning SFT", "Phi-4-reasoning SFT")]),
          ("OLMo-3-7B-Think", [("Base", "OLMo-3-7B-Think Base", "OLMo-3-7B-Think Base"), ("SFT", "OLMo-3-7B-Think SFT lr1e-4", "OLMo-3-7B-Think SFT lr1e-4"),
                               (r"SFT, lr $2\times10^{-4}$", "OLMo-3-7B-Think SFT lr2e-4", "OLMo-3-7B-Think SFT lr2e-4")])]
rows = []
for model, rr in MODELS:
    if rows:
        rows.append("MID")
    for i, (run, hk, rk) in enumerate(rr):
        d = H[hk]
        ifs, acc = rif(rk)
        row = [model if i == 0 else "", run, ifs, acc, f"{m7(d):.2f}", strict7(d)]
        key = hk.replace(" ", "_")
        NUM[f"mean7_{key}"] = round(m7(d), 2)
        NUM[f"strict7_{key}"] = strict7(d)
        NUM[f"mean10_{key}"] = round(100 * d.primary.mean(), 2)
        if hk == base_of(hk):
            row += ["--", "--", "--"]
        else:
            b = H[base_of(hk)]
            row += [put(f"gain7_{key}", gain(b, d)), put(f"gain7_first100_{key}", gain(b, d, window=100)),
                    put(f"gain7_degen0_{key}", gain(b, d, col="primary_d0"))]
            put(f"gain7_stripped_{key}", gain(b, d, col="primary_s"))
            put(f"gain10_{key}", gain(b, d, tasks=None))
        row.append(f"{100 * d.primary.mean():.2f}")
        rows.append(row)
for w in (200, 300):
    put(f"gain7_first{w}_SFT_seed_42", gain(H["Base"], H["SFT seed 42"], window=w))
tbl("table_models7", "Base and SFT models on ReasonIF (300 questions) and CoT-Control. The CoT-Control columns use the seven tasks that are not "
    "character-level (350 question--task pairs), the composite that predicts obfuscation in \\citet{haskins2026obfuscation}; the last column is the "
    "ten-task mean.", "tab:models", "llrrrrllll",
    ["Model", "Run", "IFS", "Accuracy", "Mean", "Strict", "Gain [95\\% CI]", "Gain, first 100 words", "Gain, degenerate $=0$", "10-task mean"],
    rows, "Seed 42 is the reported Qwen adapter; seeds 1 and 2 change only the training seed. OLMo SFT uses the Qwen recipe; the learning rate "
    "$2\\times10^{-4}$ run was added after the first result. Gains are paired differences in percentage points with 95\\% cluster-bootstrap "
    "intervals over the 50 questions; the first-100-words column keeps only pairs in which both traces reach 100 words; the degenerate column "
    "scores degenerate traces (fewer than 20 words of reasoning, or more than 90\\% ``meow'') as 0. Strict: pairs with a compliance score of "
    "exactly 1, out of 350.",
    pre=[r" & & \multicolumn{2}{c}{ReasonIF (\%)} & \multicolumn{5}{c}{CoT-Control, seven tasks} & \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-9}"])

# ---------------------------------------------------------------- tab:headings
HEAD_ROWS = [("Base (no fine-tuning)", "Base", "Qwen3-14B Base"),
             ("GPT-5.2 summaries (reported adapter)", "SFT seed 42", "Qwen3-14B SFT seed 42"),
             ("GPT-5.2 summaries, headings removed", "SFT no headers", "no_headers"),
             (r"GPT-5.2 summaries, one fixed \texttt{**Reasoning**} heading", "SFT neutral header", "neutral_header"),
             ("Qwen3-235B-A22B reasoning (same prompts)", "SFT teacher qwen3 235b", "t235"),
             ("Qwen3-235B-A22B reasoning with the GPT-5.2 opening heading", "SFT header transplant", "header_transplant"),
             ("Qwen3-14B's own reasoning (self-distillation)", "SFT self distill", "self_distill"),
             ("Qwen3-14B's own reasoning with the GPT-5.2 opening heading", "SFT self headers", "self_headers")]
for key, sub in (("no_headers", "data_variants/no_headers/reasonif_300/qwen3_14b/*"), ("neutral_header", "data_variants/neutral_header/reasonif_300/qwen3_14b/*"),
                 ("header_transplant", "data_variants/header_transplant/reasonif_300/qwen3_14b/*"), ("self_headers", "data_variants/self_headers/reasonif_300/qwen3_14b/*"),
                 ("t235", "reasonif_300/qwen3_14b_training_variants/qwen3_14b_qwen3_235b_high_reasoning_original"),
                 ("self_distill", "reasonif_300/qwen3_14b_training_variants/qwen3_14b_gpt52_qwen314")):
    f = first(RES / sub / "scored_responses.jsonl")
    if f:
        R[key] = F._jl(f)
base_r = {r["dataset_index"]: r for r in R["Qwen3-14B Base"]}
rows = []
for label, hk, rk in HEAD_ROWS:
    if hk not in H:
        continue
    d = H[hk]
    row = [label, int(d.had_header.sum()), f"{m7(d, 'primary_s'):.2f}"]
    if hk == "Base":
        row += ["--", "--"]
    else:
        key = hk.replace(" ", "_")
        row += [put(f"gain7_stripped_{key}", gain(H["Base"], d, col="primary_s")), put(f"gain10_stripped_{key}", gain(H["Base"], d, col="primary_s", tasks=None))]
        put(f"gain7_{key}", gain(H["Base"], d))
    recs = R.get(rk)
    if recs:
        ifs = np.array([int(bool(r["instruction_following"])) for r in recs])
        row.append(f"{100 * ifs.mean():.1f}")
        if rk == "Qwen3-14B Base":
            row.append("--")
        else:
            dd = np.array([int(bool(r["instruction_following"])) - int(bool(base_r[r["dataset_index"]]["instruction_following"])) for r in recs])
            lo, hi = F._ci(dd)
            row.append(ci(100 * dd.mean(), 100 * lo, 100 * hi, 1))
            NUM[f"rif_{hk.replace(' ', '_')}"] = round(100 * ifs.mean(), 1)
            NUM[f"rif_gain_{hk.replace(' ', '_')}"] = {"est": round(100 * dd.mean(), 1), "lo": round(100 * lo, 1), "hi": round(100 * hi, 1)}
    else:
        row += ["", ""]
    rows.append(row)
tbl("table_headings7", "Heading experiments (Qwen3-14B, one training run each except the reported adapter). CoT-Control is graded with a leading "
    "bold heading removed from every trace.", "tab:headings", "lrrllrl",
    ["Training data", "Opens with heading", "7-task mean", "7-task gain (95\\% CI)", "10-task gain (95\\% CI)", "ReasonIF IFS (\\%)", "IFS gain (95\\% CI)"],
    rows, "Opens with heading: CoT-Control traces (of 500) whose reasoning starts with a bold heading. 7-task: the seven tasks that are not "
    "character-level (350 pairs); 10-task: all 500 pairs. CoT-Control gains are paired differences with cluster-bootstrap intervals over the 50 "
    "questions; ReasonIF gains with bootstrap intervals over the 300 questions. Same recipe for every adapter (LoRA $r=32$, learning rate $10^{-4}$, "
    "3 epochs, seed 42, lowest-validation-loss checkpoint).")

# ---------------------------------------------------------------- tab:teacher
TEACH = [("Base (no fine-tuning)", "Base", "--", "--"),
         ("Claude 3.7 Sonnet reasoning", "SFT s1k claude37", "s1K, 212 questions", "51.1 / 49.9"),
         ("DeepSeek R1 reasoning", "SFT s1k deepseek r1", "s1K, 212 questions", "0.2 / 0.0"),
         ("Claude Sonnet 4.5 reasoning", "SFT teacher sonnet 3 7", "GPT-5.2 prompts", "0.2 / 0.0"),
         ("Qwen3-14B's own reasoning", "SFT self distill", "GPT-5.2 prompts", "0.1 / 0.0"),
         ("GPT-5.2 reasoning summaries (reported adapter)", "SFT seed 42", "GPT-5.2 prompts", "")]
rows = []
for label, hk, data, chen in TEACH:
    if hk not in H:
        continue
    d = H[hk]
    row = [label, data, chen, f"{m7(d):.2f}"]
    if hk == "Base":
        row += ["--", "--", "--", "--"]
    else:
        key = hk.replace(" ", "_")
        row += [put(f"gain7_{key}", gain(H["Base"], d)), put(f"gain7_first100_{key}", gain(H["Base"], d, window=100)),
                put(f"gain7_degen0_{key}", gain(H["Base"], d, col="primary_d0")), put(f"gain10_{key}", gain(H["Base"], d, tasks=None))]
    row += [int(d.words.median()), int(d.capped.sum()) if "capped" in d else ""]
    rows.append(row)
if "SFT s1k claude37" in H and "SFT s1k deepseek r1" in H:
    put("claude37_minus_r1_7", gain(H["SFT s1k deepseek r1"], H["SFT s1k claude37"]))
    put("claude37_minus_r1_10", gain(H["SFT s1k deepseek r1"], H["SFT s1k claude37"], tasks=None))
    NUM["s1k_selection"] = json.loads((REPO / "data/training/s1k_selection.json").read_text())["stats"]
    NUM["s1k_eligible"] = NUM["s1k_selection"]["eligible"]
    NUM["s1k_questions"] = NUM["s1k_selection"]["questions"]
tbl("table_teacher", "Does the teacher matter? Qwen3-14B fine-tuned with the reported recipe on reasoning from different teachers (one run each). "
    "CoT-Control on the seven non-character tasks; gains over Base.", "tab:teacher", "lllrllllrr",
    ["Teacher", "Prompts", "Teacher stylistic / addition (\\%)", "7-task mean", "Gain [95\\% CI]", "Gain, first 100 words", "Gain, degenerate $=0$",
     "10-task gain", "Median words", "Hit cap"], rows,
    "Teacher controllability: stylistic-modification and addition compliance from \\citet[Table 2]{chen2026cotcontrol} (GPT-5.2 is not in that table). "
    "The s1K sets pair the same 212 questions of \\texttt{simplescaling/s1K-claude-3-7-sonnet} and \\texttt{simplescaling/s1K-1.1} with each teacher's "
    "raw reasoning and answer (training cutoff 12,288 tokens, batch $1\\times8$; questions sharing an 8-gram with CoT-Control excluded); the "
    "other rows use the 212 GPT-5.2 prompts. Hit cap: traces (of 500) that reached the 3,000-token limit. Paired differences with cluster-bootstrap "
    "intervals over the 50 questions.")

# ---------------------------------------------------------------- tab:controls (five non-case style tasks)
FIVE = [t for t in C.STYLE_TASKS if t in SEVEN]
runs_ = {}
for key, pat in {"BB": "*2x2-base-to-base-on-10tok__*", "SB": "*2x2-sft-to-base-on-10tok__*"}.items():
    runs_[key] = C._load(C._find(pat)[0])
for cond in ("handcrafted", "named", "mismatched", "cross_question", "modal_sft"):
    hits = C._find(f"*control-{cond}-10tok__*")
    if hits:
        runs_[cond] = C._load(hits[0])
rng = np.random.default_rng(F.SEED)
rows = []
for key, df in runs_.items():
    five = df[df.task.isin(FIVE)]
    row = [C.LABELS.get(key, key), f"{100 * five.continuation_compliance.mean():.2f}", int(five.continuation_score_one.sum())]
    for ref in ("BB", "SB"):
        if key == ref:
            row.append("--")
            continue
        r5 = runs_[ref][runs_[ref].task.isin(FIVE)]
        d = C._paired(C._per_question(five), C._per_question(r5), rng)
        NUM[f"controls5_{key}_vs_{ref}"] = {"est": round(d[0], 2), "lo": round(d[1], 2), "hi": round(d[2], 2)}
        row.append(ci(d[0], d[1], d[2]))
    NUM[f"controls5_{key}_mean"] = round(100 * five.continuation_compliance.mean(), 2)
    NUM[f"controls8_{key}_mean"] = round(100 * df.continuation_compliance.mean(), 2)
    row.append(f"{100 * df.continuation_compliance.mean():.2f}")
    rows.append(row)
tbl("table_controls7", "Qwen3-14B base recipient continuing from a fixed ten-token opening, on the five style tasks that are not character-level "
    "(third person, arrow prefix, end of sentence, meow, repeat sentences; 250 items per condition, continuation scored).", "tab:controls", "lrrllr",
    ["Opening", "Mean", "Strict", "Difference from base donor (95\\% CI)", "Difference from SFT donor (95\\% CI)", "8-task mean"], rows,
    "Paired differences in percentage points with cluster-bootstrap intervals over the 50 questions (10{,}000 resamples, seed 42). The suppression "
    "tasks are not part of these runs. 8-task mean: the same runs on the eight style tasks (adding lowercase, uppercase and alternating case), as in "
    "the previous version. Openings are listed in Appendix Table~\\ref{tab:control-openings}.")

# ---------------------------------------------------------------- crossed contrasts (seven tasks; a fresh generator per interval)
L = A.load_long()
L = L[L.task.isin(SEVEN)]
on = pd.DataFrame(F._jl(RES / "haskins_500/qwen3_14b/haskins_qwen3_14b_vllm_calibrated_prefix_on_37pct.jsonl"))
on = on[on.task.isin(SEVEN)]
on["comp"] = [A.primary_score(t, float(c)) for t, c in zip(on.task, on.continuation_compliance)]
rng = np.random.default_rng(F.SEED)
for model, g in L.groupby("model"):
    pq = {c: g[g.cond == c].groupby("q").comp.mean().sort_index() for c in g.cond.unique()}
    for c in pq:
        NUM[f"cross7_{model}_{c}_mean"] = round(100 * g[g.cond == c].comp.mean(), 2)
        NUM[f"cross7_{model}_{c}_strict"] = int(g[g.cond == c].strict.sum())
    pairs = [("SB", "BB"), ("SS", "BS"), ("BS", "BB"), ("SS", "SB")] + ([("A5", "BB"), ("A5", "SB")] if "A5" in pq else [])
    for x, y in pairs:
        a, b = pq[x].align(pq[y], join="inner")
        est, lo, hi = A._boot_prompt_diff(a.to_numpy(), b.to_numpy(), np.random.default_rng(F.SEED))
        NUM[f"cross7_{model}_{x}_minus_{y}"] = {"est": round(100 * est, 2), "lo": round(100 * lo, 2), "hi": round(100 * hi, 2)}
    if model == "Qwen3-14B":
        on_q = on.assign(q=on.prompt_idx.astype(int)).groupby("q").comp.mean().sort_index()
        a, b = pq["A5"].align(on_q, join="inner")
        est, lo, hi = A._boot_prompt_diff(a.to_numpy(), b.to_numpy(), np.random.default_rng(F.SEED))
        NUM["cross7_Qwen3-14B_A5_minus_A1rerun"] = {"est": round(100 * est, 2), "lo": round(100 * lo, 2), "hi": round(100 * hi, 2)}
        NUM["cross7_Qwen3-14B_A1rerun_mean"] = round(100 * on.comp.mean(), 2)

# ---------------------------------------------------------------- tab:seven-all (appendix)
LAB = {"Base": "Qwen3-14B Base", "SFT seed 42": "Qwen3-14B SFT, seed 42 (reported)", "SFT seed 1": "Qwen3-14B SFT, seed 1",
       "SFT seed 2": "Qwen3-14B SFT, seed 2", "SFT no headers": "GPT-5.2 summaries, headings removed",
       "SFT neutral header": r"GPT-5.2 summaries, fixed \texttt{**Reasoning**} heading",
       "SFT header transplant": "Qwen3-235B-A22B reasoning + GPT-5.2 opening heading", "SFT self headers": "Qwen3-14B's own reasoning + GPT-5.2 opening heading",
       "SFT output mask": "GPT-5.2 data, loss on reasoning only", "SFT think mask": "GPT-5.2 data, loss on answer only",
       "SFT teacher qwen3 235b": "Qwen3-235B-A22B reasoning", "SFT teacher sonnet 3 7": "Claude Sonnet 4.5 reasoning",
       "SFT teacher reasonflux": "ReasonFlux reasoning", "SFT self distill": "Qwen3-14B's own reasoning",
       "SFT s1k claude37": "s1K, Claude 3.7 Sonnet reasoning", "SFT s1k deepseek r1": "s1K, DeepSeek R1 reasoning",
       "Phi-4-reasoning Base": "Phi-4-reasoning Base", "Phi-4-reasoning SFT": "Phi-4-reasoning SFT",
       "Phi-4-reasoning SFT s1k claude37": "Phi-4-reasoning, s1K Claude 3.7 Sonnet reasoning", "Phi-4-reasoning SFT s1k deepseek r1": "Phi-4-reasoning, s1K DeepSeek R1 reasoning", "OLMo-3-7B-Think Base": "OLMo-3-7B-Think Base",
       "OLMo-3-7B-Think SFT lr1e-4": "OLMo-3-7B-Think SFT", "OLMo-3-7B-Think SFT lr2e-4": r"OLMo-3-7B-Think SFT, lr $2\times10^{-4}$"}
rows = []
for k, nm in LAB.items():
    if k not in H:
        continue
    d = H[k]
    row = [nm, f"{m7(d):.2f}", strict7(d), f"{m7(d, 'primary_s'):.2f}", f"{100 * d.primary.mean():.2f}"]
    if k == base_of(k):
        row += ["--", "--", "--"]
    else:
        b = H[base_of(k)]
        key = k.replace(" ", "_")
        row += [put(f"gain7_{key}", gain(b, d)), put(f"gain7_stripped_{key}", gain(b, d, col="primary_s")), put(f"gain10_{key}", gain(b, d, tasks=None))]
    rows.append(row)
tbl("table_seven_all", "CoT-Control for every standalone run: the seven non-character tasks (350 pairs) against all ten (500 pairs).", "tab:seven-all",
    "lrrrrlll", ["Run", "7-task mean", "Strict /350", "7-task, heading removed", "10-task mean", "7-task gain", "7-task gain, heading removed", "10-task gain"],
    rows, "Gains are over the same model's Base (Qwen3-14B for every Qwen run), paired differences with 95\\% cluster-bootstrap intervals over the 50 "
    "questions. One training run each except the Qwen SFT seeds.")

for k in ("SFT seed 42", "Phi-4-reasoning SFT", "SFT s1k claude37", "SFT s1k deepseek r1"):
    b, d = H[base_of(k)], H[k]
    for t in SEVEN:
        n, g, lo, hi = gain(b, d, tasks=[t])
        NUM[f"task_gain7_{k.replace(' ', '_')}_{t}"] = round(g, 1)
        NUM[f"task_mean_{k.replace(' ', '_')}_{t}"] = round(100 * d[d.task == t].primary.mean(), 1)
        NUM[f"task_mean_{base_of(k).replace(' ', '_')}_{t}"] = round(100 * b[b.task == t].primary.mean(), 1)

for k in ("Base", "SFT seed 42", "SFT self distill", "SFT self headers", "SFT s1k claude37", "SFT s1k deepseek r1", "SFT teacher sonnet 3 7"):
    if k in H:
        NUM[f"lower_{k.replace(' ', '_')}"] = round(100 * H[k][H[k].task == "lowercase_thinking"].primary_s.mean(), 1)
for k in ("SFT s1k claude37", "SFT s1k deepseek r1"):
    if k in H:
        NUM[f"capped_{k.replace(' ', '_')}"] = int(H[k].capped.sum())
NUM["capped_Base"] = int(H["Base"].capped.sum())

PB = "Phi-4-reasoning Base"
for v in ("s1k claude37", "s1k deepseek r1"):
    k = f"Phi-4-reasoning SFT {v}"
    if k in H:
        key = k.replace(" ", "_")
        put(f"gain7_{key}", gain(H[PB], H[k]))
        put(f"gain7_first100_{key}", gain(H[PB], H[k], window=100))
if "Phi-4-reasoning SFT s1k claude37" in H and "Phi-4-reasoning SFT s1k deepseek r1" in H:
    put("phi_claude37_minus_r1_7", gain(H["Phi-4-reasoning SFT s1k deepseek r1"], H["Phi-4-reasoning SFT s1k claude37"]))
    put("phi_claude37_minus_r1_7_first100", gain(H["Phi-4-reasoning SFT s1k deepseek r1"], H["Phi-4-reasoning SFT s1k claude37"], window=100))

ov = RES / "summary_tables/s1k_reasonif_overlap.json"
if ov.exists():
    o = json.loads(ov.read_text(encoding="utf-8"))
    NUM["rif_s1k_exact_all"] = o["all_1000"]["reasonif_exact"]
    NUM["rif_s1k_exact_used"] = o["used_212"]["reasonif_exact"]
    NUM["rif_s1k_8gram_all"] = o["all_1000"]["reasonif_with_shared_8gram"]
    NUM["rif_s1k_8gram_used"] = o["used_212"]["reasonif_with_shared_8gram"]

(OUT / "numbers7.json").write_text(json.dumps(NUM, indent=1), encoding="utf-8")
print("wrote", OUT / "numbers7.json", len(NUM), "entries")
