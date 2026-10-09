"""
Generate every results table of the manuscript as LaTeX from the verified CSVs and raw records.

  python scripts/make_paper_tables.py [OUT_DIR]        (default: paper_tables/)

Run `python scripts/reproduce_tables.py --write` first. Each table is written as OUT_DIR/<name>.tex and is meant
to be pulled into main.tex with \\input{tables/<name>}. No number in these files is typed by hand.
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
T = REPO / "results" / "summary_tables"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "paper_tables"
OUT.mkdir(parents=True, exist_ok=True)


class R(str):
    """Raw LaTeX (not escaped)."""


def esc(s) -> str:
    if isinstance(s, R):
        return str(s)
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s.replace("->", r"$\to$")


def write(name, caption, label, colspec, header, rows, note=None, resize=True, rule_after=()):
    lines = [r"\begin{table}[htbp]", r"\centering\small", rf"\caption{{{caption}}}\label{{{label}}}", r"\setlength{\tabcolsep}{4pt}"]
    if resize:
        lines.append(r"\begin{adjustbox}{max width=\linewidth}")
    lines += [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(str(h) for h in header) + r" \\", r"\midrule"]
    for i, row in enumerate(rows):
        lines.append(" & ".join(esc(c) for c in row) + r" \\")
        if i in rule_after:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if resize:
        lines.append(r"\end{adjustbox}")
    if note:
        lines.append(rf"\par\smallskip\begin{{minipage}}{{0.97\linewidth}}\footnotesize {note}\end{{minipage}}")
    lines.append(r"\end{table}")
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def f1(x):
    return f"{float(x):.1f}"


def f2(x):
    return f"{float(x):.2f}"


def ci(lo, hi, nd=1):
    return f"[{float(lo):.{nd}f}, {float(hi):.{nd}f}]"


def pct(n, d, nd=1):
    return f"{100 * float(n) / float(d):.{nd}f}"


def truthy(v):
    return v is True or str(v) == "True"


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


t1 = pd.read_csv(T / "table1_reasonif_overall.csv", dtype=str, keep_default_na=False)
t2 = pd.read_csv(T / "table2_haskins_standalone.csv", dtype=str, keep_default_na=False)
t3 = pd.read_csv(T / "table3_haskins_crossed_2x2.csv", dtype=str, keep_default_na=False)
t4 = pd.read_csv(T / "table4_kl_percentiles.csv")
t5 = pd.read_csv(T / "table5_top_spike_per_task_phi4.csv", keep_default_na=False)
t6 = pd.read_csv(T / "table6_haskins_per_task.csv")
t7 = pd.read_csv(T / "table7_haskins_run_health.csv")
t8 = pd.read_csv(T / "table8_window_matched_haskins.csv", dtype={"Window_Words": str})
t9 = pd.read_csv(T / "table9_prefix_content_haskins.csv")
t9b = pd.read_csv(T / "table9b_prefix_content_per_task.csv")
t10 = pd.read_csv(T / "table10_prefix_conditional_haskins.csv")
t11 = pd.read_csv(T / "table11_prefix_examples.csv", keep_default_na=False)
t12 = pd.read_csv(T / "table12_paired_contrasts_haskins.csv")
t13 = pd.read_csv(T / "table13_paired_contrasts_reasonif.csv")
fig1 = pd.read_csv(REPO / "reproducible_figures" / "figure1_haskins_crossed_source.csv")

# ----------------------------------------------------------------------------------------------- Table 1
rows = []
for r in t1.itertuples():
    rows.append([f"{r.Model.split('-')[0]} {r.Condition.replace(' (continuation-scored)', '').replace(' (Untouched)', '').replace(' (gpt52-high)', '')}"
                 + (r"$^\dagger$" if "aggregate" in r.Status else ""),
                 f"{r.IFS_Pct} ({r.IFS_Count})", f"{r.Acc_Pct} ({r.Acc_Count})", f"{r.Joint_Pct} ({r.Joint_Count})",
                 f"{float(r.Mean_Tokens):,.1f}", r.Truncated])
rows = [[R(esc(c)) if i == 0 and "dagger" in str(c) else c for i, c in enumerate(row)] for row in rows]
write("t1_reasonif", "ReasonIF outcomes (300 questions per condition). Percent with counts. Prefix rows are scored on the recipient continuation only.",
      "tab:reasonif", "lrrrrr", ["Condition", "IFS", "Accuracy", "Joint", "Mean tokens", "Truncated"], rows,
      note=r"IFS: all instructions followed. Joint: IFS and correct answer. $^\dagger$ Aggregate CSV only; raw records for the Phi prefix runs are not in the "
           r"repository. In the Qwen prefix runs the forced prefix is shorter than 10 tokens for "
           + f"{t1.loc[t1.Condition.str.contains('Prefix-OFF') & t1.Model.eq('Qwen3-14B'), 'Prefix_Shorter_Than_10'].iloc[0]} (Prefix-OFF) and "
           + f"{t1.loc[t1.Condition.str.contains('Prefix-ON') & t1.Model.eq('Qwen3-14B'), 'Prefix_Shorter_Than_10'].iloc[0]} (Prefix-ON) questions "
           r"(15 Prefix-OFF prefixes are empty). All numbers are single-run point estimates; paired differences with intervals are in Table~\ref{tab:reasonif-paired}.")

# ----------------------------------------------------------------------------------------------- Table 2
rows = []
for r in t2.itertuples():
    scoring = r.Scoring.replace("whole trace", "whole trace").replace("full trace incl. prefix", "prefix + continuation")
    rows.append([r.Model.split("-")[0].replace("Qwen3","Qwen"), r.Condition, scoring, r.Mean_Compliance, f"{r.Strict_Count} ({pct(r.Strict_Count, r.N)})",
                 r.NonChar_Mean, r.NonChar_Strict_Count, r.Hit_Token_Limit])
write("t2_haskins_standalone", "Haskins standalone and prefix runs, 500 question--constraint pairs per condition.",
      "tab:standalone", "lllrrrrr", ["Model", "Condition", "Scoring basis", "Mean", "Strict (\\%)", "Non-char mean", "Non-char strict /350", "Hit cap"], rows,
      note=r"Mean: average per-example score in percent. Strict: score exactly 1. Non-char: the seven non-character tasks (350 pairs). "
           r"``Hit cap'': generations that reached the 3,000-token limit. For the two prefix runs the same completions are scored twice: on the continuation only "
           r"(the model-generated tokens after the injected prefix) and on prefix plus continuation. Compare Prefix-ON with Prefix-OFF within one scoring basis. "
           r"Prefix-OFF is a donor that does not see the constraint; Prefix-ON is the same SFT-to-Base condition as Table~\ref{tab:cross}, run a second time.",
      rule_after=(5,))

# ----------------------------------------------------------------------------------------------- Table 3
cimap = {("Qwen3-14B", "Base -> Base (A2)"): ("Qwen 3 14B", "Base->Base"), ("Qwen3-14B", "Base -> SFT (A3)"): ("Qwen 3 14B", "Base->SFT"),
         ("Qwen3-14B", "SFT -> Base (A1 / Prefix-ON)"): ("Qwen 3 14B", "SFT->Base"), ("Qwen3-14B", "SFT -> SFT (A4)"): ("Qwen 3 14B", "SFT->SFT"),
         ("Phi-4-reasoning", "Base -> Base (A2)"): ("Microsoft Phi-4 Reasoning", "Base->Base"), ("Phi-4-reasoning", "Base -> SFT (A3)"): ("Microsoft Phi-4 Reasoning", "Base->SFT"),
         ("Phi-4-reasoning", "SFT -> Base (A1 / Prefix-ON)"): ("Microsoft Phi-4 Reasoning", "SFT->Base"), ("Phi-4-reasoning", "SFT -> SFT (A4)"): ("Microsoft Phi-4 Reasoning", "SFT->SFT")}
rows = []
for r in t3.itertuples():
    key = cimap.get((r.Model, r.Pairing))
    if key:
        c = fig1[(fig1.model == key[0]) & (fig1.condition == key[1])].iloc[0]
        mean = f"{float(r.Cont_All_Mean):.2f} {ci(c.comp_ci95_low, c.comp_ci95_high)}"
        strict = f"{int(float(r.Cont_All_Strict_Count))} ({pct(r.Cont_All_Strict_Count, 500)}) {ci(c.strict_ci95_low, c.strict_ci95_high)}"
    else:
        mean, strict = f2(r.Cont_All_Mean), f"{int(float(r.Cont_All_Strict_Count))} ({pct(r.Cont_All_Strict_Count, 500)})"
    pair = r.Pairing.replace(" (A1 / Prefix-ON)", " (A1)")
    rows.append([r.Model.split("-")[0].replace("Qwen3","Qwen"), pair.replace("->", "->"), mean, strict, f2(r.Cont_NonChar_Mean), int(float(r.Cont_NonChar_Strict_Count)), f2(r.Full_All_Mean)])
write("t3_crossed", "Crossed donor--recipient Haskins results, scored on the recipient continuation (500 pairs per row).",
      "tab:cross", "llrrrrr", ["Model", "Donor $\\to$ recipient", "Mean [95\\% CI]", "Strict (\\%) [95\\% CI]", "Non-char mean", "Non-char strict /350", "Full-trace mean"],
      [[R(esc(c)) if i == 1 else c for i, c in enumerate(row)] for row in rows],
      note=r"Intervals are cluster-bootstrap 95\% intervals (10{,}000 resamples of the 50 underlying questions, seed 42); they do not include training-seed uncertainty "
           r"(one training run per model). The A5 row (SFT donor not shown the constraint) and the calibrated A1 rerun come from the calibrated vLLM runs of Table~\ref{tab:standalone} "
           r"and have no interval. In Qwen, 45 of 500 SFT-donor prefixes differ between the two recipient runs (vLLM batch nondeterminism); base-donor prefixes are identical in 500 of 500. "
           r"Phi strict passes occur only in non-character tasks, hence equal counts in the all-task and non-character columns.",
      rule_after=(5,))

# ----------------------------------------------------------------------------------------------- Table 4 paired Haskins
rows = []
piv = {}
for r in t12.itertuples():
    piv.setdefault((r.Model, r.Contrast), {})[r.Metric] = f"{r.Estimate_pp:+.2f} {ci(r.CI95_Low, r.CI95_High, 2)}"
for (model, contrast), d in sorted(piv.items(), key=lambda kv: (not kv[0][0].startswith("Qwen"),)):
    rows.append([model.split("-")[0].replace("Qwen3","Qwen"), contrast, d["continuation mean (pp)"], d["strict pass rate (pp)"]])
write("t4_haskins_paired", "Paired differences between crossed Haskins conditions (percentage points, continuation scoring).", "tab:haskins-paired",
      "llrr", ["Model", "Contrast", "$\\Delta$ mean [95\\% CI]", "$\\Delta$ strict [95\\% CI]"],
      [[R(esc(c)) if i == 2 or i == 3 else c for i, c in enumerate(row)] for row in rows],
      note=r"Cluster bootstrap over the 50 questions (10{,}000 resamples, seed 42), computed on per-question means over the ten tasks. Donor effects hold the recipient fixed; "
           r"recipient effects hold the donor fixed. Intervals reflect sampling variation of questions and generations, not training seeds.", rule_after=(3,))

# ----------------------------------------------------------------------------------------------- Table 5 paired ReasonIF
rows = []
piv = {}
for r in t13.itertuples():
    p = "<0.0001" if r.McNemar_Exact_P < 1e-4 else f"{r.McNemar_Exact_P:.4f}"
    piv.setdefault((r.Model, r.Contrast), {})[r.Metric] = f"{r.Estimate_pp:+.1f} {ci(r.CI95_Low, r.CI95_High)} p={p}"
for (model, contrast), d in piv.items():
    rows.append([model.split("-")[0].replace("Qwen3","Qwen"), contrast, d["IFS"], d["Accuracy"], d["Joint"]])
write("t5_reasonif_paired", "Paired differences on the 300 ReasonIF questions (percentage points).", "tab:reasonif-paired",
      "llrrr", ["Model", "Contrast", "IFS", "Accuracy", "Joint"], rows,
      note=r"Each cell: difference, 95\% bootstrap interval over questions (10{,}000 resamples), and exact two-sided McNemar $p$ on the discordant questions. "
           r"No correction for multiple comparisons; $p$-values are descriptive. Prefix rows use continuation scoring.", rule_after=(4,))

# ----------------------------------------------------------------------------------------------- Prefix content
rows = [[r.Model.split("-")[0].replace("Qwen3","Qwen"), r.Condition, r.N, f"{r.Prefix_Self_Compliant_Pct:.1f}", f"{r.Prefix_Names_Or_Demonstrates_Pct:.1f}", f"{r.Prefix_Starts_With_Bold_Heading_Pct:.1f}"]
        for r in t9.itertuples()]
rows.sort(key=lambda x: (x[0] != "Qwen", x[1]))
write("t6_prefix_content", "What the injected 10-token prefixes contain (eight non-suppression tasks, 400 prefixes per row).", "tab:prefix-content",
      "llrrrr", ["Model", "Donor $\\to$ recipient", "$N$", "Self-compliant (\\%)", "Names/shows constraint (\\%)", "Starts with bold heading (\\%)"],
      [[R(esc(c)) if i == 1 else c for i, c in enumerate(row)] for row in rows],
      note=r"Self-compliant: the prefix text alone scores exactly 1 under the task grader. Names/shows constraint: a task-specific regular expression matches the prefix "
           r"(e.g.\ ``lowercase'', ``uppercase'', ``the assistant'', ``meow'', ``this is my analysis''). The regular expressions are listed in "
           r"\texttt{scripts/analysis\_prefix\_length\_paired.py}. The prefix includes the \texttt{<think>} token, so it has nine content tokens. "
           r"Prefixes for Base$\to$Base and Base$\to$SFT (and for SFT$\to$Base and SFT$\to$SFT) come from the same donor runs; the Qwen SFT donor differs for 45 of 500 pairs.")

rows = [[r.Model.split("-")[0].replace("Qwen3","Qwen"), r.Condition, r.Prefix, r.N, f"{r.Continuation_Mean:.2f}", r.Strict_Count] for r in t10.itertuples()]
rows.sort(key=lambda x: (x[0] != "Qwen", x[1], x[2] != "prefix names/demonstrates constraint"))
write("t7_prefix_conditional", "Continuation compliance split by whether the prefix names or demonstrates the constraint (eight non-suppression tasks).",
      "tab:prefix-cond", "lllrrr", ["Model", "Donor $\\to$ recipient", "Prefix", "$N$", "Continuation mean", "Strict"],
      [[R(esc(c)) if i == 1 else c for i, c in enumerate(row)] for row in rows],
      note=r"Observational: the split is not randomised, and prefix content is confounded with task. It shows how much of the SFT-donor advantage is concentrated in "
           r"items whose prefix already contains the constraint.")

# ----------------------------------------------------------------------------------------------- Window matched
rows = []
for (model, w, n), g in sorted(t8.groupby(["Model", "Window_Words", "N_Pairs"], sort=False), key=lambda kv: kv[0][0] != "Qwen3-14B"):
    d = {r.Condition: f"{r.Mean_Compliance:.2f} ({r.Strict_Count})" for r in g.itertuples()}
    rows.append([model.split("-")[0].replace("Qwen3","Qwen"), w if w != "full continuation" else "full (all pairs)", n, d["Base->Base"], d["Base->SFT"], d["SFT->Base"], d["SFT->SFT"]])
write("t8_window_matched", "Haskins compliance when all four crossed conditions are scored on the same first $W$ words of the continuation.", "tab:window",
      "llrllll", ["Model", "$W$ (words)", "Pairs", "Base$\\to$Base", "Base$\\to$SFT", "SFT$\\to$Base", "SFT$\\to$SFT"],
      [[R(esc(c)) if i in (1, 3, 4, 5, 6) else c for i, c in enumerate(row)] for row in rows],
      note=r"Cell: mean compliance in percent (strict count). Only question--task pairs whose continuation has at least $W$ words in all four conditions are kept, "
           r"and every continuation is truncated to its first $W$ words before re-scoring with the paper grader. Equal windows remove the length advantage of shorter "
           r"continuations; they do not remove the effect of the prefix.", rule_after=(3,))

# ----------------------------------------------------------------------------------------------- KL
rows = []
for r in t4.itertuples():
    rows.append([r.Model_Benchmark, r.Traces, r.Cohort_L_ge_100, f"{r.Early_Share_Pct_First100_Cohort:.1f}", f"{r.Early_Share_Pct_Full512_Cohort:.1f}", f"{r.Mean_KL_t1:.1f}",
                 f"{r.Tokens_t_gt_10_Full512:,}", f"{r.Mean:.3f}", f"{r.P50:.3f}", f"{r.P99:.2f}", f"{r.P99_9:.2f}", f"{r.Max_Spike:.2f}", f"{r.Seq_Max_P95:.2f}"])
write("t9_kl", "Forward KL $\\mathcal{D}_{\\mathrm{KL}}(\\pi_{\\mathrm{SFT}}\\|\\pi_{\\mathrm{Base}})$ on SFT-generated histories (nats).", "tab:kl", "lrrrrrrrrrrrr",
      ["Model / benchmark", "Traces", "$L\\!\\ge\\!100$", "First-10 share, pos.\\ 1--100", "First-10 share, full", "Mean KL $t{=}1$", "Tokens $t{>}10$", "Mean", "P50", "P99", "P99.9", "Max", "Seq.\\ max P95"],
      [[R(esc(c)) if i in (3, 4, 5, 6) else c for i, c in enumerate(row)] for row in rows] if False else rows,
      note=r"Teacher-forced float32 KL over the first $\le$512 reasoning tokens. First-10 share: summed KL over positions 1--10 divided by summed KL over positions 1--100 (column 4) "
           r"or over the whole measured trace (column 5), both for traces with at least 100 tokens. Tokens $t{>}10$ through Seq.\ max P95 use every trace longer than 10 tokens and every "
           r"position up to 512. Seq.\ max P95 is the 95th percentile over traces of the largest downstream KL. Qwen's largest downstream values exceed Phi's on both benchmarks.")

# ----------------------------------------------------------------------------------------------- KL spikes / outcome groups
t14 = pd.read_csv(T / "table14_kl_spike_frequency.csv")
t15 = pd.read_csv(T / "table15_kl_outcome_groups.csv")
rows = [[f"{r.Model.split('-')[0].replace('Qwen3','Qwen')} {r.Benchmark}", f"{r.Downstream_Tokens:,}", f"{r.Pct_Tokens_KL_gt_2:.2f}", f"{r.Per_1000_Tokens_KL_gt_5:.2f}",
         f"{r.Traces_With_Any_KL_gt_5}/{r.Traces} ({r.Pct_Traces_With_Any_KL_gt_5:.1f})", f"{r.Max_KL:.2f}"] for r in t14.itertuples()]
write("t10_kl_spikes", "How often large downstream forward-KL values occur (tokens after position 10, nats).", "tab:kl-spikes", "lrrrrr",
      ["Model / benchmark", "Tokens", "\\% tokens $>2$", "Tokens $>5$ per 1000", "Traces with any $>5$ (\\%)", "Largest value"], rows,
      note=r"Computed on every trace longer than 10 tokens over the first $\le$512 positions. Qwen has the larger share of traces with a spike on both benchmarks and the larger maximum; Phi's per-token rate of values above 5 nats is slightly higher only on Haskins.")
rows = [[r.Model.split("-")[0].replace("Qwen3", "Qwen"), r.Contrast, f"{r.N_First} / {r.N_Second}", f"{r.Mean_Max_KL_First:.2f} / {r.Mean_Max_KL_Second:.2f}",
         f"{r.Difference:+.2f} [{r.CI95_Low:+.2f}, {r.CI95_High:+.2f}]"] for r in t15.itertuples()]
write("t11_kl_groups", "Largest downstream KL per trace by single-sample outcome group, Haskins (nats).", "tab:kl-groups", "llrrr",
      ["Model", "Contrast", "$N$ (first / second)", "Mean of per-trace max", "Difference [95\\% CI]"], rows,
      note=r"G1: the base model fails and the SFT model passes on that question in separate runs; G2: both pass; G3: both fail. Bootstrap over traces (5{,}000 resamples). "
           r"All intervals include zero: the data show no association between a large KL spike and the base model failing the constraint.")

# ----------------------------------------------------------------------------------------------- Appendix: lengths
q = REPO / "results" / "reasonif_300" / "qwen3_14b"
p = REPO / "results" / "reasonif_300" / "phi4_reasoning"
files = [("Qwen Base", q / "reasonif_qwen3_14b_base_untouched.jsonl"), ("Qwen SFT", q / "reasonif_qwen3_14b_sft_gpt52_high.jsonl"),
         ("Qwen Prefix-OFF", q / "reasonif_qwen3_14b_prefix_constraint_off.jsonl"), ("Qwen Prefix-ON", q / "reasonif_qwen3_14b_prefix_constraint_on.jsonl"),
         ("Phi Base", p / "base" / "scored_responses.jsonl"), ("Phi SFT", p / "sft" / "scored_responses.jsonl")]
rows = []
for name, f in files:
    rec = jl(f)
    tk = np.array([float(r["output_tokens"]) for r in rec])
    rows.append([name, f"{tk.mean():,.1f}", f"{np.median(tk):,.1f}", sum(truthy(r.get("truncated")) for r in rec)])
for name, folder in (("Phi Prefix-OFF", "prefix_off"), ("Phi Prefix-ON", "prefix_on")):
    d = pd.read_csv(p / folder / "overall.csv").iloc[0]
    rows.append([name + r"$^\dagger$", f"{d.mean_output_tokens:,.1f}", f"{d.median_output_tokens:,.1f}", round(d.truncated * d.questions)])
write("a1_reasonif_length", "ReasonIF output length and truncation.", "tab:reasonif-length", "lrrr", ["Condition", "Mean tokens", "Median tokens", "Truncated / 300"],
      [[R(esc(c)) if i == 0 and "dagger" in str(c) else c for i, c in enumerate(row)] for row in rows],
      note=r"Total output tokens as recorded (not reasoning-only, not donor plus recipient cost). $^\dagger$ aggregate CSV only.")

# ----------------------------------------------------------------------------------------------- Appendix: per task
TASK_NAMES = {"alternating_case": "Alternating case", "arrow_prefix": "Arrow prefix", "end_of_sentence": "End of sentence", "lowercase_thinking": "Lowercase",
              "meow_between_words": "Meow between words", "multiple_word_suppression": "Multiple-word suppression", "repeat_sentences": "Repeat sentences",
              "third_person": "Third person", "uppercase_thinking": "Uppercase", "word_suppression": "Word suppression"}
SHORT = {"Base -> Base (A2)": r"B$\to$B", "Base -> SFT (A3)": r"B$\to$S", "SFT -> Base (A1 / Prefix-ON)": r"S$\to$B", "SFT -> SFT (A4)": r"S$\to$S", "SFT -> Base OFF (A5)": r"S$\to$B OFF"}
for model, name, label in (("Qwen3-14B", "a2_tasks_qwen", "tab:tasks-qwen"), ("Phi-4-reasoning", "a3_tasks_phi", "tab:tasks-phi")):
    g = t6[t6.Model == model]
    pairings = list(dict.fromkeys(g.Pairing))
    rows = []
    for task in sorted(TASK_NAMES, key=lambda k: TASK_NAMES[k]):
        row = [TASK_NAMES[task]]
        for pr in pairings:
            x = g[(g.Pairing == pr) & (g.Task == task)].iloc[0]
            row.append(f"{x.Cont_Mean:.2f} / {int(x.Cont_Strict_Count)}")
        rows.append(row)
    write(name, f"{model.split('-')[0]} Haskins task-level continuation scores.", label, "l" + "r" * len(pairings),
          ["Task"] + [R(SHORT[p_]) for p_ in pairings], rows,
          note=r"Each cell: mean compliance (\%) / number of strict passes out of 50. B = Base, S = SFT, donor first.")

# ----------------------------------------------------------------------------------------------- Appendix: ReasonIF per constraint
CNAME = {"punctuation:no_comma": "No commas", "language:reasoning_language": "Language", "change_case:english_capital": "All capitals",
         "length_constraint_checkers:number_words": "Word limit", "detectable_format:json_format": "JSON", "startend:end_checker": "Ending phrase"}
order = list(CNAME)


def by_constraint(rec, flag):
    out = {c: [0, 0] for c in order}
    for r in rec:
        out[r["constraint"]][0] += 1
        out[r["constraint"]][1] += truthy(r[flag])
    return out


qb = by_constraint(jl(files[0][1]), "instruction_following")
qs = by_constraint(jl(files[1][1]), "instruction_following")
qo = by_constraint(jl(files[2][1]), "continuation_instruction_following")
qn = by_constraint(jl(files[3][1]), "continuation_instruction_following")
rows = [[CNAME[c], qb[c][0]] + [f"{pct(d[c][1], d[c][0])} ({d[c][1]})" for d in (qb, qs, qo, qn)] for c in order]
write("a4_reasonif_constraint_qwen", "Qwen ReasonIF instruction following by constraint.", "tab:reason-tasks-qwen", "lrrrrr", ["Constraint", "N", "Base", "SFT", "OFF", "ON"], rows,
      note=r"Cell: IFS percent (passing count). Prefix scores exclude the injected text.")
pb = by_constraint(jl(files[4][1]), "instruction_following")
ps = by_constraint(jl(files[5][1]), "instruction_following")
po = {c: [0, 0] for c in order}
pn = {c: [0, 0] for c in order}
for d, folder in ((po, "prefix_off"), (pn, "prefix_on")):
    for r in pd.read_csv(p / folder / "by_constraint.csv").itertuples():
        d[r.constraint] = [int(r.questions), round(r.continuation_ifs * r.questions)]
rows = [[CNAME[c], pb[c][0]] + [f"{pct(d[c][1], d[c][0])} ({d[c][1]})" for d in (pb, ps, po, pn)] for c in order]
write("a5_reasonif_constraint_phi", "Phi ReasonIF instruction following by constraint.", "tab:reason-tasks-phi", "lrrrrr", ["Constraint", "N", "Base", "SFT", "OFF", "ON"], rows,
      note=r"Cell: IFS percent (passing count). OFF/ON come from per-constraint CSVs (raw records absent); prefix scores exclude the injected text.")

# ----------------------------------------------------------------------------------------------- Appendix: prefix audit, spikes, examples, health
qdirs = {k: pd.read_csv(glob.glob(str(REPO / "results/haskins_500/qwen3_14b" / f"*{v}__*" / "results.csv"))[0]).set_index("id") for k, v in
         {"BB": "2x2-base-to-base-on-10tok", "BS": "2x2-base-to-sft-on-10tok", "SB": "2x2-sft-to-base-on-10tok", "SS": "2x2-sft-to-sft-on-10tok"}.items()}
same_b = int((qdirs["BB"].injected_prefix_token_ids == qdirs["BS"].injected_prefix_token_ids.reindex(qdirs["BB"].index)).sum())
same_s = int((qdirs["SB"].injected_prefix_token_ids == qdirs["SS"].injected_prefix_token_ids.reindex(qdirs["SB"].index)).sum())
write("a6_prefix_audit", "Qwen prefix identity across recipients (token IDs compared per task and question).", "tab:prefix-audit", "lrr", ["Donor", "Identical prefixes", "Different prefixes"],
      [["Base", f"{same_b}/500", 500 - same_b], ["SFT", f"{same_s}/500", 500 - same_s]],
      note=r"Equality is checked per task and question on the injected token IDs of the two recipient runs.")

rows = [[r.Task, r.Pos_t, f"{r.KL_nats:.2f}", r.Preceding_Context, r.Base_Token, r.SFT_Token, "yes" if r.Base_Run_Strict else "no", "yes" if r.SFT_Run_Strict else "no"] for r in t5.itertuples()]
write("a7_spikes", "Largest recorded KL spike for each Haskins task (Phi-4-reasoning).", "tab:spikes", "lrrp{3.4cm}llll",
      ["Task:question", "$t$", "KL (nats)", "Preceding context", "Base token", "SFT token", "Base run strict", "SFT run strict"],
      [[R(esc(c)) if i == 1 else (R(r"\texttt{" + esc(c) + "}") if i in (4, 5) else c) for i, c in enumerate(row)] for row in rows],
      note=r"Straight from the stored spike records (\texttt{spike\_token\_pairs.json}). Tokens are the highest-probability tokens of each model at that position under teacher forcing. "
           r"``Run strict'' is the whole-trace outcome of a separate base or SFT run for that question; it does not say that the displayed base token violates the constraint.",
      resize=True)

rows = [[r.Model.split("-")[0].replace("Qwen3","Qwen"), r.Task, r.SFT_Donor_Prefix] for r in t11.itertuples()]
write("a8_prefix_examples", "The SFT-donor prefix for the first question, by task (nine content tokens after \\texttt{<think>}).", "tab:prefix-examples", "llp{8cm}",
      ["Model", "Task", "Prefix"], [[a, b, R(r"\texttt{" + esc(c) + "}")] for a, b, c in rows], rule_after=(9,))

rows = []
for r in t9b[t9b.Condition == "SFT->Base"].itertuples():
    rows.append([r.Model.split("-")[0].replace("Qwen3","Qwen"), TASK_NAMES.get(r.Task, r.Task), r.Self_Compliant, r.Names_Or_Demonstrates, r.Bold_Heading])
rows.sort(key=lambda x: (x[0] != "Qwen", x[1]))
write("a9_prefix_per_task", "SFT-donor prefix content by task (out of 50 prefixes per task).", "tab:prefix-per-task", "llrrr",
      ["Model", "Task", "Self-compliant", "Names/shows constraint", "Bold heading"], rows, rule_after=(7,))

rows = [[r.Model.split("-")[0].replace("Qwen3","Qwen"), r.Run, r.N, r.Hit_3000_Token_Limit, r.Unfinished_Reasoning, f"{r.Mean_Continuation_Tokens:,.1f}"] for r in t7.itertuples()]
write("a10_run_health", "Haskins generation health: token-cap hits, unfinished reasoning, and mean generated tokens.", "tab:health", "llrrrr",
      ["Model", "Run", "N", "Hit 3,000-token cap", "Unfinished reasoning", "Mean tokens"], rows,
      note=r"Unfinished reasoning: no closing \texttt{</think>} was produced; the available text is scored. Tokens are continuation tokens for the crossed Qwen runs and generated tokens for the Phi standalone runs.")

# ----------------------------------------------------------------------------------------------- Appendix: prompts + token ids
inst = json.loads((REPO / "data" / "eval_prompts" / "haskins_task_instructions.json").read_text(encoding="utf-8"))
rows = [[R(r"\texttt{" + esc(k) + "}"), R(r"\emph{" + esc(v) + "}")] for k, v in inst["tasks"].items()]
write("a11_prompts", "Haskins constraint instructions exactly as used in the runs (``chain of thought'' as the CoT term).", "tab:app-haskins-prompts", "lp{11cm}", ["Task", "Instruction"], rows,
      note=r"Qwen system prompt: \emph{You are a helpful assistant.} followed by the instruction. Phi-4-reasoning system prompt: Microsoft's official reasoning prompt, then "
           r"\emph{Constraint Requirement:} and the instruction. \texttt{\{keyword\}} and \texttt{\{keywords\}} are filled per question from the calibrated keyword file; "
           r"the single-word task names one keyword, the multiple-word task lists the keyword and all its synonyms.", resize=False)

rows = []
for tag, label in (("BB", "Base donor (ON)"), ("SB", "SFT donor (ON)")):
    for i in (0, 1, 2):
        r = qdirs[tag][qdirs[tag].task == "third_person"].sort_values("prompt_idx").iloc[i]
        rows.append([label, R(r"\texttt{" + esc(r.injected_prefix_text.replace("\n", "\\n")) + "}"), R(r"\texttt{" + esc(r.injected_prefix_token_ids) + "}")])
write("a12_prefix_tokens", "Representative injected prefixes and Qwen3 token IDs (third-person task, questions 0--2).", "tab:app-prefix-tokens", "lp{5.5cm}p{5cm}",
      ["Condition", "Prefix text", "Token IDs"], rows, note=r"Every Haskins prefix has exactly ten token IDs; the first is \texttt{<think>} (151667).")

print("wrote", len(list(OUT.glob("*.tex"))), "tables to", OUT)
