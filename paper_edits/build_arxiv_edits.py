"""Text edits for the arXiv version of the paper (Overleaf project 6ad6b41f...), 2026-10-20.

De-anonymization (author block, code link, OLMo adapters), the corrected Phi-4-reasoning length claim, and the new teacher
results (OLMo-3-7B-Think s1K students; ReasonIF for the Qwen3-14B s1K students, with a ReasonIF column in the teacher
table). Every number is computed here from the result files; nothing is typed by hand. Writes
paper_edits/arxiv_edits.json: {"edits": [[old, new], ...], "teacher_rif": [[row label, value], ...]}; each old string
must occur exactly once in the document (checked when applied).

  python paper_edits/build_arxiv_edits.py
"""
import glob
import json
import re
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import analysis_followups as F  # noqa: E402
from seven_task_composite import SEVEN  # noqa: E402

BS = chr(92)


def a(s):
    return s.replace("|", BS)


def p2(est, lo, hi):
    return f"${est:+.2f}$, $[{lo:.2f}, {hi:.2f}]$"


def p1(est, lo, hi):
    return f"${est:+.1f}$, $[{lo:.1f}, {hi:.1f}]$"


def gain(b, d, tasks=SEVEN, window=None):
    n, _, _, g, lo, hi = F._paired_gain(b, d, None, col="primary", window=window, tasks=tasks)
    return g, lo, hi


# ---- CoT-Control runs -------------------------------------------------------------------------------------------
HR = F.haskins_runs()
QB = HR["Qwen3-14B Base"]
QC = F._prep(pd.read_csv(glob.glob(str(F.RES / "data_variants/s1k_claude37/haskins_500/qwen3_14b/*/results.csv"))[0]))
PB, PS = HR["Phi-4-reasoning Base"], HR["Phi-4-reasoning SFT"]
OB = HR["OLMo-3-7B-Think Base"]
OC = F._prep(pd.read_csv(glob.glob(str(F.RES / "data_variants/olmo3_s1k_claude37/haskins_500/olmo3_7b_think/*/results.csv"))[0]))
OR = F._prep(pd.read_csv(glob.glob(str(F.RES / "data_variants/olmo3_s1k_deepseek_r1/haskins_500/olmo3_7b_think/*/results.csv"))[0]))
CASE = ["lowercase_thinking", "uppercase_thinking", "alternating_case"]
ALL10 = sorted(QB.task.unique())
mean7 = lambda d: 100 * d[d.task.isin(SEVEN)].primary.mean()  # noqa: E731
q_gain = gain(QB, QC)
o_c_base, o_c_r1 = gain(OB, OC), gain(OR, OC)
o_case, o_ten = gain(OB, OC, tasks=CASE), gain(OB, OC, tasks=ALL10)
o_tp_b, o_tp_c = (100 * d[d.task == "third_person"].primary.mean() for d in (OB, OC))

# Phi length: share of seven-task SFT traces under 250 reasoning tokens, Base minimum, matched gains (length_overview.py)
phi_csv = {("SFT" if "gpt52" in p else "Base"): p for p in glob.glob(str(F.RES / "haskins_500/phi4_reasoning/extracted_runs/*/results.csv"))}
tok = {k: pd.read_csv(p, usecols=["task", "reasoning_tokens_reencoded"]) for k, p in phi_csv.items()}
tok = {k: v[v.task.isin(SEVEN)].reasoning_tokens_reencoded for k, v in tok.items()}
share_short = 100 * (tok["SFT"] < 250).mean()
assert (tok["Base"] < 250).sum() == 0
LO = (F.RES / "summary_tables/length_overview.txt").read_text(encoding="utf-8")


def matched(run):
    m = re.search(rf"^   {re.escape(run)}: unmatched gain [^;]+; length-matched within task ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\] \(n=(\d+)\)", LO, re.M)
    return float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))


ph_m, ph_c, ph_r = matched("SFT"), matched("SFT s1k claude37"), matched("SFT s1k deepseek r1")

# ---- ReasonIF (reasonif_s1k_summary.py) -------------------------------------------------------------------------
G = runpy.run_path(str(REPO / "scripts/reasonif_s1k_summary.py"), run_name="reasonif_s1k_summary")
D, T = G["D"], G["T"]
base, cl, r1 = D["Base"], D["s1K Claude 3.7"], D["s1K DeepSeek R1"]


def rdiff(x, y, col="ifs", keep=None):
    keep = np.ones(300, bool) if keep is None else keep
    v = (y[col] - x[col])[keep].to_numpy()
    lo, hi = F._ci(pd.Series(v))
    return 100 * v.mean(), 100 * lo, 100 * hi


over = base.i.isin(G["OVER"]).to_numpy()
nw = G["nw"]
fair = ~(T["Base"] | T["s1K DeepSeek R1"])
pct = lambda d, c="ifs": 100 * d[c].mean()  # noqa: E731
r_cb, r_cr = rdiff(base, cl), rdiff(r1, cl)
r_cr_over, r_cr_nw, r_cr_fair = rdiff(r1, cl, keep=~over), rdiff(r1, cl, keep=nw), rdiff(r1, cl, keep=fair)
hard_acc = 100 * base.acc[T["s1K Claude 3.7"]].mean()
assert int(over.sum()) == 24

E = []
E.append([a("On another 212 questions, raw reasoning from Claude 3.7 Sonnet raises CoT-Control compliance by 7.43 points, while reasoning from DeepSeek R1 does not. In Phi-4-reasoning the gain over the base model is explained by shorter outputs, and OLMo-3-7B-Think shows none."),
          a(f"On another 212 questions, raw reasoning from Claude 3.7 Sonnet raises CoT-Control compliance by {q_gain[0]:.2f} points and ReasonIF instruction-following success from {pct(base):.1f}|% to {pct(cl):.1f}|%, while reasoning from DeepSeek R1 does neither. In Phi-4-reasoning, compliance also rises ({mean7(PB):.2f} to {mean7(PS):.2f}), but its outputs become so much shorter than the base model's that the gain cannot be separated from length; OLMo-3-7B-Think shows no seven-task gain.")])
E.append([a("In Phi-4-reasoning the gain is explained by shorter outputs;"), a("In Phi-4-reasoning the gain cannot be separated from much shorter outputs;")])
E.append([a("and reasoning from DeepSeek R1, among the least, does not; summaries are not required."),
          a(f"and ReasonIF success by {r_cb[0]:.1f} points, and reasoning from DeepSeek R1, among the least, raises neither; summaries are not required. On OLMo-3-7B-Think the difference does not hold.")])
E.append([a("but on the first 100 words no gain remains ($-0.50$, $[-4.56, 4.60]$; Section~|ref{sec:length}), so the gain is explained by shorter outputs."),
          a(f"but on the first 100 words no gain remains ($-0.50$, $[-4.56, 4.60]$; Section~|ref{{sec:length}}), and {share_short:.0f}|% of the fine-tuned model's seven-task traces are shorter than 250 reasoning tokens, against none of the base model's, so only {ph_m[3]} traces can be matched on length within a task (gain {p1(*ph_m[:3])}): the gain cannot be separated from the shorter outputs.")])
E.append([a("ReasonIF is not evaluated, because the two question sets overlap: 4 of the 300 ReasonIF questions appear verbatim among the s1K questions (1 among the 212 used) and 37 share an 8-gram with one (24 with the 212), mostly AIME and AMC problems."),
          a("The question sets overlap on ReasonIF: 4 of the 300 ReasonIF questions appear verbatim among the s1K questions (1 among the 212 used) and 37 share an 8-gram with one (24 with the 212), mostly AIME and AMC problems, so we also report ReasonIF without those 24 questions.")])
E.append([a("the difference disappears on the first 100 words ($+1.23$, $[-2.63, 3.40]$). All of these are single runs."),
          a(f"the difference disappears on the first 100 words ($+1.23$, $[-2.63, 3.40]$), although both students' gains over Base remain when traces are matched on length within each task (${ph_c[0]:+.1f}$ and ${ph_r[0]:+.1f}$ points). "
            f"On OLMo-3-7B-Think the result does not hold on the seven tasks: the Claude 3.7 student is {p2(*o_c_base)} against Base and {p2(*o_c_r1)} against the R1 student, because its third-person compliance falls ({o_tp_b:.1f}|% to {o_tp_c:.1f}|%); its gain is on the three case tasks (${o_case[0]:+.1f}$ points; ten tasks ${o_ten[0]:+.2f}$). All of these are single runs.\n\n"
            f"|paragraph{{ReasonIF.}} On ReasonIF the Qwen3-14B teacher difference is larger than on CoT-Control (Table~|ref{{tab:teacher}}, last column). The Claude 3.7 student's instruction-following success rises from {pct(base):.1f}|% to {pct(cl):.1f}|% ({p1(*r_cb)}), the R1 student's does not ({pct(r1):.1f}|%), and the paired difference between them is {p1(*r_cr)} "
            f"(${r_cr_over[0]:+.1f}$ without the 24 overlapping questions; ${r_cr_nw[0]:+.1f}$ without the word-limit constraint, which shorter reasoning passes more easily). "
            f"Its accuracy falls further ({pct(base, 'acc'):.1f}|% to {pct(cl, 'acc'):.1f}|%, against {pct(r1, 'acc'):.1f}|% for the R1 student), mostly because {int(T['s1K Claude 3.7'].sum())} of its 300 responses reach the 16,384-token limit without an answer ({int(T['s1K DeepSeek R1'].sum())} for the R1 student, {int(T['Base'].sum())} for Base), mainly on harder questions (Base accuracy {hard_acc:.1f}|% on them). "
            f"On the {int(fair.sum())} questions on which Base and the R1 student both finish, a selection that does not depend on the Claude 3.7 student's outputs, the IFS difference is {p1(*r_cr_fair)}.")])
E.append([a("The effect is model-dependent: it is explained by shorter outputs in Phi-4-reasoning and absent in OLMo-3-7B-Think."),
          a("The effect is model-dependent: in Phi-4-reasoning it cannot be separated from much shorter outputs, and in OLMo-3-7B-Think it is absent on the seven tasks.")])
E.append([a("raw reasoning from Claude 3.7 Sonnet raises controllability and reasoning from DeepSeek R1 does not."),
          a("raw reasoning from Claude 3.7 Sonnet raises controllability on both benchmarks and reasoning from DeepSeek R1 does not.")])
E.append([a("its difference survives the length check only on Qwen3-14B (on Phi-4-reasoning it is explained by shorter outputs)"),
          a("its difference holds only on Qwen3-14B (on Phi-4-reasoning it disappears on the first 100 words, and on OLMo-3-7B-Think it appears only on the case tasks)")])
E.append([a("adapters |texttt{[anonymized]} and its |texttt{-lr2e-4} variant)"),
          a("adapters |texttt{kyleliu789/olmo3-7b-think-gpt52-high-reasoning-original} (commit |texttt{0984a2a}) and its |texttt{-lr2e-4} variant (commit |texttt{2bd074f}))")])
E.append([a("|url{https://anonymous.4open.science/r/Benign_CoT_Control-F3DF/}"), a("|url{https://github.com/Kyleliu7/Benign_CoT_Control} (commit |texttt{3f7114e})")])
E.append([a("|author{Kyle Liu}"), a("|author{Kyle Liu||Independent||[2pt]|texttt{kyleliu789@gmail.com}}")])
E.append([a("CoT-Control on the seven non-character tasks; gains over Base.}|label{tab:teacher}"),
          a("CoT-Control on the seven non-character tasks; gains over Base. Last column: ReasonIF instruction-following success on the 300 ReasonIF questions.}|label{tab:teacher}")])
E.append([a("|begin{tabular}{lllrllllrr}"), a("|begin{tabular}{lllrllllrrr}")])
E.append([a(" & Median words & Hit cap ||"), a(" & Median words & Hit cap & ReasonIF IFS (|%) ||")])
ROWS = [["Base (no fine-tuning)", "Base"], ["Claude 3.7 Sonnet reasoning", "s1K Claude 3.7"], ["DeepSeek R1 reasoning", "s1K DeepSeek R1"],
        ["Claude Sonnet 4.5 reasoning", "Sonnet 4.5 teacher (GPT-5.2 prompts)"], ["Qwen3-14B's own reasoning", "Self-distillation"],
        ["GPT-5.2 reasoning summaries (reported adapter)", "SFT GPT-5.2 (seed 42)"]]
out = {"edits": E, "teacher_rif": [[lab, f"{pct(D[k]):.1f}"] for lab, k in ROWS]}
(REPO / "paper_edits/arxiv_edits.json").write_text(json.dumps(out, ensure_ascii=True, indent=1), encoding="utf-8")
for o, n in E:
    print("-", n[:400].replace("\n", " | "))
print(out["teacher_rif"])
