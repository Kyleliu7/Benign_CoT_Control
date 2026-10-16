"""LaTeX tables added after the October 2026 review (same conventions as make_paper_tables.py: no number typed by hand).

  tab:headings        heading experiments: Base, SFT (seed 42), no headings, fixed heading, Qwen3-235B reasoning and
                      Qwen3-235B reasoning with the GPT-5.2 opening heading (rows appear when their records exist)
  tab:standalone-task standalone CoT-Control per task for Qwen3-14B (Base, seeds 42/1/2) and Phi-4-reasoning
  tab:variants-cot    CoT-Control for the Table 2 adapters (rows appear when their records exist)
  tab:mechanism       what the SFT adapters learned (per-segment surprisal) and how often they open with a heading

  python scripts/make_review_tables.py [OUT_DIR]      (default: paper_tables/)
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_followups as F  # noqa: E402
import analysis_prefix_length_paired as P  # noqa: E402
from haskins_header_stripped import rescore  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "paper_tables"
OUT.mkdir(parents=True, exist_ok=True)
LEAD = re.compile(r"^\s*\*\*[^*\n]{1,120}\*\*")
NAMED = {t: P.NAMES_CONSTRAINT[t] for t in ("lowercase_thinking", "uppercase_thinking", "third_person")}


def tbl(name, caption, label, colspec, header, rows, note, star=True):
    env = "table*" if star else "table"
    lines = [rf"\begin{{{env}}}[t]", r"\centering\small", rf"\caption{{{caption}}}\label{{{label}}}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{adjustbox}{max width=\linewidth}", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(str(c) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
              rf"\par\smallskip\begin{{minipage}}{{0.97\linewidth}}\footnotesize {note}\end{{minipage}}", rf"\end{{{env}}}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT / f"{name}.tex")


def sgn(x, nd=2):
    return f"{x:+.{nd}f}".replace("-", "$-$")


def cell_ci(est, lo, hi, nd=2):
    return f"{sgn(est, nd)} [{sgn(lo, nd)}, {sgn(hi, nd)}]".replace("[+", "[").replace(", +", ", ")


def first(pattern):
    hits = sorted(glob.glob(str(pattern)))
    return hits[0] if hits else None


# ---------------------------------------------------------------- data
H = {k: v for k, v in F.haskins_runs().items()}
R = {k: v[1] for k, v in F.reasonif_runs().items()}
ROWS = [("Base (no fine-tuning)", "Qwen3-14B Base", "Qwen3-14B Base"),
        ("GPT-5.2 summaries (reported adapter)", "Qwen3-14B SFT seed 42", "Qwen3-14B SFT seed 42")]
for v, label in (("no_headers", "GPT-5.2 summaries, headings removed"), ("neutral_header", r"GPT-5.2 summaries, one fixed \texttt{**Reasoning**} heading")):
    h, r = first(RES / "data_variants" / v / "haskins_500/qwen3_14b/*/results.csv"), first(RES / "data_variants" / v / "reasonif_300/qwen3_14b/*/scored_responses.jsonl")
    if h and r:
        H[v], R[v] = F._prep(pd.read_csv(h)), F._jl(r)
        ROWS.append((label, v, v))
t235_h = first(RES / "haskins_500/qwen3_14b_training_variants/teacher_qwen3_235b/haskins_500/qwen3_14b/*/results.csv")
t235_r = first(RES / "reasonif_300/qwen3_14b_training_variants/qwen3_14b_qwen3_235b_high_reasoning_original/scored_responses.jsonl")
if t235_r:
    R["t235"] = F._jl(t235_r)
    if t235_h:
        H["t235"] = F._prep(pd.read_csv(t235_h))
    ROWS.append(("Qwen3-235B-A22B reasoning (same prompts)", "t235" if t235_h else None, "t235"))
h, r = first(RES / "data_variants/header_transplant/haskins_500/qwen3_14b/*/results.csv"), first(RES / "data_variants/header_transplant/reasonif_300/qwen3_14b/*/scored_responses.jsonl")
if h and r:
    H["transplant"], R["transplant"] = F._prep(pd.read_csv(h)), F._jl(r)
    ROWS.append(("Qwen3-235B-A22B reasoning with the GPT-5.2 opening heading", "transplant", "transplant"))

HS = {k: rescore(v) for k, v in H.items()}
base_h = HS["Qwen3-14B Base"]
base_r = {r["dataset_index"]: r for r in R["Qwen3-14B Base"]}

rows = []
for label, hk, rk in ROWS:
    row = [label]
    if hk:
        d = HS[hk]
        row.append(f"{int(d.reasoning.map(lambda x: bool(LEAD.match(str(x)))).sum())}")
        row.append("/".join(str(sum(bool(re.search(pat, str(x)[:300], re.I)) for x in d[d.task == t].reasoning)) for t, pat in NAMED.items()))
        row.append(f"{100 * d.primary_s.mean():.2f}")
        if hk == "Qwen3-14B Base":
            row.append("--")
        else:
            m = base_h.merge(d, on=["task", "q"], suffixes=("_b", "_m"))
            diff = (m.primary_s_m - m.primary_s_b).to_numpy()
            lo, hi = F._ci(pd.Series(diff).groupby(m.q.values).mean())
            row.append(cell_ci(100 * diff.mean(), 100 * lo, 100 * hi))
    else:
        row += ["", "", "", ""]
    recs = R[rk]
    ifs = np.array([int(bool(r["instruction_following"])) for r in recs])
    row.append(f"{100 * ifs.mean():.1f}")
    if rk == "Qwen3-14B Base":
        row.append("--")
    else:
        dd = np.array([int(bool(r["instruction_following"])) - int(bool(base_r[r["dataset_index"]]["instruction_following"])) for r in recs])
        lo, hi = F._ci(dd)
        row.append(cell_ci(100 * dd.mean(), 100 * lo, 100 * hi, 1))
    row.append(f"{100 * np.mean([bool(r['answer_correct']) for r in recs]):.1f}")
    rows.append(row)
tbl("table_headings", "Heading experiments (Qwen3-14B, one training run each except the reported adapter). CoT-Control is graded with a leading bold heading removed from every trace.",
    "tab:headings", "lrrrlrlr",
    ["Training data", "Opens with heading", "Names or shows rule early (lower/upper/third)", "CoT-Control mean", "Gain vs Base (95\\% CI)", "ReasonIF IFS (\\%)", "Gain vs Base (95\\% CI)", "Accuracy (\\%)"],
    rows, "Opens with heading: CoT-Control traces (of 500) whose reasoning starts with a bold heading. Names or shows rule early: traces whose first 300 characters "
          "name or demonstrate the constraint (the paper's rule, e.g. \\texttt{the assistant} for third person), out of 50 each for lowercase, uppercase and third person. CoT-Control gains are paired differences with cluster-bootstrap "
          "intervals over the 50 questions; ReasonIF gains with bootstrap intervals over the 300 questions. Same recipe for every adapter (LoRA $r=32$, "
          "learning rate $10^{-4}$, 3 epochs, seed 42, lowest-validation-loss checkpoint).")

# ---------------------------------------------------------------- standalone per task
cols = [("Qwen Base", "Qwen3-14B Base"), ("Qwen SFT 42", "Qwen3-14B SFT seed 42"), ("Qwen SFT 1", "Qwen3-14B SFT seed 1"),
        ("Qwen SFT 2", "Qwen3-14B SFT seed 2"), ("Phi Base", "Phi-4-reasoning Base"), ("Phi SFT", "Phi-4-reasoning SFT")]
names = {"third_person": "Third person", "arrow_prefix": "Arrow prefix", "word_suppression": "Word suppression", "multiple_word_suppression": "Multiple-word suppression",
         "end_of_sentence": "End of sentence", "repeat_sentences": "Repeat sentences", "meow_between_words": "Meow between words",
         "alternating_case": "Alternating case", "lowercase_thinking": "Lowercase", "uppercase_thinking": "Uppercase"}
rows = []
for task, nm in names.items():
    row = [nm]
    for _, k in cols:
        g = H[k][H[k].task == task]
        row.append(f"{100 * g.primary.mean():.1f} / {int(g.strict.sum())}")
    rows.append(row)
tbl("table_standalone_task", "Standalone CoT-Control per task (mean compliance in percent / strict passes of 50).", "tab:standalone-task",
    "l" + "r" * len(cols), ["Task"] + [c for c, _ in cols], rows,
    "Graded as in the main tables (binary for word suppression, multiple-word suppression, end of sentence and meow, where the mean counts a "
    "compliance score of at least 0.99 as a pass; strict requires a score of exactly 1, so a binary-task cell can show a mean above zero with no strict pass).")

# ---------------------------------------------------------------- Table 2 adapters on CoT-Control
lab = {"output_mask": "Same data, loss on reasoning only (answer masked)", "think_mask": "Same data, loss on answer only (reasoning masked)",
       "teacher_qwen3_235b": "Same prompts, Qwen3-235B-A22B reasoning", "teacher_sonnet_3_7": "Same prompts, Claude 3.7 Sonnet reasoning",
       "teacher_reasonflux": "Same prompts, ReasonFlux reasoning"}
rows = [["Base (no fine-tuning)", f"{100 * base_h.primary_s.mean():.2f}", f"{100 * H['Qwen3-14B Base'].primary.mean():.2f}", "--", "0"],
        ["Reported adapter (GPT-5.2 summaries)"] + [""] * 4]
d = HS["Qwen3-14B SFT seed 42"]
m = base_h.merge(d, on=["task", "q"], suffixes=("_b", "_m"))
diff = (m.primary_s_m - m.primary_s_b).to_numpy(); lo, hi = F._ci(pd.Series(diff).groupby(m.q.values).mean())
rows[1][1:] = [f"{100 * d.primary_s.mean():.2f}", f"{100 * d.primary.mean():.2f}", cell_ci(100 * diff.mean(), 100 * lo, 100 * hi),
               str(int(d.reasoning.map(lambda x: bool(LEAD.match(str(x)))).sum()))]
for v, nm in lab.items():
    f = first(RES / f"haskins_500/qwen3_14b_training_variants/{v}/haskins_500/qwen3_14b/*/results.csv")
    if not f:
        continue
    d = rescore(F._prep(pd.read_csv(f)))
    m = base_h.merge(d, on=["task", "q"], suffixes=("_b", "_m"))
    diff = (m.primary_s_m - m.primary_s_b).to_numpy(); lo, hi = F._ci(pd.Series(diff).groupby(m.q.values).mean())
    rows.append([nm, f"{100 * d.primary_s.mean():.2f}", f"{100 * d.primary.mean():.2f}", cell_ci(100 * diff.mean(), 100 * lo, 100 * hi),
                 str(int(d.reasoning.map(lambda x: bool(LEAD.match(str(x)))).sum()))])
if len(rows) > 2:
    tbl("table_variants_cot", "The Table~\\ref{tab:variants} adapters on CoT-Control (500 question--constraint pairs; one training run each).", "tab:variants-cot",
        "lrrlr", ["Training data", "Mean, heading removed", "Mean as graded", "Gain vs Base, heading removed (95\\% CI)", "Opens with heading"], rows,
        "Paired differences with cluster-bootstrap intervals over the 50 questions. ``Heading removed'' grades each trace with a leading bold heading deleted.")

# ---------------------------------------------------------------- mechanism summary (surprisal and first-token heading probability)
mrows = []
fam = [("Qwen3-14B", "surprisal_qwen3_14b.json", "sft42", "first_token_qwen3_14b.json", "sft42"),
       ("OLMo-3-7B-Think, lr $10^{-4}$", "surprisal_olmo3_7b_think.json", "lr1e-4", "first_token_olmo3_7b_think.json", "lr1e-4"),
       ("OLMo-3-7B-Think, lr $2\\times10^{-4}$", "surprisal_olmo3_7b_think.json", "lr2e-4", "first_token_olmo3_7b_think.json", "lr2e-4"),
       ("Phi-4-reasoning", "surprisal_phi4_reasoning.json", "sft", "first_token_phi4_reasoning.json", "sft"),
       ("Qwen3.5-9B", "surprisal_qwen3.5_9b.json", "sft", "first_token_qwen3.5_9b.json", "sft")]
for nm, sf, sl, ff, fl in fam:
    sp, fp = RES / "mechanism" / sf, RES / "mechanism" / ff
    if not (sp.exists() and fp.exists()):
        continue
    s = pd.DataFrame(json.loads(sp.read_text())["rows"])
    s = s[s.split == "val"]
    cells = [nm]
    for seg in ("lead_header", "first_sentence", "body"):
        n = s[f"base__{seg}__n"].to_numpy(); k = n > 0
        red = (-s[f"base__{seg}__sum"].to_numpy()[k] + s[f"{sl}__{seg}__sum"].to_numpy()[k]) / n[k]
        cells.append(f"{red.mean():.2f}")
    fr = json.loads(fp.read_text())["rows"]
    cells.append(f"{np.mean([r['base__p_star'] for r in fr]):.3f}")
    cells.append(f"{np.mean([r[f'{fl}__p_star'] for r in fr]):.3f}")
    mrows.append(cells)
if mrows:
    tbl("table_mechanism", "What fine-tuning learned from the training data, and whether the model starts its reasoning with a heading.", "tab:mechanism",
        "lrrrrr", ["Model", "Heading", "First sentence", "Body", "P(heading) base", "P(heading) SFT"], mrows,
        "Columns 2--4: reduction in surprisal (nats per token) of each part of the 22 held-out training targets after SFT, teacher-forced. "
        "P(heading): probability that the first reasoning token, right after \\texttt{<think>}, starts a bold heading, averaged over the 500 CoT-Control prompts.",
        star=False)
