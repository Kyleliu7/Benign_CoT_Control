"""Main results table for the three fine-tuned models (Base vs SFT on ReasonIF and CoT-Control), from raw records.

CoT-Control gains are paired SFT - Base differences with 95% cluster-bootstrap intervals over the 50 questions
(analysis_followups._paired_gain): full trace, first 100 words (pairs where both traces reach 100 words), and with
degenerate traces scored 0. ReasonIF values come from the summary tables built from the raw records.

  python scripts/make_models_table.py [OUT_TEX]   (default paper_tables/t15_models.tex)
"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_followups as F  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
T = REPO / "results" / "summary_tables"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "paper_tables" / "t15_models.tex"


def truthy(v):
    return str(v).strip().lower() == "true"


H = F.haskins_runs()
for d in H.values():
    d["clean"] = d.primary.where(~d.degen, 0.0)

seeds = pd.read_csv(T / "table26_training_seeds.csv").set_index("Condition")
t1 = pd.read_csv(T / "table1_reasonif_overall.csv")
olmo = pd.read_csv(T / "table28_olmo_summary.csv").set_index("Run")


def reasonif_olmo_lr2():
    f = sorted(glob.glob(str(REPO / "results" / "olmo3_7b_think" / "reasonif_300_sft_lr2e-4" / "*" / "scored_responses.jsonl")))
    if not f:
        return None, None
    r = [json.loads(l) for l in Path(f[0]).read_text(encoding="utf-8").split(chr(10)) if l.strip()]
    return 100 * sum(truthy(x["instruction_following"]) for x in r) / len(r), 100 * sum(truthy(x["answer_correct"]) for x in r) / len(r)


def phi_rif(cond):
    r = t1[(t1.Model == "Phi-4-reasoning") & (t1.Condition == cond)].iloc[0]
    return r.IFS_Pct, r.Acc_Pct


o2_ifs, o2_acc = reasonif_olmo_lr2()
rows = [  # (model, run label, haskins label, base haskins label, ReasonIF IFS, ReasonIF acc)
    ("Qwen3-14B", "Base", "Qwen3-14B Base", None, seeds.loc["Base", "ReasonIF_IFS_Pct"], seeds.loc["Base", "ReasonIF_Accuracy_Pct"]),
    ("", "SFT (seed 42)", "Qwen3-14B SFT seed 42", "Qwen3-14B Base", seeds.loc["SFT seed 42", "ReasonIF_IFS_Pct"], seeds.loc["SFT seed 42", "ReasonIF_Accuracy_Pct"]),
    ("", "SFT (seed 1)", "Qwen3-14B SFT seed 1", "Qwen3-14B Base", seeds.loc["SFT seed 1", "ReasonIF_IFS_Pct"], seeds.loc["SFT seed 1", "ReasonIF_Accuracy_Pct"]),
    ("", "SFT (seed 2)", "Qwen3-14B SFT seed 2", "Qwen3-14B Base", seeds.loc["SFT seed 2", "ReasonIF_IFS_Pct"], seeds.loc["SFT seed 2", "ReasonIF_Accuracy_Pct"]),
    ("Phi-4-reasoning", "Base", "Phi-4-reasoning Base", None, *phi_rif("Base (Untouched)")),
    ("", "SFT", "Phi-4-reasoning SFT", "Phi-4-reasoning Base", *phi_rif("SFT (gpt52-high)")),
    ("OLMo-3-7B-Think", "Base", "OLMo-3-7B-Think Base", None, 100 * olmo.loc["Base", "ReasonIF_IFS"] / 300, 100 * olmo.loc["Base", "ReasonIF_Accuracy"] / 300),
    ("", "SFT", "OLMo-3-7B-Think SFT lr1e-4", "OLMo-3-7B-Think Base", 100 * olmo.loc["SFT lr1e-4", "ReasonIF_IFS"] / 300, 100 * olmo.loc["SFT lr1e-4", "ReasonIF_Accuracy"] / 300),
    ("", "SFT, lr $2\\times10^{-4}$", "OLMo-3-7B-Think SFT lr2e-4", "OLMo-3-7B-Think Base", o2_ifs, o2_acc),
]


def g(base, sft, **kw):
    n, mb, ms, d, lo, hi = F._paired_gain(H[base], H[sft], None, **kw)
    return f"{d:+.2f} [{lo:.2f}, {hi:.2f}]".replace("-", "$-$")


lines = []
for model, run, hl, bl, ifs, acc in rows:
    d = H[hl]
    cells = [model, run, "--" if ifs is None else f"{ifs:.1f}", "--" if acc is None else f"{acc:.1f}",
             f"{100 * d.primary.mean():.2f}", f"{int((d.upstream == 1.0).sum())}"]
    if bl is None:
        cells += ["--", "--", "--"]
    else:
        cells += [g(bl, hl), g(bl, hl, window=100), g(bl, hl, col="clean")]
    lines.append(" & ".join(cells) + " \\\\")
    if model == "" and run.startswith("SFT (seed 2)") or hl == "Phi-4-reasoning SFT":
        lines.append("\\midrule")

tex = "\n".join([
    "\\begin{table}[htbp]",
    "\\centering\\small",
    "\\caption{Base and SFT models on ReasonIF (300 questions) and CoT-Control (500 question--constraint pairs). Gains are SFT minus Base in CoT-Control mean compliance.}\\label{tab:models}",
    "\\setlength{\\tabcolsep}{4pt}",
    "\\begin{adjustbox}{max width=\\linewidth}",
    "\\begin{tabular}{llrrrrlll}",
    "\\toprule",
    " & & \\multicolumn{2}{c}{ReasonIF (\\%)} & \\multicolumn{5}{c}{CoT-Control} \\\\",
    "\\cmidrule(lr){3-4}\\cmidrule(lr){5-9}",
    "Model & Run & IFS & Accuracy & Mean & Strict & Gain [95\\% CI] & Gain, first 100 words & Gain, degenerate $=0$ \\\\",
    "\\midrule",
    *lines,
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{adjustbox}",
    "\\par\\smallskip\\begin{minipage}{0.97\\linewidth}\\footnotesize Seed 42 is the reported Qwen adapter; seeds 1 and 2 change only the training seed. "
    "OLMo SFT uses the Qwen recipe; the learning rate $2\\times10^{-4}$ run was added after the first result. "
    "Gains are paired differences with 95\\% cluster-bootstrap intervals over the 50 questions; the first-100-words column keeps only pairs in which both traces reach 100 words; "
    "the last column scores degenerate traces (fewer than 20 words of reasoning, or more than 90\\% ``meow'') as 0. Strict: pairs with a compliance score of exactly 1, out of 500.\\end{minipage}",
    "\\end{table}",
    "",
])
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(tex, encoding="utf-8")
print(tex)
