"""Overleaf edits for the seven-task figures, the updated opening analysis and the teacher-test setup note
(paper_edits/figures_payload.json). Same conventions as build_payload.py ('|' = backslash, '~N~' = newline in anchors).

  python paper_edits/build_payload_figures.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = dict(re.findall(r"%%%% (\w+)\n(.*?)(?=\n%%%% |\Z)", (ROOT / "paper_edits/seven_task_text.tex").read_text(encoding="utf-8"), re.S))
T = {k: v.rstrip("\n") for k, v in T.items()}
BS = chr(92)


def a(s):
    return s.replace("~N~", "\n").replace("|", BS)


E = [
    {"start": a("figures/figure0_teaser.pdf}"), "end": a("~N~"), "text": a("figures/figure0_teaser7.pdf}")},
    {"start": a("and CoT-Control (ten-task mean compliance, 500 question--constraint pairs; the seven-task composite is in Table~|ref{tab:models})"),
     "end": a(". The Qwen SFT bar"), "text": a("and CoT-Control (mean compliance on the seven non-character tasks, 350 question--constraint pairs)")},
    {"start": a("figures/figure1_cotcontrol_crossed.pdf}"), "end": a("~N~|caption"), "text": a("figures/figure1_cotcontrol_crossed7.pdf}")},
    {"start": a("{CoT-Control continuation compliance and strict pass rates under crossed donor--recipient conditions (500 question--constraint pairs per condition)."),
     "end": a(" Counts above strict bars"),
     "text": a("{CoT-Control continuation compliance and strict pass rates under crossed donor--recipient conditions, on the seven non-character tasks (350 question--task pairs per condition).")},
    {"start": a("restricted to the 455 identical-prefix pairs it is essentially unchanged"), "end": a(" (29.15"),
     "text": a("over all ten tasks and restricted to the 455 identical-prefix pairs it is essentially unchanged")},
    {"start": a("(ten tasks: 16.11|%, 15.76|%, 29.51|% and 25.26|%; Figure~|ref{fig:haskins-crossed})"), "end": a(". Changing the donor"),
     "text": a("(Figure~|ref{fig:haskins-crossed}; ten tasks: 16.11|%, 15.76|%, 29.51|% and 25.26|%)")},
    {"start": a("Across all twelve Qwen3-14B runs"), "end": a(" Claude Sonnet 4.5's reasoning opens with a short"), "text": T["OPEN"]},
    {"start": a("On the case tasks, compliance across twelve Qwen3-14B runs tracks"), "end": a(" how often the model"), "text": T["INTROOPEN"]},
    {"before": a("~N~~N~|paragraph{Training data.}"), "text": T["APPRUNS"]},
]
out = ROOT / "paper_edits/figures_payload.json"
out.write_text(json.dumps(E, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out, len(E))
