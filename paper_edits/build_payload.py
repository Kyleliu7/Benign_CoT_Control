"""Build the list of Overleaf edits for the seven-task revision (paper_edits/seven_task_payload.json) from the rendered text
(paper_edits/seven_task_text.tex) and the generated tables (paper_tables/). Each edit replaces the text from the start of
`start` up to (not including) `end`, or inserts `text` before `before`; anchors must be unique in the document.
Anchors are written with "|" for a backslash and "~N~" for a newline to keep them readable.

  python paper_edits/build_payload.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = dict(re.findall(r"%%%% (\w+)\n(.*?)(?=\n%%%% |\Z)", (ROOT / "paper_edits/seven_task_text.tex").read_text(encoding="utf-8"), re.S))
T = {k: v.rstrip("\n") for k, v in T.items()}
BS = chr(92)


def a(s):
    """Anchor: '|' stands for a backslash and '~N~' for a newline."""
    return s.replace("~N~", "\n").replace("|", BS)


def tab(name):
    return (ROOT / "paper_tables" / f"{name}.tex").read_text(encoding="utf-8").rstrip("\n")


TABLE = "|begin{table*}[t]~N~|centering|small~N~|caption{"
E = [
    {"start": a("CoT-Control mean compliance rises from 15.50 to 24.21--26.79."), "end": a(" The gain costs ReasonIF accuracy"), "text": T["INTRO1"]},
    {"start": a("In Phi-4-reasoning the gain is mostly explained by shorter outputs; in OLMo-3-7B-Think it is absent under the same recipe once degenerate outputs are removed, and at a higher learning rate"),
     "end": a(" its ReasonIF gain (8.3"), "text": T["INTRO2"]},
    {"start": a("|item |textbf{The opening of the reasoning carries the effect}"), "end": a("|item |textbf{In Qwen, ten tokens carry much of the effect}"), "text": T["INTRO3"] + "\n"},
    {"start": a("An opening written by the SFT model raises the unchanged base model's continuation compliance from 16.11|% to 29.51|%"), "end": a("~N~|end{enumerate}"), "text": T["INTRO4"]},
    {"start": a("and CoT-Control (mean compliance, 500 question--constraint pairs)"), "end": a(". The Qwen SFT bar is the reported adapter"),
     "text": a("and CoT-Control (ten-task mean compliance, 500 question--constraint pairs; the seven-task composite is in Table~|ref{tab:models})")},
    {"start": a("We report mean compliance, graded binary for four tasks as in the original paper, and strict passes."), "end": a(" Intervals are paired 95|% bootstrap intervals over questions"), "text": T["EVAL"]},
    {"start": a("CoT-Control mean compliance rises from 15.50 to 24.21, 25.95 and 26.79"), "end": a(" ReasonIF accuracy falls from 80.3"), "text": T["QWEN1"]},
    {"start": a("Most of the CoT-Control gain is on third person, uppercase, lowercase"), "end": a("~N~~N~" + TABLE + "Base and SFT models on ReasonIF"), "text": T["QWEN2"]},
    {"start": a(TABLE + "Base and SFT models on ReasonIF"), "end": a("~N~~N~|subsection{Other Models}"), "text": tab("table_models7")},
    {"start": a("|paragraph{Phi-4-reasoning.} SFT raises ReasonIF IFS from 5.0"), "end": a("~N~~N~|paragraph{OLMo-3-7B-Think.}"), "text": T["PHI"]},
    {"start": a("|paragraph{OLMo-3-7B-Think.} With the Qwen recipe"), "end": a(" however, this run reaches 20.3"), "text": T["OLMO"]},
    {"start": a("For Qwen the gains are $+8.60$"), "end": a(" Because this selects pairs on a length"), "text": T["LENGTH"]},
    {"start": a("Length therefore explains only a small part of the Qwen gain."), "end": a("~N~~N~|subsection{Is the Gain Degenerate Output?}"), "text": T["LENGTH2"]},
    {"before": a(" On ReasonIF, excluding passes with no or incoherent reasoning"), "text": T["DEGEN"]},
    {"start": a("The CoT-Control grader treats a heading line as a sentence"), "end": a(" Under a constraint, the transplanted model writes"), "text": T["HEAD1"]},
    {"before": a(" The heading is also what fine-tuning learns most"), "text": T["HEAD2"]},
    {"start": a(TABLE + "Heading experiments"), "end": a("~N~~N~|subsection{Do the First Tokens Carry the Effect?}"),
     "text": tab("table_headings7") + "\n\n" + T["TEACHER"] + "\n\n" + tab("table_teacher")},
    {"start": a("With base-derived openings, Qwen's continuation compliance is 16.11"), "end": a(" On ReasonIF, Prefix-ON (SFT donor, base recipient) reaches 42.0"), "text": T["CROSS1"]},
    {"start": a("Prefix-OFF, in which the donor is not given the constraint and its opening names no rule, still reaches 32.0"),
     "end": a(" although this advantage disappears at equal word windows (Appendix Table~|ref{tab:window}), and Phi's SFT donor names"), "text": T["CROSS2"]},
    {"start": a("The transferred ingredient is therefore mainly constraint-specific text at the start of the reasoning, which the SFT donor writes more often, not the SFT model's style."),
     "end": a(" Naming the rule recovers the mean"), "text": T["CTRL1"]},
    {"before": a("~N~~N~" + TABLE + "Qwen3-14B base recipient continuing from a fixed ten-token opening"), "text": "\n\n" + T["CTRL2"]},
    {"start": a(TABLE + "Qwen3-14B base recipient continuing from a fixed ten-token opening"), "end": a("~N~~N~|section{Related Work}"), "text": tab("table_controls7")},
    {"start": a("The ablations narrow down why."), "end": a(" ``Benign'' describes the intended supervision"), "text": T["DISC"]},
    {"before": a("~N~~N~|section*{Use of AI Assistants}"), "text": T["LIMIT"]},
    {"before": a(" The numbers in this paper are regenerated from the raw records"), "text": T["AIUSE"]},
    {"before": a(TABLE + "Qwen3-14B SFT with three training seeds"), "text": tab("table_seven_all") + "\n\n"},
]
out = ROOT / "paper_edits/seven_task_payload.json"
out.write_text(json.dumps(E, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out, len(E), "edits", sum(len(e["text"]) for e in E), "chars")
