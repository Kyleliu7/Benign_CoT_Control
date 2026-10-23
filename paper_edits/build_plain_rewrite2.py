r"""Second pass on the plain main text (Kyle: "can we add the teacher in here too?" on the "Is the Gain Real?" section).

- The teacher test moves up into Results, right after Qwen3-14B ("Raw Reasoning Traces: Does the Teacher Matter?").
- "Is the Gain Real?" covers both training sets: the Claude 3.7 student's first-100-words and degenerate=0 results, its
  ReasonIF result without the word-limit constraint, and a new "Reasoning Loops" subsection (label sec:loops).
- The teacher section's "Other models and teachers" paragraph is split: Phi and OLMo teacher results go into their model
  paragraphs, the Sonnet 4.5 point into the training-data section.

Input: the text applied in the first pass (rewrite/plain_payload.json "new"); every edit is an exact, unique replacement.
Output: rewrite/plain_payload2.json {"old_sha256", "start", "end", "new"}; numbers checked against the snapshot and loop JSON.
"""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
P1 = json.loads((HERE / "plain_payload.json").read_text(encoding="utf-8"))
OLD = P1["new"]
S = (HERE / "main_snapshot.tex").read_text(encoding="utf-8")
LOOPS = json.loads((HERE.parent / "repo/results/summary_tables/reasoning_loops.json").read_text(encoding="utf-8"))
L0 = LOOPS["qwen_claude_minus_r1_loops0"]
LOOPS_CI = f"${L0['est']:+.2f}$, $[{L0['lo']:.2f}, {L0['hi']:.2f}]$"
T = OLD


def rep(old, new):
    global T
    assert T.count(old) == 1, old[:80]
    T = T.replace(old, new)


# 1. cut the teacher subsection (with its table) out of "Is the Gain Real?"
ts, te = T.index("\\subsection{Does the Teacher Matter?}"), T.index("\\section{Related Work}")
TEACH = T[ts:te]
T = T[:ts] + T[te:]
tab = TEACH[TEACH.index("\\begin{table*}"): TEACH.index("\\end{table*}") + len("\\end{table*}")]
first = TEACH[TEACH.index("The public s1K releases"): TEACH.index("\n\n\\paragraph{CoT-Control.}")]
cot = TEACH[TEACH.index("\\paragraph{CoT-Control.}"): TEACH.index("\n\n\\paragraph{ReasonIF.}")]
rif = TEACH[TEACH.index("\\paragraph{ReasonIF.}"): TEACH.index("\n\n\\paragraph{Other models and teachers.}")]
for s in (" It holds on the first 100 words ($+9.07$) and with degenerate traces scored as failures ($+7.72$).",
          " The Claude 3.7 student often falls into repetitive loops (" + str(LOOPS["qwen_claude37_looping"]) + " of " + str(LOOPS["qwen_claude37_n"])
          + " traces), which the R1 student almost never does; with looping traces scored as failures it remains above the R1 student (" + LOOPS_CI + ")."):
    assert cot.count(s) == 1, s[:60]
    cot = cot.replace(s, "")
s = " and without the word-limit constraint ($+13.8$)"
assert rif.count(s) == 1
rif = rif.replace(s, "")
NEWTEACH = ("\\subsection{Raw Reasoning Traces: Does the Teacher Matter?}\n\\label{sec:teacher}\n"
            "The GPT-5.2 data are short summaries of software-engineering reasoning. To test raw reasoning traces, and whether the teacher matters, "
            "we kept the questions fixed and changed only the teacher. " + first + "\n\n" + cot + "\n\n" + rif + "\n\n" + tab + "\n\n")
rep("\\subsection{Other Models}\n", NEWTEACH + "\\subsection{Other Models}\n")

# 2. Phi, OLMo and Sonnet 4.5 parts of the old "Other models and teachers" paragraph
rep("too few traces can be matched on length to test further. The gain cannot be separated from length.",
    "too few traces can be matched on length to test further. The gain cannot be separated from length. Trained on the s1K data, "
    "Phi's Claude 3.7 student also gains more than its R1 student ($+5.15$), but the difference disappears on the first 100 words ($+1.23$); "
    "both students still gain over Base at matched length ($+5.7$ and $+4.0$).")
rep("It is a single run, with a learning rate chosen after the first result.",
    "It is a single run, with a learning rate chosen after the first result. With the s1K data, OLMo's Claude 3.7 student is $-1.50$ below its R1 "
    "student on the seven tasks: its third-person compliance falls (87.2\\% to 73.3\\%), and its gain is on the case tasks.")
rep("Claude Sonnet 4.5 reasoning gains as much as the GPT-5.2 summaries, mostly on the suppression tasks.",
    "Claude Sonnet 4.5 reasoning gains as much as the GPT-5.2 summaries ($+12.13$), mostly on the suppression tasks and partly through degenerate "
    "passes ($+6.70$ with those scored as failures), although Sonnet 4.5 is near zero on stylistic and addition constraints in "
    "\\citet{chen2026cotcontrol} (0.2\\% and 0.0\\%).")

# 3. "Is the Gain Real?" covers both training sets
rep("and also hold at 200 and 300 words.",
    "and also hold at 200 and 300 words. The Claude 3.7 student's gain holds as well ($+9.07$ on the first 100 words).")
rep("and on question pairs of similar length (17.5\\% to 36.5\\%). Length explains part of the Qwen gain, not all of it.",
    "and on question pairs of similar length (17.5\\% to 36.5\\%). Without the word-limit constraint, which shorter reasoning passes most easily, "
    "the Claude 3.7 student stays $+13.8$ above the R1 student on ReasonIF. Length explains part of the Qwen gains, not all of them.")
rep("leaves the Qwen seven-task gains almost unchanged ($+6.16$, $+5.47$ and $+5.36$).",
    "leaves the Qwen seven-task gains almost unchanged ($+6.16$, $+5.47$ and $+5.36$ for the GPT-5.2 seeds, $+7.72$ for the Claude 3.7 student).")
rep(" Reasoning loops, which this rule does not catch, are examined with the teacher test (Section~\\ref{sec:teacher}).",
    "\n\n\\subsection{Reasoning Loops}\n\\label{sec:loops}\nThe degenerate-output rule does not catch reasoning that keeps repeating itself. "
    "We count a trace as looping if most of its last 300 words repeat 8-grams from earlier in the same trace. The Claude 3.7 student loops in "
    + str(LOOPS["qwen_claude37_looping"]) + " of its " + str(LOOPS["qwen_claude37_n"]) + " CoT-Control traces; the R1 student almost never does. "
    "With looping traces scored as failures, the Claude 3.7 student remains above the R1 student (" + LOOPS_CI + ").")
rep("(Sections~\\ref{sec:length}, \\ref{sec:degenerate} and~\\ref{sec:teacher})", "(Section~\\ref{sec:investigating})")

NEW = T
num = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
added = set(num.findall(NEW)) - set(num.findall(OLD))
missing = sorted(n for n in added if n not in S + json.dumps(LOOPS))
lab = lambda s: set(re.findall(r"\\label\{([^}]+)\}", s))  # noqa: E731
print("new numbers not in sources:", missing, "| labels dropped:", sorted(lab(OLD) - lab(NEW)), "| added:", sorted(lab(NEW) - lab(OLD)))
assert not missing and not (lab(OLD) - lab(NEW))
order = [m.group(0) for m in re.finditer(r"\\(?:sub)?section\{[^}]+\}", NEW)]
print("\n".join(order))
(HERE / "plain_payload2.json").write_text(json.dumps({"old_sha256": hashlib.sha256(OLD.encode("utf-8")).hexdigest(), "start": "Our main findings:",
                                                       "end": "\\bibliographystyle{acl_natbib}", "new": NEW}, ensure_ascii=True), encoding="utf-8")
