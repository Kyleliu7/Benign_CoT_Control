r"""Plainer main text for the arXiv paper (Emergent-Misalignment style: short sentences, 1-2 numbers per claim, intervals
mostly in tables). Kyle approved "Intro + main text"; the abstract, the two opening intro paragraphs, the three main
tables, Related Work, Reproducibility and the appendix are kept verbatim from the snapshot.

Replaces the region of main.tex from "In summary:" up to (not including) "\bibliographystyle". Every number in the new
prose must already occur in the snapshot or in results/summary_tables/reasoning_loops.json (checked below).

  python rewrite/build_plain_rewrite.py    -> rewrite/plain_payload.json {"old_sha256", "start", "end", "new"}
"""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = (HERE / "main_snapshot.tex").read_text(encoding="utf-8")
LOOPS = json.loads((HERE.parent / "repo/results/summary_tables/reasoning_loops.json").read_text(encoding="utf-8"))

START, END = "In summary:", "\\bibliographystyle{acl_natbib}"
a, b = S.index(START), S.index(END)
assert S.count(START) == 1 and S.count(END) == 1
OLD = S[a:b]


def block(begin, end_marker):
    i = S.index(begin)
    j = S.index(end_marker, i) + len(end_marker)
    return S[i:j]


TAB_MODELS = block("\\begin{table*}[t]\n\\centering\\small\n\\caption{Base and SFT models on ReasonIF", "\\end{table*}")
TAB_VARIANTS = block("\\begin{table*}[t]\n\\centering\\small\n\\caption{Other Qwen3-14B fine-tunes", "\\end{table*}")
TAB_TEACHER = block("\\begin{table*}[t]\n\\centering\\small\n\\caption{Does the teacher matter?", "\\end{table*}")
RELATED = S[S.index("\\section{Related Work}"): S.index("\\section{Discussion}")]
REPRO = S[S.index("\\section*{Reproducibility}"): b]
AIUSE_OLD = S[S.index("\\section*{Use of AI Assistants}"): S.index("\\section*{Reproducibility}")]
AIUSE = AIUSE_OLD.rstrip() + " An AI agent also restructured this version and rewrote the main text in plainer language.\n\n"

NEW = r"""Our main findings:
\begin{enumerate}
\item \textbf{Benign distillation raises controllability.} Fine-tuning Qwen3-14B on 212 GPT-5.2 reasoning summaries more than doubles ReasonIF instruction following (14.0\% to 33.3\%) and raises seven-task CoT-Control compliance (13.55 to 19.71). Raw reasoning traces from Claude 3.7 Sonnet do the same (14.0\% to 29.3\% on ReasonIF; $+7.43$ on CoT-Control) (Sections~\ref{sec:results} and~\ref{sec:teacher}).
\item \textbf{The teacher matters.} The same 212 questions with DeepSeek R1's reasoning give no gain on either benchmark (Section~\ref{sec:teacher}).
\item \textbf{The gain is not an artifact.} It survives grading only the first 100 words and scoring degenerate or looping outputs as failures, and the GPT-5.2 gain repeats across three training seeds (Sections~\ref{sec:length}, \ref{sec:degenerate} and~\ref{sec:teacher}).
\item \textbf{It comes from the reasoning text, and it costs accuracy.} Training on the answers alone gives no gain. ReasonIF accuracy falls from 80.3\% to 59.3--65.3\% (Section~\ref{sec:variants}).
\item \textbf{It depends on the model.} In Phi-4-reasoning the gain cannot be separated from much shorter outputs, and OLMo-3-7B-Think shows no seven-task gain (Section~\ref{sec:other-models}).
\end{enumerate}
Appendix~\ref{app:additional} traces part of the Qwen gain to the opening of the reasoning: the summaries' bold headings and the first ten tokens. We do not test a monitor, so these results do not show that CoT monitoring fails. They show that ordinary post-training can move controllability up as well as down, so it should be re-measured after distillation.

\section{Setup}
\label{sec:methods}
\subsection{Training}
\label{sec:questions}
\paragraph{Models.} We fine-tune Qwen3-14B \citep{yang2025qwen3}, Phi-4-reasoning \citep{abdin2025phi4reasoning} and OLMo-3-7B-Think \citep{olmo2025olmo3} with LoRA ($r=32$, $\alpha=64$; learning rate $10^{-4}$; 3 epochs; seed 42), keeping the checkpoint with the lowest validation loss. \emph{Base} is the model before fine-tuning. For Qwen we add seeds 1 and 2; for OLMo, a run with learning rate $2\times10^{-4}$. Full settings are in Appendices~\ref{app:setup} and~\ref{app:training-settings}.

\paragraph{Data.}\label{sec:data} The main training set is the 212 examples of \texttt{TeichAI/gpt-5.2-high-reasoning-250x} \citep{teichaihighreasoning} with a complete reasoning block (190 for training, 22 for validation). The prompts are software-engineering requests. The reasoning is a GPT-5.2 \emph{reasoning summary}: short (mean 222 words), written in the first person, and always opening with a bold heading. It contains no controllability instructions. The teacher test (Section~\ref{sec:teacher}) uses a second set: 212 mathematics and science questions from s1K with raw reasoning from Claude 3.7 Sonnet or DeepSeek R1.

\paragraph{Runs.} Ablations change one element of the training data at a time (Section~\ref{sec:variants}). The length and degenerate-output checks, the extra seeds and the OLMo runs were added after the main results. Neither benchmark was used for tuning, except the second OLMo learning rate, chosen after the first OLMo result.

\subsection{Evaluation}
\emph{ReasonIF} \citep{kwon2026reasonif} has 300 questions, each with one constraint on the reasoning (ending phrase, word limit, language, no commas, JSON, all capitals). We report instruction-following success (IFS) and accuracy. \emph{CoT-Control} \citep{chen2026cotcontrol}, in the version of \citet{haskins2026obfuscation}, crosses 50 questions with ten constraints, such as writing in the third person, avoiding a word, or writing in lowercase. Our main number is mean compliance on the seven tasks that are not character-level (350 pairs), because only these predict obfuscation across models \citep{haskins2026obfuscation}; every table also gives the ten-task mean. Intervals are paired 95\% bootstrap intervals over questions.

\section{Results}
\label{sec:results}
\subsection{Qwen3-14B}
Fine-tuning on the GPT-5.2 summaries raises ReasonIF IFS from 14.0\% to 33.3\% ($+19.3$, $[13.7, 25.0]$), and the two extra seeds reach 33.7\% and 33.0\% (Table~\ref{tab:models}). Seven-task CoT-Control compliance rises from 13.55 to 19.71 ($+6.16$, $[3.73, 8.80]$), and to 19.02 and 19.77 for the other seeds. The gain is mostly on third person, word suppression and multiple-word suppression (Appendix Table~\ref{tab:standalone-task}). ReasonIF accuracy falls from 80.3\% to 65.3\%, but joint success, following the instruction and answering correctly, still rises from 11.7\% to 20.7\%.

""" + TAB_MODELS + r"""

\subsection{Other Models}
\label{sec:other-models}
\label{sec:olmo}
\paragraph{Phi-4-reasoning.} Fine-tuning raises ReasonIF IFS from 5.0\% to 12.0\%, almost all on the word-limit constraint, and seven-task compliance from 12.40 to 20.66. But the fine-tuned model writes far shorter reasoning: 84\% of its traces are under 250 tokens, against none of the base model's. Graded on the first 100 words, no gain remains ($-0.50$), and too few traces can be matched on length to test further. The gain cannot be separated from length.

\paragraph{OLMo-3-7B-Think.} With the same recipe, ReasonIF IFS rises slightly (8.3\% to 11.7\%) and seven-task compliance does not change ($+0.70$, $[-0.78, 2.28]$). At learning rate $2\times10^{-4}$, ReasonIF IFS rises to 20.3\%, but the reasoning becomes about three times shorter, and at matched length the gain is small ($+2.6$) except on the language constraint. On CoT-Control this run gains on the case tasks and loses on third person, so its seven-task mean falls ($-3.01$; Appendix Tables~\ref{tab:olmo} and~\ref{tab:olmo-task}). It is a single run, with a learning rate chosen after the first result.

\section{Is the Gain Real?}
\label{sec:investigating}
\subsection{Shorter Reasoning}
\label{sec:length}
Shorter reasoning makes compliance easier \citep{chen2026cotcontrol}, and fine-tuning shortens Qwen's reasoning. Following \citet{chen2026cotcontrol}, we grade every trace on its first 100 words only. The Qwen gains shrink but remain for all three seeds ($+4.50$, $+3.94$ and $+2.53$, against $+6.16$, $+5.47$ and $+6.22$ on full traces), and also hold at 200 and 300 words. On ReasonIF, the gain holds with reasoning length as a covariate (odds ratio 2.37, $[1.41, 3.97]$) and on question pairs of similar length (17.5\% to 36.5\%). Length explains part of the Qwen gain, not all of it. For Phi, the same check removes the gain.

\subsection{Degenerate Output}
\label{sec:degenerate}
Rule-based graders can reward broken text, such as empty reasoning or reasoning made of ``meow''. Scoring such traces as failures leaves the Qwen seven-task gains almost unchanged ($+6.16$, $+5.47$ and $+5.36$). On ReasonIF, a manual review of suspicious passes leaves 31.3\% for every seed. For OLMo with the Qwen recipe, the same rule removes its CoT-Control gain entirely. Reasoning loops, which this rule does not catch, are examined with the teacher test (Section~\ref{sec:teacher}).

\subsection{Which Part of the Training Data Matters?}
\label{sec:variants}
Table~\ref{tab:variants} changes one element of the training data at a time (one run each). Training only on the reasoning still raises IFS (24.7\%); training only on the answers does not (12.7\%), and it keeps accuracy at the base level (82.3\%). The gain, and most of the accuracy cost, come from the reasoning text. Replacing the GPT-5.2 reasoning with other teachers' reasoning for the same prompts gives little or no gain on ReasonIF: 18.0\% for Qwen3-235B-A22B, 12.3\% for Claude Sonnet 4.5 and 10.3\% for ReasonFlux. CoT-Control differs (Appendix Table~\ref{tab:variants-cot}): Claude Sonnet 4.5 reasoning gains as much as the GPT-5.2 summaries, mostly on the suppression tasks.

""" + TAB_VARIANTS + r"""

\subsection{Does the Teacher Matter?}
\label{sec:teacher}
The other-teacher runs above change several things at once. For a cleaner test, we kept the questions fixed and changed only the teacher. The public s1K releases \citep{muennighoff2025s1} pair the same mathematics and science questions with raw reasoning from Claude 3.7 Sonnet and from DeepSeek R1 (\texttt{simplescaling/s1K-claude-3-7-sonnet} and \texttt{simplescaling/s1K-1.1}). The two teachers sit at opposite ends of the controllability ranking of \citet{chen2026cotcontrol} (51.1\% against 0.2\% stylistic compliance). We drew 212 questions that fit a 12,288-token training cutoff and share no 8-gram with CoT-Control, and trained Qwen3-14B on each teacher's reasoning with the same recipe (one run each).

\paragraph{CoT-Control.} The Claude 3.7 student gains $+7.43$ ($[4.89, 10.14]$) on the seven tasks; the R1 student does not change ($+0.31$). Paired, the Claude 3.7 student is $+7.12$ ($[4.66, 9.81]$) above the R1 student (Table~\ref{tab:teacher}). Its gain is mostly on word suppression and the arrow prefix, which the GPT-5.2 summaries do not move. It holds on the first 100 words ($+9.07$) and with degenerate traces scored as failures ($+7.72$). The Claude 3.7 student often falls into repetitive loops (""" + str(LOOPS["qwen_claude37_looping"]) + " of " + str(LOOPS["qwen_claude37_n"]) + r""" traces), which the R1 student almost never does; with looping traces scored as failures it remains above the R1 student ($""" + f"{LOOPS['qwen_claude_minus_r1_loops0']['est']:+.2f}" + r"""$, $[""" + f"{LOOPS['qwen_claude_minus_r1_loops0']['lo']:.2f}, {LOOPS['qwen_claude_minus_r1_loops0']['hi']:.2f}" + r"""]$). Fine-tuning on Qwen3-14B's own reasoning gives no gain ($+0.98$). Both s1K students and the self-distilled model lose most of the base model's lowercase compliance (45.7\% to 16.0\%, 7.8\% and 6.8\%); in \citet{haskins2026obfuscation}, lowercase compliance correlates negatively with obfuscation.

\paragraph{ReasonIF.} The difference is larger on ReasonIF. The Claude 3.7 student's IFS rises from 14.0\% to 29.3\%, while the R1 student stays at 13.0\%; paired, the difference is $+16.3$ ($[11.0, 21.7]$). It holds without the 24 ReasonIF questions that overlap with the s1K training questions ($+17.0$) and without the word-limit constraint ($+13.8$). The cost is accuracy: 57.3\%, against 80.3\% for Base and 70.7\% for the R1 student, mostly because 73 of its 300 responses reach the 16,384-token limit without an answer.

\paragraph{Other models and teachers.} On Phi-4-reasoning the Claude 3.7 student also gains more than the R1 student ($+5.15$), but the difference disappears on the first 100 words ($+1.23$); both students still gain over Base at matched length ($+5.7$ and $+4.0$). On OLMo-3-7B-Think the result does not hold: the Claude 3.7 student is $-1.50$ below the R1 student on the seven tasks, because its third-person compliance falls (87.2\% to 73.3\%); its gain is on the case tasks. Claude Sonnet 4.5 is near zero on stylistic and addition constraints in \citet{chen2026cotcontrol} (0.2\% and 0.0\%), yet its student gains on CoT-Control ($+12.13$), mostly on suppression and partly through degenerate passes ($+6.70$ with those scored as failures); its data also differ in prompts, length and format. All teacher runs are single runs.

""" + TAB_TEACHER + "\n\n" + RELATED + r"""\section{Discussion}
\label{sec:discussion}
\paragraph{The direction of change depends on the data.} Prior work finds that reasoning post-training lowers controllability. Here, distillation with no controllability objective raises it, with some teachers and not others. The analyses in Appendix~\ref{app:additional} show that on the case tasks, compliance is largely decided at the start of the reasoning: training data that opens with a short, free-form span the student can write in the required style raises it. On the seven tasks that predict obfuscation, what in the training reasoning carries the effect remains open. ``Benign'' describes the intended supervision, not the effect. We do not claim that such data generally raise controllability, only that they can, as a byproduct of training. Like \citet{chen2026cotcontrol}, we encourage measuring controllability throughout post-training.

\paragraph{Relationship to CoT monitorability.} Low compliance in natural rollouts is not an upper bound on what a model does after further training. Following formatting constraints does not show an ability or incentive to hide harmful intent, and we test no monitor; we rely on the correlation between controllability and obfuscation found in prior work.

\section{Conclusion}
Ordinary supervised fine-tuning on benign reasoning can raise chain-of-thought controllability. In Qwen3-14B, 212 GPT-5.2 reasoning summaries and 212 raw Claude 3.7 Sonnet reasoning traces both raise it, on ReasonIF and on the CoT-Control tasks that predict obfuscation, while DeepSeek R1 reasoning on the same questions does not. The gains survive checks for shorter outputs and degenerate text, the Claude 3.7 result survives scoring reasoning loops as failures, and the GPT-5.2 gain repeats across three training seeds. The effect does not appear in every model: in Phi-4-reasoning it cannot be separated from shorter outputs, and OLMo-3-7B-Think shows none. Controllability should be re-measured after ordinary distillation, not assumed to fall.

\section*{Limitations}
The main result rests on one training set of 212 examples, and the teacher test adds two more on a second set of 212 questions; all are small. Of three models, the effect is clear in one. Only the main Qwen result has three training runs; the teacher test, Phi, OLMo, the ablations and the appendix analyses are single runs. CoT-Control has 50 underlying questions. The graders reward some degenerate text, and our degenerate-output and loop rules are post hoc. Grading a fixed window cannot rule out longer-range length effects, and no model was trained on length-matched data. We chose the seven-task composite after seeing the ten-task results; both are in every table. The teacher test uses one pair of teachers, its difference holds only on Qwen3-14B, and its Claude 3.7 student often loops and loses accuracy; its data also differ from the main data in questions and training cutoff, so it cannot isolate the teacher's controllability. The link between an opening written in the required style and the reasoning after it (Appendix~\ref{app:additional}) is observational.

""" + AIUSE + REPRO

# every number in the new prose (tables and verbatim blocks excluded) must already occur in the snapshot or the loop JSON
prose = NEW
for verb in (TAB_MODELS, TAB_VARIANTS, TAB_TEACHER, RELATED, REPRO, AIUSE):
    prose = prose.replace(verb, " ")
num = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
sources = S + json.dumps(LOOPS)
missing = sorted({n for n in num.findall(prose) if n not in sources})
labels_old = set(re.findall(r"\\label\{([^}]+)\}", OLD))
labels_new = set(re.findall(r"\\label\{([^}]+)\}", NEW))
print("numbers not found in sources:", missing)
print("labels dropped:", sorted(labels_old - labels_new), "| labels added:", sorted(labels_new - labels_old))
words = lambda s: len(re.sub(r"\\[a-zA-Z]+|[{}$]", " ", s).split())  # noqa: E731
print("prose words old -> new:", words(OLD.replace(TAB_MODELS, "").replace(TAB_VARIANTS, "").replace(TAB_TEACHER, "")), "->", words(prose))
assert not missing
(HERE / "plain_payload.json").write_text(json.dumps({"old_sha256": hashlib.sha256(OLD.encode("utf-8")).hexdigest(), "start": START, "end": END, "new": NEW}, ensure_ascii=True), encoding="utf-8")
