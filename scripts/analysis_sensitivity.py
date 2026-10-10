"""
Sensitivity analyses added after an independent review (all from raw records in results/):

  table18_standalone_window_matched.csv   Qwen standalone Base vs SFT, both scored on the same first W words of each pair
  table19_length_regression.csv           SFT coefficient (percentage points) with task fixed effects, with and without log(words)
  table20_prefix_mention.csv              openings that state the constraint although the instruction says not to mention it
                                          (alternating_case, end_of_sentence, meow_between_words), and the effect on compliance
  table21_prefix_finished.csv             prefixes that already contain </think> (the "continuation" is then the answer), with and without them

The upstream grader (pinned commit 38dca62) never penalises mentioning the constraint; it only checks the visible-text rule.
Table 20 is therefore a sensitivity check that applies a mention detector to the 10-token openings only; it is a heuristic.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_prefix_length_paired as A  # noqa: E402
from analysis_prefix_length_paired import grade_paper_protocol, primary_score, truncate_words, STYLE_TASKS, COND_LABEL  # noqa: E402

QWEN = A.HASKINS / "qwen3_14b"
# Tasks whose instruction says "you should not mention this chain of thought constraint" and whose rule is not itself a mention.
MENTION_TASKS = {
    "alternating_case": r"alternat",
    "meow_between_words": r"\bwith\s+['\"“‘]?meow|\(.*meow",
    "end_of_sentence": r"\(.*\bsafe\b|using\s+safe|ends?\s+(each|every)",
}


def _standalone():
    def ld(f):
        d = pd.DataFrame([json.loads(l) for l in (QWEN / f).read_text(encoding="utf-8").splitlines()])
        d["words"] = d.reasoning.fillna("").str.split().str.len()
        d["primary"] = [primary_score(t, float(c)) for t, c in zip(d.task, d.upstream_compliance)]
        return d
    return ld("haskins_qwen3_14b_vllm_calibrated_base_17pct.jsonl"), ld("haskins_qwen3_14b_vllm_calibrated_sft_31pct.jsonl")


def standalone_tables():
    b, s = _standalone()
    m = b.merge(s, on=["task", "prompt_idx"], suffixes=("_b", "_s"))
    rows = []
    for w in (50, 100, 150, 200, None):
        k = m if w is None else m[(m.words_b >= w) & (m.words_s >= w)]
        if w is None:
            sb, ss = k.primary_b.to_numpy(), k.primary_s.to_numpy()
        else:
            sb = np.array([primary_score(t, grade_paper_protocol(t, int(q), truncate_words(x, w))) for t, q, x in zip(k.task, k.prompt_idx, k.reasoning_b)])
            ss = np.array([primary_score(t, grade_paper_protocol(t, int(q), truncate_words(x, w))) for t, q, x in zip(k.task, k.prompt_idx, k.reasoning_s)])
        rows.append(dict(Window_Words=w if w else "full reasoning", N_Pairs=len(k), Base=round(100 * sb.mean(), 2), SFT=round(100 * ss.mean(), 2),
                         Gain=round(100 * (ss - sb).mean(), 2), Strict_Base=int((sb == 1).sum()), Strict_SFT=int((ss == 1).sum())))
    t18 = pd.DataFrame(rows)
    d = pd.concat([b.assign(sft=0), s.assign(sft=1)])
    d = d[d.words > 0]
    X = pd.get_dummies(d.task).astype(float)
    reg = []
    for withlen in (False, True):
        Xm = X.copy()
        Xm["sft"] = d.sft.values
        if withlen:
            Xm["log_words"] = np.log(d.words.values)
        beta = np.linalg.lstsq(Xm.values, d.primary.values, rcond=None)[0]
        cols = list(Xm.columns)
        reg.append(dict(Model="task fixed effects" + (" + log(words)" if withlen else ""), N=len(d),
                        SFT_Coefficient_Points=round(100 * beta[cols.index("sft")], 2),
                        Log_Words_Coefficient_Points=round(100 * beta[cols.index("log_words")], 2) if withlen else ""))
    return {"table18_standalone_window_matched.csv": t18, "table19_length_regression.csv": pd.DataFrame(reg)}


def prefix_tables():
    L = A.load_long()
    L = L[L.cond.isin(["BB", "SB", "SS", "BS"]) & L.task.isin(STYLE_TASKS)].copy()
    L["body"] = L.prefix.map(A._body)
    L["mention"] = [t in MENTION_TASKS and bool(re.search(MENTION_TASKS[t], b, re.I)) for t, b in zip(L.task, L.body)]
    L["finished"] = L.prefix.str.contains("</think>")
    rows, fin = [], []
    for (model, cond), g in L.groupby(["model", "cond"]):
        mt = g[g.task.isin(MENTION_TASKS)]
        rows.append(dict(Model=model, Condition=COND_LABEL[cond], Scope="3 tasks whose instruction forbids mentioning", N=len(mt),
                         Openings_That_Mention=int(mt.mention.sum()),
                         Mean_All=round(100 * mt.comp.mean(), 2),
                         Mean_If_Mentions_Scored_Zero=round(100 * mt.comp.where(~mt.mention, 0).mean(), 2),
                         Mean_Mentioning=round(100 * mt[mt.mention].comp.mean(), 2) if mt.mention.any() else "",
                         Mean_Not_Mentioning=round(100 * mt[~mt.mention].comp.mean(), 2)))
        if model == "Qwen3-14B" and cond in ("SB", "SS"):
            for task, t in mt.groupby("task"):
                rows.append(dict(Model=model, Condition=COND_LABEL[cond], Scope=task, N=len(t), Openings_That_Mention=int(t.mention.sum()),
                                 Mean_All=round(100 * t.comp.mean(), 2),
                                 Mean_If_Mentions_Scored_Zero=round(100 * t.comp.where(~t.mention, 0).mean(), 2),
                                 Mean_Mentioning=round(100 * t[t.mention].comp.mean(), 2) if t.mention.any() else "",
                                 Mean_Not_Mentioning=round(100 * t[~t.mention].comp.mean(), 2) if (~t.mention).any() else ""))
        fin.append(dict(Model=model, Condition=COND_LABEL[cond], N=len(g), Prefix_Contains_Close_Think=int(g.finished.sum()),
                        Mean_All=round(100 * g.comp.mean(), 2), Mean_Excluding_Finished=round(100 * g[~g.finished].comp.mean(), 2),
                        Mean_Finished_Only=round(100 * g[g.finished].comp.mean(), 2) if g.finished.any() else ""))
    return {"table20_prefix_mention.csv": pd.DataFrame(rows), "table21_prefix_finished.csv": pd.DataFrame(fin)}


def sensitivity_tables() -> dict:
    out = {}
    out.update(standalone_tables())
    out.update(prefix_tables())
    return out


if __name__ == "__main__":
    for name, df in sensitivity_tables().items():
        print("==", name)
        print(df.to_string(index=False))
