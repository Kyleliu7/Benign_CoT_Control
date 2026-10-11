"""
Length-controlled controllability (standalone Base vs SFT), all from raw records in results/:

  table23_length_controlled_curve.csv      Haskins compliance scored on the first L words of each reasoning trace
                                           (L = 100, 200, ..., 1000), following Chen et al. (2026, Sec. 5.3): at each length L,
                                           controllability is computed from the first word up to word L. Two task sets:
                                           the three consistency tasks Chen et al. use (lowercase, uppercase, alternating case)
                                           and all ten tasks. "Paired" rows keep only question-task pairs where both the base and
                                           the SFT trace reach L words, with a cluster-bootstrap CI over the 50 questions.
  table24_length_adjusted_regression.csv   SFT effect with and without log(reasoning words) as a covariate:
                                           Haskins (linear on the headline score and logistic on strict success, task fixed effects)
                                           and ReasonIF (logistic on instruction-following success, constraint fixed effects;
                                           the word-limit constraint is excluded because it is defined by length, and constraints
                                           with no pass in either condition are excluded).
  table25_reasonif_length_matched.csv      ReasonIF items whose base and SFT reasoning lengths are within a factor of 1.5,
                                           word-limit constraint excluded.

Scores use the paper scorer (pinned upstream grader + calibrated suppression checks) and the original paper's grading rule
(binary for word suppression, multiple word suppression, end-of-sentence and meow; fractional otherwise).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_prefix_length_paired as A  # noqa: E402
from analysis_prefix_length_paired import grade_paper_protocol, primary_score, truncate_words  # noqa: E402

LENGTHS = list(range(100, 1001, 100))
CONSISTENCY_TASKS = ["lowercase_thinking", "uppercase_thinking", "alternating_case"]
TASK_SETS = {"lowercase, uppercase, alternating case": CONSISTENCY_TASKS, "all ten tasks": A.ALL_TASKS}
N_BOOT, SEED = 10_000, 42


def _haskins_standalone() -> dict:
    """{model: {"Base": df, "SFT": df}} with columns task, q, reasoning, words, primary."""
    out = {}
    q = A.HASKINS / "qwen3_14b"
    qwen = {}
    for cond, name in (("Base", "haskins_qwen3_14b_vllm_calibrated_base_17pct.jsonl"), ("SFT", "haskins_qwen3_14b_vllm_calibrated_sft_31pct.jsonl")):
        d = pd.DataFrame([json.loads(l) for l in (q / name).read_text(encoding="utf-8").split(chr(10)) if l.strip()])
        qwen[cond] = d.assign(q=d.prompt_idx.astype(int), upstream=d.upstream_compliance.astype(float))
    out["Qwen3-14B"] = qwen
    phi = {}
    for run in sorted((A.HASKINS / "phi4_reasoning" / "extracted_runs").glob("*/results.csv")):
        d = pd.read_csv(run)
        phi["SFT" if "gpt52" in run.parent.name else "Base"] = d.assign(q=d.prompt_idx.astype(int), upstream=d.upstream_compliance.astype(float))
    out["Phi-4-reasoning"] = phi
    for model in out:
        for cond, d in out[model].items():
            d["reasoning"] = d.reasoning.fillna("").astype(str)
            d["words"] = d.reasoning.str.split().str.len()
            d["primary"] = [primary_score(t, c) for t, c in zip(d.task, d.upstream)]
            out[model][cond] = d[["task", "q", "reasoning", "words", "primary"]].reset_index(drop=True)
    return out


def _windowed(d: pd.DataFrame, w: int) -> pd.Series:
    return pd.Series([primary_score(t, grade_paper_protocol(t, int(q), truncate_words(x, w))) for t, q, x in zip(d.task, d.q, d.reasoning)],
                     index=d.index)


def _cluster_ci(diff_by_q: pd.Series, rng) -> tuple:
    v = diff_by_q.to_numpy()
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    boots = v[idx].mean(axis=1)
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def table_curve(H: dict) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for model, conds in H.items():
        b, s = conds["Base"], conds["SFT"]
        for set_name, tasks in TASK_SETS.items():
            bt, st = b[b.task.isin(tasks)], s[s.task.isin(tasks)]
            rows.append(dict(Model=model, Tasks=set_name, Length_Words="full trace", Basis="all traces",
                             N_Base=len(bt), N_SFT=len(st), Base=round(100 * bt.primary.mean(), 2), SFT=round(100 * st.primary.mean(), 2),
                             Gain=round(100 * (st.primary.mean() - bt.primary.mean()), 2), CI_Low="", CI_High=""))
            for w in LENGTHS:
                bw, sw = bt[bt.words >= w], st[st.words >= w]
                if len(bw) == 0 or len(sw) == 0:
                    continue
                bsc, ssc = _windowed(bw, w), _windowed(sw, w)
                rows.append(dict(Model=model, Tasks=set_name, Length_Words=w, Basis="traces reaching L (unpaired, as Chen et al.)",
                                 N_Base=len(bw), N_SFT=len(sw), Base=round(100 * bsc.mean(), 2), SFT=round(100 * ssc.mean(), 2),
                                 Gain=round(100 * (ssc.mean() - bsc.mean()), 2), CI_Low="", CI_High=""))
                m = bw.assign(score=bsc).merge(sw.assign(score=ssc), on=["task", "q"], suffixes=("_b", "_s"))
                if len(m) < 20:
                    continue
                m["diff"] = m.score_s - m.score_b
                lo, hi = _cluster_ci(m.groupby("q")["diff"].mean(), rng)
                rows.append(dict(Model=model, Tasks=set_name, Length_Words=w, Basis="paired: both traces reach L",
                                 N_Base=len(m), N_SFT=len(m), Base=round(100 * m.score_b.mean(), 2), SFT=round(100 * m.score_s.mean(), 2),
                                 Gain=round(100 * m["diff"].mean(), 2), CI_Low=round(100 * lo, 2), CI_High=round(100 * hi, 2)))
    return pd.DataFrame(rows)


def _logit(X: np.ndarray, y: np.ndarray, iters: int = 100):
    """Logistic regression by Newton-Raphson with a tiny ridge for stability; returns (beta, standard errors)."""
    beta = np.zeros(X.shape[1])
    ridge = 1e-6 * np.eye(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + ridge
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]) + ridge)
    return beta, np.sqrt(np.diag(cov))


def _design(df: pd.DataFrame, fe: str, with_len: bool):
    X = pd.get_dummies(df[fe]).astype(float)
    X["sft"] = df.sft.values
    if with_len:
        X["log_words"] = np.log(df.words.values)
    return X


def _reasonif() -> dict:
    files = {"Qwen3-14B": (A.REASONIF / "qwen3_14b" / "reasonif_qwen3_14b_base_untouched.jsonl", A.REASONIF / "qwen3_14b" / "reasonif_qwen3_14b_sft_gpt52_high.jsonl"),
             "Phi-4-reasoning": (A.REASONIF / "phi4_reasoning" / "base" / "scored_responses.jsonl", A.REASONIF / "phi4_reasoning" / "sft" / "scored_responses.jsonl")}
    out = {}
    for model, (fb, fs) in files.items():
        parts = []
        for sft, f in ((0, fb), (1, fs)):
            d = pd.DataFrame([json.loads(l) for l in f.read_text(encoding="utf-8").split(chr(10)) if l.strip()])
            d = d.assign(sft=sft, ifs=d.instruction_following.astype(bool).astype(int),
                         words=d.reasoning_content.fillna("").astype(str).str.split().str.len(),
                         constraint=d.constraint.astype(str))
            parts.append(d[["dataset_index", "constraint", "sft", "ifs", "words"]])
        out[model] = pd.concat(parts, ignore_index=True)
    return out


def table_regression(H: dict, R: dict) -> pd.DataFrame:
    rows = []
    for model, conds in H.items():
        d = pd.concat([conds["Base"].assign(sft=0), conds["SFT"].assign(sft=1)], ignore_index=True)
        d = d[d.words > 0].reset_index(drop=True)
        for with_len in (False, True):
            X = _design(d, "task", with_len)
            beta = np.linalg.lstsq(X.values, d.primary.values, rcond=None)[0]
            cols = list(X.columns)
            rows.append(dict(Benchmark="Haskins", Model=model, Outcome="headline score (linear, percentage points)", N=len(d),
                             Covariates="task fixed effects" + (" + log(words)" if with_len else ""),
                             SFT_Effect=round(100 * beta[cols.index("sft")], 2), SFT_Effect_CI="",
                             Log_Words_Coefficient=round(100 * beta[cols.index("log_words")], 2) if with_len else ""))
        strict = (d.primary == 1.0).astype(int).values
        keep = d.task.isin(d.loc[strict == 1, "task"].unique()).values
        for with_len in (False, True):
            X = _design(d[keep], "task", with_len)
            beta, se = _logit(X.values, strict[keep])
            i = list(X.columns).index("sft")
            rows.append(dict(Benchmark="Haskins", Model=model, Outcome="strict success (logistic, odds ratio)", N=int(keep.sum()),
                             Covariates="task fixed effects" + (" + log(words)" if with_len else ""),
                             SFT_Effect=round(float(np.exp(beta[i])), 2),
                             SFT_Effect_CI=f"[{np.exp(beta[i] - 1.96 * se[i]):.2f}, {np.exp(beta[i] + 1.96 * se[i]):.2f}]",
                             Log_Words_Coefficient=round(float(np.exp(beta[list(X.columns).index('log_words')])), 2) if with_len else ""))
    for model, d in R.items():
        d = d[(d.constraint != "length_constraint_checkers:number_words") & (d.words > 0)]
        d = d[d.constraint.isin(d.loc[d.ifs == 1, "constraint"].unique())].reset_index(drop=True)
        for with_len in (False, True):
            X = _design(d, "constraint", with_len)
            beta, se = _logit(X.values, d.ifs.values)
            i = list(X.columns).index("sft")
            rows.append(dict(Benchmark="ReasonIF (word limit excluded)", Model=model, Outcome="instruction-following success (logistic, odds ratio)",
                             N=len(d), Covariates="constraint fixed effects" + (" + log(words)" if with_len else ""),
                             SFT_Effect=round(float(np.exp(beta[i])), 2),
                             SFT_Effect_CI=f"[{np.exp(beta[i] - 1.96 * se[i]):.2f}, {np.exp(beta[i] + 1.96 * se[i]):.2f}]",
                             Log_Words_Coefficient=round(float(np.exp(beta[list(X.columns).index('log_words')])), 2) if with_len else ""))
    return pd.DataFrame(rows)


def table_reasonif_matched(R: dict) -> pd.DataFrame:
    rows = []
    for model, d in R.items():
        d = d[d.constraint != "length_constraint_checkers:number_words"]
        w = d.pivot_table(index=["dataset_index", "constraint"], columns="sft", values=["ifs", "words"])
        w = w[(w[("words", 0)] > 0) & (w[("words", 1)] > 0)]
        ratio = np.abs(np.log(w[("words", 1)] / w[("words", 0)]))
        for label, sub in (("all items (word limit excluded)", w), ("length within a factor of 1.5", w[ratio <= np.log(1.5)])):
            rows.append(dict(Model=model, Items=label, N=len(sub), Base_Mean_Words=round(sub[("words", 0)].mean(), 1),
                             SFT_Mean_Words=round(sub[("words", 1)].mean(), 1), Base_IFS=round(100 * sub[("ifs", 0)].mean(), 2),
                             SFT_IFS=round(100 * sub[("ifs", 1)].mean(), 2), Gain=round(100 * (sub[("ifs", 1)].mean() - sub[("ifs", 0)].mean()), 2)))
    return pd.DataFrame(rows)


def length_controlled_tables() -> dict:
    H, R = _haskins_standalone(), _reasonif()
    return {"table23_length_controlled_curve.csv": table_curve(H),
            "table24_length_adjusted_regression.csv": table_regression(H, R),
            "table25_reasonif_length_matched.csv": table_reasonif_matched(R)}


if __name__ == "__main__":
    out_dir = A.REPO_ROOT / "results" / "summary_tables"
    for name, df in length_controlled_tables().items():
        if "--write" in sys.argv:
            df.to_csv(out_dir / name, index=False)
        print(f"\n=== {name}\n{df.to_string(index=False)}")
