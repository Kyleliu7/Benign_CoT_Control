"""
Additional analyses requested by the audit (all derived from raw records in results/):

  table8_window_matched_haskins.csv        compliance when every condition is scored on the same first W words
  table9_prefix_content_haskins.csv        what the injected prefixes contain (self-compliance, names the constraint, bold heading)
  table10_prefix_conditional_haskins.csv   continuation compliance split by whether the prefix names/demonstrates the constraint
  table11_prefix_examples.csv              the first SFT-donor prefix per task (Qwen and Phi)
  table12_paired_contrasts_haskins.csv     paired cluster-bootstrap CIs (resampling the 50 questions, 10,000 draws, seed 42)
  table13_paired_contrasts_reasonif.csv    paired differences on the 300 ReasonIF items: bootstrap CI + exact McNemar p
  table14_kl_spike_frequency.csv           how often large downstream forward-KL values occur, per model and benchmark
  table15_kl_outcome_groups.csv            per-trace maximum downstream KL by single-sample outcome group (G1 vs G3 / G2, bootstrap CI)

These are descriptive. The prefix-conditional split is observational (prefix content and task difficulty are confounded).
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from evaluators.haskins_evaluator import grade_paper_protocol, primary_score  # noqa: E402

HASKINS = REPO_ROOT / "results" / "haskins_500"
REASONIF = REPO_ROOT / "results" / "reasonif_300"
N_BOOT, SEED = 10_000, 42

STYLE_TASKS = ["third_person", "arrow_prefix", "end_of_sentence", "meow_between_words", "repeat_sentences",
               "alternating_case", "lowercase_thinking", "uppercase_thinking"]
ALL_TASKS = STYLE_TASKS + ["word_suppression", "multiple_word_suppression"]
NAMES_CONSTRAINT = {
    "arrow_prefix": r">>|arrow", "third_person": r"third.person|the assistant", "alternating_case": r"alternat",
    "lowercase_thinking": r"lowercase|lower.case", "uppercase_thinking": r"uppercase|upper.case|capital",
    "repeat_sentences": r"this is my analysis", "end_of_sentence": r"\bsafe\b", "meow_between_words": r"meow",
}
COND_LABEL = {"BB": "Base->Base", "BS": "Base->SFT", "SB": "SFT->Base", "SS": "SFT->SFT", "A5": "SFT->Base (OFF)"}
QWEN_TAGS = {"BB": "2x2-base-to-base-on-10tok", "BS": "2x2-base-to-sft-on-10tok",
             "SB": "2x2-sft-to-base-on-10tok", "SS": "2x2-sft-to-sft-on-10tok"}
PHI_KEYS = {"BB": "a2", "BS": "a3", "SB": "a1", "SS": "a4", "A5": "a5"}


def _body(prefix: str) -> str:
    return re.sub(r"^\s*<think>\s*", "", prefix or "")


def load_long() -> pd.DataFrame:
    """One row per (model, condition, task, question): prefix, continuation text, stored score."""
    rows = []
    for cond, tag in QWEN_TAGS.items():
        df = pd.read_csv(glob.glob(str(HASKINS / "qwen3_14b" / f"*{tag}__*" / "results.csv"))[0]).fillna("")
        for r in df.itertuples():
            rows.append(("Qwen3-14B", cond, r.task, int(r.prompt_idx), r.injected_prefix_text, r.continuation_reasoning,
                         float(r.continuation_compliance), int(float(r.continuation_score_one)), float(r.continuation_tokens)))
    f = HASKINS / "qwen3_14b" / "haskins_qwen3_14b_vllm_calibrated_prefix_off_21pct.jsonl"
    for line in f.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows.append(("Qwen3-14B", "A5", r["task"], int(r["prompt_idx"]), r["injected_prefix_text"], r["continuation_reasoning"],
                     float(r["continuation_compliance"]), int(float(r["continuation_compliance"]) == 1.0), float(r.get("generated_tokens", np.nan))))
    bundle = json.loads((HASKINS / "phi4_reasoning" / "phi4_haskins_500_bundle.json").read_text(encoding="utf-8"))
    for item in bundle["items"]:
        for cond, key in PHI_KEYS.items():
            m = item["models"][key]
            cot, prefix = m.get("cot") or "", m.get("prefix") or ""
            cont = cot[len(prefix):] if prefix and cot.startswith(prefix) else cot
            rows.append(("Phi-4-reasoning", cond, item["task"], int(item["prompt_idx"]), prefix, cont,
                         float(m["comp"]) / 100.0, int(bool(m["strict"])), float(m.get("tokens", np.nan))))
    df = pd.DataFrame(rows, columns=["model", "cond", "task", "q", "prefix", "cont", "comp", "strict", "ntok"])
    # headline score follows the original paper's rule (binary for word suppression, end-of-sentence, meow); the plain fractional score is kept
    df["comp_frac"] = df["comp"]
    df["comp"] = [primary_score(t, c) for t, c in zip(df.task, df.comp_frac)]
    return df


def truncate_words(text: str, w: int) -> str:
    ends = [m.end() for m in re.finditer(r"\S+", text)]
    return text if len(ends) <= w else text[: ends[w - 1]]


# ------------------------------------------------------------------------------------------------
def table_window_matched(L: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in L[L.cond.isin(["BB", "BS", "SB", "SS"])].groupby("model"):
        g = g.assign(words=g.cont.map(lambda t: len(t.split())))
        wide_words = g.pivot_table(index=["task", "q"], columns="cond", values="words")
        for w in (50, 100, 150, None):
            keep = wide_words.index if w is None else wide_words.index[(wide_words >= w).all(axis=1)]
            if len(keep) == 0:
                continue
            sub = g.set_index(["task", "q"]).loc[keep].reset_index()
            for cond in ["BB", "BS", "SB", "SS"]:
                s = sub[sub.cond == cond]
                if w is None:
                    comp, strict = s.comp.to_numpy(), s.strict.to_numpy()
                else:
                    sc = np.array([grade_paper_protocol(t, int(q), truncate_words(x, w)) for t, q, x in zip(s.task, s.q, s.cont)])
                    comp, strict, frac = np.array([primary_score(t, c) for t, c in zip(s.task, sc)]), (sc == 1.0).astype(int), sc
                if w is None:
                    frac = s.comp_frac.to_numpy()
                rows.append(dict(Model=model, Window_Words=w if w else "full continuation", N_Pairs=len(keep), Condition=COND_LABEL[cond],
                                 Mean_Compliance=round(100 * comp.mean(), 2), Mean_Compliance_Fractional=round(100 * frac.mean(), 2),
                                 Strict_Count=int(strict.sum())))
    return pd.DataFrame(rows)


def table_prefix_content(L: pd.DataFrame):
    rows, per_task = [], []
    for model, g in L.groupby("model"):
        for cond, s in g.groupby("cond"):
            s = s[s.task.isin(STYLE_TASKS)].copy()
            body = s.prefix.map(_body)
            s["self_ok"] = [grade_paper_protocol(t, int(q), b) == 1.0 for t, q, b in zip(s.task, s.q, body)]
            s["names"] = [bool(re.search(NAMES_CONSTRAINT[t], b, re.I)) for t, b in zip(s.task, body)]
            s["heading"] = body.str.startswith("**")
            rows.append(dict(Model=model, Condition=COND_LABEL[cond], N=len(s),
                             Prefix_Self_Compliant_Pct=round(100 * s.self_ok.mean(), 1),
                             Prefix_Names_Or_Demonstrates_Pct=round(100 * s.names.mean(), 1),
                             Prefix_Starts_With_Bold_Heading_Pct=round(100 * s.heading.mean(), 1)))
            if cond in ("SB", "BB"):
                for task, t in s.groupby("task"):
                    per_task.append(dict(Model=model, Condition=COND_LABEL[cond], Task=task, N=len(t),
                                         Self_Compliant=int(t.self_ok.sum()), Names_Or_Demonstrates=int(t.names.sum()),
                                         Bold_Heading=int(t.heading.sum())))
    return pd.DataFrame(rows), pd.DataFrame(per_task)


def table_prefix_conditional(L: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, cond), s in L[L.cond.isin(["BB", "SB", "SS", "BS"]) & L.task.isin(STYLE_TASKS)].groupby(["model", "cond"]):
        body = s.prefix.map(_body)
        names = np.array([bool(re.search(NAMES_CONSTRAINT[t], b, re.I)) for t, b in zip(s.task, body)])
        for flag, label in ((True, "prefix names/demonstrates constraint"), (False, "prefix does not")):
            t = s[names == flag]
            if len(t):
                rows.append(dict(Model=model, Condition=COND_LABEL[cond], Prefix=label, N=len(t),
                                 Continuation_Mean=round(100 * t.comp.mean(), 2), Strict_Count=int(t.strict.sum())))
    return pd.DataFrame(rows)


def table_prefix_examples(L: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in L[L.cond == "SB"].groupby("model"):
        for task in ALL_TASKS:
            t = g[g.task == task].sort_values("q").iloc[0]
            rows.append(dict(Model=model, Task=task, Question_Index=int(t.q), SFT_Donor_Prefix=_body(t.prefix).replace("\n", "\\n")))
    return pd.DataFrame(rows)


def _boot_prompt_diff(a: np.ndarray, b: np.ndarray, rng) -> tuple:
    d = a - b
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boots = d[idx].mean(axis=1)
    return d.mean(), *np.percentile(boots, [2.5, 97.5])


def table_paired_haskins(L: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    contrasts = [("SFT->Base minus Base->Base (donor effect, base recipient)", "SB", "BB"),
                 ("SFT->SFT minus Base->SFT (donor effect, SFT recipient)", "SS", "BS"),
                 ("Base->SFT minus Base->Base (recipient effect, base donor)", "BS", "BB"),
                 ("SFT->SFT minus SFT->Base (recipient effect, SFT donor)", "SS", "SB")]
    rows = []
    for model, g in L.groupby("model"):
        per_q = {c: g[g.cond == c].groupby("q")[["comp", "comp_frac", "strict"]].mean().sort_index() for c in ["BB", "BS", "SB", "SS"]}
        for label, x, y in contrasts:
            for metric, name in (("comp", "continuation mean (pp)"), ("strict", "strict pass rate (pp)"),
                                 ("comp_frac", "continuation mean, fractional scoring for all tasks (pp)")):
                est, lo, hi = _boot_prompt_diff(per_q[x][metric].to_numpy(), per_q[y][metric].to_numpy(), rng)
                rows.append(dict(Model=model, Contrast=label, Metric=name, Estimate_pp=round(100 * est, 2),
                                 CI95_Low=round(100 * lo, 2), CI95_High=round(100 * hi, 2)))
    return pd.DataFrame(rows)


def _truthy(v):
    return v is True or str(v) == "True"


def _paired(a: np.ndarray, b: np.ndarray, rng):
    d = a.astype(int) - b.astype(int)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return d.mean(), lo, hi, n10, n01, p


def table_paired_reasonif() -> pd.DataFrame:
    def load(path, ifs, joint):
        recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        recs.sort(key=lambda r: int(r["dataset_index"]))
        return {"IFS": np.array([_truthy(r[ifs]) for r in recs]), "Accuracy": np.array([_truthy(r["answer_correct"]) for r in recs]),
                "Joint": np.array([_truthy(r[joint]) for r in recs])}
    q = REASONIF / "qwen3_14b"
    qwen = {"Base": load(q / "reasonif_qwen3_14b_base_untouched.jsonl", "instruction_following", "joint_success"),
            "SFT": load(q / "reasonif_qwen3_14b_sft_gpt52_high.jsonl", "instruction_following", "joint_success"),
            "Prefix-OFF": load(q / "reasonif_qwen3_14b_prefix_constraint_off.jsonl", "continuation_instruction_following", "continuation_joint_success"),
            "Prefix-ON": load(q / "reasonif_qwen3_14b_prefix_constraint_on.jsonl", "continuation_instruction_following", "continuation_joint_success")}
    p = REASONIF / "phi4_reasoning"
    phi = {"Base": load(p / "base" / "scored_responses.jsonl", "instruction_following", "joint_success"),
           "SFT": load(p / "sft" / "scored_responses.jsonl", "instruction_following", "joint_success")}
    rng = np.random.default_rng(SEED)
    rows = []
    for model, data, pairs in (("Qwen3-14B", qwen, [("SFT", "Base"), ("Prefix-OFF", "Base"), ("Prefix-ON", "Base"), ("Prefix-ON", "SFT"), ("Prefix-ON", "Prefix-OFF")]),
                               ("Phi-4-reasoning", phi, [("SFT", "Base")])):
        for x, y in pairs:
            for metric in ("IFS", "Accuracy", "Joint"):
                est, lo, hi, n10, n01, pv = _paired(data[x][metric], data[y][metric], rng)
                rows.append(dict(Model=model, Contrast=f"{x} minus {y}", Metric=metric, Estimate_pp=round(100 * est, 1),
                                 CI95_Low=round(100 * lo, 1), CI95_High=round(100 * hi, 1), Only_First_Succeeds=n10,
                                 Only_Second_Succeeds=n01, McNemar_Exact_P=round(pv, 4)))
    return pd.DataFrame(rows)


KL_DIR = REPO_ROOT / "results" / "kl_divergence"
KL_SETS = [("Qwen3-14B", "Haskins", "qwen3_14b_haskins_500"), ("Phi-4-reasoning", "Haskins", "phi4_haskins_500"),
           ("Qwen3-14B", "ReasonIF", "qwen3_14b_reasonif_300"), ("Phi-4-reasoning", "ReasonIF", "phi4_reasonif_300")]


def _kl_records(stem):
    return json.loads((KL_DIR / "caches_float32" / f"kl_cache_{stem}.json").read_text(encoding="utf-8"))["records"]


def table_kl_spike_frequency() -> pd.DataFrame:
    rows = []
    for model, bench, stem in KL_SETS:
        traces = [np.asarray(r["reasoning_kl"], dtype=np.float64) for r in _kl_records(stem).values() if len(r["reasoning_kl"]) > 10]
        down = np.concatenate([t[10:] for t in traces])
        any5 = sum(bool((t[10:] > 5).any()) for t in traces)
        rows.append(dict(Model=model, Benchmark=bench, Traces=len(traces), Downstream_Tokens=len(down),
                         Pct_Tokens_KL_gt_2=round(100 * float((down > 2).mean()), 3), Per_1000_Tokens_KL_gt_5=round(1000 * float((down > 5).mean()), 2),
                         Per_1000_Tokens_KL_gt_8=round(1000 * float((down > 8).mean()), 3), Traces_With_Any_KL_gt_5=any5,
                         Pct_Traces_With_Any_KL_gt_5=round(100 * any5 / len(traces), 1), Max_KL=round(float(down.max()), 2)))
    return pd.DataFrame(rows)


def table_kl_outcome_groups() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for model, bench, stem in KL_SETS:
        if bench != "Haskins":
            continue
        recs = _kl_records(stem)
        csv = pd.read_csv(KL_DIR / "tabular_results_csv" / f"per_sample_reasoning_kl_{stem}.csv")
        vals = []
        for r in csv.itertuples():
            rec = recs.get(str(r.key))
            if rec is None or len(rec["reasoning_kl"]) <= 10:
                continue
            vals.append((r.q_code, float(np.max(rec["reasoning_kl"][10:]))))
        g = pd.DataFrame(vals, columns=["grp", "maxkl"])
        for a_label, b_label in (("G1", "G3"), ("G1", "G2")):
            a, b = g[g.grp == a_label].maxkl.to_numpy(), g[g.grp == b_label].maxkl.to_numpy()
            if len(a) < 3 or len(b) < 3:
                continue
            boots = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(5000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append(dict(Model=model, Benchmark=bench, Contrast=f"{a_label} (base fails, SFT passes) minus {b_label}",
                             N_First=len(a), N_Second=len(b), Mean_Max_KL_First=round(float(a.mean()), 2), Mean_Max_KL_Second=round(float(b.mean()), 2),
                             Difference=round(float(a.mean() - b.mean()), 2), CI95_Low=round(float(lo), 2), CI95_High=round(float(hi), 2)))
    return pd.DataFrame(rows)


def analysis_tables() -> dict:
    L = load_long()
    content, per_task = table_prefix_content(L)
    return {
        "table8_window_matched_haskins.csv": table_window_matched(L),
        "table9_prefix_content_haskins.csv": content,
        "table9b_prefix_content_per_task.csv": per_task,
        "table10_prefix_conditional_haskins.csv": table_prefix_conditional(L),
        "table11_prefix_examples.csv": table_prefix_examples(L),
        "table12_paired_contrasts_haskins.csv": table_paired_haskins(L),
        "table13_paired_contrasts_reasonif.csv": table_paired_reasonif(),
        "table14_kl_spike_frequency.csv": table_kl_spike_frequency(),
        "table15_kl_outcome_groups.csv": table_kl_outcome_groups(),
    }


if __name__ == "__main__":
    for name, df in analysis_tables().items():
        print("\n== " + name + "\n" + df.to_string(index=False))
