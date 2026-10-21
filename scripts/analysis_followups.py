"""
Follow-up analyses added on 2026-10-10/11 (all from raw records in results/):

  table26_training_seeds.csv        Qwen3-14B SFT with training seeds 42 (published adapter), 1 and 2: Haskins standalone and ReasonIF,
                                    each compared with the same Base run (paired cluster bootstrap over questions; first-L-words windows).
                                    Only the training seed differs; the validation split (seed 42), hyperparameters, checkpoint rule
                                    (lowest validation loss) and evaluation sampling seeds are identical.
  table27_olmo_haskins_per_task.csv OLMo-3-7B-Think Base, SFT (lr 1e-4, same recipe as Qwen) and SFT (lr 2e-4, chosen after seeing the
                                    1e-4 result): Haskins mean and strict count per task.
  table28_olmo_summary.csv          OLMo summary: Haskins mean, strict, degenerate traces, output length and opening style, gains vs Base
                                    (full trace, degenerate traces scored 0, first-L-words); ReasonIF for Base and SFT lr 1e-4 (the
                                    lr 2e-4 adapter was not run on ReasonIF).
  table29_haskins_degenerate.csv    Standalone Haskins runs of all models: strict passes whose reasoning is degenerate (fewer than 20 words,
                                    or, for meow_between_words, more than 90% of words are "meow"), and the mean if degenerate traces
                                    are scored 0. The rule was written after seeing OLMo outputs; it is a post-hoc sensitivity check.
  table30_reasonif_excluded.csv     ReasonIF IFS with and without the passes listed as no_reasoning / incoherent in
                                    data/manual_review/reasonif_pass_review.json (manual review; grader output unchanged).
  table31_qwen_training_variants.csv ReasonIF for the other Qwen3-14B fine-tunes (data/training_variants.json): loss on reasoning only or
                                    answer only, plain-prose rewrite, other reasoning sources, answers only, and two excluded runs.

Scores use the paper scorer and the original paper's grading rule (binary for word suppression, multiple word suppression,
end-of-sentence and meow; fractional otherwise). Missing runs are skipped, so the tables are empty until the runs are added.
"""
import glob
import json
import re
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_prefix_length_paired as A  # noqa: E402
from analysis_prefix_length_paired import grade_paper_protocol, primary_score, truncate_words  # noqa: E402

RES = A.REPO_ROOT / "results"
N_BOOT, SEED = 10_000, 42
CASE_TASKS = ["lowercase_thinking", "uppercase_thinking", "alternating_case"]
REVIEW = A.REPO_ROOT / "data" / "manual_review" / "reasonif_pass_review.json"


def _jl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split(chr(10)) if l.strip()]


def _first(pattern):
    hits = sorted(glob.glob(str(pattern)))
    return hits[0] if hits else None


def degenerate(task: str, text: str) -> bool:
    w = str(text).split()
    if len(w) < 20:
        return True
    return task == "meow_between_words" and sum(x.strip(".,!").lower() == "meow" for x in w) / len(w) > 0.9


def _prep(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["reasoning"] = d.reasoning.fillna("").astype(str)
    d["q"] = d.prompt_idx.astype(int)
    d["upstream"] = d.upstream_compliance.astype(float)
    d["primary"] = [primary_score(t, c) for t, c in zip(d.task, d.upstream)]
    d["strict"] = d.upstream == 1.0
    d["degen"] = [degenerate(t, x) for t, x in zip(d.task, d.reasoning)]
    d["words"] = d.reasoning.str.split().str.len()
    if "hit_token_limit" in d:
        d["capped"] = d.hit_token_limit.astype(str).str.lower().eq("true")
    return d[[c for c in ["task", "q", "reasoning", "upstream", "primary", "strict", "degen", "words", "capped"] if c in d]]


def haskins_runs() -> dict:
    """{label: DataFrame} for every standalone Haskins run that is present."""
    q = RES / "haskins_500" / "qwen3_14b"
    out = {}
    for label, name in (("Qwen3-14B Base", "haskins_qwen3_14b_vllm_calibrated_base_17pct.jsonl"),
                        ("Qwen3-14B SFT seed 42", "haskins_qwen3_14b_vllm_calibrated_sft_31pct.jsonl")):
        out[label] = _prep(pd.DataFrame(_jl(q / name)))
    for s in (1, 2):
        f = _first(RES / "training_seeds" / f"seed{s}" / "haskins_500" / "qwen3_14b" / "*" / "results.csv")
        if f:
            out[f"Qwen3-14B SFT seed {s}"] = _prep(pd.read_csv(f))
    for run in sorted((RES / "haskins_500" / "phi4_reasoning" / "extracted_runs").glob("*/results.csv")):
        out["Phi-4-reasoning SFT" if "gpt52" in run.parent.name else "Phi-4-reasoning Base"] = _prep(pd.read_csv(run))
    o = RES / "olmo3_7b_think" / "haskins_500"
    for label, pat in (("OLMo-3-7B-Think Base", "allenai__Olmo-3-7B-Think__[0-9a-f]*"),
                       ("OLMo-3-7B-Think SFT lr1e-4", "*high-reasoning-original__*"),
                       ("OLMo-3-7B-Think SFT lr2e-4", "*high-reasoning-original-lr2e-4__*")):
        f = _first(o / pat / "results.csv")
        if f:
            out[label] = _prep(pd.read_csv(f))
    return out


def reasonif_runs() -> dict:
    """{label: (review_key, records)} for every ReasonIF run that is present."""
    r = RES / "reasonif_300"
    out = {"Qwen3-14B Base": ("qwen3_14b/base", _jl(r / "qwen3_14b" / "reasonif_qwen3_14b_base_untouched.jsonl")),
           "Qwen3-14B SFT seed 42": ("qwen3_14b/seed42", _jl(r / "qwen3_14b" / "reasonif_qwen3_14b_sft_gpt52_high.jsonl"))}
    for s in (1, 2):
        f = _first(RES / "training_seeds" / f"seed{s}" / "reasonif_300" / "qwen3_14b" / "*" / "scored_responses.jsonl")
        if f:
            out[f"Qwen3-14B SFT seed {s}"] = (f"qwen3_14b/seed{s}", _jl(f))
    out["Phi-4-reasoning Base"] = ("phi4_reasoning/base", _jl(r / "phi4_reasoning" / "base" / "scored_responses.jsonl"))
    out["Phi-4-reasoning SFT"] = ("phi4_reasoning/sft", _jl(r / "phi4_reasoning" / "sft" / "scored_responses.jsonl"))
    for label, key, pat in (("OLMo-3-7B-Think Base", "olmo3_7b_think/base", "full_*"),
                            ("OLMo-3-7B-Think SFT lr1e-4", "olmo3_7b_think/sft_lr1e-4", "lora_*")):
        f = _first(RES / "olmo3_7b_think" / "reasonif_300" / pat / "scored_responses.jsonl")
        if f:
            out[label] = (key, _jl(f))
    return out


def _ci(values, rng=None):
    # a fresh generator per interval, so the same paired differences always give the same interval in every table
    v = np.asarray(values, dtype=float)
    b = v[np.random.default_rng(SEED).integers(0, len(v), (N_BOOT, len(v)))].mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _paired_gain(base: pd.DataFrame, sft: pd.DataFrame, rng, col="primary", window=None, tasks=None):
    m = base.merge(sft, on=["task", "q"], suffixes=("_b", "_s"))
    if tasks:
        m = m[m.task.isin(tasks)]
    if window:
        m = m[(m.words_b >= window) & (m.words_s >= window)]
        sb = np.array([primary_score(t, grade_paper_protocol(t, q, truncate_words(x, window))) for t, q, x in zip(m.task, m.q, m.reasoning_b)])
        ss = np.array([primary_score(t, grade_paper_protocol(t, q, truncate_words(x, window))) for t, q, x in zip(m.task, m.q, m.reasoning_s)])
    else:
        sb, ss = m[f"{col}_b"].to_numpy(dtype=float), m[f"{col}_s"].to_numpy(dtype=float)
    lo, hi = _ci(pd.Series(ss - sb).groupby(m.q.values).mean(), rng)
    return len(m), 100 * sb.mean(), 100 * ss.mean(), 100 * (ss - sb).mean(), 100 * lo, 100 * hi


def _r2(x):
    return round(float(x), 2)


def table_training_seeds(H: dict, R: dict) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    base_h = H["Qwen3-14B Base"]
    base_r = {r["dataset_index"]: r for r in R["Qwen3-14B Base"][1]}
    for label in ["Qwen3-14B Base"] + [k for k in H if k.startswith("Qwen3-14B SFT")]:
        d = H[label]
        row = dict(Condition=label.replace("Qwen3-14B ", ""), Haskins_Mean=_r2(100 * d.primary.mean()),
                   Haskins_Mean_Fractional=_r2(100 * d.upstream.mean()), Haskins_Strict=int(d.strict.sum()))
        if label != "Qwen3-14B Base":
            for tag, kw in (("Full", {}), ("First100", {"window": 100}), ("First200", {"window": 200})):
                n, _, _, g, lo, hi = _paired_gain(base_h, d, rng, **kw)
                row.update({f"Haskins_Gain_{tag}": _r2(g), f"Haskins_Gain_{tag}_CI": f"[{lo:.2f}, {hi:.2f}]", f"Haskins_N_{tag}": n})
        recs = R.get(label, (None, None))[1]
        if recs:
            n = len(recs)
            ifs = np.array([int(bool(r["instruction_following"])) for r in recs])
            row.update(ReasonIF_IFS=int(ifs.sum()), ReasonIF_IFS_Pct=_r2(100 * ifs.mean()),
                       ReasonIF_Accuracy_Pct=_r2(100 * np.mean([bool(r["answer_correct"]) for r in recs])),
                       ReasonIF_Joint=int(sum(bool(r["joint_success"]) for r in recs)),
                       ReasonIF_Mean_Tokens=round(float(np.mean([r["output_tokens"] for r in recs])), 1))
            if label != "Qwen3-14B Base":
                diff = np.array([int(bool(r["instruction_following"])) - int(bool(base_r[r["dataset_index"]]["instruction_following"])) for r in recs])
                lo, hi = _ci(diff, rng)
                row.update(ReasonIF_IFS_Gain=_r2(100 * diff.mean()), ReasonIF_IFS_Gain_CI=f"[{100*lo:.2f}, {100*hi:.2f}]")
        rows.append(row)
    return pd.DataFrame(rows)


OLMO = ["OLMo-3-7B-Think Base", "OLMo-3-7B-Think SFT lr1e-4", "OLMo-3-7B-Think SFT lr2e-4"]
OLMO_BASE = OLMO[0]


def table_olmo_per_task(H: dict) -> pd.DataFrame:
    runs = [k for k in OLMO if k in H]
    if not runs:
        return pd.DataFrame()
    rows = []
    for t in A.ALL_TASKS:
        row = dict(Task=t, Grading="binary" if t in ("word_suppression", "multiple_word_suppression", "end_of_sentence", "meow_between_words") else "fractional")
        for k in runs:
            g = H[k][H[k].task == t]
            tag = k.replace("OLMo-3-7B-Think ", "").replace(" ", "_")
            row[f"Mean_{tag}"] = _r2(100 * g.primary.mean())
            row[f"Strict_{tag}"] = int(g.strict.sum())
            row[f"Strict_Degenerate_{tag}"] = int((g.strict & g.degen).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _mcnemar(b01, b10):
    n, k = b01 + b10, min(b01, b10)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def table_olmo_summary(H: dict, R: dict) -> pd.DataFrame:
    runs = [k for k in OLMO if k in H]
    if not runs:
        return pd.DataFrame()
    rng = np.random.default_rng(SEED)
    base = H[OLMO[0]]
    rows = []
    for k in runs:
        d = H[k]
        start = d.reasoning.str.lstrip()
        row = dict(Run=k.replace("OLMo-3-7B-Think ", ""), Haskins_Mean=_r2(100 * d.primary.mean()), Haskins_Strict=int(d.strict.sum()),
                   Haskins_Strict_Degenerate=int((d.strict & d.degen).sum()), Haskins_Degenerate_Traces=int(d.degen.sum()),
                   Haskins_Mean_Degenerate_Scored_0=_r2(100 * d.primary.where(~d.degen, 0.0).mean()),
                   Haskins_Hit_Cap=int(d.capped.sum()) if "capped" in d else "", Haskins_Median_Words=float(d.words.median()),
                   Haskins_Starts_Bold_Heading=int(start.str.startswith("**").sum()),
                   Haskins_Starts_Okay=int(start.str.lower().str.startswith(("okay", "alright", "hmm")).sum()))
        if k != OLMO[0]:
            b = base.assign(clean=base.primary.where(~base.degen, 0.0))
            s = d.assign(clean=d.primary.where(~d.degen, 0.0))
            for tag, kw in (("Full", {}), ("Degenerate_Scored_0", {"col": "clean"}), ("First100_All", {"window": 100}),
                            ("First100_Case", {"window": 100, "tasks": CASE_TASKS}), ("First300_Case", {"window": 300, "tasks": CASE_TASKS})):
                n, _, _, g, lo, hi = _paired_gain(b, s, rng, **kw)
                row.update({f"Gain_{tag}": _r2(g), f"Gain_{tag}_CI": f"[{lo:.2f}, {hi:.2f}]", f"N_{tag}": n})
        if k in R:
            recs = R[k][1]
            row.update(ReasonIF_IFS=int(sum(bool(r["instruction_following"]) for r in recs)),
                       ReasonIF_Accuracy=int(sum(bool(r["answer_correct"]) for r in recs)),
                       ReasonIF_Joint=int(sum(bool(r["joint_success"]) for r in recs)),
                       ReasonIF_Mean_Tokens=round(float(np.mean([r["output_tokens"] for r in recs])), 1))
            if k != OLMO_BASE and OLMO_BASE in R:
                bb = {r["dataset_index"]: bool(r["instruction_following"]) for r in R[OLMO_BASE][1]}
                ss = {r["dataset_index"]: bool(r["instruction_following"]) for r in recs}
                b01 = sum(1 for i in bb if not bb[i] and ss[i])
                b10 = sum(1 for i in bb if bb[i] and not ss[i])
                row.update(ReasonIF_Fail_To_Pass=b01, ReasonIF_Pass_To_Fail=b10, ReasonIF_McNemar_P=round(_mcnemar(b01, b10), 4))
        rows.append(row)
    return pd.DataFrame(rows)



def table_haskins_degenerate(H: dict) -> pd.DataFrame:
    return pd.DataFrame([dict(Run=k, N=len(d), Mean=_r2(100 * d.primary.mean()), Strict=int(d.strict.sum()),
                              Strict_Degenerate=int((d.strict & d.degen).sum()), Degenerate_Traces=int(d.degen.sum()),
                              Mean_Degenerate_Scored_0=_r2(100 * d.primary.where(~d.degen, 0.0).mean())) for k, d in H.items()])


def table_reasonif_excluded(R: dict) -> pd.DataFrame:
    review = json.loads(REVIEW.read_text(encoding="utf-8"))["excluded"]
    base_of = {"Qwen3-14B": "Qwen3-14B Base", "Phi-4-reasoning": "Phi-4-reasoning Base", "OLMo-3-7B-Think": "OLMo-3-7B-Think Base"}
    ifs_clean = {}
    rows = []
    for label, (key, recs) in R.items():
        excl = {int(i) for i in review.get(key, {})}
        passes = {r["dataset_index"] for r in recs if r["instruction_following"]}
        assert excl <= passes, f"{key}: excluded items are not passes: {sorted(excl - passes)}"
        n = len(recs)
        ifs_clean[label] = (len(passes) - len(excl)) / n
        rows.append(dict(Run=label, N=n, IFS=len(passes), Excluded=len(excl), Excluded_Items=" ".join(str(i) for i in sorted(excl)),
                         IFS_Pct=_r2(100 * len(passes) / n), IFS_Pct_Excluding=_r2(100 * ifs_clean[label])))
    df = pd.DataFrame(rows)
    df["Gain_vs_Base_Excluding"] = [
        "" if r.Run in base_of.values() or base_of[r.Run.split(" ")[0]] not in ifs_clean
        else _r2(100 * (ifs_clean[r.Run] - ifs_clean[base_of[r.Run.split(" ")[0]]])) for r in df.itertuples()]
    return df


def table_training_variants(R: dict) -> pd.DataFrame:
    """ReasonIF for the other Qwen3-14B fine-tunes listed in data/training_variants.json (one run each)."""
    meta = json.loads((A.REPO_ROOT / "data" / "training_variants.json").read_text(encoding="utf-8"))["variants"]
    root = RES / "reasonif_300" / "qwen3_14b_training_variants"
    rng = np.random.default_rng(SEED)
    base = {r["dataset_index"]: bool(r["instruction_following"]) for r in R["Qwen3-14B Base"][1]}
    runs = [("reference", "Base (no fine-tuning)", "Qwen/Qwen3-14B", R["Qwen3-14B Base"][1], ""),
            ("reference", "Reported adapter: GPT-5.2 reasoning summaries, loss on reasoning and answer",
             "kyleliu789/qwen3-14b-gpt52-high-reasoning-original", R["Qwen3-14B SFT seed 42"][1], "")]
    for v in meta:
        f = root / v["folder"] / "scored_responses.jsonl"
        if f.exists():
            runs.append((v["group"], v["label"], v["adapter"], _jl(f), v.get("note", "")))
    rows = []
    for group, label, adapter, recs, note in runs:
        n = len(recs)
        ifs = np.array([int(bool(r["instruction_following"])) for r in recs])
        tok = [r.get("output_tokens", r.get("total_tokens")) for r in recs]
        row = dict(Group=group, Condition=label, Adapter=adapter, N=n, IFS=int(ifs.sum()), IFS_Pct=_r2(100 * ifs.mean()),
                   Accuracy_Pct=_r2(100 * np.mean([bool(r["answer_correct"]) for r in recs])),
                   Joint=int(sum(bool(r["joint_success"]) for r in recs)), Mean_Tokens=round(float(np.mean(tok)), 1),
                   Starts_Bold_Heading=int(sum(str(r.get("reasoning_content") or "").lstrip().startswith("**") for r in recs)), Note=note)
        if label.startswith("Base"):
            row.update(IFS_Gain_vs_Base="", IFS_Gain_CI="")
        else:
            diff = np.array([int(bool(r["instruction_following"])) - int(base[r["dataset_index"]]) for r in recs])
            lo, hi = _ci(diff, rng)
            row.update(IFS_Gain_vs_Base=_r2(100 * diff.mean()), IFS_Gain_CI=f"[{100*lo:.2f}, {100*hi:.2f}]")
        rows.append(row)
    return pd.DataFrame(rows)


def followup_tables() -> dict:
    H, R = haskins_runs(), reasonif_runs()
    tables = {"table26_training_seeds.csv": table_training_seeds(H, R),
              "table27_olmo_haskins_per_task.csv": table_olmo_per_task(H),
              "table28_olmo_summary.csv": table_olmo_summary(H, R),
              "table29_haskins_degenerate.csv": table_haskins_degenerate(H),
              "table30_reasonif_excluded.csv": table_reasonif_excluded(R),
              "table31_qwen_training_variants.csv": table_training_variants(R)}
    # integer counts stay integers; cells that do not apply (e.g. gains for the Base row) are empty strings
    return {k: v.convert_dtypes().astype(object).fillna("") for k, v in tables.items() if not v.empty}


if __name__ == "__main__":
    out_dir = A.REPO_ROOT / "results" / "summary_tables"
    for name, df in followup_tables().items():
        if "--write" in sys.argv:
            df.to_csv(out_dir / name, index=False)
        print(f"\n=== {name}\n{df.to_string(index=False)}")
