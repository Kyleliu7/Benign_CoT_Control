"""
Recompute every manuscript table from the raw run files and compare with results/summary_tables/.

Nothing is read from a hard-coded constant: each number below is derived from

  results/haskins_500/**/results.csv, records/, phi4_haskins_500_bundle.json   (Haskins)
  results/reasonif_300/**/*.jsonl, overall.csv                                  (ReasonIF)
  results/kl_divergence/caches_float32/*.json, spike_analysis/*.json            (KL)

Usage
  python scripts/reproduce_tables.py            # recompute, compare with committed CSVs, exit 1 on mismatch
  python scripts/reproduce_tables.py --write    # (re)write the CSVs in results/summary_tables/
  python scripts/reproduce_tables.py --rescore-reasonif   # also re-run the official ReasonIF grader

Rows whose raw records are not in this repository are emitted with Status=UNAVAILABLE and empty numbers
instead of a value copied from elsewhere. A missing value is never a zero.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
TABLES = RESULTS / "summary_tables"
HASKINS = RESULTS / "haskins_500"
REASONIF = RESULTS / "reasonif_300"
KL = RESULTS / "kl_divergence"

NON_CHAR_7 = ["third_person", "arrow_prefix", "word_suppression", "multiple_word_suppression",
              "end_of_sentence", "meow_between_words", "repeat_sentences"]
ALL_10 = NON_CHAR_7 + ["alternating_case", "lowercase_thinking", "uppercase_thinking"]
QWEN_RUNS = {  # pairing label -> run-directory tag
    "Base -> Base (A2)": "2x2-base-to-base-on-10tok",
    "Base -> SFT (A3)": "2x2-base-to-sft-on-10tok",
    "SFT -> Base (A1 / Prefix-ON)": "2x2-sft-to-base-on-10tok",
    "SFT -> SFT (A4)": "2x2-sft-to-sft-on-10tok",
}
PHI_CONDS = {
    "Base -> Base (A2)": "a2", "Base -> SFT (A3)": "a3", "SFT -> Base (A1 / Prefix-ON)": "a1",
    "SFT -> SFT (A4)": "a4", "SFT -> Base OFF (A5)": "a5",
}
NA = ""


def truthy(v) -> bool:
    return v is True or str(v) == "True"


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------------------------
# Table 1: ReasonIF
# --------------------------------------------------------------------------------------------
def reasonif_row(model, condition, recs, ifs_key, joint_key, source):
    n = len(recs)
    ifs = sum(truthy(r[ifs_key]) for r in recs)
    acc = sum(truthy(r["answer_correct"]) for r in recs)
    joint = sum(truthy(r[joint_key]) for r in recs)
    toks = [float(r["output_tokens"]) for r in recs if r.get("output_tokens") not in (None, "")]
    trunc = sum(truthy(r.get("truncated")) for r in recs)
    prefix_lens = [len(r["forced_prefix_token_ids"]) for r in recs] if "forced_prefix_token_ids" in recs[0] else []
    return dict(Model=model, Condition=condition, Status="OK", N=n,
                IFS_Count=ifs, IFS_Pct=round(100 * ifs / n, 1), Acc_Count=acc, Acc_Pct=round(100 * acc / n, 1),
                Joint_Count=joint, Joint_Pct=round(100 * joint / n, 1),
                Mean_Tokens=round(float(np.mean(toks)), 1), Truncated=trunc,
                Prefix_Shorter_Than_10=sum(1 for l in prefix_lens if l < 10) if prefix_lens else NA,
                Source=source)


def phi_prefix_row(condition, folder):
    df = pd.read_csv(REASONIF / "phi4_reasoning" / folder / "overall.csv").iloc[0]
    n = int(df["questions"])
    ifs, acc = round(df["continuation_instruction_following"] * n), round(df["answer_correct"] * n)
    joint, trunc = round(df["continuation_joint_success"] * n), round(df["truncated"] * n)
    return dict(Model="Phi-4-reasoning", Condition=condition, Status="OK (aggregate CSV only; raw records absent)", N=n,
                IFS_Count=ifs, IFS_Pct=round(100 * ifs / n, 1), Acc_Count=acc, Acc_Pct=round(100 * acc / n, 1),
                Joint_Count=joint, Joint_Pct=round(100 * joint / n, 1),
                Mean_Tokens=round(float(df["mean_output_tokens"]), 1), Truncated=trunc,
                Prefix_Shorter_Than_10=NA, Source=f"phi4_reasoning/{folder}/overall.csv")


def table1():
    q = REASONIF / "qwen3_14b"
    rows = [reasonif_row("Qwen3-14B", "Base (Untouched)", load_jsonl(q / "reasonif_qwen3_14b_base_untouched.jsonl"),
                         "instruction_following", "joint_success", "reasonif_qwen3_14b_base_untouched.jsonl")]
    rows.append(reasonif_row("Qwen3-14B", "SFT (gpt52-high)", load_jsonl(q / "reasonif_qwen3_14b_sft_gpt52_high.jsonl"),
                             "instruction_following", "joint_success", "reasonif_qwen3_14b_sft_gpt52_high.jsonl"))
    rows.append(reasonif_row("Qwen3-14B", "Prefix-OFF (continuation-scored)", load_jsonl(q / "reasonif_qwen3_14b_prefix_constraint_off.jsonl"),
                             "continuation_instruction_following", "continuation_joint_success", "reasonif_qwen3_14b_prefix_constraint_off.jsonl"))
    rows.append(reasonif_row("Qwen3-14B", "Prefix-ON (continuation-scored)", load_jsonl(q / "reasonif_qwen3_14b_prefix_constraint_on.jsonl"),
                             "continuation_instruction_following", "continuation_joint_success", "reasonif_qwen3_14b_prefix_constraint_on.jsonl"))
    p = REASONIF / "phi4_reasoning"
    rows.append(reasonif_row("Phi-4-reasoning", "Base (Untouched)", load_jsonl(p / "base" / "scored_responses.jsonl"),
                             "instruction_following", "joint_success", "phi4_reasoning/base/scored_responses.jsonl"))
    rows.append(reasonif_row("Phi-4-reasoning", "SFT (gpt52-high)", load_jsonl(p / "sft" / "scored_responses.jsonl"),
                             "instruction_following", "joint_success", "phi4_reasoning/sft/scored_responses.jsonl"))
    rows.append(phi_prefix_row("Prefix-OFF (continuation-scored)", "prefix_off"))
    rows.append(phi_prefix_row("Prefix-ON (continuation-scored)", "prefix_on"))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Haskins helpers
# --------------------------------------------------------------------------------------------
def agg(df, comp_col, strict_col):
    nc = df[df.task.isin(NON_CHAR_7)]
    return dict(All_Mean=round(100 * df[comp_col].mean(), 2), All_Strict_Count=int(df[strict_col].sum()), All_N=len(df),
                NonChar_Mean=round(100 * nc[comp_col].mean(), 2), NonChar_Strict_Count=int(nc[strict_col].sum()), NonChar_N=len(nc))


def qwen_frames():
    out = {}
    for label, tag in QWEN_RUNS.items():
        path = glob.glob(str(HASKINS / "qwen3_14b" / f"*{tag}__*" / "results.csv"))
        assert len(path) == 1, (label, path)
        out[label] = pd.read_csv(path[0])
    return out


def phi_bundle():
    return json.loads((HASKINS / "phi4_reasoning" / "phi4_haskins_500_bundle.json").read_text(encoding="utf-8"))


def phi_standalone():
    out = {}
    for run in sorted((HASKINS / "phi4_reasoning" / "extracted_runs").glob("*/results.csv")):
        out["SFT" if "gpt52" in run.parent.name else "Base"] = pd.read_csv(run)
    return out


QWEN_Q = HASKINS / "qwen3_14b"


def _jsonl_df(name: str) -> pd.DataFrame:
    return pd.DataFrame(load_jsonl(QWEN_Q / name))


def _qwen_standalone_rows():
    """Qwen3-14B standalone / prefix runs from the calibrated vLLM protocol (haskins-vllm-offline-v3-calibrated).

    Base and SFT have one scoring basis (the whole reasoning). The prefix runs store BOTH a continuation-only and a
    full-trace (prefix + continuation) score; the manuscript's Prefix-OFF row used the continuation score and its
    Prefix-ON row used the full-trace score, so both are listed for both."""
    specs = [("Base", "haskins_qwen3_14b_vllm_calibrated_base_17pct.jsonl", [("whole trace", "upstream_compliance")]),
             ("SFT", "haskins_qwen3_14b_vllm_calibrated_sft_31pct.jsonl", [("whole trace", "upstream_compliance")]),
             ("Prefix-OFF (SFT donor, no constraint)", "haskins_qwen3_14b_vllm_calibrated_prefix_off_21pct.jsonl",
              [("continuation only", "continuation_compliance"), ("full trace incl. prefix", "full_compliance")]),
             ("Prefix-ON (SFT donor, constraint)", "haskins_qwen3_14b_vllm_calibrated_prefix_on_37pct.jsonl",
              [("continuation only", "continuation_compliance"), ("full trace incl. prefix", "full_compliance")])]
    rows = []
    for cond, fname, scorings in specs:
        df = _jsonl_df(fname)
        df["prompt_idx"] = df["prompt_idx"].astype(int)
        for basis, col in scorings:
            df["_c"] = df[col].astype(float)
            df["_s"] = (df["_c"] == 1.0).astype(int)
            a = agg(df, "_c", "_s")
            rows.append(dict(Model="Qwen3-14B", Condition=cond, Scoring=basis, Status="OK", Mean_Compliance=a["All_Mean"],
                             Strict_Count=a["All_Strict_Count"], N=a["All_N"], NonChar_Mean=a["NonChar_Mean"],
                             NonChar_Strict_Count=a["NonChar_Strict_Count"],
                             Hit_Token_Limit=int(df.hit_token_limit.map(truthy).sum())))
    return rows


def table2():
    rows = _qwen_standalone_rows()
    for cond, df in phi_standalone().items():
        a = agg(df, "upstream_compliance", "score_equals_one")
        rows.append(dict(Model="Phi-4-reasoning", Condition=cond, Scoring="whole trace", Status="OK", Mean_Compliance=a["All_Mean"],
                         Strict_Count=a["All_Strict_Count"], N=a["All_N"], NonChar_Mean=a["NonChar_Mean"],
                         NonChar_Strict_Count=a["NonChar_Strict_Count"], Hit_Token_Limit=int(df.hit_token_limit.sum())))
    return pd.DataFrame(rows)


def table3():
    rows = []
    for label, df in qwen_frames().items():
        c = agg(df, "continuation_compliance", "continuation_score_one")
        f = agg(df, "full_compliance", "full_score_one")
        rows.append(dict(Model="Qwen3-14B", Pairing=label, Status="OK", **{f"Cont_{k}": v for k, v in c.items()},
                         Full_All_Mean=f["All_Mean"], Full_All_Strict_Count=f["All_Strict_Count"]))
    for label, fname in [("SFT -> Base OFF (A5; calibrated vLLM run)", "haskins_qwen3_14b_vllm_calibrated_prefix_off_21pct.jsonl"),
                         ("SFT -> Base (A1 rerun; calibrated vLLM run)", "haskins_qwen3_14b_vllm_calibrated_prefix_on_37pct.jsonl")]:
        df = _jsonl_df(fname)
        df["prompt_idx"] = df["prompt_idx"].astype(int)
        for col in ("continuation_compliance", "full_compliance"):
            df[col] = df[col].astype(float)
        df["_cs"] = (df["continuation_compliance"] == 1.0).astype(int)
        df["_fs"] = (df["full_compliance"] == 1.0).astype(int)
        c = agg(df, "continuation_compliance", "_cs")
        f = agg(df, "full_compliance", "_fs")
        rows.append(dict(Model="Qwen3-14B", Pairing=label, Status="OK", **{f"Cont_{k}": v for k, v in c.items()},
                         Full_All_Mean=f["All_Mean"], Full_All_Strict_Count=f["All_Strict_Count"]))
    bundle = phi_bundle()
    for label, key in PHI_CONDS.items():
        aggs = {a["aggregate"]: a for a in bundle["stats"][key]["aggregates"]}
        a10, a7 = aggs["all_10"], aggs["seven_non_character"]
        rows.append(dict(
            Model="Phi-4-reasoning", Pairing=label, Status="OK",
            Cont_All_Mean=round(100 * a10["continuation_mean_compliance_all_scheduled"], 2),
            Cont_All_Strict_Count=round(500 * a10["continuation_strict_pass_rate_all_scheduled"]), Cont_All_N=500,
            Cont_NonChar_Mean=round(100 * a7["continuation_mean_compliance_all_scheduled"], 2),
            Cont_NonChar_Strict_Count=round(350 * a7["continuation_strict_pass_rate_all_scheduled"]), Cont_NonChar_N=350,
            Full_All_Mean=round(100 * a10["full_mean_compliance_all_scheduled"], 2),
            Full_All_Strict_Count=round(500 * a10["full_strict_pass_rate_all_scheduled"])))
    return pd.DataFrame(rows)


def table6_per_task():
    rows = []
    for label, df in qwen_frames().items():
        for task in ALL_10:
            d = df[df.task == task]
            rows.append(dict(Model="Qwen3-14B", Pairing=label, Task=task, N=len(d),
                             Cont_Mean=round(100 * d.continuation_compliance.mean(), 2), Cont_Strict_Count=int(d.continuation_score_one.sum())))
    bundle = phi_bundle()
    for label, key in PHI_CONDS.items():
        for task in ALL_10:
            t = bundle["stats"][key]["tasks"][task]
            rows.append(dict(Model="Phi-4-reasoning", Pairing=label, Task=task, N=t["planned"],
                             Cont_Mean=round(100 * t["mean_continuation_compliance_all_scheduled"], 2),
                             Cont_Strict_Count=round(t["planned"] * t["rate_continuation_strict_all_scheduled"])))
    return pd.DataFrame(rows)


def table7_run_health():
    rows = []
    for label, df in qwen_frames().items():
        rows.append(dict(Model="Qwen3-14B", Run=label, N=len(df), Hit_3000_Token_Limit=int(df.hit_token_limit.map(truthy).sum()),
                         Unfinished_Reasoning=int((df.extraction_status == "unfinished_reasoning").sum()),
                         Mean_Continuation_Tokens=round(df.continuation_tokens.mean(), 1)))
    for cond, df in phi_standalone().items():
        rows.append(dict(Model="Phi-4-reasoning", Run=f"standalone {cond}", N=len(df),
                         Hit_3000_Token_Limit=int(df.hit_token_limit.map(truthy).sum()),
                         Unfinished_Reasoning=int((df.extraction_status == "unfinished_reasoning").sum()),
                         Mean_Continuation_Tokens=round(df.generated_tokens.mean(), 1)))
    return pd.DataFrame(rows)


def table8_legacy_records():
    """Older-protocol Qwen records (different schema, SFT labelled model=gpt52_short). Their stored scores do not
    reproduce with the pinned scorer, so they are listed for transparency only."""
    sys.path.insert(0, str(REPO_ROOT))
    from evaluators.haskins_evaluator import grade_paper_protocol
    specs = [("standalone_base", "haskins_qwen3_14b_standalone_base.jsonl", "compliance", "compliant_binary", "extracted_reasoning"),
             ("standalone_sft", "haskins_qwen3_14b_standalone_sft.jsonl", "compliance", "compliant_binary", "extracted_reasoning"),
             ("sft_donor_to_base_on_a1 (continuation)", "haskins_qwen3_14b_sft_donor_to_base_on_a1.jsonl", "continuation_compliance", "continuation_binary", "continuation_reasoning"),
             ("sft_donor_to_base_off_a5 (continuation)", "haskins_qwen3_14b_sft_donor_to_base_off_a5.jsonl", "continuation_compliance", "continuation_binary", "continuation_reasoning")]
    rows = []
    for label, fname, comp, strict, text in specs:
        recs = load_jsonl(QWEN_Q / fname)
        stored = np.array([float(r[comp]) for r in recs])
        stored_s = int(sum(int(float(r[strict])) for r in recs))
        new = np.array([grade_paper_protocol(r["task"], int(r["prompt_idx"]), r[text] or "") for r in recs])
        labels = sorted({str(r.get("model", r.get("condition", ""))) for r in recs})
        rows.append(dict(Records=label, Model_Label="/".join(labels), N=len(recs), Stored_Mean=round(100 * stored.mean(), 2),
                         Stored_Strict=stored_s, Pinned_Scorer_Mean=round(100 * new.mean(), 2), Pinned_Scorer_Strict=int((new == 1.0).sum()),
                         Rows_Disagreeing=int((np.abs(new - stored) > 1e-6).sum()),
                         Status="NOT reproducible with the pinned scorer; do not cite"))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Table 4: KL (both windows, stated explicitly)
# --------------------------------------------------------------------------------------------
KL_CACHES = [("Qwen Haskins", "qwen3_14b_haskins_500"), ("Phi-4 Haskins", "phi4_haskins_500"),
             ("Qwen ReasonIF", "qwen3_14b_reasonif_300"), ("Phi-4 ReasonIF", "phi4_reasonif_300")]


def table4():
    rows = []
    for name, stem in KL_CACHES:
        data = json.loads((KL / "caches_float32" / f"kl_cache_{stem}.json").read_text(encoding="utf-8"))
        traces = [np.asarray(r["reasoning_kl"], dtype=np.float64) for r in data["records"].values() if r.get("reasoning_kl")]
        cohort = [t for t in traces if len(t) >= 100]
        first100_share = 100 * sum(t[:10].sum() for t in cohort) / sum(t[:100].sum() for t in cohort)
        full_share = 100 * sum(t[:10].sum() for t in cohort) / sum(t.sum() for t in cohort)
        down = np.concatenate([t[10:] for t in traces if len(t) > 10])
        down100 = np.concatenate([t[10:100] for t in cohort])
        seq_max = [t[10:].max() for t in traces if len(t) > 10]
        p = np.percentile(down, [50, 90, 95, 99, 99.9])
        rows.append(dict(Model_Benchmark=name, Traces=len(traces), Cohort_L_ge_100=len(cohort),
                         Early_Share_Pct_First100_Cohort=round(first100_share, 2),
                         Early_Share_Pct_Full512_Cohort=round(full_share, 2),
                         Mean_KL_t1=round(float(np.mean([t[0] for t in traces])), 2),
                         Tokens_t_gt_10_Full512=len(down), Mean=round(float(down.mean()), 3), P50=round(p[0], 3),
                         P90=round(p[1], 3), P95=round(p[2], 3), P99=round(p[3], 3), P99_9=round(p[4], 3),
                         Max_Spike=round(float(down.max()), 2), Seq_Max_P95=round(float(np.percentile(seq_max, 95)), 2),
                         Tokens_t11_to_100_Cohort=len(down100), Mean_t11_to_100_Cohort=round(float(down100.mean()), 3),
                         Max_Spike_t11_to_100_Cohort=round(float(down100.max()), 2)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Table 5: strongest recorded spike per Haskins task (Phi-4), straight from the spike records
# --------------------------------------------------------------------------------------------
def table5():
    spikes = json.loads((KL / "spike_analysis" / "spike_token_pairs.json").read_text(encoding="utf-8"))
    best = {}
    for s in spikes:
        if s["task"] not in best or s["kl"] > best[s["task"]]["kl"]:
            best[s["task"]] = s
    rows = []
    for task in ALL_10:
        s = best[task]
        rows.append(dict(Task=f"{task}:{s['prompt_idx']}", Pos_t=s["t"], KL_nats=round(s["kl"], 2),
                         Preceding_Context=s["context"], Base_Token=repr(s["base_token"]), SFT_Token=repr(s["sft_token"]),
                         Base_Run_Strict=bool(s["base_strict"]), SFT_Run_Strict=bool(s["sft_strict"])))
    return pd.DataFrame(rows).sort_values("KL_nats", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------------
def rescore_reasonif() -> int:
    """Re-run the pinned official ReasonIF grader and compare with stored flags (needs fast-langdetect model)."""
    sys.path.insert(0, str(REPO_ROOT))
    from evaluators.reasonif_evaluator import evaluate_reasonif_record, LANGDETECT_FAILURES
    bad = 0
    language_dependent = {"language:reasoning_language", "change_case:english_capital"}
    files = {**{f"qwen/{p.name}": p for p in (REASONIF / "qwen3_14b").glob("*.jsonl")},
             "phi/base": REASONIF / "phi4_reasoning/base/scored_responses.jsonl",
             "phi/sft": REASONIF / "phi4_reasoning/sft/scored_responses.jsonl"}
    for name, path in files.items():
        recs = load_jsonl(path)
        new = [evaluate_reasonif_record(r) for r in recs]
        flag = "official_instruction_following" if "official_instruction_following" in recs[0] else "instruction_following"
        degraded = bool(LANGDETECT_FAILURES)
        pairs = [(a, r) for a, r in zip(new, recs) if not (degraded and a["constraint"] in language_dependent)]
        diff = sum(1 for a, r in pairs if a["instruction_following"] != truthy(r[flag]))
        note = f" (language/english_capital items excluded: {len(recs) - len(pairs)})" if degraded else ""
        print(f"  {name:<70s} IFS flag mismatches: {diff}/{len(pairs)}{note}")
        bad += diff
    if LANGDETECT_FAILURES:
        print(f"  WARNING: language detection failed {len(LANGDETECT_FAILURES)} times (fast-langdetect model unavailable); "
              "language/english_capital items could not be re-scored and were excluded. Re-run with the model available "
              "to verify those two constraints.")
    return bad


def _norm(value) -> str:
    """Canonical cell text so 25, 25.0 and '25' compare equal and NaN equals ''."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value) == "nan":
        return ""
    try:
        return format(float(value), ".10g")
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write recomputed CSVs to results/summary_tables/")
    ap.add_argument("--rescore-reasonif", action="store_true", help="re-run the official ReasonIF grader on the raw records")
    args = ap.parse_args()

    tables = {
        "table1_reasonif_overall.csv": table1(),
        "table2_haskins_standalone.csv": table2(),
        "table3_haskins_crossed_2x2.csv": table3(),
        "table4_kl_percentiles.csv": table4(),
        "table5_top_spike_per_task_phi4.csv": table5(),
        "table6_haskins_per_task.csv": table6_per_task(),
        "table7_haskins_run_health.csv": table7_run_health(),
        "table8_legacy_qwen_haskins_records.csv": table8_legacy_records(),
    }
    mismatches = 0
    for name, df in tables.items():
        print("\n" + "=" * 100 + f"\n {name}\n" + "=" * 100)
        print(df.to_string(index=False))
        target = TABLES / name
        if args.write:
            TABLES.mkdir(parents=True, exist_ok=True)
            df.to_csv(target, index=False)
        elif not target.exists():
            print(f"[MISSING] {target.name} is not committed (run with --write)")
            mismatches += 1
        else:
            old = pd.read_csv(target, dtype=str, keep_default_na=False)
            same = old.shape == df.shape and list(old.columns) == list(df.columns) and all(
                _norm(a) == _norm(b) for a, b in zip(old.values.ravel(), df.values.ravel()))
            print(f"[{'MATCH' if same else 'MISMATCH'}] committed {target.name} vs recomputed")
            mismatches += 0 if same else 1
    if args.write:
        print(f"\nWrote {len(tables)} tables to {TABLES}")
    if args.rescore_reasonif:
        print("\nRe-scoring ReasonIF with the official grader ...")
        mismatches += rescore_reasonif()
    print("\n" + ("ALL TABLES MATCH RAW DATA." if mismatches == 0 else f"{mismatches} TABLE/RESCORE MISMATCH(ES)."))
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
