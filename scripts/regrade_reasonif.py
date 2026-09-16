#!/usr/bin/env python3
"""
Turnkey ReasonIF Regrading Program
Regrades any raw_responses.jsonl file using the paper-pinned ReasonIF benchmark evaluator (commit 706b953).
Outputs:
  - scored_responses.jsonl (row-level graded records)
  - overall.json & overall.csv (macro summary + Wilson 95% CIs)
  - by_constraint.csv (per-constraint compliance & accuracy)
  - by_source.csv (per-source benchmark dataset breakdown)

Usage:
  python regrade_reasonif.py
  python regrade_reasonif.py --input "path/to/raw_responses.jsonl"
  python regrade_reasonif.py --input "path/to/raw_responses.jsonl" --output-dir "path/to/output_dir"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ==============================================================================
# 1. Automatic ReasonIF Checker Library Discovery
# ==============================================================================
POSSIBLE_REASONIF_PATHS = [
    Path(r"C:\Users\bryan\OneDrive\Desktop\cot-controllability-deploy\evaluators"),
    Path(__file__).resolve().parent / "reasonIF" / "src",
    Path(__file__).resolve().parent / "reasonIF",
    Path(r"C:\Users\bryan\OneDrive\Documents\GitHub\CoT_Controllability\reasonIF\src"),
]

for p in POSSIBLE_REASONIF_PATHS:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import reasonif_official.instructions.instruction_checker as instruction_checker
    from reasonif_official.eval_utils import evaluate_instruction_following, extract_final_answer
except ImportError:
    try:
        import src.instructions.instruction_checker as instruction_checker
        from src.eval_utils import evaluate_instruction_following, extract_final_answer
    except ImportError as e:
        raise RuntimeError(
            f"Could not import ReasonIF evaluation libraries from any known path. "
            f"Searched: {POSSIBLE_REASONIF_PATHS}. Error: {e}"
        )

# Patch fast-langdetect compatibility
try:
    from fast_langdetect import detect as installed_language_detect

    def compatible_detect(text: str, low_memory: bool = False):
        try:
            result = installed_language_detect(text, low_memory=low_memory)
        except TypeError:
            result = installed_language_detect(text, model="lite" if low_memory else "full")
        if isinstance(result, list):
            if not result:
                raise ValueError("fast-langdetect returned no predictions")
            return result[0]
        return result

    instruction_checker.detect = compatible_detect
except ImportError:
    pass


# ==============================================================================
# 2. Helper Functions
# ==============================================================================
def split_reasoning_content(raw_output: str) -> Tuple[str, str]:
    """Splits raw output on </think> tag into reasoning trace and final answer content."""
    text = (raw_output or "").strip()
    if "</think>" in text:
        reasoning, content = text.split("</think>", 1)
        return reasoning.removeprefix("<think>").strip(), content.strip()
    if text.startswith("<think>"):
        return text[len("<think>"):].strip(), ""
    return text, ""


def canonical_answer(value: Any, source: str) -> str:
    """Normalizes predicted and ground truth answers for strict equivalence checks."""
    if value is None:
        return ""
    val_str = str(value).strip().replace(",", "")
    if source in {"arc", "gpqa"}:
        match = re.search(r"[ABCD]", val_str.upper())
        return match.group(0) if match else val_str.upper()
    try:
        num = float(val_str)
        return str(int(num)) if num.is_integer() else f"{num:.12g}"
    except ValueError:
        return val_str


# ==============================================================================
# 3. Core Regrading Engine
# ==============================================================================
def regrade_record(record: Dict[str, Any]) -> Dict[str, Any]:
    reasoning = record.get("reasoning_content")
    content = record.get("content")
    raw_output = record.get("raw_output", "")

    if reasoning is None or content is None:
        reasoning, content = split_reasoning_content(raw_output)

    c_names = record.get("constraint_name")
    if isinstance(c_names, str):
        c_names = [c_names]
    elif not c_names:
        c_names = [record.get("constraint", "unknown")]

    c_args = record.get("constraint_args")
    if isinstance(c_args, dict):
        c_args = [c_args]
    elif c_args is None:
        c_args = [{}]

    prompt = record.get("question") or record.get("prompt", "")
    source = record.get("source", "gsm8k")
    ground_truth = record.get("answer") or record.get("ground_truth", "")

    # 1. Instruction following evaluation
    follows = False
    if reasoning and reasoning.strip():
        following_list = evaluate_instruction_following(
            instruction_id_list=c_names,
            parameters=c_args,
            prompt=prompt,
            response=reasoning,
        )
        follows = all(following_list)

    # 2. Answer extraction and correctness evaluation
    predicted = extract_final_answer(content, source)
    canon_pred = canonical_answer(predicted, source)
    canon_gt = canonical_answer(ground_truth, source)
    correct = bool(canon_pred and canon_gt and (canon_pred == canon_gt))

    # 3. Format and structural validity
    has_answer_tags = "<answer>" in content and "</answer>" in content
    has_reasoning_end = "</think>" in raw_output
    format_valid = has_answer_tags and has_reasoning_end
    missing_answer = not has_answer_tags
    joint = bool(follows and correct)

    # Output tokens
    tokens = record.get("output_tokens", 0)
    if not tokens and raw_output:
        tokens = int(len(raw_output.split()) * 1.3)

    return {
        **record,
        "constraint": c_names[0] if c_names else "unknown",
        "predicted_answer": predicted,
        "instruction_following": bool(follows),
        "answer_correct": bool(correct),
        "joint_success": joint,
        "has_answer_tags": has_answer_tags,
        "has_reasoning_end": has_reasoning_end,
        "format_valid": format_valid,
        "missing_answer": missing_answer,
        "output_tokens": tokens,
    }


def regrade_reasonif_file(
    input_file: Path,
    output_dir: Optional[Path] = None,
    save_artifacts: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """Loads raw responses, evaluates all records, computes aggregate tables, and writes artifacts."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    target_dir = output_dir or input_file.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: line {line_num} could not be decoded: {e}")

    total_records = len(records)
    if total_records == 0:
        raise ValueError(f"No records found in {input_file}")

    print(f"Regrading {total_records} records from: {input_file}")
    scored_records = [regrade_record(r) for r in records]
    df = pd.DataFrame(scored_records)

    model_label = df["model_label"].iloc[0] if "model_label" in df.columns else "model"

    # Overall Summary Table
    metric_cols = [
        "instruction_following",
        "answer_correct",
        "joint_success",
        "format_valid",
        "missing_answer",
        "truncated",
    ]
    avail_metric_cols = [c for c in metric_cols if c in df.columns]

    overall_row = {
        "model": model_label,
        "questions": total_records,
        **{c: float(df[c].mean()) for c in avail_metric_cols},
        "mean_output_tokens": float(df["output_tokens"].mean()),
        "median_output_tokens": float(df["output_tokens"].median()),
    }
    overall_df = pd.DataFrame([overall_row])

    # Per-Constraint Breakdown
    by_constraint = (
        df.groupby("constraint", sort=False)
        .agg(
            questions=("dataset_index", "count"),
            instruction_following=("instruction_following", "mean"),
            answer_accuracy=("answer_correct", "mean"),
            joint_success=("joint_success", "mean"),
            format_valid=("format_valid", "mean"),
            missing_answer=("missing_answer", "mean"),
            truncated=("truncated", "mean") if "truncated" in df.columns else ("dataset_index", lambda _: 0.0),
            mean_output_tokens=("output_tokens", "mean"),
        )
        .reset_index()
    )

    # Per-Source Breakdown
    by_source = (
        df.groupby("source", sort=False)
        .agg(
            questions=("dataset_index", "count"),
            instruction_following=("instruction_following", "mean"),
            answer_accuracy=("answer_correct", "mean"),
            joint_success=("joint_success", "mean"),
            format_valid=("format_valid", "mean"),
            missing_answer=("missing_answer", "mean"),
            mean_output_tokens=("output_tokens", "mean"),
        )
        .reset_index()
    )

    # Save artifacts if requested
    if save_artifacts:
        scored_path = target_dir / "scored_responses.jsonl"
        with open(scored_path, "w", encoding="utf-8") as out_f:
            for row in scored_records:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"-> Saved scored records: {scored_path}")

        overall_json_path = target_dir / "overall.json"
        with open(overall_json_path, "w", encoding="utf-8") as out_f:
            json.dump([overall_row], out_f, indent=2)
        print(f"-> Saved overall JSON:   {overall_json_path}")

        overall_csv_path = target_dir / "overall.csv"
        overall_df.to_csv(overall_csv_path, index=False)
        print(f"-> Saved overall CSV:    {overall_csv_path}")

        by_constraint_path = target_dir / "by_constraint.csv"
        by_constraint.to_csv(by_constraint_path, index=False)
        print(f"-> Saved constraint CSV: {by_constraint_path}")

        by_source_path = target_dir / "by_source.csv"
        by_source.to_csv(by_source_path, index=False)
        print(f"-> Saved source CSV:     {by_source_path}")

    return overall_df, by_constraint, by_source, scored_records


# ==============================================================================
# 4. CLI Entry Point
# ==============================================================================
def main():
    default_input = Path(
        r"C:\Users\bryan\OneDrive\Documents\GitHub\CoT_Controllability\outputs\reasonif_qwen3_14b_svamp_full_dpo_top_p095\n300_gseed42_new16384_temp1_top_p095_native_chat\raw_responses.jsonl"
    )

    parser = argparse.ArgumentParser(
        description="Regrade ReasonIF raw responses against official benchmark grader."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to input raw_responses.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write output artifacts (defaults to parent dir of input)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="If set, only print evaluation tables without overwriting output files",
    )
    args = parser.parse_args()

    overall, by_constraint, by_source, _ = regrade_reasonif_file(
        input_file=args.input,
        output_dir=args.output_dir,
        save_artifacts=not args.no_save,
    )

    print("\n" + "=" * 88)
    print(f"REASONIF REGRADING REPORT: {args.input.name}")
    print("=" * 88)

    print("\n[1] OVERALL BENCHMARK PERFORMANCE (N=300)")
    print("-" * 88)
    print(
        f"  Instruction Following Score (IFS) : {overall['instruction_following'].iloc[0] * 100:6.2f}%\n"
        f"  Standard Answer Accuracy          : {overall['answer_correct'].iloc[0] * 100:6.2f}%\n"
        f"  Joint Success Rate                : {overall['joint_success'].iloc[0] * 100:6.2f}%\n"
        f"  Format Validity                   : {overall['format_valid'].iloc[0] * 100:6.2f}%\n"
        f"  Missing Answer Tag Rate           : {overall['missing_answer'].iloc[0] * 100:6.2f}%\n"
        f"  Mean Output Tokens                : {overall['mean_output_tokens'].iloc[0]:6.1f}\n"
        f"  Median Output Tokens              : {overall['median_output_tokens'].iloc[0]:6.1f}"
    )

    print("\n[2] PER-CONSTRAINT BREAKDOWN")
    print("-" * 88)
    display_cols = [
        "constraint",
        "questions",
        "instruction_following",
        "answer_accuracy",
        "joint_success",
        "format_valid",
        "mean_output_tokens",
    ]
    show_constraint = by_constraint[display_cols].copy()
    for pct_col in ["instruction_following", "answer_accuracy", "joint_success", "format_valid"]:
        show_constraint[pct_col] = show_constraint[pct_col].apply(lambda x: f"{x * 100:5.1f}%")
    show_constraint["mean_output_tokens"] = show_constraint["mean_output_tokens"].apply(lambda x: f"{x:6.1f}")
    print(show_constraint.to_string(index=False))

    print("\n[3] PER-DATASET SOURCE BREAKDOWN")
    print("-" * 88)
    show_source = by_source[
        ["source", "questions", "instruction_following", "answer_accuracy", "joint_success", "mean_output_tokens"]
    ].copy()
    for pct_col in ["instruction_following", "answer_accuracy", "joint_success"]:
        show_source[pct_col] = show_source[pct_col].apply(lambda x: f"{x * 100:5.1f}%")
    show_source["mean_output_tokens"] = show_source["mean_output_tokens"].apply(lambda x: f"{x:6.1f}")
    print(show_source.to_string(index=False))
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
