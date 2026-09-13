#!/usr/bin/env python3
"""
Turnkey ReasonIF Benchmark Evaluator
Evaluates row-level ReasonIF generation JSONL files against the pinned official grader (commit 706b953).
Outputs Instruction Following Score (IFS), Answer Accuracy, and Joint Success Rate.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    import reasonif_official.instructions.instruction_checker as instruction_checker
    from fast_langdetect import detect as installed_language_detect

    def compatible_detect(text, low_memory=False):
        try:
            result = installed_language_detect(text, low_memory=low_memory)
        except TypeError:
            result = installed_language_detect(text, model='lite' if low_memory else 'full')
        if isinstance(result, list):
            if not result:
                raise ValueError('fast-langdetect returned no predictions')
            return result[0]
        return result

    instruction_checker.detect = compatible_detect
except ImportError as e:
    print(f"Warning: could not patch fast_langdetect ({e})")

from reasonif_official.eval_utils import evaluate_instruction_following, extract_final_answer


def split_reasoning_content(raw_output: str) -> Tuple[str, str]:
    text = (raw_output or "").strip()
    if "</think>" in text:
        reasoning, content = text.split("</think>", 1)
        return reasoning.removeprefix("<think>").strip(), content.strip()
    if text.startswith("<think>"):
        return text[len("<think>"):].strip(), ""
    return text, ""


def canonical_answer(value: Any, source: str) -> str:
    val_str = str(value).strip().replace(',', '')
    if source in {'arc', 'gpqa'}:
        match = re.search(r'[ABCD]', val_str.upper())
        return match.group(0) if match else val_str.upper()
    try:
        num = float(val_str)
        return str(int(num)) if num.is_integer() else f"{num:.12g}"
    except ValueError:
        return val_str


def evaluate_reasonif_record(record: Dict[str, Any]) -> Dict[str, Any]:
    c_name = record.get("constraint_name") or record.get("constraint")
    c_args = record.get("constraint_args", {})
    prompt = record.get("prompt") or record.get("original_prompt", "")
    source = record.get("source", "gsm8k")
    ground_truth = record.get("answer", "")

    c_name_list = [c_name] if isinstance(c_name, str) else (c_name or [])
    c_args_list = [c_args] if isinstance(c_args, dict) else (c_args or [])

    reasoning = record.get("reasoning_content")
    content = record.get("content")
    if reasoning is None or content is None:
        raw_output = record.get("raw_output", "")
        reasoning, content = split_reasoning_content(raw_output)

    following_list = evaluate_instruction_following(
        c_name_list,
        c_args_list,
        prompt,
        reasoning
    )
    is_following = following_list[0] if following_list else False

    pred_ans = record.get("predicted_answer")
    if pred_ans is None:
        pred_ans = extract_final_answer(content or reasoning, source)
    
    canon_pred = canonical_answer(pred_ans, source)
    canon_gt = canonical_answer(ground_truth, source)
    is_correct = (canon_pred == canon_gt)

    joint = bool(is_following and is_correct)

    tokens = record.get("output_tokens", 0)
    if not tokens:
        raw_output = record.get("raw_output", "")
        tokens = int(len(raw_output.split()) * 1.3)

    return {
        "constraint": c_name_list[0] if c_name_list else "unknown",
        "source": source,
        "instruction_following": bool(is_following),
        "answer_correct": bool(is_correct),
        "joint_success": joint,
        "tokens": tokens,
        "predicted_answer": canon_pred,
        "ground_truth": canon_gt
    }


def evaluate_reasonif_file(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    evaluated = [evaluate_reasonif_record(r) for r in records]
    df = pd.DataFrame(evaluated)

    by_constraint = df.groupby("constraint").agg(
        N=("instruction_following", "count"),
        IFS_pct=("instruction_following", lambda x: np.mean(x) * 100),
        Accuracy_pct=("answer_correct", lambda x: np.mean(x) * 100),
        Joint_pct=("joint_success", lambda x: np.mean(x) * 100),
        Mean_Tokens=("tokens", "mean")
    ).reset_index()

    overall = pd.DataFrame([{
        "Scope": "Overall Benchmark (N=300)",
        "N": len(df),
        "IFS_pct": float(np.mean(df["instruction_following"]) * 100),
        "Accuracy_pct": float(np.mean(df["answer_correct"]) * 100),
        "Joint_pct": float(np.mean(df["joint_success"]) * 100),
        "Mean_Tokens": float(np.mean(df["tokens"]))
    }])

    return overall, by_constraint


def main():
    parser = argparse.ArgumentParser(description="Evaluate ReasonIF benchmark JSONL responses.")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output-csv", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    overall, by_constraint = evaluate_reasonif_file(args.input)

    print("\n" + "=" * 80)
    print(f"REASONIF BENCHMARK EVALUATION RESULTS: {Path(args.input).name}")
    print("=" * 80)
    print("\n--- OVERALL SUMMARY ---")
    print(overall.to_string(index=False))
    print("\n--- PER-CONSTRAINT BREAKDOWN ---")
    print(by_constraint.to_string(index=False))
    print("=" * 80 + "\n")

    if args.output_csv:
        by_constraint.to_csv(args.output_csv, index=False)
        print(f"Saved results to {args.output_csv}")


if __name__ == "__main__":
    main()
