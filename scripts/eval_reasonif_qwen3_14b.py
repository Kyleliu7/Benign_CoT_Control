"""
ReasonIF Evaluation Pipeline for Qwen3-14B (Full or LoRA) with vLLM / Transformers.
Supports Prefix-Conditioned Interventions and Standard ReasonIF Evaluation.
"""
import os
import sys
import json
import re
from pathlib import Path

import argparse
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from evaluators.reasonif_evaluator import evaluate_reasonif_file

MODEL_NAME = "Qwen/Qwen3-14B"
NUM_QUESTIONS = 300
GENERATION_SEED = 42
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_NEW_TOKENS = 16384


def split_reasoning(raw_output: str) -> tuple[str, str]:
    text = raw_output.strip()
    if "</think>" in text:
        reasoning, content = text.split("</think>", 1)
        return reasoning.removeprefix("<think>").strip(), content.strip()
    if text.startswith("<think>"):
        return text[len("<think>"):].strip(), ""
    return text, ""


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-14B responses on ReasonIF benchmark.")
    parser.add_argument("--input", "-i", type=str, required=False, default=None,
                        help="Path to JSONL file containing responses to evaluate.")
    parser.add_argument("--output", "-o", type=str, required=False, default=None,
                        help="Path to save scored output.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input is None:
        print(f"ReasonIF Evaluation Pipeline for {MODEL_NAME}")
        print("Usage: python scripts/eval_reasonif_qwen3_14b.py --input <path_to_responses.jsonl>")
        print("\nAvailable pre-scored Qwen3-14B ReasonIF runs in results/scored_runs/:")
        scored_dir = PKG_DIR / "results" / "scored_runs"
        if scored_dir.exists():
            for f in sorted(scored_dir.glob("reasonif_qwen3_14b_*.jsonl")):
                print(f"  - {f.name}")
        return

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Evaluating ReasonIF responses from: {input_path}")
    overall, by_constraint = evaluate_reasonif_file(str(input_path))

    print("\n" + "=" * 80)
    print(f"REASONIF BENCHMARK RESULTS ({input_path.stem})")
    print("=" * 80)
    print("\n--- Overall Benchmark Performance ---")
    print(overall.to_string(index=False))
    print("\n--- Breakdown by Constraint Type ---")
    print(by_constraint.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
