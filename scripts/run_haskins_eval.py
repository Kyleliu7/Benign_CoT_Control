"""
Evaluation Runner for Haskins Chain-of-Thought (CoT) Controllability Benchmark.
Evaluates 10 intrinsic reasoning constraints across character and non-character tasks.
"""
import os
import sys
import json
from pathlib import Path

import argparse
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from evaluators.haskins_evaluator import evaluate_haskins_file

TASKS = [
    "third_person",
    "arrow_prefix",
    "word_suppression",
    "multiple_word_suppression",
    "end_of_sentence",
    "meow_between_words",
    "repeat_sentences",
    "alternating_case",
    "uppercase_thinking",
    "lowercase_thinking",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate responses on Haskins CoT Controllability benchmark.")
    parser.add_argument("--input", "-i", type=str, required=False, default=None,
                        help="Path to JSONL file containing responses to evaluate.")
    parser.add_argument("--output", "-o", type=str, required=False, default=None,
                        help="Path to save scored JSONL file.")
    parser.add_argument("--continuation-only", action="store_true", default=False,
                        help="Evaluate compliance only on continuation tokens (excluding prefix).")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input is None:
        print("Haskins CoT Controllability Benchmark: 10 Tasks Supported")
        print("Usage: python scripts/run_haskins_eval.py --input <path_to_responses.jsonl> [--continuation-only]")
        print("\nAvailable pre-scored runs in results/scored_runs/:")
        scored_dir = PKG_DIR / "results" / "scored_runs"
        if scored_dir.exists():
            for f in sorted(scored_dir.glob("haskins_*.jsonl")):
                print(f"  - {f.name}")
        return

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Evaluating Haskins responses from: {input_path}")
    overall, by_task = evaluate_haskins_file(str(input_path), continuation_only=args.continuation_only)

    print("\n" + "=" * 80)
    print("HASKINS BENCHMARK EVALUATION RESULTS")
    print("=" * 80)
    print("\n--- Summary by Aggregate Scope ---")
    print(overall.to_string(index=False))
    print("\n--- Per-Task Performance ---")
    print(by_task.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
