"""
ReasonIF 300 Evaluation CLI
Evaluates model completions on the official ReasonIF benchmark suite (300 tasks).
Calculates Instruction Following Score (IFS), Answer Accuracy, and Joint Success Rate.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluators.reasonif_evaluator import evaluate_reasonif_file


def main():
    parser = argparse.ArgumentParser(description="Evaluate model completions on ReasonIF 300 benchmark.")
    parser.add_argument("--input_file", "--input", "-i", type=str, required=True, help="Path to JSONL completions file.")
    parser.add_argument("--output_file", "--output_csv", "--output", "-o", type=str, default=None, help="Path to save scored breakdown CSV.")
    args = parser.parse_args()

    in_path = Path(args.input_file)
    if not in_path.exists():
        print(f"Error: {in_path} does not exist.")
        sys.exit(1)

    print(f"Evaluating ReasonIF completions from: {in_path}")
    overall, by_constraint = evaluate_reasonif_file(str(in_path))

    print("\n" + "=" * 80)
    print(f"REASONIF BENCHMARK EVALUATION RESULTS: {in_path.name}")
    print("=" * 80)
    print("\n--- OVERALL SUMMARY ---")
    print(overall.to_string(index=False))
    print("\n--- PER-CONSTRAINT BREAKDOWN ---")
    print(by_constraint.to_string(index=False))
    print("=" * 80 + "\n")

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        by_constraint.to_csv(out_path, index=False)
        print(f"Saved per-constraint results to {out_path}")


if __name__ == "__main__":
    main()
