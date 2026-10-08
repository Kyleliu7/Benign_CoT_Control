"""
Haskins 500 Evaluation CLI
Scores model completions against the 10 Haskins CoT-Control constraints.
Supports single JSON/JSONL/CSV files as well as run directories containing record files.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluators.haskins_evaluator import evaluate_haskins_record


def main():
    parser = argparse.ArgumentParser(description="Evaluate model completions on Haskins 500 benchmark.")
    parser.add_argument("--input_file", "--input", "-i", type=str, required=True, help="Path to JSON/JSONL/CSV completions file or run directory.")
    parser.add_argument("--output_file", "--output", "-o", type=str, default=None, help="Path to save scored outputs.")
    parser.add_argument("--continuation_only", action="store_true", help="Score only the post-prefix continuation reasoning.")
    parser.add_argument("--scorer", choices=["paper", "semantic"], default="paper",
                        help="paper (default) reproduces the manuscript scores; semantic is the legacy strict diagnostic.")
    args = parser.parse_args()

    in_path = Path(args.input_file)
    if not in_path.exists():
        print(f"Error: {in_path} does not exist.")
        sys.exit(1)

    records = []
    if in_path.is_dir():
        # Check for records subfolder or direct json files
        rec_dir = in_path / "records" if (in_path / "records").is_dir() else in_path
        json_files = sorted(list(rec_dir.glob("*.json")))
        meta_names = {"aggregates.json", "manifest.json", "COMPLETE.json", "run_status.json", "summary_by_task.json", "prefix_cache.json"}
        for jf in json_files:
            if jf.name not in meta_names:
                with open(jf, "r", encoding="utf-8") as f:
                    records.append(json.load(f))
    elif in_path.suffix == ".jsonl":
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    elif in_path.suffix == ".csv":
        with open(in_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)
    else:
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            records = data if isinstance(data, list) else data.get("items", [])

    print(f"Loaded {len(records)} records from {in_path}. Evaluating...")
    scored = []
    for r in records:
        scored.append(evaluate_haskins_record(r, continuation_only=args.continuation_only, scorer=args.scorer))

    strict_pass = sum(1 for s in scored if s.get("strict_binary") == 1 or s.get("strict_binary_pass") is True)
    avg_compliance = sum(s.get("compliance", s.get("defined_partial_compliance", 0.0)) for s in scored) / max(len(scored), 1)

    print("\n" + "=" * 60)
    print(" HASKINS EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Evaluated: {len(scored)}")
    print(f"Strict Pass:     {strict_pass}/{len(scored)} ({strict_pass/len(scored)*100:.2f}%)")
    print(f"Mean Compliance: {avg_compliance*100:.2f}%")
    print("=" * 60)

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for s in scored:
                f.write(json.dumps(s) + "\n")
        print(f"Saved scored results to {out_path}")


if __name__ == "__main__":
    main()
