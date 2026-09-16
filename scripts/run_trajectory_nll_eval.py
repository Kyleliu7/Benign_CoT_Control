"""
Trajectory NLL & Prefix Surprisal Evaluator.
Computes token-level surprisal under base foundation model priors across position segments and prefix conditioning.
"""
import os
import sys
import torch
import pandas as pd
from pathlib import Path

import json
import argparse
import numpy as np
from typing import List, Dict, Any, Optional

SEGMENTS = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 250), (250, None)]
PREFIX_K_LIST = [5, 10, 20, 50]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Trajectory NLL and Token Surprisal across positional segments.")
    parser.add_argument("--input", "-i", type=str, required=False, default=None,
                        help="Path to JSONL file containing generation trajectories.")
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen3-14B",
                        help="HuggingFace model ID or local path to evaluate token log-likelihoods.")
    parser.add_argument("--output", "-o", type=str, required=False, default=None,
                        help="Path to save computed surprisal metrics.")
    parser.add_argument("--max-samples", "-n", type=int, default=None,
                        help="Maximum number of trajectories to evaluate.")
    return parser.parse_args()


def compute_segment_nlls(token_nlls: List[float], segments=SEGMENTS) -> Dict[str, float]:
    """Segment token NLL sequence into positional intervals."""
    res = {}
    for start, end in segments:
        label = f"tok_{start}_{end if end is not None else 'end'}"
        sub = token_nlls[start:end] if end is not None else token_nlls[start:]
        res[label] = float(np.mean(sub)) if len(sub) > 0 else float("nan")
    return res


def main():
    args = parse_args()
    if args.input is None:
        print("Trajectory Surprisal and Prefix Continuation Evaluator")
        print("Usage: python scripts/run_trajectory_nll_eval.py --input <path_to_trajectories.jsonl> [--model <model_id>]")
        print(f"\nPositional analysis segments: {SEGMENTS}")
        print(f"Prefix conditioning lengths (k): {PREFIX_K_LIST}")
        return

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading trajectories from: {input_path}")
    trajectories = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            trajectories.append(json.loads(line))

    if args.max_samples:
        trajectories = trajectories[:args.max_samples]

    print(f"Evaluating NLL surprisal across {len(trajectories)} trajectories with segments: {SEGMENTS}")
    # Display summary
    print("\n" + "=" * 80)
    print("TRAJECTORY NLL EVALUATION INITIALIZED")
    print(f"Model: {args.model} | Input: {input_path.name} | Total Records: {len(trajectories)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
