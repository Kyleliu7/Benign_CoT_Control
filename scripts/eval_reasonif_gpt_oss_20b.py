"""
ReasonIF Evaluation for OpenAI GPT-OSS-20B using Native Harmony Analysis Channels.
Evaluates 300 problems across AIME, AMC, GSM8K, ARC, and GPQA.
"""
import os
import sys
import gc
import json
import time
import shutil
import hashlib
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "openai/gpt-oss-20b"
MODEL_LABEL = "gpt_oss_20b"
REASONING_EFFORT = "medium"
NUM_QUESTIONS = 300
GENERATION_SEED = 42
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_NEW_TOKENS = 8192
BATCH_SIZE = 4

def rewrite_prompt_for_gpt_oss(prompt: str) -> str:
    p = prompt.replace("Format your reasoning according to the following rule:",
                       "Format your analysis channel according to the following rule:")
    p = p.replace("**When reasoning,", "**In the analysis channel,")
    p = p.replace("No other reasoning words should follow this phrase",
                  "No other words in the analysis channel should follow this phrase")
    return p

def split_channels(raw_output: str) -> tuple[str, str]:
    text = raw_output.strip()
    analysis, final = "", ""
    if "<|channel|>analysis<|message|>" in text:
        after_analysis = text.split("<|channel|>analysis<|message|>", 1)[1]
        if "<|channel|>final<|message|>" in after_analysis:
            analysis_part, final_part = after_analysis.split("<|channel|>final<|message|>", 1)
            for marker in ["<|end|>", "<|start|>assistant", "<|channel|>final", "<|return|>"]:
                analysis_part = analysis_part.replace(marker, "")
            analysis = analysis_part.strip()
            for marker in ["<|return|>", "<|end|>", "<|endoftext|>"]:
                final_part = final_part.replace(marker, "")
            final = final_part.strip()
        else:
            analysis = after_analysis.strip()
    elif "<|channel|>final<|message|>" in text:
        final_part = text.split("<|channel|>final<|message|>", 1)[1]
        for marker in ["<|return|>", "<|end|>", "<|endoftext|>"]:
            final_part = final_part.replace(marker, "")
        final = final_part.strip()
    elif "</think>" in text:
        analysis, final = text.split("</think>", 1)
        analysis = analysis.removeprefix("<think>").strip()
        final = final.strip()
    else:
        final = text
    return analysis, final


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate GPT-OSS-20B responses on ReasonIF benchmark.")
    parser.add_argument("--input", "-i", type=str, required=False, default=None,
                        help="Path to JSONL file containing responses to evaluate.")
    parser.add_argument("--output", "-o", type=str, required=False, default=None,
                        help="Path to save scored output.")
    return parser.parse_args()


def main():
    PKG_DIR = Path(__file__).resolve().parent.parent
    if str(PKG_DIR) not in sys.path:
        sys.path.insert(0, str(PKG_DIR))

    from evaluators.reasonif_evaluator import evaluate_reasonif_file

    args = parse_args()
    if args.input is None:
        print(f"ReasonIF Evaluation Pipeline for {MODEL_ID} (Reasoning effort: {REASONING_EFFORT})")
        print("Usage: python scripts/eval_reasonif_gpt_oss_20b.py --input <path_to_responses.jsonl>")
        print("\nAvailable pre-scored GPT-OSS-20B ReasonIF runs in results/scored_runs/:")
        scored_dir = PKG_DIR / "results" / "scored_runs"
        if scored_dir.exists():
            for f in sorted(scored_dir.glob("reasonif_gpt_oss_*.jsonl")):
                print(f"  - {f.name}")
        return

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Evaluating GPT-OSS ReasonIF responses from: {input_path}")
    overall, by_constraint = evaluate_reasonif_file(str(input_path))

    print("\n" + "=" * 80)
    print(f"GPT-OSS REASONIF BENCHMARK RESULTS ({input_path.stem})")
    print("=" * 80)
    print("\n--- Overall Benchmark Performance ---")
    print(overall.to_string(index=False))
    print("\n--- Breakdown by Constraint Type ---")
    print(by_constraint.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
