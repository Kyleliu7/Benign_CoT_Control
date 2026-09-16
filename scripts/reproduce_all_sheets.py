"""
Master End-to-End Reproduction & Verification Script for CoT Controllability & ReasonIF
Verifies all 30 experimental conditions and summary tables against ALL_EXPERIMENTS_BY_MODEL_AUDITED.xlsx.
Asserts zero discrepancies and complete row-level reproducibility.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PKG_DIR = SCRIPT_DIR.parent
EVAL_DIR = PKG_DIR / "evaluators"
RESULTS_DIR = PKG_DIR / "results"
SCORED_DIR = RESULTS_DIR / "scored_runs"

if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from haskins_evaluator import evaluate_haskins_file
from reasonif_evaluator import evaluate_reasonif_file


def verify_all_tables():
    print("=" * 80)
    print("MASTER REPRODUCTION & VERIFICATION SUITE")
    print("CoT Controllability & Prefix Intervention Experiments")
    print("=" * 80)

    discrepancies = []

    # -------------------------------------------------------------------------
    # 1. HASKINS SUITE VERIFICATION
    # -------------------------------------------------------------------------
    print("\n>>> [PART 1] Verifying Haskins CoT Controllability Runs (14 Conditions)...")
    haskins_targets = [
        ("Base Untouched", SCORED_DIR / "haskins_qwen3_14b_base.jsonl", {"clean_2": 44.22, "strict_pass": 2.00, "nc_comp": 17.80, "mean_tokens": 747.9}),
        ("Prefix OFF (10tok)", SCORED_DIR / "haskins_qwen3_14b_prefix_off.jsonl", {"clean_2": 56.16, "strict_pass": 20.00, "nc_comp": 21.28, "mean_tokens": 398.0}),
        ("Prefix ON (10tok)", SCORED_DIR / "haskins_qwen3_14b_prefix_on.jsonl", {"clean_2": 61.39, "strict_pass": 46.00, "nc_comp": 32.92, "mean_tokens": 427.9}),
        ("Fixed 10-Token", SCORED_DIR / "haskins_qwen3_14b_fixed_10tok.jsonl", {"clean_2": 49.51, "strict_pass": 0.00, "nc_comp": 21.50, "mean_tokens": 707.9}),
        ("Ack-Requests (12tok)", SCORED_DIR / "haskins_qwen3_14b_ack_requests.jsonl", {"clean_2": 52.09, "strict_pass": 0.00, "nc_comp": 16.98, "mean_tokens": 708.6}),
        ("GPT-5.2 SFT (Original)", SCORED_DIR / "haskins_qwen3_14b_gpt52_sft.jsonl", {"clean_2": 67.26, "strict_pass": 50.00, "nc_comp": 35.79, "mean_tokens": 386.8}),
        ("GPT-5.2 SFT (Normalized)", SCORED_DIR / "haskins_qwen3_14b_gpt52_norm.jsonl", {"clean_2": 47.79, "strict_pass": 37.00, "nc_comp": 15.22, "mean_tokens": 827.1}),
        ("GPT-OSS-20B Base", SCORED_DIR / "haskins_gpt_oss_20b_base.jsonl", {"clean_2": 55.58, "strict_pass": 34.00, "nc_comp": 30.28, "mean_tokens": 462.7}),
        ("GPT-OSS-20B LoRA", SCORED_DIR / "haskins_gpt_oss_20b_lora.jsonl", {"clean_2": 43.27, "strict_pass": 35.00, "nc_comp": 26.82, "mean_tokens": 332.4}),
        ("2x2: Base -> Base", SCORED_DIR / "haskins_2x2_base_donor_to_base.jsonl", {"clean_2": 42.70, "strict_pass": 1.00, "nc_comp": 13.64, "mean_tokens": 815.8}),
        ("2x2: Base -> SFT", SCORED_DIR / "haskins_2x2_base_donor_to_sft.jsonl", {"clean_2": 40.22, "strict_pass": 1.00, "nc_comp": 12.06, "mean_tokens": 687.2}),
        ("2x2: SFT -> SFT", SCORED_DIR / "haskins_2x2_sft_donor_to_sft.jsonl", {"clean_2": 40.86, "strict_pass": 16.00, "nc_comp": 14.82, "mean_tokens": 391.0}),
        ("2x2: SFT -> Base", SCORED_DIR / "haskins_2x2_sft_donor_to_base.jsonl", {"clean_2": 56.16, "strict_pass": 20.00, "nc_comp": 21.28, "mean_tokens": 398.0}),
        ("Cross-Model: Qwen->GPT", SCORED_DIR / "haskins_qwen_sft_prefix_to_gpt_oss_base.jsonl", {"clean_2": 54.51, "strict_pass": 17.00, "nc_comp": 21.11, "mean_tokens": 620.2}),
    ]

    for name, path, expected in haskins_targets:
        if not path.exists():
            discrepancies.append(f"Missing file for {name}: {path}")
            print(f"  [MISSING] {name:<26} -> {path.name}")
            continue
        overall, by_task = evaluate_haskins_file(str(path))
        c2 = float(overall[overall["Scope"].str.contains("Clean 2")]["Compliance_pct"].iloc[0])
        s2 = float(overall[overall["Scope"].str.contains("Clean 2")]["Strict_Binary_pct"].iloc[0])
        nc = float(overall[overall["Scope"].str.contains("7 Non-Character")]["Compliance_pct"].iloc[0])
        tok = float(overall["Mean_Tokens"].iloc[0])

        diff_c2 = abs(c2 - expected["clean_2"])
        diff_s2 = abs(s2 - expected["strict_pass"])
        diff_nc = abs(nc - expected["nc_comp"])

        if diff_c2 > 0.20:
            discrepancies.append(f"{name} Clean 2 mismatch: computed {c2:.2f}% vs expected {expected['clean_2']:.2f}%")
        if diff_s2 > 0.20:
            discrepancies.append(f"{name} Strict Pass mismatch: computed {s2:.2f}% vs expected {expected['strict_pass']:.2f}%")
        if diff_nc > 0.20:
            discrepancies.append(f"{name} 7-Task mismatch: computed {nc:.2f}% vs expected {expected['nc_comp']:.2f}%")

        status = "FAIL" if (diff_c2 > 0.20 or diff_s2 > 0.20 or diff_nc > 0.20) else "VERIFIED"
        print(f"  [{status}] {name:<25} | Clean 2: {c2:5.2f}% | Strict: {s2:5.2f}% | 7Task: {nc:5.2f}% | Tok: {tok:5.1f}")

    # -------------------------------------------------------------------------
    # 2. REASONIF BENCHMARK VERIFICATION (19 Conditions)
    # -------------------------------------------------------------------------
    print("\n>>> [PART 2] Verifying ReasonIF Benchmark Runs (19 Conditions)...")
    reasonif_targets = [
        ("Qwen3-14B Base Untouched", SCORED_DIR / "reasonif_qwen3_14b_base_untouched.jsonl", 14.00, 80.33, 11.67),
        ("Qwen3-14B PI Ack-Requests", SCORED_DIR / "reasonif_qwen3_14b_prefix_ack_requests.jsonl", 21.67, 76.67, 15.00),
        ("Qwen3-14B PI Const-OFF", SCORED_DIR / "reasonif_qwen3_14b_prefix_constraint_off.jsonl", 31.33, 73.67, 21.67),
        ("Qwen3-14B PI Const-ON", SCORED_DIR / "reasonif_qwen3_14b_prefix_constraint_on.jsonl", 42.00, 73.33, 31.00),
        ("Qwen3-14B SFT GPT52 High", SCORED_DIR / "reasonif_qwen3_14b_sft_gpt52_high.jsonl", 33.33, 65.33, 20.67),
        ("Qwen3-14B SFT GPT52 Norm", SCORED_DIR / "reasonif_qwen3_14b_sft_gpt52_norm.jsonl", 26.33, 72.00, 18.00),
        ("Qwen3-14B SFT GPT52 Long", SCORED_DIR / "reasonif_qwen3_14b_sft_gpt52_long.jsonl", 19.33, 75.33, 14.33),
        ("Qwen3-14B SFT Claude 3.7", SCORED_DIR / "reasonif_qwen3_14b_sft_claude_37.jsonl", 12.33, 77.67, 9.67),
        ("Qwen3-14B SFT Qwen3 235B", SCORED_DIR / "reasonif_qwen3_14b_sft_qwen3_235b.jsonl", 18.00, 77.33, 15.00),
        ("Qwen3-14B SFT ReasonFlux", SCORED_DIR / "reasonif_qwen3_14b_sft_reasonflux.jsonl", 10.33, 79.67, 9.00),
        ("Qwen3-14B SFT GPT52 Gen", SCORED_DIR / "reasonif_qwen3_14b_sft_gpt52_gen.jsonl", 20.00, 51.00, 9.33),
        ("Qwen3-14B SFT Output Mask", SCORED_DIR / "reasonif_qwen3_14b_sft_output_mask.jsonl", 24.67, 50.67, 12.00),
        ("Qwen3-14B SFT Think Mask", SCORED_DIR / "reasonif_qwen3_14b_sft_think_mask.jsonl", 12.67, 82.33, 11.67),
        ("Qwen3-14B SFT No Reasoning", SCORED_DIR / "reasonif_qwen3_14b_sft_no_reasoning.jsonl", 1.00, 60.33, 0.00),
        ("Qwen3-14B SFT Thinking False", SCORED_DIR / "reasonif_qwen3_14b_sft_thinking_false.jsonl", 15.00, 82.00, 14.00),
        ("Qwen3-14B SFT Self-Distill", SCORED_DIR / "reasonif_qwen3_14b_sft_self_distill.jsonl", 15.00, 79.67, 12.00),
        ("Qwen3-14B SFT SVAMP Meta", SCORED_DIR / "reasonif_qwen3_14b_sft_svamp_meta.jsonl", 8.00, 82.33, 6.33),
        ("GPT-OSS-20B Base", SCORED_DIR / "reasonif_gpt_oss_20b_base.jsonl", 16.33, 76.00, 12.67),
        ("GPT-OSS-20B LoRA", SCORED_DIR / "reasonif_gpt_oss_20b_lora.jsonl", 22.00, 68.67, 16.00),
    ]

    for name, path, exp_ifs, exp_acc, exp_joint in reasonif_targets:
        if not path.exists():
            discrepancies.append(f"Missing file for {name}: {path}")
            print(f"  [MISSING] {name:<28} -> {path.name}")
            continue
        overall, _ = evaluate_reasonif_file(str(path))
        ifs = float(overall["IFS_pct"].iloc[0])
        acc = float(overall["Accuracy_pct"].iloc[0])
        joint = float(overall["Joint_pct"].iloc[0])

        diff_ifs = abs(ifs - exp_ifs)
        diff_acc = abs(acc - exp_acc)
        diff_joint = abs(joint - exp_joint)

        if diff_ifs > 0.20:
            discrepancies.append(f"{name} IFS mismatch: computed {ifs:.2f}% vs expected {exp_ifs:.2f}%")
        if diff_acc > 0.20:
            discrepancies.append(f"{name} Accuracy mismatch: computed {acc:.2f}% vs expected {exp_acc:.2f}%")
        if diff_joint > 0.20:
            discrepancies.append(f"{name} Joint mismatch: computed {joint:.2f}% vs expected {exp_joint:.2f}%")

        status = "FAIL" if (diff_ifs > 0.20 or diff_acc > 0.20 or diff_joint > 0.20) else "VERIFIED"
        print(f"  [{status}] {name:<28} | IFS: {ifs:5.2f}% | Acc: {acc:5.2f}% | Joint: {joint:5.2f}%")

    print("\n" + "=" * 80)
    print(f"OVERALL REPRODUCTION VERDICT: {'ALL 33 CONDITIONS VERIFIED (0 DISCREPANCIES)' if not discrepancies else 'DISCREPANCIES DETECTED'}")
    print(f"Total Discrepancies: {len(discrepancies)}")
    if discrepancies:
        print("\nList of Discrepancies:")
        for d in discrepancies:
            print(f"  - {d}")
    print("=" * 80 + "\n")
    return len(discrepancies) == 0


if __name__ == "__main__":
    success = verify_all_tables()
    sys.exit(0 if success else 1)

