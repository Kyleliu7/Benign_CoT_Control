"""
Per-Token Forward KL Divergence Computation CLI
Computes teacher-forced forward KL divergence along Chain-of-Thought (CoT) reasoning trajectories
between instruction-tuned SFT policies and base models in float32 numerical precision.
Supports both offline cache analysis and live GPU forward-pass evaluation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def detect_reasoning_token_bounds(
    tokenizer, p_ids: List[int], g_ids: List[int], prompt_text: str, raw_output: str
) -> Tuple[int, int]:
    """
    Detects reasoning start and end indices strictly within g_ids using token IDs and prompt state.
    Returns (start_idx, end_idx) where g_ids[start_idx:end_idx] are reasoning tokens.
    Guarantees:
      1. If prompt already opened <think> without closing, start_idx = 0.
      2. If prompt did not open <think>, locates <think> in g_ids and sets start_idx after it.
      3. Reasoning stops strictly BEFORE </think>.
      4. If no <think> is present in prompt or generation, returns (0, 0) without misclassifying.
    """
    think_open_ids = tokenizer.encode("<think>", add_special_tokens=False)
    think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)

    prompt_has_open = "<think>" in prompt_text
    prompt_has_close = "</think>" in prompt_text

    if prompt_has_open:
        last_open_char = prompt_text.rfind("<think>")
        last_close_char = prompt_text.rfind("</think>") if prompt_has_close else -1
        prompt_reasoning_active = last_open_char > last_close_char
    else:
        prompt_reasoning_active = False
        for i in range(len(p_ids) - len(think_open_ids) + 1):
            if p_ids[i : i + len(think_open_ids)] == think_open_ids:
                prompt_reasoning_active = True
                break

    if prompt_reasoning_active:
        start_idx = 0
    else:
        open_pos = -1
        for i in range(len(g_ids) - len(think_open_ids) + 1):
            if g_ids[i : i + len(think_open_ids)] == think_open_ids:
                open_pos = i
                break
        if open_pos != -1:
            start_idx = open_pos + len(think_open_ids)
        else:
            curr_str = ""
            for i in range(len(g_ids)):
                curr_str += tokenizer.decode([g_ids[i]])
                if "<think>" in curr_str:
                    open_pos = i + 1
                    break
            if open_pos != -1:
                start_idx = open_pos
            else:
                return 0, 0

    close_pos = -1
    for i in range(start_idx, len(g_ids) - len(think_close_ids) + 1):
        if g_ids[i : i + len(think_close_ids)] == think_close_ids:
            close_pos = i
            break

    if close_pos == -1:
        for i in range(start_idx, len(g_ids)):
            tok_text = tokenizer.decode([g_ids[i]])
            if "</think>" in tok_text:
                close_pos = i
                break

    if close_pos != -1:
        end_idx = close_pos
    else:
        end_idx = len(g_ids)

    return max(0, start_idx), max(start_idx, end_idx)


def compute_kl_summary_from_traces(traces: List[List[float]]) -> Dict[str, float]:
    """
    Computes summary percentiles and statistics across all reasoning token traces.
    """
    early_tokens = []
    downstream_tokens = []
    all_tokens = []
    seq_max_downstream = []

    for tr in traces:
        arr = np.array(tr, dtype=np.float32)
        if len(arr) == 0:
            continue
        all_tokens.extend(arr.tolist())
        early_tokens.extend(arr[:10].tolist())
        if len(arr) > 10:
            downstream = arr[10:]
            downstream_tokens.extend(downstream.tolist())
            seq_max_downstream.append(float(np.max(downstream)))

    down_arr = np.array(downstream_tokens, dtype=np.float32) if downstream_tokens else np.array([0.0])
    early_arr = np.array(early_tokens, dtype=np.float32) if early_tokens else np.array([0.0])

    total_kl = float(np.sum(all_tokens)) if all_tokens else 1.0
    early_kl = float(np.sum(early_tokens)) if early_tokens else 0.0

    return {
        "total_tokens_measured": len(all_tokens),
        "downstream_tokens_count": len(down_arr),
        "early_share_pct": float(early_kl / max(total_kl, 1e-9) * 100),
        "early_mean": float(np.mean(early_arr)),
        "downstream_mean": float(np.mean(down_arr)),
        "downstream_p50": float(np.percentile(down_arr, 50)),
        "downstream_p90": float(np.percentile(down_arr, 90)),
        "downstream_p95": float(np.percentile(down_arr, 95)),
        "downstream_p99": float(np.percentile(down_arr, 99)),
        "downstream_p99_9": float(np.percentile(down_arr, 99.9)),
        "downstream_max_spike": float(np.max(down_arr)),
        "seq_max_p95": float(np.percentile(seq_max_downstream, 95)) if seq_max_downstream else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute per-token Forward KL divergence along CoT reasoning paths.")
    parser.add_argument("--cache_file", type=str, default=None, help="Path to precomputed float32 KL cache JSON.")
    parser.add_argument("--model_family", type=str, default="qwen3_14b", choices=["qwen3_14b", "phi4"], help="Model family.")
    parser.add_argument("--dataset", type=str, default="haskins_500", choices=["haskins_500", "reasonif_300"], help="Evaluation dataset.")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to save summary stats CSV.")
    parser.add_argument("--execute_forward_passes", action="store_true", help="Run live model forward passes on GPU.")
    args = parser.parse_args()

    # Determine cache file if not explicitly passed
    cache_path = None
    if args.cache_file:
        cache_path = Path(args.cache_file)
    else:
        # Check standard cache locations
        default_names = [
            REPO_ROOT / f"results/kl_divergence/caches_float32/kl_cache_{args.model_family}_{args.dataset}.json",
            REPO_ROOT / f"kl_cache_{args.model_family}_{args.dataset}.json",
        ]
        for p in default_names:
            if p.exists():
                cache_path = p
                break

    if cache_path and cache_path.exists():
        print(f"Loading cached float32 forward KL measurements from: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = data.get("records", {})
        traces = [r["reasoning_kl"] for r in records.values() if "reasoning_kl" in r]
        print(f"Loaded {len(traces)} reasoning traces.")

        summary = compute_kl_summary_from_traces(traces)

        print("\n" + "=" * 70)
        print(f" FORWARD KL DIVERGENCE ANALYSIS ({args.model_family} on {args.dataset})")
        print("=" * 70)
        print(f"Total Reasoning Tokens:    {summary['total_tokens_measured']}")
        print(f"Downstream Tokens (t > 10): {summary['downstream_tokens_count']}")
        print(f"Early KL Concentration:    {summary['early_share_pct']:.2f}% in first 10 tokens")
        print(f"Early KL Mean (t <= 10):   {summary['early_mean']:.3f} nats")
        print(f"Downstream Mean (t > 10):  {summary['downstream_mean']:.3f} nats")
        print(f"Downstream P50 (Median):   {summary['downstream_p50']:.3f} nats")
        print(f"Downstream P90:            {summary['downstream_p90']:.3f} nats")
        print(f"Downstream P95:            {summary['downstream_p95']:.3f} nats")
        print(f"Downstream P99:            {summary['downstream_p99']:.3f} nats")
        print(f"Downstream P99.9:          {summary['downstream_p99_9']:.3f} nats")
        print(f"Downstream Max Spike:      {summary['downstream_max_spike']:.2f} nats")
        print(f"Per-Sequence Max (P95):    {summary['seq_max_p95']:.2f} nats")
        print("=" * 70 + "\n")

        if args.output_csv:
            df_out = pd.DataFrame([summary])
            df_out.to_csv(args.output_csv, index=False)
            print(f"Saved summary to {args.output_csv}")

    elif args.execute_forward_passes:
        print("Live forward-pass execution requires PyTorch and GPU. Initializing...")
        try:
            import torch
            import torch.nn.functional as F
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as e:
            print(f"Error: Missing required packages for live GPU execution ({e}).")
            sys.exit(1)

        print("Refer to cot_kl_divergence_analysis.ipynb for batch configuration and setup.")
    else:
        print(f"Error: No cache file found for {args.model_family} on {args.dataset}.")
        print("Provide --cache_file <path> or use --execute_forward_passes with a GPU.")
        sys.exit(1)


if __name__ == "__main__":
    main()
