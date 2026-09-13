#!/usr/bin/env python3
"""
Faithful ReasonIF Evaluation Pipeline for Qwen3-14B + LoRA (GPT-5.2 High Reasoning Normalized)
Evaluates all 300 official ReasonIF questions using vLLM with LoRA adapter.
"""

import argparse
import gc
import json
import math
import os
import re
import sys
import time
import platform
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Paths
WORKSPACE = Path("/home/kyleliu789/workspace")
REASONIF_ROOT = WORKSPACE / "reasonIF_pinned_final_eval"
if str(REASONIF_ROOT) not in sys.path:
    sys.path.insert(0, str(REASONIF_ROOT))

from src.utils import prepare_message_list
from src.eval_utils import evaluate_instruction_following, extract_final_answer
import src.instructions.instruction_checker as reasonif_instruction_checker
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
reasonif_instruction_checker.detect = compatible_detect

def canonical_answer(value, source):
    value = str(value).strip().replace(',', '')
    if source in {'arc', 'gpqa'}:
        match = re.search(r'[ABCD]', value.upper())
        return match.group(0) if match else value.upper()
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else f'{number:.12g}'
    except ValueError:
        return value

def split_reasoning(raw_output):
    text = raw_output.strip()
    if '</think>' in text:
        reasoning, content = text.split('</think>', 1)
        return reasoning.removeprefix('<think>').strip(), content.strip()
    if text.startswith('<think>'):
        return text[len('<think>'):].strip(), ''
    return text, ''

def main():
    parser = argparse.ArgumentParser(description="Evaluate ReasonIF on Qwen LoRA Model")
    parser.add_argument("--base-model", default="Qwen/Qwen3-14B", help="Base model ID")
    parser.add_argument("--lora-path", default=str(WORKSPACE / "LlamaFactory/qwen3-14b-gpt52-high-reasoning-normalized"), help="Path to LoRA adapter")
    parser.add_argument("--output-dir", default=str(WORKSPACE / "outputs/qwen3_14b_gpt52_high_reasoning_normalized_reasonif"), help="Output directory")
    parser.add_argument("--sanity-check", action="store_true", help="Run 3-item sanity check")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("REASONIF EVALUATION: QWEN3-14B + SFT NORMALIZED LORA")
    print("=" * 80)
    print(f"Base Model: {args.base_model}")
    print(f"LoRA Path:  {args.lora_path}")
    print(f"Output Dir: {out_dir}")

    # 1. Load official ReasonIF prompts
    dataset_path = REASONIF_ROOT / "data" / "reasonIF_dataset.json"
    official_messages, official_dataset = prepare_message_list("Qwen3-14B", input_path=str(dataset_path))
    num_questions = 3 if args.sanity_check else 300
    official_messages = official_messages[:num_questions]
    official_dataset = official_dataset[:num_questions]
    print(f"Evaluating {len(official_dataset)} questions.")

    # 2. Tokenizer & render prompts
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    rendered_prompts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        for msgs in official_messages
    ]

    # 3. Initialize vLLM engine
    print("\nInitializing vLLM engine with LoRA support...")
    llm = LLM(
        model=args.base_model,
        enable_lora=True,
        max_lora_rank=32,
        max_model_len=18432,
        gpu_memory_utilization=0.90,
        max_num_seqs=64,
        seed=42,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    lora_request = LoRARequest("normalized_lora", 1, args.lora_path)

    # 4. Generate outputs with per-item seeds
    print(f"\nGenerating {len(rendered_prompts)} responses...", flush=True)
    t0 = time.time()
    
    sampling_params_list = [
        SamplingParams(
            temperature=1.0,
            top_p=0.95,
            max_tokens=16384,
            seed=42 + idx,
        )
        for idx in range(num_questions)
    ]

    outputs = []
    chunk_size = 64
    raw_save_path = out_dir / "raw_generations_partial.jsonl"
    with open(raw_save_path, "w", encoding="utf-8") as raw_f:
        for c_start in range(0, num_questions, chunk_size):
            c_end = min(c_start + chunk_size, num_questions)
            c_prompts = rendered_prompts[c_start:c_end]
            c_params = sampling_params_list[c_start:c_end]
            print(f"Generating chunk {c_start} to {c_end} ({len(c_prompts)} prompts batched)...", flush=True)
            chunk_outputs = llm.generate(c_prompts, c_params, lora_request=lora_request)
            outputs.extend(chunk_outputs)
            for item_idx, c_out in enumerate(chunk_outputs, start=c_start):
                raw_f.write(json.dumps({
                    "dataset_index": item_idx,
                    "text": c_out.outputs[0].text,
                    "finish_reason": c_out.outputs[0].finish_reason,
                    "token_ids": c_out.outputs[0].token_ids,
                }) + "\n")
            raw_f.flush()
            print(f"Completed {len(outputs)}/{num_questions} items...", flush=True)

    gen_time = time.time() - t0
    print(f"Generation completed in {gen_time:.1f}s ({gen_time/len(outputs):.2f}s/item)", flush=True)

    # 5. Process & score responses
    print("\nScoring responses...", flush=True)
    scored_rows = []
    for idx, (data_item, out) in enumerate(zip(official_dataset, outputs)):
        raw_text = out.outputs[0].text
        finish_reason = out.outputs[0].finish_reason
        total_tokens = len(out.outputs[0].token_ids)

        reasoning, content = split_reasoning(raw_text)
        reasoning_tokens = len(tokenizer.encode(reasoning, add_special_tokens=False)) if reasoning else 0

        # Instruction following check
        c_name = data_item["constraint_name"]
        c_args = data_item["constraint_args"]
        prompt_str = data_item["question"]

        follows = all(evaluate_instruction_following(
            instruction_id_list=c_name,
            parameters=c_args,
            prompt=prompt_str,
            response=reasoning,
        )) if reasoning.strip() else False

        # Answer accuracy check
        predicted = extract_final_answer(content, data_item["source"])
        correct = canonical_answer(predicted, data_item["source"]) == canonical_answer(data_item["answer"], data_item["source"])
        has_answer_tags = "<answer>" in content and "</answer>" in content
        has_reasoning_end = "</think>" in raw_text

        row = {
            "dataset_index": idx,
            "source": data_item["source"],
            "constraint": c_name[0],
            "question": prompt_str,
            "ground_truth": data_item["answer"],
            "predicted_answer": predicted,
            "instruction_following": bool(follows),
            "answer_correct": bool(correct),
            "joint_success": bool(follows and correct),
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
            "finish_reason": finish_reason,
            "has_answer_tags": bool(has_answer_tags),
            "has_reasoning_end": bool(has_reasoning_end),
            "format_valid": bool(has_answer_tags and has_reasoning_end),
            "missing_answer": bool(not has_answer_tags),
            "truncated": bool((finish_reason == "length") or (total_tokens >= 16384)),
            "raw_output": raw_text,
            "reasoning_content": reasoning,
            "content": content,
        }
        scored_rows.append(row)

    # Save scored JSONL
    scored_path = out_dir / "scored_responses.jsonl"
    with open(scored_path, "w", encoding="utf-8") as f:
        for r in scored_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved scored responses to: {scored_path}", flush=True)

    # Aggregations
    df = pd.DataFrame(scored_rows)
    overall_ifs = float(df["instruction_following"].mean() * 100)
    overall_acc = float(df["answer_correct"].mean() * 100)
    overall_joint = float(df["joint_success"].mean() * 100)
    mean_reasoning_tok = float(df["reasoning_tokens"].mean())
    mean_total_tok = float(df["total_tokens"].mean())
    format_valid = float(df["format_valid"].mean() * 100)

    print("\n" + "=" * 80, flush=True)
    print("REASONIF BENCHMARK RESULTS: QWEN3-14B + SFT NORMALIZED LORA", flush=True)
    print("=" * 80, flush=True)
    print(f"Instruction Following Success (IFS): {overall_ifs:.2f}% ({int(df['instruction_following'].sum())}/{len(df)})", flush=True)
    print(f"Answer Accuracy:                    {overall_acc:.2f}% ({int(df['answer_correct'].sum())}/{len(df)})", flush=True)
    print(f"Joint Success Rate:                 {overall_joint:.2f}% ({int(df['joint_success'].sum())}/{len(df)})", flush=True)
    print(f"Format Valid:                       {format_valid:.2f}%", flush=True)
    print(f"Mean Reasoning Tokens:              {mean_reasoning_tok:.1f}", flush=True)
    print(f"Mean Total Generated Tokens:        {mean_total_tok:.1f}", flush=True)
    print("=" * 80, flush=True)

    # Constraint breakdown
    c_breakdown = []
    for c_type in df["constraint"].unique():
        cdf = df[df["constraint"] == c_type]
        c_breakdown.append({
            "constraint": c_type,
            "N": len(cdf),
            "ifs": float(cdf["instruction_following"].mean() * 100),
            "accuracy": float(cdf["answer_correct"].mean() * 100),
            "joint": float(cdf["joint_success"].mean() * 100),
            "mean_reasoning_tokens": float(cdf["reasoning_tokens"].mean()),
        })
    c_df = pd.DataFrame(c_breakdown)
    print("\nPER-CONSTRAINT BREAKDOWN:", flush=True)
    print(c_df.to_string(index=False), flush=True)

    c_df.to_csv(out_dir / "by_constraint.csv", index=False)

    summary = {
        "model": "Qwen/Qwen3-14B + GPT-5.2 High Reasoning Normalized SFT (LoRA)",
        "lora_path": args.lora_path,
        "questions": len(df),
        "ifs": overall_ifs,
        "accuracy": overall_acc,
        "joint_success": overall_joint,
        "format_valid": format_valid,
        "mean_reasoning_tokens": mean_reasoning_tok,
        "mean_total_tokens": mean_total_tok,
        "constraint_breakdown": c_breakdown,
    }
    with open(out_dir / "overall.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved overall metrics to: {out_dir / 'overall.json'}", flush=True)
    print("\nREASONIF EVALUATION COMPLETE!", flush=True)
    os._exit(0)

if __name__ == "__main__":
    main()
