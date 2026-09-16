import os
import sys
import gc
import re
import math
import json
import time
import shutil
import hashlib
import platform
import argparse
from pathlib import Path
from typing import Any, Optional

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===================== DEFAULT CONFIGURATION =====================
DEFAULT_BASE_MODEL = "microsoft/Phi-4-reasoning"
DEFAULT_LORA_ADAPTER = "/home/kyleliu789/workspace/LlamaFactory/outputs/phi4-reasoning-14b-gpt52-high-reasoning-original"

NUM_QUESTIONS = 300
GENERATION_SEED = 42
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_NEW_TOKENS = 8192
BATCH_SIZE = 4

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


def split_channels(raw_output: str) -> tuple[str, str]:
    text = raw_output.strip()
    analysis = ""
    final = ""

    if "</think>" in text:
        analysis, final = text.split("</think>", 1)
        analysis = analysis.replace("<think>", "").strip()
        final = final.replace("<|im_end|>", "").strip()
    elif "<think>" in text:
        parts = text.split("<think>", 1)
        analysis = parts[1].strip()
        final = parts[0].strip()
    else:
        analysis = ""
        final = text

    return analysis, final


def read_jsonl(path: Path) -> dict[int, dict]:
    completed = {}
    if not path.is_file():
        return completed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            idx = row.get("dataset_index", line_number - 1)
            completed[idx] = row
        except json.JSONDecodeError:
            continue
    return completed


def append_records_atomic(path: Path, records: list[dict]):
    existing = path.read_bytes() if path.exists() else b""
    addition = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as sink:
        sink.write(existing)
        sink.write(addition)
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, path)


def wilson_interval(successes, total, z=1.959963984540054):
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    margin = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return center - margin, center + margin


def resolve_adapter_path(adapter_arg: Optional[str]) -> Optional[str]:
    if not adapter_arg or adapter_arg.strip().lower() in ("none", "false", "0", ""):
        return None
    adapter_str = adapter_arg.strip()
    try:
        p = Path(adapter_str)
        if p.is_absolute() and p.exists():
            return str(p.resolve())
    except (OSError, PermissionError):
        pass
    return adapter_str


def load_model_and_tokenizer(base_model_id: str, adapter_path: Optional[str]):
    print(f"\nLoading tokenizer from {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if not adapter_path:
        print(f"\nLoading base model {base_model_id} in bfloat16 (no LoRA adapter)...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print(f"\nLoading base model {base_model_id} in bfloat16...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"Applying LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path)

    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="ReasonIF Benchmark Evaluation for Phi-4-reasoning")
    parser.add_argument("--base_model_id", type=str, default=DEFAULT_BASE_MODEL,
                        help=f"Base model ID (default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--adapter_path", type=str, default=DEFAULT_LORA_ADAPTER,
                        help=f"LoRA adapter path (default: {DEFAULT_LORA_ADAPTER}). Pass '' or 'none' for base model only.")
    parser.add_argument("--no_lora", action="store_true",
                        help="Convenience flag to evaluate base model only without LoRA adapter.")
    parser.add_argument("--model_label", type=str, default="phi4_reasoning_14b_gpt52_high",
                        help="Label for output directory and summary table.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    args = parser.parse_args()

    base_model_id = args.base_model_id
    adapter_arg = None if args.no_lora else args.adapter_path
    resolved_adapter = resolve_adapter_path(adapter_arg)
    model_label = args.model_label

    print("=" * 80)
    print(f"REASONIF BENCHMARK EVALUATION: {model_label}")
    print("=" * 80)
    print(f"Base Model:       {base_model_id}")
    print(f"Adapter:          {resolved_adapter or 'None (Base Model)'}")
    print(f"Questions:        {NUM_QUESTIONS}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Device:           {torch.cuda.get_device_name(0)}")
    print(f"VRAM:             {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 80)

    run_key = f"full_n{NUM_QUESTIONS}_gseed{GENERATION_SEED}_new{MAX_NEW_TOKENS}_temp1_top_p095_native_chat"
    output_dir = WORKSPACE / "outputs" / "reasonif_phi4_reasoning_final_paper" / model_label / run_key
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_responses.jsonl"
    scored_path = output_dir / "scored_responses.jsonl"

    dataset_path = REASONIF_ROOT / "data" / "reasonIF_dataset.json"
    # Benchmark word-limit calibration: Standardized to Qwen3-14B reference table limits as the matched 14B parameter scale baseline
    calibration_model = getattr(args, "calibration_model", "Qwen3-14B")
    official_messages, official_dataset = prepare_message_list(calibration_model, input_path=str(dataset_path))
    assert len(official_dataset) == NUM_QUESTIONS, f"Expected {NUM_QUESTIONS} questions, got {len(official_dataset)}"

    prepared_rows = []
    for idx, row in enumerate(official_dataset):
        row = dict(row)
        row["dataset_index"] = idx
        row["original_prompt"] = row["prompt"]
        prepared_rows.append(row)

    prep_file = output_dir / "reasonif_dataset_after_official_preparation.json"
    if not prep_file.exists():
        prep_file.write_text(json.dumps(prepared_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    completed_raw = read_jsonl(raw_path)
    pending_indices = [i for i in range(NUM_QUESTIONS) if i not in completed_raw]
    print(f"\n[Generation Status] Total: {NUM_QUESTIONS}, Completed: {len(completed_raw)}, Pending: {len(pending_indices)}")

    if pending_indices:
        model, tokenizer = load_model_and_tokenizer(base_model_id, resolved_adapter)

        manifest = {
            "model_label": model_label,
            "base_model": base_model_id,
            "adapter_path": resolved_adapter,
            "num_questions": NUM_QUESTIONS,
            "generation_seed": GENERATION_SEED,
            "calibration_model": calibration_model,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "batch_size": args.batch_size,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
        }
        (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        eos_token_ids = [100265, 100257]

        pbar = tqdm(total=NUM_QUESTIONS, initial=len(completed_raw), desc=f"ReasonIF ({model_label})")
        batch_size = args.batch_size
        for b_start in range(0, len(pending_indices), batch_size):
            batch_idxs = pending_indices[b_start: b_start + batch_size]
            batch_rows = [prepared_rows[i] for i in batch_idxs]

            batch_prompts = []
            for row in batch_rows:
                msgs = [{"role": "user", "content": row["prompt"]}]
                formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                batch_prompts.append(formatted)

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")
            input_len = inputs.input_ids.shape[1]

            # Deterministic generator seed matching recorded seed metadata
            torch.manual_seed(GENERATION_SEED + batch_idxs[0])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(GENERATION_SEED + batch_idxs[0])

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_ids,
                )

            new_records = []
            for row, out in zip(batch_rows, outputs):
                gen_ids = out[input_len:]
                while len(gen_ids) > 0 and gen_ids[-1].item() in (tokenizer.pad_token_id, 100265, 100257):
                    gen_ids = gen_ids[:-1]

                raw_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
                analysis_text, content_text = split_channels(raw_text)

                new_records.append({
                    **row,
                    "model_id": base_model_id,
                    "adapter_path": resolved_adapter,
                    "raw_output": raw_text,
                    "reasoning_content": analysis_text,
                    "content": content_text,
                    "prompt_tokens": int(input_len),
                    "output_tokens": int(len(gen_ids)),
                    "finish_reason": "length" if len(gen_ids) >= MAX_NEW_TOKENS else "stop",
                    "truncated": len(gen_ids) >= MAX_NEW_TOKENS,
                    "seed": GENERATION_SEED + row["dataset_index"],
                })

            append_records_atomic(raw_path, new_records)
            completed_raw.update({r["dataset_index"]: r for r in new_records})
            pbar.update(len(batch_rows))

        pbar.close()
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("SCORING GENERATED RESPONSES WITH OFFICIAL REASONIF EVALUATOR")
    print("=" * 80)

    all_records = [completed_raw[i] for i in range(NUM_QUESTIONS)]
    scored_rows = []

    for record in all_records:
        eval_item = {
            "prompt": record["prompt"],
            "instruction_id_list": record["instruction_id_list"],
            "kwargs": record["kwargs"],
            "response": record["content"],
        }
        eval_res = evaluate_instruction_following(eval_item)
        passed = eval_res["all_satisfied"]

        predicted = extract_final_answer(record["content"], record["source"])
        correct = canonical_answer(predicted, record["source"]) == canonical_answer(record["answer"], record["source"])

        scored_rows.append({
            **record,
            "constraint": record["constraint_name"][0],
            "predicted_answer": predicted,
            "canonical_target": canonical_answer(record["answer"], record["source"]),
            "canonical_predicted": canonical_answer(predicted, record["source"]),
            "instruction_following": float(passed),
            "answer_correct": float(correct),
            "joint_success": float(passed and correct),
            "missing_answer": float(predicted is None or str(predicted).strip() == ""),
            "format_valid": float(extract_final_answer(record["content"], record["source"]) is not None),
        })

    scored_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scored_rows), encoding="utf-8")

    frame = pd.DataFrame(scored_rows)
    metric_columns = ["instruction_following", "answer_accuracy", "joint_success", "format_valid", "missing_answer", "truncated"]
    frame["answer_accuracy"] = frame["answer_correct"]

    overall = pd.DataFrame([{
        "model": model_label,
        "questions": len(frame),
        **{column: frame[column].mean() for column in metric_columns},
        "mean_output_tokens": frame["output_tokens"].mean(),
        "median_output_tokens": frame["output_tokens"].median(),
    }])

    category = frame.groupby("constraint").apply(lambda g: pd.Series({
        "questions": len(g),
        "instruction_following": g["instruction_following"].mean(),
        "answer_accuracy": g["answer_correct"].mean(),
        "joint_success": g["joint_success"].mean(),
        "format_valid": g["format_valid"].mean(),
        "missing_answer": g["missing_answer"].mean(),
        "truncated": g["truncated"].mean(),
        "mean_output_tokens": g["output_tokens"].mean(),
        "median_output_tokens": g["output_tokens"].median(),
    }), include_groups=False).reset_index()

    source_analysis = frame.groupby("source").apply(lambda g: pd.Series({
        "questions": len(g),
        "instruction_following": g["instruction_following"].mean(),
        "answer_accuracy": g["answer_correct"].mean(),
        "joint_success": g["joint_success"].mean(),
        "format_valid": g["format_valid"].mean(),
        "missing_answer": g["missing_answer"].mean(),
        "truncated": g["truncated"].mean(),
        "mean_output_tokens": g["output_tokens"].mean(),
        "median_output_tokens": g["output_tokens"].median(),
    }), include_groups=False).reset_index()

    print("\nOVERALL RESULTS:")
    print(overall.to_string(index=False))
    print("\nBY CONSTRAINT:")
    print(category.to_string(index=False))
    print("\nBY SOURCE TASK:")
    print(source_analysis.to_string(index=False))

    overall.to_csv(output_dir / "overall_metrics.csv", index=False)
    category.to_csv(output_dir / "by_constraint.csv", index=False)
    source_analysis.to_csv(output_dir / "by_source.csv", index=False)

    summary_lines = [
        f"# ReasonIF Result: {model_label}", "",
        f"- Base Model: `{base_model_id}`",
        f"- LoRA Adapter: `{resolved_adapter or 'None'}`",
        f"- Questions Evaluated: `{len(frame)}`",
        "",
        "## Overall Metrics",
        overall.to_markdown(index=False),
        "",
        "## By Constraint Category",
        category.to_markdown(index=False),
        "",
        "## By Source Task",
        source_analysis.to_markdown(index=False),
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    hashes = {}
    for p in output_dir.iterdir():
        if p.is_file() and p.name != "artifact_sha256.json":
            hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (output_dir / "artifact_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")

    archive_base = output_dir.parent / f"{model_label}_{run_key}_reasonif_results"
    archive = shutil.make_archive(str(archive_base), "zip", output_dir)
    print(f"\nCreated result archive: {archive}")

    marker_file = WORKSPACE / "eval_reasonif" / f"EVALUATION_COMPLETE_{model_label}"
    marker_file.write_text(f"COMPLETED at {time.ctime()}\nArchive: {archive}\n", encoding="utf-8")
    print(f"Created completion marker: {marker_file}")


if __name__ == "__main__":
    main()
