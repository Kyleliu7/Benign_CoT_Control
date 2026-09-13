# Benign CoT Control & Prefix Intervention

A research repository investigating Chain-of-Thought (CoT) controllability and prefix intervention across Qwen3-14B and OpenAI GPT-OSS-20B. Evaluates whether prefilling reasoning headers steers reasoning length, formatting, and constraint compliance without compromising base reasoning accuracy.

This package contains all fine-tuning datasets, custom LoRA training configurations, raw and scored generation outputs across 30 experimental conditions, and turnkey CLI re-scoring evaluators.

---

## Setup and Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.4+ (CUDA 12.1+ recommended for GPU inference/training)

### Installation
```bash
pip install -r requirements.txt
```

---

## Repository Structure

```text
.
├── configs/                               # Custom LoRA training configurations
│   ├── qwen3_14b_sft_lora-claude-37.yaml
│   ├── qwen3_14b_sft_lora-deepseek-r1.yaml
│   ├── qwen3_14b_sft_lora-gpt-52-high-reasoning-normalized.yaml
│   ├── qwen3_14b_sft_lora-gpt-52-high-reasoning-original.yaml
│   ├── qwen3_14b_sft_lora-gpt52-long.yaml
│   ├── qwen3_14b_sft_lora-no-reasoning.yaml
│   ├── qwen3_14b_sft_lora-qwen3-235b.yaml
│   ├── qwen3_14b_sft_lora-reasonflux-f1.yaml
│   ├── qwen3_14b_sft_lora-self-distill.yaml
│   ├── qwen3_14b_sft_lora-svamp_meta_discussion.yaml
│   └── train_gpt_oss_20b_lora.yaml
├── data/
│   ├── eval_prompts/                      # Benchmark evaluation datasets
│   │   ├── diverse_prompts.json           # 50 Haskins diverse test prompts
│   │   ├── haskins_exact_prompt_keywords.json # 50 calibrated suppression keywords
│   │   └── reasonif_dataset_300.json      # 300 official ReasonIF benchmark problems
│   └── training/                          # SFT training datasets & registry
│       ├── dataset_info.json              # LlamaFactory dataset registration
│       ├── claude_sonnet_gpt52_prompts_sft.json
│       ├── deepseek_r1_gpt52_prompts_sft.json
│       ├── gpt52_combined_short_long_424.json
│       ├── gpt52_high_reasoning_glm53_plain_prose.json
│       ├── gpt52_high_reasoning_original.json
│       ├── gpt52_hybrid_50short_50long.json
│       ├── gpt52_long_reasoning_prompts_sft.json
│       ├── gpt52_no_reasoning.json
│       ├── qwen3_14b_gpt52_prompts_sft.json
│       ├── qwen3_235b_gpt52_prompts_sft.json
│       └── reasonif_14b_meta_reasoning_sft.json
├── evaluators/                            # Turnkey re-scoring engines
│   ├── haskins_evaluator.py               # Haskins 10-task strict & partial compliance CLI
│   ├── reasonif_evaluator.py              # Official ReasonIF constraint & accuracy CLI
│   ├── test_evaluators.py                 # Evaluator unit test suite
│   └── reasonif_official/                 # Pinned official ReasonIF grader modules
├── results/
│   ├── ALL_EXPERIMENTS_BY_MODEL_AUDITED.xlsx # Master 36-tab audited workbook
│   ├── audited-results-import-ready.csv   # Primary candidate summary table
│   ├── pending-validation-import-ready.csv# Known anomaly and defect ledger
│   ├── all-tables-verification-log.json   # Machine-readable verification log (0 discrepancies)
│   ├── haskins_per_task_comparison.csv    # Haskins per-task results
│   ├── prefix_transfer_2x2_matrix.csv     # 2x2 prefix transfer matrix
│   ├── qwen_prefix_transfer_comparison.csv# Cross-model transfer summary
│   ├── reasonif_by_constraint_comparison.csv # ReasonIF constraint breakdown
│   ├── reasonif_by_source_comparison.csv  # ReasonIF task source breakdown
│   └── scored_runs/                       # 30 row-level scored JSONLs (~168 MB total)
│       ├── haskins_qwen3_14b_base.jsonl
│       ├── haskins_qwen3_14b_gpt52_sft.jsonl
│       ├── haskins_2x2_base_donor_to_base.jsonl
│       ├── haskins_2x2_sft_donor_to_base.jsonl
│       ├── haskins_qwen_sft_prefix_to_gpt_oss_base.jsonl
│       ├── reasonif_qwen3_14b_base_untouched.jsonl
│       ├── reasonif_qwen3_14b_prefix_constraint_on.jsonl
│       ├── reasonif_qwen3_14b_sft_gpt52_high.jsonl
│       ├── reasonif_gpt_oss_20b_base.jsonl
│       └── ... (all 30 experimental conditions)
├── scripts/                               # Verification & execution scripts
│   ├── reproduce_all_sheets.py            # Master verifier across all 30 conditions
│   ├── reproduce_tables.py                # Reproduces Tables 1–5 from raw runs
│   ├── run_cross_prefix_transfer_2x2.py   # Haskins 2x2 transfer inference script
│   └── run_cross_prefix_transfer_gpt_oss.py # Cross-model transfer inference script
├── requirements.txt                       # Python dependencies
└── README.md                              # This document
```

---

## Evaluation & Reproduction

### 1. Evaluator Unit Tests
Verify all scoring rules and boundary behaviors against unit tests:
```bash
python -m unittest evaluators.test_evaluators
```

### 2. Verify All 30 Conditions Against the Master Spreadsheet
Independently re-evaluates all 30 conditions and checks for zero discrepancies against ALL_EXPERIMENTS_BY_MODEL_AUDITED.xlsx:
```bash
python scripts/reproduce_all_sheets.py
```

### 3. Reproduce Benchmark Tables 1 through 5
Regenerates the summary tables directly from raw JSONL outputs:
```bash
python scripts/reproduce_tables.py
```

---

## Re-Scoring Model Outputs

### Haskins Controllability Benchmark
Evaluate model generations across all 10 Haskins tasks with strict and partial compliance scoring:

```bash
# Evaluate entire generation trace
python -m evaluators.haskins_evaluator --input results/scored_runs/haskins_qwen3_14b_base.jsonl

# Evaluate post-prefix continuation tokens only
python -m evaluators.haskins_evaluator --input results/scored_runs/haskins_qwen3_14b_base.jsonl --continuation-only
```

### ReasonIF Benchmark
Evaluate model generations against official ReasonIF constraints and mathematical correctness:

```bash
python -m evaluators.reasonif_evaluator --input results/scored_runs/reasonif_qwen3_14b_sft_gpt52_high.jsonl
```

---

## Model Training

Models are fine-tuned using [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) with the provided LoRA configurations and datasets:

```bash
# Fine-tune Qwen3-14B on Original GPT-5.2 High Reasoning
llamafactory-cli train configs/qwen3_14b_sft_lora-gpt-52-high-reasoning-original.yaml

# Fine-tune Qwen3-14B on Normalized Reasoning (GLM-5.3 Plain Prose)
llamafactory-cli train configs/qwen3_14b_sft_lora-gpt-52-high-reasoning-normalized.yaml

# Fine-tune GPT-OSS-20B on Harmony Reasoning
llamafactory-cli train configs/train_gpt_oss_20b_lora.yaml
```
