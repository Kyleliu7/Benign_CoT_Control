# Benign CoT Control & Prefix Intervention

A research repository investigating Chain-of-Thought (CoT) controllability and prefix intervention across Qwen3-14B and OpenAI GPT-OSS-20B. Evaluates whether prefilling reasoning headers steers reasoning length, formatting, and constraint compliance without compromising task performance.

Contains all fine-tuning datasets, custom training YAMLs, row-level generation outputs across 30 experimental conditions, and turnkey CLI re-scoring evaluators.

---

## 📁 Repository Structure

`	ext
.
├── configs/                               # Custom LoRA training configs for fine-tuned models
│   ├── qwen3_14b_sft_lora-gpt-52-high-reasoning-original.yaml
│   ├── qwen3_14b_sft_lora-gpt-52-high-reasoning-normalized.yaml
│   ├── qwen3_14b_sft_lora-gpt52-long.yaml
│   ├── qwen3_14b_sft_lora-claude-37.yaml
│   ├── qwen3_14b_sft_lora-qwen3-235b.yaml
│   ├── qwen3_14b_sft_lora-deepseek-r1.yaml
│   ├── qwen3_14b_sft_lora-self-distill.yaml
│   ├── qwen3_14b_sft_lora-no-reasoning.yaml
│   ├── qwen3_14b_sft_lora-reasonflux-f1.yaml
│   ├── qwen3_14b_sft_lora-svamp_meta_discussion.yaml
│   └── train_gpt_oss_20b_lora.yaml
├── data/
│   ├── eval_prompts/                      # Evaluation benchmarks
│   │   ├── diverse_prompts.json           # 50 Haskins diverse test prompts
│   │   ├── haskins_exact_prompt_keywords.json # 50 calibrated suppression keywords
│   │   └── reasonif_dataset_300.json      # 300 official ReasonIF benchmark problems
│   └── training/                          # SFT training datasets & registry
│       ├── dataset_info.json              # LlamaFactory dataset registration
│       ├── gpt52_high_reasoning_original.json
│       ├── gpt52_high_reasoning_glm53_plain_prose.json
│       ├── gpt52_combined_short_long_424.json
│       ├── gpt52_long_reasoning_prompts_sft.json
│       ├── gpt52_hybrid_50short_50long.json
│       ├── gpt52_no_reasoning.json
│       ├── claude_sonnet_gpt52_prompts_sft.json
│       ├── qwen3_235b_gpt52_prompts_sft.json
│       ├── deepseek_r1_gpt52_prompts_sft.json
│       ├── qwen3_14b_gpt52_prompts_sft.json
│       └── reasonif_14b_meta_reasoning_sft.json
├── evaluators/                            # Turnkey re-scoring engines
│   ├── haskins_evaluator.py               # Haskins 10-task strict & partial compliance CLI
│   ├── reasonif_evaluator.py              # Official ReasonIF constraint & accuracy CLI
│   ├── test_evaluators.py                 # Evaluator unit tests
│   └── reasonif_official/                 # Official ReasonIF benchmark grading modules
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
├── requirements.txt                       # Minimal Python dependencies
└── README.md                              # This documentation
`

---

## 🚀 How to Run & Re-Score

### 1. Installation
`ash
pip install -r requirements.txt
`

### 2. Run Evaluator Unit Tests
`ash
python -m unittest evaluators.test_evaluators
`

### 3. Re-Score Haskins Controllability
Evaluate any Haskins model output file across all 10 tasks with strict and partial compliance:
`ash
# Evaluate full trace
python -m evaluators.haskins_evaluator --input results/scored_runs/haskins_qwen3_14b_base.jsonl

# Evaluate continuation tokens only (post-prefix)
python -m evaluators.haskins_evaluator --input results/scored_runs/haskins_qwen3_14b_base.jsonl --continuation-only
`

### 4. Re-Score ReasonIF Responses
Evaluate any ReasonIF output file against official constraints and math correctness:
`ash
python -m evaluators.reasonif_evaluator --input results/scored_runs/reasonif_qwen3_14b_sft_gpt52_high.jsonl
`

### 5. Verify All 30 Conditions Against Spreadsheet
Cross-verify all 30 conditions against ALL_EXPERIMENTS_BY_MODEL_AUDITED.xlsx:
`ash
python scripts/reproduce_all_sheets.py
`

### 6. Reproduce Benchmark Tables 1–5
`ash
python scripts/reproduce_tables.py
`

---

## 🛠️ Model Training

Fine-tune models using [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) with the provided custom configs:

`ash
# Fine-tune Qwen3-14B on Original GPT-5.2 High Reasoning
llamafactory-cli train configs/qwen3_14b_sft_lora-gpt-52-high-reasoning-original.yaml

# Fine-tune Qwen3-14B on Normalized Reasoning (GLM-5.3 Plain Prose)
llamafactory-cli train configs/qwen3_14b_sft_lora-gpt-52-high-reasoning-normalized.yaml

# Fine-tune GPT-OSS-20B on Harmony Reasoning
llamafactory-cli train configs/train_gpt_oss_20b_lora.yaml
`
