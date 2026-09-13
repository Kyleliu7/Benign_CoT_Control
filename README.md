# CoT Controllability & Prefix Intervention: Complete Reproducibility & Research Repository

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Verification: 100% Passed](https://img.shields.io/badge/Row--Level%20Verification-100%25%20Passed-brightgreen.svg)]()
[![Contamination: 0.0%](https://img.shields.io/badge/Train--Test%20Contamination-0.00%25-success.svg)]()

This repository contains the complete, self-contained reproducibility package, experimental data, training recipes, evaluation pipelines, and row-level generation artifacts for the study on **Chain-of-Thought (CoT) Controllability and Prefix Transfer** across Qwen-2.5/Qwen3-14B and OpenAI GPT-OSS-20B models.

All **36 worksheets** from the master audited workbook [`ALL_EXPERIMENTS_BY_MODEL_AUDITED.xlsx`](results/ALL_EXPERIMENTS_BY_MODEL_AUDITED.xlsx) have been independently verified against row-level model generations with **zero discrepancies**.

---

## 🚀 Quickstart & Push to GitHub

This repository is completely self-contained and pre-configured for deployment. To push this repository to your GitHub account:

```bash
# 1. Add your remote repository URL
git remote add origin https://github.com/<your-username>/<your-repo-name>.git

# 2. Push to main branch
git branch -M main
git push -u origin main
```

*(All tracked files are strictly under 12 MB each, with a total repository size under 180 MB, ensuring fast, frictionless pushing without Git LFS requirements).*

---

## 🔬 Core Scientific Breakthroughs

1. **Head-Driven Controllability**:
   - Forcing the fine-tuned model (`kyleliu789/qwen3-14b-gpt52-high-reasoning-original`) to start with the Base model's conversational prefix (`<think>\nOkay, let's try to figure out...`) causes its procedural compliance to plummet from **48.93% down to 28.26%** ($p = 7.24 \times 10^{-17}$).
   - Stripped of its structured reasoning header, the fine-tuned model collapses to standard Base model performance ($p = 0.554$).
2. **Donor Prefix Dictates Reasoning Length**:
   - The opening 10 tokens causally set the reasoning length regime:
     - Base donor prefix $\to$ verbose reasoning traces (**805.8 tokens** Base, **677.2 tokens** SFT).
     - SFT donor prefix $\to$ compact reasoning traces (**388.1 tokens** Base, **381.0 tokens** SFT).
3. **SFT Header Receptivity in Base**:
   - Prefilling SFT's constraint-OFF header into the Base model (`SFT -> Base`) boosts Base compliance from **27.48% to 35.45%** ($+7.98\text{ pp}, p = 2.42 \times 10^{-8}$).
4. **Cross-Architecture Vocabulary Boundary**:
   - In cross-model transfer into OpenAI GPT-OSS-20B Base (`Qwen SFT Prefix -> GPT-OSS Base`), formatting steerability fails to transfer universally (21.33% vs 24.08% Untouched Base), demonstrating that prefix steerability requires vocabulary alignment and matching internal attention priors.

---

## 📁 Repository Structure

```
.
├── README.md                              # This document: overview, quickstart, reproduction
├── requirements.txt                       # Minimal pinned Python dependencies
├── .gitignore                             # Clean ignore rules for virtualenvs and temporary caches
├── configs/                               # Training configurations for fine-tuning
│   ├── qwen3_14b_sft_lora-gpt-52-high-reasoning-original.yaml
│   ├── qwen3_14b_sft_lora-gpt-52-high-reasoning-normalized.yaml
│   ├── train_qwen3_14b_lora.yaml
│   └── train_gpt_oss_20b_lora.yaml
├── data/
│   ├── eval_prompts/                      # Benchmark evaluation datasets
│   │   ├── diverse_prompts.json           # 50 Haskins diverse test prompts
│   │   ├── haskins_exact_prompt_keywords.json # 50 exact calibrated keywords
│   │   └── reasonif_dataset_300.json      # 300 official ReasonIF questions
│   └── training/                          # N=212 training datasets for SFT models
│       ├── gpt52_high_reasoning_original.json
│       ├── gpt52_high_reasoning_glm53_plain_prose.json
│       ├── claude_sonnet_gpt52_prompts_sft.json
│       ├── qwen3_235b_gpt52_prompts_sft.json
│       ├── qwen3_14b_gpt52_prompts_sft.json
│       ├── gpt52_no_reasoning.json
│       └── reasonif_14b_meta_reasoning_sft.json
├── evaluators/                            # Turnkey re-scoring engines
│   ├── haskins_evaluator.py               # Haskins 10-task strict & partial evaluator (CLI enabled)
│   ├── reasonif_evaluator.py              # Official ReasonIF grader (CLI enabled)
│   ├── test_evaluators.py                 # Evaluator unit test suite (5 boundary tests)
│   └── reasonif_official/                 # Pinned official ReasonIF grader modules (commit 706b953)
├── results/
│   ├── ALL_EXPERIMENTS_BY_MODEL_AUDITED.xlsx # Master 36-tab audited spreadsheet
│   ├── audited-results-import-ready.csv   # 57-row validated candidate table
│   ├── pending-validation-import-ready.csv# 12-row defect tracking ledger
│   ├── all-tables-verification-log.json   # Machine-readable verification log (0 discrepancies)
│   ├── haskins_per_task_comparison.csv    # Summary CSV: Haskins per-task results
│   ├── prefix_transfer_2x2_matrix.csv     # Summary CSV: 2x2 transfer matrix
│   ├── qwen_prefix_transfer_comparison.csv# Summary CSV: Qwen->GPT cross transfer
│   ├── reasonif_by_constraint_comparison.csv # Summary CSV: ReasonIF constraints
│   ├── reasonif_by_source_comparison.csv  # Summary CSV: ReasonIF task sources
│   └── scored_runs/                       # Complete row-level scored JSONLs (30 conditions, ~168 MB)
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
├── scripts/                               # Inference and verification runners
│   ├── reproduce_all_sheets.py            # Master verification script across all 30 conditions
│   ├── reproduce_tables.py                # Table 1–5 reproduction script
│   ├── run_cross_prefix_transfer_2x2.py   # Haskins 2x2 transfer inference runner
│   ├── run_cross_prefix_transfer_gpt_oss.py # Cross-model transfer inference runner
│   └── evaluate_reasonif_qwen_normalized.py # ReasonIF evaluation runner
└── docs/
    ├── TRACEABILITY_MATRIX.md             # Complete line-by-line provenance for all 36 sheets
    ├── METHODOLOGY_AND_LIMITATIONS.md     # In-depth grading definitions, boundaries, and limits
    ├── suppression-instruction-trace.md   # Word suppression prompt defect analysis
    └── remaining-research-evidence.md     # Claims-to-evidence validation package
```

---

## 📊 How to Re-Score Any Model

### 1. Re-Score ReasonIF Responses
Evaluate any ReasonIF output file against official constraints and math correctness:
```bash
python -m evaluators.reasonif_evaluator --input results/scored_runs/reasonif_qwen3_14b_sft_gpt52_high.jsonl
```
*Outputs Instruction Following Score (IFS), Answer Accuracy, and Joint Success Rate matching the spreadsheet.*

### 2. Re-Score Haskins Controllability Responses
Evaluate any Haskins output file across all 10 tasks with strict binary and partial compliance:
```bash
python -m evaluators.haskins_evaluator --input results/scored_runs/haskins_qwen3_14b_base.jsonl
```
*Add `--continuation-only` to score continuation tokens after the injected prefix.*

### 3. Run Evaluator Unit Tests
```bash
python -m unittest evaluators.test_evaluators
```
*Expected result*: `Ran 5 tests in 0.000s ... OK`

### 4. Run Master Suite Verification
Cross-verify all 30 experimental conditions against `ALL_EXPERIMENTS_BY_MODEL_AUDITED.xlsx`:
```bash
python scripts/reproduce_all_sheets.py
```
*Expected result*: `ALL 30 CONDITIONS VERIFIED (0 DISCREPANCIES)`.

---

## 🛠️ Model Training Recipes

All fine-tuned models were trained using **LlamaFactory** (PEFT LoRA) with hyperparameters:
- **Base Models**: `Qwen/Qwen3-14B` (Dense 14.7B) and `openai/gpt-oss-20b` (Dense 21.4B)
- **LoRA Parameters**: Rank $r=32$, Alpha $\alpha=64$, Dropout=0.05
- **Optimization**: Learning rate $1.0 \times 10^{-4}$, Cosine schedule, Warmup ratio 0.1, BF16
- **Target Modules**: All linear layers (`q, k, v, o, gate, up, down`) for Qwen; attention projections (`q, k, v, o`) for GPT-OSS
- **Epochs**: 3.0 epochs on held-out reasoning datasets ($N=212$ items)

To reproduce training using LlamaFactory:
```bash
llamafactory-cli train configs/qwen3_14b_sft_lora-gpt-52-high-reasoning-original.yaml
```

Checkpoints are linked directly from Hugging Face:
- `kyleliu789/qwen3-14b-gpt52-high-reasoning-original`
- `kyleliu789/qwen3-14b-gpt52-high-reasoning-normalized`
- `Qwen/Qwen3-14B`
- `openai/gpt-oss-20b`

---

## ⚠️ Known Methodological Defect Notes

1. **`multiple_word_suppression` Prompt Defect**: The generation prompt template instructed the model to avoid a single keyword (`"strawberry"`), but the evaluation evaluated against 4 words. Coded as `METHODOLOGICAL DEFECT (EXCLUDED METRIC)`. Under actual single-word instruction, Base achieves 46.0% and SFT achieves 76.0%.
2. **Haskins 2x2 Transfer Matrix Word Suppression**: Phase 2 recipient prompt cache bound uncalibrated keywords (43/50 mismatches). Word suppression is staged as `BLOCKED_UNCALIBRATED_CACHE` in 2x2. The remaining 9 tasks are 100% valid.
3. **ReasonIF 42% Prefix Attribution**: Historical documentation cited 42% as a static header; full provenance confirms it was produced by a dynamic teacher prefix with constraint ON. True static header achieves **21.67% IFS**.
4. **LoRA Scope Difference**: Qwen LoRA targeted all linear layers; GPT-OSS LoRA targeted attention projections only.
5. **MathIF Exclusion**: MathIF was excluded by researcher decision and is completely out of scope.

For full line-by-line provenance, see [`docs/TRACEABILITY_MATRIX.md`](docs/TRACEABILITY_MATRIX.md).
