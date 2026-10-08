# Benign Reasoning Distillation and Early-Token Steering of Chain-of-Thought Controllability

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Precision: Float32](https://img.shields.io/badge/Precision-Float32%20Deterministic-orange.svg)]()
[![Repository: Benign_CoT_Control](https://img.shields.io/badge/GitHub-Kyleliu7%2FBenign__CoT__Control-blue)](https://github.com/Kyleliu7/Benign_CoT_Control)

Official research repository and turnkey reproduction workspace for the manuscript:
**"Benign Reasoning Distillation and Early-Token Steering of Chain-of-Thought Controllability"**

---

## 🌟 Executive Summary

Instruction-tuned reasoning models often exhibit complex internal dynamics when enforcing constraints during Chain-of-Thought (CoT) generation. This repository provides the complete, pristine codebase, evaluation harnesses, datasets, pre-computed Float32 KL caches, and raw generation artifacts supporting all empirical claims in the paper:

1. **Early-Token Steering & Prefix Controllability**:
   Injecting a 10-token compliant reasoning prefix from an aligned SFT policy into an untouched base model transfers compliance downstream—elevating strict Haskins compliance from **5.6% to 24.8%** (and from **8.0% to 35.4%** on non-character constraints) in Qwen3-14B without any adapter active during recipient decoding.
2. **Dual-Architecture Cross-Evaluation**:
   - **Qwen3-14B**: Exhibits an opening phase transition ($D_{\text{KL}} \approx 25\text{--}28$ nats at $t=0$), collapsing by $94.3\%\text{--}97.5\%$ downstream ($D_{\text{KL}} \le 0.18$ nats). The adapter acts as a structural gatekeeper.
   - **Microsoft Phi-4 Reasoning**: Exhibits a modest opening spike ($4.52$ nats) but maintains high residual persistence ($25.7\%\text{--}31.7\%$) via continuous token-by-token policing. Downstream constraint preservation requires recurrent steering spikes ($P_{99.9} = 6.52$ nats, max spike $11.89$ nats).
3. **Rigorous Benchmark Suites**:
   - **Haskins 500**: 10 CoT-control tasks across 50 questions evaluated over 4-way crossed donor-recipient matrices ($N=500$ completions per condition).
   - **ReasonIF 300**: Official benchmark suite (commit `706b953`) measuring Instruction Following Score (IFS), Answer Accuracy, and Joint Success Rate.
   - **Float32 Forward KL Divergence**: 1,600 verified trajectories analyzed token-by-token strictly inside `<think>...</think>` tags.

---

## 📦 HuggingFace Pretrained Weights & Artifacts

All models and LoRA adapters evaluated in this work are publicly available on HuggingFace:

| Model Family | Component | HuggingFace Hub Identifier | Description |
| :--- | :--- | :--- | :--- |
| **Qwen3-14B** | Base Model | [`Qwen/Qwen3-14B`](https://huggingface.co/Qwen/Qwen3-14B) | Untouched base foundation model |
| **Qwen3-14B** | SFT Adapter | [`kyleliu789/qwen3-14b-gpt52-high-reasoning-original`](https://huggingface.co/kyleliu789/qwen3-14b-gpt52-high-reasoning-original) | LoRA fine-tuned on 212 TeicHAI reasoning trajectories |
| **Phi-4 Reasoning** | Base Model | [`microsoft/phi-4`](https://huggingface.co/microsoft/phi-4) | Native reasoning base model |
| **Phi-4 Reasoning** | SFT Adapter | [`kyleliu789/phi4-reasoning-14b-gpt52-high-reasoning-original`](https://huggingface.co/kyleliu789/phi4-reasoning-14b-gpt52-high-reasoning-original) | LoRA fine-tuned on identical 212 TeicHAI dataset |

The training data (`gpt52_high_reasoning_original.json`, 212 high-reasoning trajectories) is provided in `data/training/`.

---

## 🗂️ Repository Structure

```text
Benign_CoT_Control/
├── configs/                             # Pinned fine-tuning configurations
│   ├── qwen3_14b_sft_lora-gpt-52-high-reasoning-original.yaml
│   └── phi4_reasoning_14b_sft_lora-gpt-52-high-reasoning-original.yaml
│
├── data/                                # Benchmark datasets and evaluation prompts
│   ├── training/
│   │   ├── gpt52_high_reasoning_original.json  # 212 TeicHAI distillation trajectories
│   │   └── dataset_info.json
│   └── eval_prompts/
│       ├── diverse_prompts.json                # 50 base questions for Haskins
│       ├── haskins_exact_prompt_keywords.json  # 50 calibrated suppression keywords
│       └── reasonif_dataset_300.json           # Pinned ReasonIF 300 benchmark
│
├── evaluators/                          # Official & verified deterministic evaluators
│   ├── haskins_evaluator.py             # Haskins 10-constraint checker (strict + partial)
│   ├── reasonif_evaluator.py            # Official ReasonIF grader (IFS, Acc, Joint)
│   ├── test_evaluators.py               # Unit tests verifying edge cases (6/6 pass)
│   └── reasonif_official/               # Pinned official ReasonIF instruction checkers
│
├── reproducible_figures/                # Vector PDF, PNG, SVG + exact CSV sources
│   ├── figure1_haskins_crossed.*        # 4-way crossed donor-recipient transfer
│   ├── figure2_reasonif.*               # ReasonIF IFS, accuracy, and joint success
│   ├── figure3_kl_divergence.*          # Per-token forward KL trajectory curves
│   └── figure4_appendix_heatmaps.*      # Per-task compliance & divergence heatmaps
│
├── results/                             # Full empirical results & scored completions
│   ├── summary_tables/                  # Tables 1-5 from manuscript (CSVs)
│   ├── haskins_500/                     # Scored Haskins runs (Qwen & Phi-4 2x2 matrices)
│   ├── reasonif_300/                    # Scored ReasonIF completions and breakdowns
│   └── kl_divergence/                   # Float32 KL caches (1,600 runs) & spike pairs
│       ├── caches_float32/
│       └── spike_analysis/
│
├── notebooks/                           # 16 interactive Jupyter replication notebooks
│   ├── cot_kl_divergence_analysis.ipynb # Interactive KL distribution & spike explorer
│   ├── Haskins_500_vLLM_Evaluation.ipynb
│   └── ...
│
├── scripts/                             # Turnkey CLI reproduction tools
│   ├── reproduce_tables.py              # Reproduces all 5 manuscript tables
│   ├── generate_reproducible_figures.py # Re-renders Figures 1-4 from raw data
│   ├── evaluate_haskins.py              # Haskins CLI evaluation runner
│   ├── evaluate_reasonif.py             # ReasonIF CLI evaluation runner
│   └── compute_forward_kl.py            # Forward KL computation & percentile CLI
│
├── requirements.txt                     # Pinned Python dependencies
├── LICENSE                              # MIT License
└── README.md
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation
Clone the repository and install required packages in a Python 3.10+ environment:

```bash
git clone https://github.com/Kyleliu7/Benign_CoT_Control.git
cd Benign_CoT_Control
pip install -r requirements.txt
```

### 2. Verify All Manuscript Tables (100% Exact Parity)
Run the automated table reproduction script:

```bash
python scripts/reproduce_tables.py
```

This verifies and displays all 5 manuscript tables against the raw scored logs with zero discrepancies.

### 3. Generate All Publication Figures
Re-generate Figures 1–4 along with their exact source CSV files in PDF, PNG, and SVG vector formats:

```bash
python scripts/generate_reproducible_figures.py
```

Figures are saved directly to `reproducible_figures/`.

### 4. Evaluate Scored Completions via CLI

**Evaluate ReasonIF 300:**
```bash
python scripts/evaluate_reasonif.py --input_file results/reasonif_300/qwen3_14b/reasonif_qwen3_14b_sft_gpt52_high.jsonl
```

**Evaluate Haskins 500:**
```bash
# Evaluate continuation compliance on prefix-transferred run:
python scripts/evaluate_haskins.py --input_file results/haskins_500/qwen3_14b/Qwen__Qwen3-14B__qwen__qwen3-14b__2x2-sft-to-base-on-10tok__81ef2980ca47a7c8 --continuation_only
```

**Compute Forward KL Divergence & Percentiles:**
```bash
python scripts/compute_forward_kl.py --model_family qwen3_14b --dataset haskins_500
python scripts/compute_forward_kl.py --model_family phi4 --dataset haskins_500
```

---

## 📊 Key Manuscript Results

### Table 1: ReasonIF Benchmark Outcomes (300 Tasks)
Official evaluation measuring Instruction Following Score (IFS), Answer Accuracy, and Joint Success Rate:

| Model Family | Condition | IFS (%) | Accuracy (%) | Joint Success (%) | Mean Tokens |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Qwen3-14B** | Base (Untouched) | 37.0% | **80.3%** | 11.7% | 4,402 |
| **Qwen3-14B** | SFT (`gpt52-high`) | 33.3% | 65.3% | 20.7% | 2,024 |
| **Qwen3-14B** | Prefix-OFF (Control) | 32.3% | 73.7% | 22.3% | 2,724 |
| **Qwen3-14B** | **Prefix-ON (Transferred)** | **42.0%** | 73.3% | **31.0%** | 2,571 |
| **Phi-4** | Base (Untouched) | 5.0% | — | — | — |
| **Phi-4** | SFT (`gpt52-high`) | 12.0% | — | — | — |
| **Phi-4** | Prefix-OFF (Control) | 5.0% | — | — | — |
| **Phi-4** | Prefix-ON (Transferred) | 6.0% | — | — | — |

### Table 2: Haskins Standalone Benchmark (500 Pairs)
Constraint compliance without prefix injection:

| Architecture | Model Condition | Continuation Compliance (%) | Strict Binary Pass (%) |
| :--- | :--- | :---: | :---: |
| **Qwen3-14B** | Base (Untouched) | 23.40% | 13.0% (65/500) |
| **Qwen3-14B** | SFT (`gpt52-high`) | **37.89%** | **22.8%** (114/500) |
| **Phi-4** | Base (Untouched) | 9.87% | 4.0% (20/500) |
| **Phi-4** | SFT (`gpt52-high`) | **16.21%** | **9.6%** (48/500) |

### Table 3: Crossed Haskins 4-Way Prefix Transfer
Evaluating early-token steering ($k=10$ prefix tokens transferred from donor to recipient):

| Model | Pairing Condition | All 500 Strict (%) | Non-Char 350 Strict (%) |
| :--- | :--- | :---: | :---: |
| **Qwen3-14B** | Base $\to$ Base (A2) | 5.6% (28/500) | 8.0% (28/350) |
| **Qwen3-14B** | Base $\to$ SFT (A3) | 6.8% (34/500) | 9.7% (34/350) |
| **Qwen3-14B** | **SFT $\to$ Base (A1, Prefix-ON)** | **24.8% (124/500)** | **35.4% (124/350)** |
| **Qwen3-14B** | SFT $\to$ SFT (A4) | 23.2% (116/500) | 33.1% (116/350) |
| **Phi-4** | Base $\to$ Base (A2) | 3.4% (17/500) | 4.9% (17/350) |
| **Phi-4** | Base $\to$ SFT (A3) | 7.2% (36/500) | 10.3% (36/350) |
| **Phi-4** | SFT $\to$ Base (A1, Prefix-ON) | 3.0% (15/500) | 4.3% (15/350) |
| **Phi-4** | SFT $\to$ SFT (A4) | **13.8% (69/500)** | **19.7% (69/350)** |

### Table 4: Downstream Forward KL Divergence Percentiles ($t > 10$)
Float32 per-token forward KL divergence $D_{\text{KL}}(\pi_{\text{SFT}} \parallel \pi_{\text{Base}})$ across downstream reasoning tokens:

| Model & Benchmark | Tokens ($t > 10$) | Early Share (%) | Mean (nats) | P50 | P90 | P95 | P99 | P99.9 | Max Spike | Seq Max P95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen Haskins** | 141,147 | **66.66%** | 0.186 | 0.055 | 0.470 | 0.728 | 1.684 | 6.042 | 15.01 | 9.72 |
| **Phi-4 Haskins** | 76,061 | 33.70% | 0.290 | 0.099 | 0.794 | 1.131 | 2.195 | 6.518 | 11.89 | 8.75 |
| **Qwen ReasonIF** | 106,765 | **81.37%** | 0.077 | 0.025 | 0.196 | 0.278 | 0.568 | 2.196 | 17.75 | 5.49 |
| **Phi-4 ReasonIF** | 105,435 | 28.09% | 0.050 | 0.015 | 0.121 | 0.188 | 0.466 | 1.872 | 11.57 | 3.98 |

---

## 🔬 Qualitative Token-Level Policing Spikes (Table 5)

When base models experience constraint relapse downstream, the SFT policy applies targeted, discrete probability mass reallocation ($D_{\text{KL}} \gg 1.0$ nats):

| Benchmark Task | Constraint | Preceding Context | Pos $t$ | $D_{\text{KL}}$ | Base Token | SFT Token | Mechanism |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `third_person:4` | No 1st-person pronouns | `...proof by contradiction.` | 18 | 1.59 nats | `I` [FAIL] | `It` [PASS] | Impersonal pronoun substitution; blocks self-narration |
| `third_person:4` | No 1st-person pronouns | `...format in Markdown.` | 147 | 11.88 nats | `the` [FAIL] | `**` [PASS] | Structural header; suppresses 1st-person relapse |
| `ReasonIF #84` | Suppress commas | `...process bits sequentially` | 25 | 7.13 nats | `,` [FAIL] | `\n` [PASS] | Clause-boundary newline break; avoids comma violation |
| `multi_word:22` | Suppress *solve*, *solution* | `...3(x^2 - 4x + 3) = 0.` | 45 | 3.96 nats | `Solve` [FAIL] | `Factor` [PASS] | Operational verb substitution; circumvents banned term |
| `word_suppr:68` | Suppress *weigh* | `Step 2: On the scale, we` | 68 | 2.45 nats | `weigh` [FAIL] | `compare` [PASS] | Action circumlocution; overrides canonical scale term |
| `repeat_sent:19` | Repeat sentences twice | `...viral spike protein.` | 90 | 11.65 nats | `Next` [FAIL] | `mRNA` [PASS] | Forces duplicate sentence restart |
| `arrow_prefix:29` | Mandatory `>>>` prefix | `...in lowest terms.` | 74 | 4.12 nats | `We` [FAIL] | `>>>` [PASS] | Injects required syntactic marker at sentence boundary |

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).

---

## 📖 Citation

If you find this work, codebase, or pre-computed caches helpful in your research, please cite:

```bibtex
@article{liu2026benign,
  title={Benign Reasoning Distillation and Early-Token Steering of Chain-of-Thought Controllability},
  author={Liu, Kyle and Contributors},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/Kyleliu7/Benign_CoT_Control}
}
```
