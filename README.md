# Benign Reasoning Distillation and Early-Token Steering of Chain-of-Thought Controllability

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)

Code, scored completions and KL caches for the manuscript of the same name. Everything quoted below is
**recomputed from the raw files in `results/`** by `scripts/reproduce_tables.py`; nothing is copied from the paper.
See [`AUDIT.md`](AUDIT.md) for the audit that produced this version, what changed, and what is still open.

## What is and is not verified in this repository

| Result block | Status |
| :--- | :--- |
| Haskins crossed 2×2, Qwen3-14B (A1–A4), 4 × 500 completions | Raw records present; scores re-derived from text (`scripts/verify_scorer_parity.py`) |
| Haskins crossed 2×2, Phi-4-reasoning (A1–A5), 5 × 500 | Per-item text in `phi4_haskins_500_bundle.json`; scores re-derived |
| Haskins standalone, Phi-4-reasoning Base / SFT | Raw records present; re-derived |
| Haskins standalone Base / SFT / Prefix-OFF / Prefix-ON, Qwen3-14B (`haskins_qwen3_14b_vllm_calibrated_*.jsonl`) | Raw records present; scores re-derived (0 mismatches). Prefix-ON/OFF are reported under **both** continuation and full-trace scoring. (Earlier exploratory records with a different scorer, adapter label `gpt52_short` and numbers such as 23.40 / 37.89 were removed as stale.) |
| ReasonIF Qwen3-14B SFT, Prefix-OFF, Prefix-ON | Raw records present |
| ReasonIF Qwen3-14B Base | Raw records present (`base_model_id: Qwen/Qwen3-14B`, revision `40c06982…`). The file previously under this name was a GPT-OSS-20B run; it is kept as `results/reasonif_300/gpt_oss_20b/reasonif_gpt_oss_20b_medium.jsonl` |
| ReasonIF Phi-4 Base / SFT | Raw records present |
| ReasonIF Phi-4 Prefix-OFF / Prefix-ON | Aggregate CSV only; raw records absent |
| Forward-KL caches (4 × float32) | Present; Table 4 reproduces exactly |
| Token-level "policing" examples | Previous table did not match the stored spike records and was removed (see `AUDIT.md`) |

Missing values are left empty in the summary tables; an empty cell is never a zero.

## Quickstart

```bash
git clone https://github.com/Kyleliu7/Benign_CoT_Control.git && cd Benign_CoT_Control
pip install -r requirements.txt            # CPU analysis; requirements-gpu.txt adds vLLM/torch for generation
python -c "import nltk; nltk.download('punkt_tab')"

# 1. recompute every table from raw records and compare with results/summary_tables/ (exit 1 on mismatch)
python scripts/reproduce_tables.py

# 2. re-score all 7,500 stored Haskins completions with the paper scorer (needs the pinned upstream grader, see below)
python scripts/verify_scorer_parity.py

# 3. re-run the official ReasonIF grader on the raw records (needs the fast-langdetect model to check language items)
python scripts/reproduce_tables.py --rescore-reasonif

# 4. regenerate Figures 1-4 and their source CSVs into reproducible_figures/
python scripts/generate_reproducible_figures.py

python -m pytest evaluators
```

### Pinned upstream Haskins grader

Haskins scores come from `grade_compliance` in [Reih02/cot_obfuscation_code](https://github.com/Reih02/cot_obfuscation_code) at commit
`38dca62e407679ea361e04590ae3409de3f58ec7`, plus this repo's "calibrated" suppression checkers. The upstream file is not vendored (it has no
license). The evaluator loads it from `$HASKINS_UPSTREAM_ROOT` (a local clone), else a cached copy under `data/_upstream/`, else downloads it once
from raw.githubusercontent.com; its SHA-256 is always checked
(`evaluate.py` = `8367226b…aad45`, `diverse_prompts.json` = `1fdebb6b…94529d69`).

`evaluators/haskins_evaluator.py` has two scorers: `scorer="paper"` (default, reproduces every stored score) and `scorer="semantic"` (an older, stricter
re-implementation that does **not** match the stored numbers; kept only as a diagnostic). Notable differences are listed in `AUDIT.md`.

## Results (recomputed)

### ReasonIF, 300 questions (`table1_reasonif_overall.csv`)
Prefix rows are scored on the recipient continuation only. IFS = all instructions followed; Joint = IFS and correct.

| Model | Condition | IFS | Accuracy | Joint | Mean tokens | Truncated |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen3-14B | Base | 14.0 % (42) | 80.3 % (241) | 11.7 % (35) | 4,401.7 | 19 |
| Qwen3-14B | SFT | 33.3 % (100) | 65.3 % (196) | 20.7 % (62) | 2,023.7 | 6 |
| Qwen3-14B | Prefix-OFF | 32.0 % (96) | 73.7 % (221) | 22.3 % (67) | 2,926.6 | 11 |
| Qwen3-14B | Prefix-ON | 42.0 % (126) | 73.3 % (220) | 31.0 % (93) | 2,571.0 | 13 |
| Phi-4-reasoning | Base | 5.0 % (15) | 75.3 % (226) | 3.0 % (9) | 4,044.8 | 12 |
| Phi-4-reasoning | SFT | 12.0 % (36) | 69.0 % (207) | 8.0 % (24) | 2,512.2 | 7 |
| Phi-4-reasoning | Prefix-OFF† | 4.7 % (14) | 73.7 % (221) | 3.0 % (9) | 3,993.3 | 14 |
| Phi-4-reasoning | Prefix-ON† | 6.0 % (18) | 74.3 % (223) | 3.7 % (11) | 3,793.4 | 11 |

† aggregate CSV only. In the Qwen prefix runs the forced prefix is shorter than 10 tokens for 16 (OFF) and 15 (ON) of 300 questions
(15 OFF prefixes are empty), so those items are effectively unprefixed. Scores re-checked with the official grader: 0 mismatches on the 205
non-language items per file; language / english_capital items need the fast-langdetect model (see Quickstart step 3).

### Haskins, crossed donor → recipient, continuation-scored (`table3_haskins_crossed_2x2.csv`)
500 question–constraint pairs (50 questions × 10 constraints). Strict = score exactly 1.0.

| Model | Donor → Recipient | Mean | Strict | Non-char mean (350) | Non-char strict | Full-trace mean |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen3-14B | Base → Base | 17.82 | 25 (5.0 %) | 16.69 | 10 | 17.48 |
| Qwen3-14B | Base → SFT | 16.70 | 21 (4.2 %) | 15.18 | 8 | 16.55 |
| Qwen3-14B | SFT → Base (Prefix-ON) | 37.34 | 109 (21.8 %) | 32.30 | 67 | 36.28 |
| Qwen3-14B | SFT → SFT | 32.07 | 96 (19.2 %) | 28.68 | 62 | 30.93 |
| Phi-4-reasoning | Base → Base | 10.38 | 11 (2.2 %) | 12.64 | 11 | 10.25 |
| Phi-4-reasoning | Base → SFT | 14.46 | 36 (7.2 %) | 17.18 | 36 | 13.08 |
| Phi-4-reasoning | SFT → Base (Prefix-ON) | 10.92 | 15 (3.0 %) | 13.36 | 15 | 10.44 |
| Phi-4-reasoning | SFT → SFT | 18.07 | 69 (13.8 %) | 22.01 | 69 | 16.99 |
| Phi-4-reasoning | SFT → Base, constraint OFF | 9.89 | 12 (2.4 %) | 11.94 | 12 | 9.74 |

Caveats that apply to this table:
* Haskins "10-token" prefixes **include the `<think>` token** (9 content tokens).
* Qwen SFT-donor prefixes differ between the two recipient runs for 45 / 500 items (vLLM batch nondeterminism); base-donor prefixes are identical for 500 / 500.
* Many SFT-donor prefixes already restate or demonstrate the constraint (e.g. `**Thinking in lowercase only**`, `**Chain of Thought in Uppercase`), so the
  continuation is conditioned on constraint-aware text, not on a generic "SFT-style" opening.
* Three tasks (`arrow_prefix`, `repeat_sentences`, `alternating_case`) are at or near floor in every condition, and a large part of the gain comes from `third_person`, `lowercase_thinking`, `uppercase_thinking`, `meow_between_words` and word suppression (`table6_haskins_per_task.csv`).
* Run health (`table7_haskins_run_health.csv`): generations that hit the 3,000-token cap range from 7 to 27 of 500 (Qwen) and 52 vs 3 (Phi base vs SFT standalone).
  Response length differs substantially between conditions (Qwen continuation mean 986 → 677 tokens), which is a confound for sentence-level compliance.

### Haskins standalone, Phi-4-reasoning (`table2_haskins_standalone.csv`)
Phi-4-reasoning: Base 9.87 % mean, 13 / 500 strict; SFT 16.21 % mean, 62 / 500 strict.

Qwen3-14B (calibrated vLLM protocol, 500 pairs each; strict = score exactly 1.0):

| Condition | Scoring | Mean | Strict | Non-char mean (350) | Non-char strict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Base | whole trace | 17.21 | 23 (4.6 %) | 16.01 | 7 |
| SFT | whole trace | 30.63 | 80 (16.0 %) | 28.89 | 63 |
| Prefix-OFF | continuation only | 20.97 | 59 (11.8 %) | 21.39 | 52 |
| Prefix-OFF | full trace incl. prefix | 20.12 | 41 (8.2 %) | 20.92 | 41 |
| Prefix-ON | continuation only | 37.97 | 112 (22.4 %) | 33.23 | 72 |
| Prefix-ON | full trace incl. prefix | 36.94 | 89 (17.8 %) | 32.64 | 70 |

Compare Prefix-ON with Prefix-OFF **within the same scoring basis** (continuation 37.97 vs 20.97; full trace 36.94 vs 20.12). The manuscript's earlier
table paired Prefix-OFF's continuation score with Prefix-ON's full-trace score. The 2×2 table above contains an independent rerun of the same SFT→Base
condition (37.34 / 109 continuation), so this condition has been run twice with similar results.

### Forward KL (`table4_kl_percentiles.csv`, `results/kl_divergence/kl_divergence_executive_report.md`)
D_KL(π_SFT ‖ π_Base) per token on SFT-generated histories (teacher forcing, float32, first ≤ 512 reasoning tokens). Two windows are reported because
they give very different "early share" numbers: the first-10 share is 66.7 / 81.4 / 33.7 / 28.1 % (Qwen Haskins / Qwen ReasonIF / Phi Haskins / Phi ReasonIF)
over positions 1–100 of traces with ≥ 100 tokens, and 41.9 / 56.5 / 18.7 / 9.5 % over the full measured length. Downstream (t > 10) maxima and per-sequence-max
P95 are *larger* for Qwen than for Phi on both benchmarks, so the table does not by itself show a Qwen-vs-Phi difference in downstream spikes.
Teacher-forced agreement does not establish rollout transfer.

## Model and data pins

| Component | Identifier | Revision |
| :--- | :--- | :--- |
| Qwen3-14B base | `Qwen/Qwen3-14B` | `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| Qwen3-14B SFT LoRA | `kyleliu789/qwen3-14b-gpt52-high-reasoning-original` | `2ab1f5bfd74447cb3616be9004e286f677273502` (= `checkpoint-70`, best eval loss 1.5947, per the authors; step 72 has a higher eval loss) |
| Phi-4-reasoning base | `microsoft/Phi-4-reasoning` (**not** `microsoft/phi-4`) | `1de18ec97600877ce63dbf60c73b998da99f0195` |
| Phi-4-reasoning SFT LoRA | `kyleliu789/phi4-reasoning-14b-gpt52-high-reasoning-original` | `eb831e623581632779d515558686d6975c4fa0b4` (recorded as `lora_revision` in the Haskins run manifests; the ReasonIF run manifest says `local`, reported by the authors to be the same commit) |
| Training data | `data/training/gpt52_high_reasoning_original.json` (212 examples) from `TeichAI/gpt-5.2-high-reasoning-250x`, whose train split has 249 rows (not 250); 37 lacking complete `<think>` tags were dropped, 249 − 37 = 212 | — |
| ReasonIF | official grader, `evaluators/reasonif_official/` | commit `706b953` |

Training configs: `configs/*.yaml` (LoRA r = 32, α = 64, lr 1e-4, 3 epochs, seed 42, `val_size: 0.1`, `load_best_model_at_end: true` on `eval_loss`).

## Prompts actually used

`data/eval_prompts/haskins_task_instructions.json` contains the exact instruction text per Haskins task and the two system-prompt templates
(Qwen: `You are a helpful assistant. <instruction>`; Phi: Microsoft's official reasoning system prompt + `Constraint Requirement: <instruction>`).
The two suppression tasks follow the prompt wording (single keyword / keyword + synonyms) rather than upstream's, so Haskins numbers are **not directly comparable**
to numbers produced with the upstream scorer and prompts.

## Repository layout

```text
configs/                SFT configs (Qwen3-14B, Phi-4-reasoning)
data/training/          212-example training set + dataset_info.json (only the shipped dataset is registered)
data/eval_prompts/      50 Haskins questions, calibrated keywords, exact task instructions, ReasonIF-300
evaluators/             haskins_evaluator.py (paper + semantic scorers), reasonif_evaluator.py, official ReasonIF checkers, tests
notebooks/              17 generation / evaluation notebooks (vLLM)
reproducible_figures/   Figures 1-4 (PDF/PNG/SVG) and their source CSVs
results/
  haskins_500/          Qwen 2x2 runs (results.csv + records/), Phi standalone runs, Phi crossed bundle
  reasonif_300/         scored ReasonIF completions
  kl_divergence/        float32 caches, spike_analysis/spike_token_pairs.json, per-sample CSVs, executive report
  summary_tables/       table1..table13 CSVs (regenerated by scripts/reproduce_tables.py --write)
scripts/                analysis_prefix_length_paired.py (prefix content, window-matched scoring, paired CIs), make_paper_tables.py (LaTeX tables),
                        reproduce_tables.py, verify_scorer_parity.py, generate_reproducible_figures.py,
                        evaluate_haskins.py, evaluate_reasonif.py, compute_forward_kl.py, export_haskins_prompts.py
```

## License and citation

MIT License. The manuscript is not yet on arXiv; cite the repository until it is.

```bibtex
@misc{liu2026benign,
  title  = {Benign Reasoning Distillation and Early-Token Steering of Chain-of-Thought Controllability},
  author = {Liu, Kyle},
  year   = {2026},
  url    = {https://github.com/Kyleliu7/Benign_CoT_Control}
}
```
