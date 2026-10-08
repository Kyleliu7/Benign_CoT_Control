# ReasonIF Benchmark: Phi-4-reasoning Base vs. SFT LoRA

This report details the head-to-head evaluation of **`microsoft/Phi-4-reasoning`** (Untouched Base) versus **`kyleliu789/phi4-reasoning-14b-gpt52-high-reasoning-original`** (LoRA SFT, rank 32, alpha 64) across all 300 official **ReasonIF** benchmark questions under the standardized 14B parameter calibration protocol.

---

## 1. Executive Summary & Overall Metrics

| Metric | Base (Phi-4-reasoning) | SFT LoRA (GPT-5.2 High) | Absolute Diff | Relative Change | McNemar $\chi^2$ | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Instruction Following (IFS)** | **5.0%** (15/300) | **12.0%** (36/300) | **+7.0%** | **+140.0%** | **17.391** | **$p < 0.0001$** |
| **Joint Success (IFS + Answer)** | **3.0%** (9/300) | **8.0%** (24/300) | **+5.0%** | **+166.7%** | **10.316** | **$p = 0.0013$** |
| **Answer Accuracy** | **75.3%** (226/300) | **69.0%** (207/300) | -6.3% | -8.4% | 5.492 | $p = 0.0191$ |
| **Format Validity** | **94.3%** (283/300) | **87.3%** (262/300) | -7.0% | -7.4% | 8.511 | $p = 0.0035$ |
| **Missing Answer Tags** | **5.7%** (17/300) | **12.7%** (38/300) | +7.0% | +123.5% | 8.511 | $p = 0.0035$ |
| **Length Truncation** | **4.0%** (12/300) | **2.3%** (7/300) | -1.7% | -41.7% | — | — |

> [!IMPORTANT]
> **Statistical Significance**: SFT improves both pure **Instruction Following** ($p < 0.0001$) and **Joint Success** ($p = 0.0013$) by more than **$2.4\times$** over Base with strong statistical significance (McNemar paired $\chi^2$). Out of the discordant pairs, SFT satisfied instructions that Base failed in **22 cases**, whereas Base only outperformed SFT in **1 case**.

---

## 2. Breakdown by Constraint Category

| Constraint Category | Count | Base IFS | SFT IFS | IFS Diff | Base Acc | SFT Acc | Acc Diff | Base Joint | SFT Joint | Joint Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Length (Word Limit)** | 53 | **0.0%** | **32.1%** | **+32.1%** | 75.5% | 71.7% | -3.8% | **0.0%** | **24.5%** | **+24.5%** |
| **Reasoning Language** | 52 | 26.9% | **32.7%** | **+5.8%** | 73.1% | 67.3% | -5.8% | 15.4% | **17.3%** | **+1.9%** |
| **Punctuation (No Comma)** | 56 | 0.0% | **3.6%** | **+3.6%** | 69.6% | 62.5% | -7.1% | 0.0% | **3.6%** | **+3.6%** |
| **Change Case (All Caps)** | 43 | 0.0% | 0.0% | 0.0% | 81.4% | 74.4% | -7.0% | 0.0% | 0.0% | 0.0% |
| **Detectable Format (JSON)** | 47 | 0.0% | 0.0% | 0.0% | 80.9% | 72.3% | -8.5% | 0.0% | 0.0% | 0.0% |
| **Start/End Checker** | 49 | 2.0% | 0.0% | -2.0% | 73.5% | 67.3% | -6.1% | 2.0% | 0.0% | -2.0% |

---

## 3. Breakdown by Source Benchmark Task

| Source Benchmark | Count | Base IFS | SFT IFS | IFS Diff | Base Acc | SFT Acc | Acc Diff | Base Joint | SFT Joint | Joint Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GPQA** (Graduate QA) | 73 | 4.1% | **17.8%** | **+13.7%** | 43.8% | 38.4% | -5.5% | 0.0% | **8.2%** | **+8.2%** |
| **ARC** (Reasoning Challenge) | 59 | 5.1% | **11.9%** | **+6.8%** | 79.7% | **81.4%** | **+1.7%** | 5.1% | **11.9%** | **+6.8%** |
| **GSM8K** (Grade Math) | 53 | 3.8% | **9.4%** | **+5.7%** | 100.0% | 98.1% | -1.9% | 3.8% | **9.4%** | **+5.7%** |
| **AIME** (Math Olympiad) | 61 | 3.3% | **8.2%** | **+4.9%** | 70.5% | 55.7% | -14.8% | 0.0% | **3.3%** | **+3.3%** |
| **AMC** (Math Competition) | 54 | 9.3% | **11.1%** | **+1.9%** | 94.4% | 83.3% | -11.1% | 7.4% | 7.4% | 0.0% |

---

## 4. Output Length & Efficiency Analysis

| Model | Mean Output Tokens | Median Output Tokens | Truncated at 16,384 limit |
| :--- | :---: | :---: | :---: |
| **Phi-4-reasoning (Base)** | 4,044.8 | 2,094.5 | 4.0% (12 questions) |
| **Phi-4 SFT LoRA (GPT-5.2 High)** | **2,512.2** | **780.0** | **2.3%** (7 questions) |

> [!NOTE]
> The Base model exhibits severe runaway chain-of-thought rambling, averaging over 4,000 tokens per generation. The SFT model produces far more concise, disciplined reasoning (median 780 tokens), reducing inference overhead while dramatically increasing constraint compliance.
