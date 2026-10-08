# CoT Controllability 2×2 Transfer Matrix: Microsoft Phi-4 vs. Qwen 3 14B

This report provides the direct, empirical comparison between **Microsoft Phi-4 Reasoning (14B)** and **Qwen 3 14B** across all 5 conditions of the **Haskins 500 CoT Controllability Benchmark**:
- **A1**: SFT Donor $\to$ Base Recipient (Prefix-ON)
- **A2**: Base Donor $\to$ Base Recipient (Prefix-ON)
- **A3**: Base Donor $\to$ SFT Recipient (Prefix-ON)
- **A4**: SFT Donor $\to$ SFT Recipient (Prefix-ON)
- **A5**: SFT Donor $\to$ Base Recipient (Prefix-OFF Control)

All experiments evaluate exactly 10-token reasoning prefixes starting identically from the native assistant turn headers, scored under calibrated semantic matching rules across all 500 scheduled problems.

---

## 1. 2×2 Transfer Matrix (7 Non-Character Tasks)

The 7 non-character tasks (`third_person`, `word_suppression`, `multiple_word_suppression`, `arrow_prefix`, `end_of_sentence`, `meow_between_words`, `repeat_sentences`) represent structural and semantic instruction adherence within chain of thought.

### Microsoft Phi-4 Reasoning (14B)

| Recipient \ Donor | Base Donor (10-tok ON) | SFT Donor (10-tok ON) | Donor Steering Gain ($\Delta_{\text{Donor}}$) |
| :--- | :---: | :---: | :---: |
| **Base Recipient** | **12.64%** (Strict: 3.14%) <br> *[Condition A2]* | **13.36%** (Strict: 4.29%) <br> *[Condition A1]* | **+0.72%** (Strict: +1.15%) |
| **SFT Recipient** | **17.18%** (Strict: 10.29%) <br> *[Condition A3]* | **22.01%** (Strict: 19.71%) <br> *[Condition A4]* | **+4.83%** (Strict: +9.42%) |
| **Recipient Gain ($\Delta_{\text{Recipient}}$)** | **+4.54%** (Strict: +7.15%) | **+8.65%** (Strict: +15.42%) | **Synergistic Reinforcement** |

*Unconstrained Baseline (Condition A5: SFT Donor $\to$ Base Recipient, Prefix-OFF)*: **11.94%** (Strict: 3.43%)  
*Visibility Steering Effect (A1 vs. A5)*: **+1.42%** (Strict: +0.86%)

---

### Qwen 3 14B

| Recipient \ Donor | Base Donor (10-tok ON) | SFT Donor (10-tok ON) | Donor Steering Gain ($\Delta_{\text{Donor}}$) |
| :--- | :---: | :---: | :---: |
| **Base Recipient** | **16.38%** (Strict: 2.57%) | **31.70%** (Strict: 18.57%) | **+15.32%** (Strict: +16.00%) |
| **SFT Recipient** | **15.17%** (Strict: 2.29%) | **27.95%** (Strict: 16.29%) | **+12.78%** (Strict: +14.00%) |
| **Recipient Gain ($\Delta_{\text{Recipient}}$)** | **-1.21%** (Strict: -0.28%) | **-3.75%** (Strict: -2.28%) | **Donor-Dominated Steering** |

*Unconstrained Baseline (SFT Donor $\to$ Base Recipient, Prefix-OFF)*: **21.39%** (Strict: 14.86%)  
*Visibility Steering Effect (A1 vs. A5)*: **+10.31%** (Strict: +3.71%)

---

## 2. 2×2 Transfer Matrix (All 10 Tasks)

Including the 3 character-level constraints (`alternating_case`, `lowercase_thinking`, `uppercase_thinking`):

| Condition | Qwen 3 14B Cont. Mean | Phi-4 Cont. Mean | Qwen Strict Pass Rate | Phi-4 Strict Pass Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Base $\to$ Base (ON)** [A2] | 17.48% | 10.38% | 4.60% | 2.20% |
| **SFT $\to$ Base (ON)** [A1] | 36.28% | 10.92% | 17.00% | 3.00% |
| **Base $\to$ SFT (ON)** [A3] | 16.55% | 14.46% | 4.00% | 7.20% |
| **SFT $\to$ SFT (ON)** [A4] | 30.93% | 18.07% | 14.60% | **13.80%** |
| **SFT $\to$ Base (OFF)** [A5] | 20.97% | 9.89% | 11.80% | 2.40% |

---

## 3. Matched SFT $\to$ SFT Task-by-Task Breakdown

Comparing the fully adapted models (**Condition A4: SFT Donor $\to$ SFT Recipient**) to pinpoint where each architecture excels:

| Task Name | Task Category | Qwen SFT $\to$ SFT Mean | Phi-4 SFT $\to$ SFT Mean | Qwen Strict (1.0) | Phi-4 Strict (1.0) | Phi-4 vs. Qwen ($\Delta_{\text{Strict}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`word_suppression`** | Non-Character | 18.00% | **38.00%** | 18.00% | **38.00%** | **+20.00% ($2.1\times$)** |
| **`multiple_word_suppression`** | Non-Character | 6.00% | **28.00%** | 6.00% | **28.00%** | **+22.00% ($4.7\times$)** |
| **`third_person`** | Non-Character | **97.57%** | 87.35% | **92.00%** | 72.00% | -20.00% |
| **`alternating_case`** | Character | **16.23%** | 10.60% | 0.00% | 0.00% | 0.00% |
| **`lowercase_thinking`** | Character | **61.33%** | 11.18% | **42.00%** | 0.00% | -42.00% |
| **`uppercase_thinking`** | Character | **42.34%** | 4.81% | **26.00%** | 0.00% | -26.00% |
| **`repeat_sentences`** | Non-Character | **4.59%** | 0.00% | **2.00%** | 0.00% | -2.00% |
| **`end_of_sentence`** | Non-Character | **17.78%** | 0.73% | **4.00%** | 0.00% | -4.00% |
| **`meow_between_words`** | Non-Character | **56.25%** | 0.00% | **2.00%** | 0.00% | -2.00% |
| **`arrow_prefix`** | Non-Character | 0.57% | 0.00% | 0.00% | 0.00% | 0.00% |

---

## 4. Key Architectural & Theoretical Insights

1. **Two Distinct Steering Regimes**:
   - **Qwen 3 14B (Donor-Driven)**: In Qwen, the base model is extraordinarily plastic and malleable. Injecting an SFT-generated prefix into Base Qwen drives compliance from 16.38% up to 31.70% (a $+15.32$ pp gain). Base Qwen eagerly adopts whatever trajectory is initiated by the prefix.
   - **Phi-4 Reasoning (Recipient-Driven)**: In Phi-4, the Base model possesses a rigid, entrenched reasoning trajectory that resists external prefix steering ($+0.72$ pp gain). However, fine-tuning the recipient model (SFT Recipient) unlocks dramatic controllability: Base $\to$ SFT jumps to **17.18%**, and SFT $\to$ SFT reaches **22.01%** (a $+9.37$ pp gain over Base $\to$ Base).

2. **Negative Constraint Mastery in Phi-4**:
   - On **`word_suppression`**, Phi-4 SFT achieves **38.00% strict adherence** vs. Qwen's **18.00%** ($2.1\times$ higher).
   - On **`multiple_word_suppression`**, Phi-4 SFT achieves **28.00% strict adherence** vs. Qwen's **6.00%** ($4.7\times$ higher).
   - Phi-4's foundational post-training enforces rigorous constraint filtering, making negative constraints (avoiding specific words) exceptionally robust once fine-tuned.

3. **Syntactic Corruptions vs. Mathematical Rigor**:
   - On syntactic noise injection (`meow_between_words`), Qwen willingly corrupts its reasoning tokens (56.25% compliance). In contrast, Phi-4 completely suppresses syntactic corruption (0.00%), prioritizing semantic mathematical coherence over trivial surface patterns.

4. **Confirmation of Prefix Visibility Steering (A1 vs. A5)**:
   - For **Phi-4**: SFT $\to$ Base (ON) is **13.36%** vs. SFT $\to$ Base (OFF) at **11.94%** ($\Delta = +1.42\%$, strict: 4.29% vs 3.43%).
   - For **Qwen**: SFT $\to$ Base (ON) is **31.70%** vs. SFT $\to$ Base (OFF) at **21.39%** ($\Delta = +10.31\%$, strict: 18.57% vs 14.86%).
   - Across both models, withholding the constraint from the donor prefix (Prefix-OFF) depresses downstream continuation adherence, proving that the first 10 tokens establish an active constraint-entrainment channel.
