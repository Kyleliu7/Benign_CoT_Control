# Scientific Methodology, Grading Specifications, and Research Limitations

## 1. Scope Boundaries & Exclusions

### 1.1 MathIF Exclusion
By explicit researcher decision, **MathIF is completely excluded from the research scope**. 
- MathIF is removed from all future candidate final-results tables, aggregate summaries, claims, and pending-work ledgers.
- Historical evaluation files and raw outputs for MathIF are strictly preserved in their original disk locations, but are not treated as validated or failed.
- Denominators and task averages across the suite reflect only valid, active tasks.

---

## 2. Rigorous Grading Specifications: Strict vs. Partial Compliance

To eliminate ambiguity between binary constraint satisfaction and continuous proxy scores, all evaluators enforce dual metrics:

| Task Name | Constraint Type | Strict Binary Pass Criterion | Defined Partial Compliance Metric |
| :--- | :--- | :--- | :--- |
| **`third_person`** | Lexical Negative | $100\%$ sentences contain zero first-person pronouns (`I`, `me`, `my`, `myself`). | Fraction of sentences without first-person pronouns. |
| **`arrow_prefix`** | Formatting Strict | $100\%$ sentences start with triple arrow (`>>>`). Double arrow (`>>`) strictly fails. | Fraction of sentences matching `^\s*>{3,}`. |
| **`end_of_sentence`** | Lexical Boundary | $100\%$ sentences terminate with the word `"safe"`. | Fraction of sentences ending with `"safe"`. |
| **`meow_between_words`** | Syntactic Alternation | Strict alternating placement: `word meow word meow ... word`. | Slot saturation proxy: $\min(\text{meows} / (\text{non-meow words} - 1), 1.0)$. |
| **`repeat_sentences`** | Structural Bookends | Exact phrase `"This is my analysis."` at BOTH opening sentence and closing sentence. | $(M_{\text{open}} + M_{\text{close}}) / 2.0$. |
| **`alternating_case`** | Orthographic | $100\%$ words have strict alternating upper/lower casing across adjacent alphabetic characters. | Fraction of compliant words. |
| **`uppercase_thinking`** | Orthographic | $100\%$ alphabetic characters in reasoning are uppercase. | Fraction of compliant sentences. |
| **`lowercase_thinking`** | Orthographic | $100\%$ alphabetic characters in reasoning are lowercase. | Fraction of compliant sentences. |
| **`word_suppression`** | Lexical Negative | Target forbidden keyword does not appear anywhere in reasoning trace (`\bkeyword\b`). | Binary pass/fail (1.0 or 0.0). |
| **`multiple_word_suppression`** | Defective Prompt | **NA (Excluded)**: Generation prompt specified only 1 target word, omitting 3 forbidden words. | **NA (Excluded from aggregate denominators)**. |

### 2.1 Resolution of Specific Grading Discrepancies
1. **`repeat_sentences` (Base vs SFT)**:
   - **Strict Binary Pass**: **0.0%** (Base) vs **0.0%** (SFT). Neither model satisfies the joint opening AND closing bookend requirement on any trace.
   - **Partial Compliance**: **0.0%** (Base) vs **26.0%** (SFT). Under SFT, exactly 26 out of 50 traces match the closing bookend, while 0 match the opening bookend. The partial score is calculated as $0.5 \times \frac{26}{50} = 26.0\%$.
2. **`meow_between_words` (Base vs SFT)**:
   - **Strict Placement Compliance**: **0.0%** (Base) vs **0.0%** (SFT). Both models fail the strict token-by-token alternation test.
   - **Slot Saturation Proxy**: **32.17%** (Base) vs **71.99%** (SFT). SFT frequently inserts the token `"meow"`, achieving high frequency/saturation, but fails syntactic alternating grammar. This metric is explicitly labeled as a proxy and never claimed as placement compliance.

---

## 3. Data Integrity & Provenance Standards

1. **Non-Fabrication of Token Counts**:
   - Token lengths are extracted directly from generation logs (`output_tokens`, `reasoning_token_count`).
   - If token length metadata is absent in historical files, the cell is explicitly marked `NA`. Fabricating zero (`0.0`) is strictly prohibited.
2. **Word Suppression Disentanglement (Task 3)**:
   - In the Haskins 2x2 continuation experiments, donor generation was executed unconstrained (`prompt_off`).
   - When evaluating continuation compliance on recipient models, checking for unrequested benchmark keywords measures observational vocabulary avoidance rather than constraint-following.
   - The historical suppression values (50%, 58%, 64%, 74%) are formally marked `BLOCKED_UNCALIBRATED_CACHE` until a targeted rerun with constraint-ON recipient instructions is conducted (~1.0–1.5 GPU-hours).
3. **LoRA Fine-Tuning Comparability**:
   - Both Qwen-2.5-Coder-14B and GPT-OSS-20B comparisons utilized Parameter-Efficient Fine-Tuning (PEFT/LoRA).
   - Qwen LoRA targeted all linear projections (`q, k, v, o, gate, up, down`). GPT-OSS LoRA targeted attention projections only (`q_proj, v_proj`).
   - These architectural differences are documented as an observational limitation rather than an invalidating execution defect.

---

## 4. Core Scientific Limitations & Boundary Conditions

### 4.1 Reasoning Length Mediation Confounder
A critical finding across all interventions is that controllability gains strongly correlate with reasoning length compression:
- On **ReasonIF**, Qwen-14B Base produces an average of **4,402 tokens** per reasoning trace.
- Static Header prefix reduces length to **2,955 tokens** (-33%).
- Dynamic Teacher (Constraint ON) prefix reduces length to **2,571 tokens** (-42%).
- SFT LoRA reduces length to **2,024 tokens** (-54%).

Because negative constraints (e.g., word suppression, third-person pronoun avoidance) are evaluated over generated text length, shorter generation naturally reduces the probability of violating a constraint. Shorter traces may also truncate multi-step verification, directly contributing to accuracy drops.

### 4.2 The Controllability vs. Accuracy Trade-Off
Enhancing reasoning controllability via fine-tuning or prefix intervention imposes an empirical tax on reasoning capability:

```
Condition                     Instruction Following (IFS)    Answer Accuracy    Joint Success
---------------------------------------------------------------------------------------------
Qwen3-14B Base Untouched                 14.00%                   80.33%            11.67%
Qwen3-14B Static Header (k=8)            21.67%                   76.67%            15.00%
Qwen3-14B Dyn Teacher (Const OFF)        31.33%                   73.67%            21.67%
Qwen3-14B Dyn Teacher (Const ON)         42.00%                   73.33%            31.00%
Qwen3-14B SFT Targeted LoRA              33.33%                   65.33%            20.67%
GPT-OSS-20B Base Untouched               16.33%                   76.00%            12.67%
GPT-OSS-20B LoRA (Attn Only)             22.00%                   68.67%            16.00%
```

While Dynamic Teacher (Constraint ON) achieves the highest Joint Success (31.00%), task accuracy drops from 80.33% to 73.33% (-7.0%). SFT LoRA experiences an even steeper accuracy penalty to 65.33% (-15.0%).

### 4.3 Training / Evaluation Overlap (Contamination Bounds)
- Exact string matching and lowercase substring searches across evaluation prompts and training datasets confirm **zero verbatim memorization** (0/50 Haskins prompts, 0/300 ReasonIF prompts appeared in training datasets).
- However, substring searches cannot rule out conceptual or semantic distribution overlap. The degree of semantic transfer across mathematical and synthetic reasoning domains remains an observational limitation.
