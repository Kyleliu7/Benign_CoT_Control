# Repository audit

Scope: every number in the README, summary tables, executive report and figure sources was recomputed from the raw files in `results/`
and compared with the manuscript draft. Environment: CPU sandbox, no model downloads (so the fast-langdetect model was unavailable).

## 1. What was wrong (before this branch)

| # | Problem | Evidence | Fixed here? |
| :-- | :--- | :--- | :---: |
| 1 | `reproduce_tables.py` only asserted constants read from the committed CSVs ("0 discrepancies" was tautological). | Script compared `df[...] == 37.0` etc. against the same CSVs; never opened a raw file. | ✅ rewritten |
| 2 | README tables disagreed with raw data **and** with the paper (see §2). | Qwen crossed strict counts 28/34/124/116 appear in no raw file; the raw files give 25/21/109/96. | ✅ |
| 3 | The "Qwen3-14B base ReasonIF" file was not Qwen3-14B (it was the GPT-OSS-20B run). **Resolved:** the authors supplied the real Qwen3-14B base records; they give IFS 42 / acc 241 / joint 35 / 4,401.7 tokens, i.e. exactly the paper's row. The GPT-OSS file is kept under `results/reasonif_300/gpt_oss_20b/`. | All 300 records: `model_id: openai/gpt-oss-20b`, `reasoning_effort: medium`, `<\|channel\|>` format. Its stored scores (IFS 49, acc 228, joint 38, 3,569 tokens) match neither the paper (42 / 241 / 35 / 4,401.7) nor the README (111 / 241 / 35 / 4,402). | ✅ |
| 4 | `haskins_evaluator.py` did not reproduce the stored scores. | Re-scoring S→Base: 98 strict vs 109 stored; `word_suppression` 4/50 vs 15/50 (it also forbade synonyms), `end_of_sentence`, `repeat_sentences`, `meow` and the `>>>` threshold differ from the pinned upstream grader. | ✅ `scorer="paper"` reproduces all 7,500 stored scores (0 mismatches) |
| 5 | Figure script hard-coded `C:\Users\bryan\.gemini\...` and typed-in ReasonIF counts (including the unverifiable Qwen base 42/241/35). | `generate_reproducible_figures.py` lines 44-53, 126-135. | ✅ repo-relative, Fig. 2 computed from Table 1 |
| 6 | Spike "Table 5" and Figure 3 C1/C2 labels did not match stored spike records. | Of 7 rows: 1 had the right KL and position but different tokens, 4 had no record at all, `repeat_sentences:19` pos 90 holds `' legal' → '**'` (table said `Next → mRNA`), and `[FAIL]/[PASS]` were whole-trace flags presented as token-level. | ✅ replaced by `table5_top_spike_per_task_phi4.csv` generated from the records |
| 7 | KL executive report claimed "85–92 % in the first 10 tokens", had links to `C:/Users/bryan/...` and unsupported mechanism wording. | Actual first-10 share is 28–81 % (first 100 positions) or 10–56 % (full length). | ✅ rewritten |
| 8 | `requirements.txt` missed `immutabledict` and `nltk`; the ReasonIF grader failed to import on a clean install. `vllm>=0.4.0` vs the 0.29.0 actually used. | `ModuleNotFoundError: immutabledict`. | ✅ split into `requirements.txt` / `requirements-gpu.txt` |
| 9 | Official ReasonIF checker scores a failed language detection as **compliant**; offline, every language item silently passes (52/52). | Observed here. | ✅ failures are counted and warned about |
| 10 | README listed Phi base as `microsoft/phi-4`; runs used `microsoft/Phi-4-reasoning`. `dataset_info.json` registered 10 datasets that are not shipped (some named like earlier CoT-control experiments). Typos (`TeicHAI`). | | ✅ |
| 11 | Notebooks contain absolute paths (`/home/kyleliu789/...`). | Harmless but noisy. | ❌ left as is |

## 2. Old README vs paper vs raw data

| Quantity | Old README | Paper draft | Raw data |
| :--- | :---: | :---: | :---: |
| Qwen crossed strict, Base→Base / Base→SFT / SFT→Base / SFT→SFT | 28 / 34 / 124 / 116 | 25 / 21 / 109 / 96 | **25 / 21 / 109 / 96** |
| Qwen crossed mean, Base→SFT / SFT→SFT | 14.88 / 37.64 | 16.70 / 32.07 | **16.70 / 32.07** |
| Phi crossed strict Base→Base | 17 | 11 | **11** |
| Phi standalone strict, Base / SFT | 20 / 48 | 13 / 62 | **13 / 62** |
| Qwen Haskins standalone, Base | 23.40 %, 65/500 | 17.21 %, 23/500 | **17.21 %, 23/500** (`vllm_calibrated_base`); the old-protocol file stores 21.99 % / 45 and rescored 14.39 % / 7, so 23.40 % matches no record |
| Qwen ReasonIF base IFS | 37.0 % (111) | 14.0 % (42) | **14.0 % (42)** |
| Qwen ReasonIF Prefix-OFF IFS / joint | 97 / 67 | 96 / 67 | **96 / 67** (continuation fields); 94 / 65 with full-trace "official" fields |
| Phi ReasonIF Prefix-OFF IFS | 15 | 14 | **14** |
| Qwen Prefix-OFF mean tokens | 2,724.3 | 2,926.6 | **2,926.6** |

The paper's Qwen crossed, Phi crossed, Phi standalone and Qwen SFT/ON ReasonIF numbers are correct; the README was the inconsistent document.

## 3. Facts surfaced by the recomputation that affect the paper's claims

* **Haskins prefixes contain `<think>`** (all 2,000 Qwen prefixes have 10 token IDs starting with 151667), so only 9 tokens are generated content. Methods says the 10 tokens follow `<think>`.
* **SFT-donor prefixes are constraint-aware.** Examples: `**Thinking in lowercase only**`, `**Chain of Thought in Uppercase`, `The assistant is presented…`. The recipient continues from a prefix that already demonstrates or names the constraint.
* **ReasonIF prefixes are not always 10 tokens**: Prefix-OFF has 15 empty prefixes and 1 of 4 tokens; Prefix-ON has 15 prefixes of 1–6 tokens.
* **Length confound**: Qwen continuation mean 986 → 677 tokens (Base→Base vs SFT→Base); Phi base hits the 3,000-token cap on 52/500 standalone runs vs 3/500 for SFT; Phi's SFT gain on ReasonIF is mostly the word-limit constraint (0 → 17 of 53). Training CoTs average 222 words and all 212 begin with a `**Heading**`.
* **Two scoring bases in the paper's Table 2.** The Qwen Prefix-OFF row (20.97 / 59) is continuation-scored, but the Prefix-ON row (36.94 / 89) is full-trace-scored (`upstream_compliance` means different things in the two files). Like-for-like: continuation 37.97 / 112 vs 20.97 / 59; full trace 36.94 / 89 vs 20.12 / 41.
* **The SFT→Base condition exists in three versions**: the 2×2 run (37.34 / 109 continuation), the calibrated rerun (37.97 / 112) and an older-protocol run (stored 44.81 / 152, not reproducible with the pinned scorer).
* **Older-protocol Qwen records (removed at the authors' request)** (`standalone_*`, `*_a1`, `*_a5`): the stored scores do not reproduce (e.g. standalone base stored 21.99 / 45 vs 14.39 / 7 with the pinned scorer), the SFT records are labelled `model: gpt52_short`, and none of these files, under either scorer, gives the old README values (23.40 / 37.89 / 37.34 / 19.38). Treat the old README values as unsourced.
* **Run health**: 7–27/500 Qwen crossed generations hit the cap and 11–70/500 end as `unfinished_reasoning` (70 for SFT→SFT).
* **KL**: Table 4 reproduces exactly, but "early share" uses positions 1–100 of traces with ≥ 100 tokens while the downstream statistics use all tokens up to 512. Qwen's per-sequence-max P95 (9.72 / 5.49) and global max (15.01 / 17.75) exceed Phi's (8.75 / 3.98; 11.89 / 11.57). Outcome groups are small (Phi ReasonIF G1 n = 22, G2 n = 14) and dominated by "both fail".
* **Prompts**: the Haskins instruction text actually used is the upstream text (e.g. `repeat_sentences`: "mention this string at the very beginning and at the very end…"), not the paraphrases in the manuscript's Appendix table; Phi runs use Microsoft's official reasoning system prompt, Qwen runs `You are a helpful assistant.`.
* **Scorer semantics** (pinned upstream): `repeat_sentences` score = fraction of sentences equal to the target (so "strict" requires every sentence to be the target); `arrow_prefix` accepts ≥ 2 arrows; `end_of_sentence` does not strip markdown headers; `meow` strict means meow-count ≥ expected slots, not alternation. Floor effects follow from these definitions.
* **Eval sets** (checked against the shipped files): Haskins prompts are 10 math / 8 coding / 8 science / 8 history-social / 8 logic / 8 ethics (not GSM8K/MATH/ARC/CommonsenseQA); ReasonIF sources are GPQA 73, AIME 61, ARC 59, AMC 54, GSM8K 53 (no "synthetic"). 0 of 50 Haskins and 0 of 300 ReasonIF questions share an 8-gram with the training prompts or outputs.
* **Training data**: 212 examples, all DevOps / infrastructure / security design requests (not "systems programming, algorithms, applied mathematics"); 0 contain the audited constraint phrases in the instruction except one use of "Monitor" in an operational sense; `monitor*` and `must not` occur in 46 outputs in an operational sense (e.g. SLO monitoring, "must not regress"). 250 − 37 − 2 = 211, not 212.

## 4. Changes in this branch

* `evaluators/haskins_evaluator.py`: `scorer="paper"` (default) = SHA-verified pinned upstream grader + calibrated suppression checkers; legacy checks kept as `scorer="semantic"`.
* `scripts/verify_scorer_parity.py`: re-scores 4,000 Qwen + 1,000 Phi standalone + 2,500 Phi crossed rows → 0 mismatches.
* `scripts/reproduce_tables.py`: recomputes Tables 1–7 from raw files, `--write`, `--rescore-reasonif`; non-zero exit on mismatch.
* `results/summary_tables/`: regenerated; new Tables 5 (verified spikes), 6 (per-task), 7 (run health), 8 (older-protocol Qwen records, not reproducible); Qwen base ReasonIF and the Qwen Haskins standalone/prefix rows are now filled from the supplied raw records.
* `scripts/generate_reproducible_figures.py` and `reproducible_figures/`: regenerated from repo files; Qwen base omitted from Fig. 2; interpretive annotations ("adapter suppresses comma") replaced by neutral ones.
* `data/eval_prompts/haskins_task_instructions.json`, `scripts/export_haskins_prompts.py`: exact prompts.
* README, KL executive report, requirements, `dataset_info.json`, tests (`evaluators/test_paper_scorer.py`), `.gitignore`.

## 5. Open items

Resolved with the authors' answers and the supplied files:
* Qwen3-14B base ReasonIF records (item 1) and the Qwen Haskins standalone / Prefix records (item 2): present and verified. Qwen A5 = the calibrated Prefix-OFF run.
* Checkpoint: the published Qwen adapter is `checkpoint-70` (best eval loss; step 72 is worse), consistent with `load_best_model_at_end: true`. SHA-256 values quoted by the authors (root adapter = checkpoint-70 = `d3161ea4…5218`; Phi adapter `355fb00c…dfc3`, commit `eb831e62…`) are **author-reported and not checked here**; the Phi commit does appear as `lora_revision` in the Haskins manifests.
* Training-set size: the HF split has 249 rows, 37 dropped → 212. The paper's "250 − 37 − 2" text is wrong.

* Old README numbers (23.40 / 37.89 / 37.34 with 124 strict / 19.38 with 34 strict): the authors confirm they are stale exploratory artefacts, explain 124/500 and 34/500 as non-character numerators copied into the 500 denominator, and say 23.40 / 37.89 came from a pre-calibration keyword run. I could not verify that explanation (no retained file gives 124/350, 48.02 or 23.40), so they are treated as unsourced; the older-protocol Qwen files and Table 8 were deleted. Git history keeps them.
* Language items: on the authors' VM (fast-langdetect available) `--rescore-reasonif` was run on `main` at `9404952`. For the genuine Qwen3-14B base file the recomputed counts equal the stored flags exactly (Lang-52 22/22, Cap-43 0/0, NonLang-205 20/20; total IFS 42/300 = 14.0 %, 0 mismatches), and the run ended with "0 mismatch(es)" across all files. Earlier VM output for the SFT, Prefix-OFF, Prefix-ON and Phi files agreed with the stored language flags as well. Only the Qwen base row was shown in the screenshot supplied to the audit, so the other rows rest on that earlier output. Note the printed "st/re" cells are stored/recomputed pass counts, not passes/total.

Still open:
1. Phi ReasonIF Prefix-OFF / Prefix-ON raw JSONL (only summary CSVs exist; the notebooks can regenerate them).
2. KL: cache metadata says `base_model_revision: main`; confirm the revisions equal the pins in the README. Token alignment between SFT and base tokenisations is not checked in code here.
3. Source of the Figure 3 panel B outcome groups (G1–G4): numbers exist in the executive report, but the labelling rule is not in a script.
4. Decide whether to scrub `/home/kyleliu789/...` paths from notebook outputs.

## 6. Manuscript edits implied by this audit (applied to the author's `main.tex`, which is not part of this repository)

The new analyses behind the revised paper are `scripts/analysis_prefix_length_paired.py` (Tables 8-13: window-matched scoring, prefix content, paired cluster-bootstrap and McNemar contrasts) and `scripts/make_paper_tables.py` (LaTeX for every results table). Training sequences longer than 4,096 tokens are truncated by the trainer (answer part only; the longest reasoning block is 4,285 characters), so the dataset-size arithmetic is 249 - 37 = 212 with no length exclusions.

* The Qwen ReasonIF base numbers are now sourced, so Table 1 / Fig. 2 stand; update Table 12 (truncated = 19, no longer "—"), Table 17 (add the base column from `by_constraint`) and the provenance table (rows now "raw records present").
* Table 2: report Prefix-OFF and Prefix-ON under the **same** scoring basis (or both), and state it. Replace "Qwen rows … not rederived from raw records" with the verified status.
* Methods: "212 curated examples… excluding 37 … and 2 items exceeding sequence length" → 249 rows, 37 excluded, 212 kept.
* Abstract / Intro / §5: the Qwen downstream-mean range "0.098–0.224" is the positions-11–100 window (Table 4 uses 0.077–0.186); "peaking up to 7.13 nats on ReasonIF" conflicts with Table 4's 11.57; Intro "9.42 nats" has no source; "P99.9 ≥ 6.5" is true for 1 of 4 rows.
* Remove or rewrite the "continuous downstream policing" / "initial-state trajectory steering" mechanism claims and the "direct, causal explanation" sentence; KL on SFT histories does not establish rollout behaviour, and Qwen's downstream spikes are not smaller than Phi's.
* Replace Table 6 and the Fig. 3 C1/C2 captions with the generated spike table; do not label single tokens PASS/FAIL.
* Methods/Appendix: Phi system prompt, exact task instructions, `<think>` counted inside the 10 prefix tokens, shorter ReasonIF prefixes, 10,000-iteration paired cluster bootstrap (not 1,000), training-data description, Haskins / ReasonIF source datasets, pinned revisions, calibrated suppression scoring and its non-comparability with upstream, Table 13 token IDs (the first row lists 8 IDs for a "10-token" prefix).
* Add the length, truncation and prefix-content facts of §3 to Results / Limitations.
