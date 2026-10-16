# results/mechanism

Outputs of `scripts/mechanism/` (2026-10-16 VM session; logs in the VM folder `~/mechanism_runs/`). Summaries:
`python scripts/mechanism/summarize_mechanism.py`.

| File | What | Script |
|---|---|---|
| `surprisal_<model>.json` | Per-segment log-probability of the 212 GPT-5.2 training targets under base and SFT adapters | surprisal_by_segment.py |
| `first_token_<model>.json` | Probability that the first reasoning token (right after `<think>`) starts a bold heading | first_token_probe.py |
| `attention_qwen3_14b/` | Attention from the first reasoning position to the constraint and the question; attention-pattern patching | attention_reasoning_start.py |
| `openings_<model>.jsonl` | 50 sampled reasoning openings on CoT-Control prompts | sample_openings.py |

Superseded files are kept for the record and are not used in any table:
* `attention_qwen3_14b_after_newline/`, `first_token_qwen3_14b_superseded.json`, `first_token_phi4_reasoning_superseded.json`:
  measured after `<think>\n`. The SFT models write `<think>**` with no newline, so the decision point is right after
  `<think>`; after a newline the Qwen SFT model almost never starts a heading (P = 0.001), which is itself a sign that the
  habit is tied to the exact training format.
* `surprisal_qwen3.5_9b_superseded.json`: the Qwen3.5-9B adapter was loaded on the text-only class, so LoRA did not
  attach (identical base and SFT numbers). Rerun with the multimodal class (`haskins_prompts.load_causal_lm`).

Qwen3.5-9B newline test (`*_no_newline.*`): LlamaFactory trained the Qwen3.5-9B adapter on `<think>**...`, but Qwen3.5's chat
template ends the prompt with `<think>\n`. With the template prompt the adapter never starts a heading (P = 0.001; 50/50 openings
are `Thinking Process:`); with the newline removed P = 0.979 and 50/50 openings start with a heading. `vm_logs/` holds the
session's step status and the adapter paths and script commits used.
