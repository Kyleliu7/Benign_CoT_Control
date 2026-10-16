#!/bin/bash
# Second part of the 2026-10-16 session, after two corrections:
#  (1) the first-reasoning-position analyses had appended "<think>\n"; the SFT models write "<think>**" (no newline),
#      so the decision point is right after "<think>" (scripts/mechanism/haskins_prompts.OPENER);
#  (2) Qwen3.5-9B adapters must be loaded on the multimodal class (haskins_prompts.load_causal_lm), else LoRA does not
#      attach. Waits for the header-transplant run (GPU memory), then runs the corrected analyses and the CoT-Control
#      evaluations of the Table 2 adapters.   usage: mechanism_chain2.sh <snapshot root>
S=$1; cd $S
M=~/mechanism_runs; LF=~/llamafactory-env/bin/python; PY5=~/venvs/haskins-vllm/bin/python
HUB=~/.cache/huggingface/hub; OUT=$S/results/mechanism; R=~/Benign_CoT_Control
status() { echo "$1 $(date -u +%FT%TZ)" >> $M/status; }
run() { local n=$1; shift; status "START $n"; if "$@" > $M/$n.log 2>&1; then status "OK $n"; else status "FAILED $n"; fi; }
snap() { ls -d $HUB/models--$1/snapshots/* | head -1; }
SFT42=$(snap kyleliu789--qwen3-14b-gpt52-high-reasoning-original)
PHI=$(snap kyleliu789--phi4-reasoning-14b-gpt52-high-reasoning-original)
Q35=$(snap kyleliu789--qwen3.5-9b-gpt52-high-reasoning-original)

# the Qwen3.5 SFT steps of qwen35_steps.sh loaded the adapter on the wrong class; stop them after the base openings
until grep -qE "(OK|FAILED) openings_qwen35_base_v5" $M/status; do sleep 30; done
pkill -f "scripts/vm/qwen35_steps.sh"; sleep 2; pkill -f "sample_openings.py --base Qwen/Qwen3.5-9B --adapter"
status "STOPPED qwen35_steps (adapter steps rerun below)"

until grep -q "EVALS_DONE header_transplant" ~/variant_runs/status 2>/dev/null; do sleep 60; done
status "transplant run finished; starting corrected analyses"

[ -d $OUT/attention_qwen3_14b ] && mv $OUT/attention_qwen3_14b $OUT/attention_qwen3_14b_after_newline
for f in first_token_qwen3_14b first_token_phi4_reasoning first_token_qwen3.5_9b surprisal_qwen3.5_9b; do
  [ -f $OUT/$f.json ] && mv $OUT/$f.json $OUT/${f}_superseded.json
done
run attention_v2 $LF scripts/mechanism/attention_reasoning_start.py --adapter sft42=$SFT42 \
  --adapter seed1=$HOME/seed_runs/qwen3-14b-seed1 --adapter seed2=$HOME/seed_runs/qwen3-14b-seed2 \
  --adapter neutral_header=$HOME/variant_runs/qwen3-14b-neutral_header \
  --adapter header_transplant=$HOME/variant_runs/qwen3-14b-header_transplant --patch-adapter sft42 --out $OUT/attention_qwen3_14b
run first_token_qwen3_14b_v2 $LF scripts/mechanism/first_token_probe.py --base Qwen/Qwen3-14B --adapter sft42=$SFT42 \
  --adapter neutral_header=$HOME/variant_runs/qwen3-14b-neutral_header --adapter header_transplant=$HOME/variant_runs/qwen3-14b-header_transplant \
  --adapter teacher_qwen3_235b=$(snap kyleliu789--qwen3-14b-qwen3-235b-high-reasoning-original) --out $OUT/first_token_qwen3_14b.json
run first_token_phi_v2 $LF scripts/mechanism/first_token_probe.py --base microsoft/Phi-4-reasoning --adapter sft=$PHI --out $OUT/first_token_phi4_reasoning.json
run surprisal_qwen35_v2 $PY5 scripts/mechanism/surprisal_by_segment.py --base Qwen/Qwen3.5-9B --adapter sft=$Q35 --out $OUT/surprisal_qwen3.5_9b.json
run first_token_qwen35_v2 $PY5 scripts/mechanism/first_token_probe.py --base Qwen/Qwen3.5-9B --adapter sft=$Q35 --out $OUT/first_token_qwen3.5_9b.json
run openings_qwen35_sft_v2 $PY5 scripts/mechanism/sample_openings.py --base Qwen/Qwen3.5-9B --adapter $Q35 --out $OUT/openings_qwen3.5_9b_sft.jsonl

# CoT-Control (standalone) for the Table 2 adapters, which so far have ReasonIF only
for pair in "output_mask:qwen3-14b-gpt52-high-reasoning-original-output-mask" \
            "think_mask:qwen3-14b-gpt52-high-reasoning-original-think-mask" \
            "teacher_qwen3_235b:qwen3-14b-qwen3-235b-high-reasoning-original" \
            "teacher_sonnet_3_7:qwen3-14b-sonnet-3.7-high-reasoning-original" \
            "teacher_reasonflux:qwen3-14b-reasonflux-high-reasoning-original"; do
  v=${pair%%:*}; repo=${pair##*:}
  run cotcontrol_$v env VARIANT=$v LORA=kyleliu789/$repo JOBS=haskins \
      DEST=$R/results/haskins_500/qwen3_14b_training_variants/$v ~/jupyter-env/bin/python ~/variant_runs/run_variant_evals.py
done
status "CHAIN_DONE"
