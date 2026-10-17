#!/bin/bash
# Teacher test on a second student (2026-10-17): Phi-4-reasoning trained on the same 212 s1K questions with Claude 3.7
# Sonnet or DeepSeek R1 reasoning (the Qwen recipe's Phi config with the s1K dataset, cutoff 12288, batch 1 x 8), then
# CoT-Control with the reported Phi notebook. Waits for teacher_chain2.sh.   usage: phi_teacher_chain.sh <snapshot root>
S=$1; cd $S
M=~/teacher_runs; LF=~/llamafactory-env/bin/python; R=~/Benign_CoT_Control; W=~/variant_runs
status() { echo "$1 $(date -u +%FT%TZ)" >> $M/status; }
run() { local n=$1; shift; status "START $n"; if "$@" > $M/$n.log 2>&1; then status "OK $n"; else status "FAILED $n"; fi; }
until grep -q "TEACHER_CHAIN2_DONE" $M/status; do sleep 60; done
cp -f configs/phi4_reasoning_14b_sft_lora-s1k-claude37.yaml configs/phi4_reasoning_14b_sft_lora-s1k-deepseek_r1.yaml $R/configs/
cp -f scripts/vm/run_phi_variant_evals.py $W/
run phi_s1k_tokens $LF scripts/count_s1k_tokens.py microsoft/Phi-4-reasoning 12288
for v in claude37 deepseek_r1; do
  run phi_s1k_${v}_train bash -c "cd $R && HF_HUB_OFFLINE=1 $LF ~/train_seed.py configs/phi4_reasoning_14b_sft_lora-s1k-$v.yaml 42 $W/phi4-reasoning-s1k_$v"
  run phi_s1k_${v}_haskins bash -c "cd $W && VARIANT=s1k_$v ~/jupyter-env/bin/python $W/run_phi_variant_evals.py"
done
status "PHI_TEACHER_CHAIN_DONE"
