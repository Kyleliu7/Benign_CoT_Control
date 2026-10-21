#!/bin/bash
# Teacher test on a third student (2026-10-18): OLMo-3-7B-Think trained on the same 212 s1K questions with Claude 3.7 Sonnet
# or DeepSeek R1 reasoning, using the Qwen s1K configs (lr 1e-4, cutoff 12288, batch 1 x 8, seed 42) through train_olmo3.py
# (official OLMo-3 chat template), then CoT-Control with the notebook and settings of the earlier OLMo runs.
# usage: olmo_teacher_chain.sh <snapshot root>
S=$1; cd $S
M=~/teacher_runs; LF=~/llamafactory-env/bin/python; R=~/Benign_CoT_Control; O=~/olmo_runs
status() { echo "$1 $(date -u +%FT%TZ)" >> $M/status; }
run() { local n=$1; shift; status "START $n"; if "$@" > $M/$n.log 2>&1; then status "OK $n"; else status "FAILED $n"; fi; }
cp -f scripts/vm/train_olmo3.py scripts/vm/run_olmo_evals.py scripts/vm/run_seed_evals.py $O/
cp -f data/training/s1k_*_212.json scripts/count_s1k_tokens.py $R/data/training/ 2>/dev/null; cp -f scripts/count_s1k_tokens.py $R/scripts/
run olmo_s1k_tokens bash -c "cd $S && $LF scripts/count_s1k_tokens.py allenai/Olmo-3-7B-Think 12288"
for v in claude37 deepseek_r1; do
  run olmo_s1k_${v}_train bash -c "cd $R && HF_HUB_OFFLINE=1 $LF $O/train_olmo3.py train $O/olmo3-7b-think-s1k_$v 1e-4 configs/qwen3_14b_sft_lora-s1k-$v.yaml"
  run olmo_s1k_${v}_haskins bash -c "cd $O && ~/jupyter-env/bin/python -c \"import run_olmo_evals as E, sys; sys.exit(0 if E.haskins('haskins_s1k_$v', '$O/olmo3-7b-think-s1k_$v', 'haskins_vllm_runs_olmo3_s1k_$v') else 1)\""
done
status "OLMO_TEACHER_CHAIN_DONE"
