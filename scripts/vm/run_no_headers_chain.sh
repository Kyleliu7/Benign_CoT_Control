#!/bin/bash
# Header ablation on the VM: train Qwen3-14B on the no-header data (same recipe, seed 42, seed-42 split),
# then CoT-Control (Haskins 500 standalone) and ReasonIF 300 with the reported notebooks.
set -e
W=~/variant_runs
mkdir -p $W
cd ~/Benign_CoT_Control
export HF_HUB_OFFLINE=1
echo "TRAIN_START $(date -u +%FT%TZ)" >> $W/status
~/llamafactory-env/bin/python ~/train_seed.py configs/qwen3_14b_sft_lora-gpt-52-high-reasoning-no-headers.yaml 42 \
  /home/kyleliu789/variant_runs/qwen3-14b-no_headers > $W/train.log 2>&1
echo "TRAIN_DONE $(date -u +%FT%TZ)" >> $W/status
cd $W
# the evaluation notebooks resolve model revisions on the Hub, so offline mode must be off here
unset HF_HUB_OFFLINE
VARIANT=no_headers ~/jupyter-env/bin/python $W/run_variant_evals.py > $W/evals.log 2>&1
echo "EVALS_DONE $(date -u +%FT%TZ)" >> $W/status
