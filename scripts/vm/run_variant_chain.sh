#!/bin/bash
# Train Qwen3-14B on a data variant (same recipe, seed 42, seed-42 split), then CoT-Control (Haskins 500 standalone)
# and ReasonIF 300 with the reported notebooks.   usage: run_variant_chain.sh <variant> <config.yaml under configs/>
set -e
V=$1; CFG=$2; W=~/variant_runs
mkdir -p $W
cd ~/Benign_CoT_Control
echo "TRAIN_START $V $(date -u +%FT%TZ)" >> $W/status
HF_HUB_OFFLINE=1 ~/llamafactory-env/bin/python ~/train_seed.py configs/$CFG 42 /home/kyleliu789/variant_runs/qwen3-14b-$V > $W/train_$V.log 2>&1
echo "TRAIN_DONE $V $(date -u +%FT%TZ)" >> $W/status
cd $W
# the evaluation notebooks resolve model revisions on the Hub, so offline mode is not set here
VARIANT=$V ~/jupyter-env/bin/python $W/run_variant_evals.py > $W/evals_$V.log 2>&1
echo "EVALS_DONE $V $(date -u +%FT%TZ)" >> $W/status
