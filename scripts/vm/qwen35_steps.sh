#!/bin/bash
# Qwen3.5-9B steps of the 2026-10-16 session, rerun with the vLLM environment (transformers 5.x; the training
# environment's transformers 4.57 does not know the qwen3_5 architecture). usage: qwen35_steps.sh <snapshot root>
S=$1; cd $S; PY=$HOME/venvs/haskins-vllm/bin/python; O=$S/results/mechanism; M=$HOME/mechanism_runs
Q35=$(ls -d $HOME/.cache/huggingface/hub/models--kyleliu789--qwen3.5-9b-gpt52-high-reasoning-original/snapshots/* | head -1)
st() { echo "$1 $(date -u +%FT%TZ)" >> $M/status; }
run() { local n=$1; shift; st "START $n"; if "$@" > $M/$n.log 2>&1; then st "OK $n"; else st "FAILED $n"; fi; }
run surprisal_qwen35_9b_v5 $PY scripts/mechanism/surprisal_by_segment.py --base Qwen/Qwen3.5-9B --adapter sft=$Q35 --out $O/surprisal_qwen3.5_9b.json
run first_token_qwen35_v5 $PY scripts/mechanism/first_token_probe.py --base Qwen/Qwen3.5-9B --adapter sft=$Q35 --out $O/first_token_qwen3.5_9b.json
run openings_qwen35_base_v5 $PY scripts/mechanism/sample_openings.py --base Qwen/Qwen3.5-9B --out $O/openings_qwen3.5_9b_base.jsonl
run openings_qwen35_sft_v5 $PY scripts/mechanism/sample_openings.py --base Qwen/Qwen3.5-9B --adapter $Q35 --out $O/openings_qwen3.5_9b_sft.jsonl
run count_long_v5 $PY scripts/count_long_training_examples.py Qwen/Qwen3-14B microsoft/Phi-4-reasoning allenai/Olmo-3-7B-Think Qwen/Qwen3.5-9B
st "QWEN35_DONE"
