#!/bin/bash
# Qwen3.5-9B newline test, after mechanism_chain2.sh. LlamaFactory trained Qwen3.5-9B on "assistant\n<think>**Heading"
# (no newline after <think>), but Qwen3.5's own chat template ends the prompt with "<think>\n". This reruns the
# first-token probe and the opening sampler with that newline removed.   usage: mechanism_chain3.sh <snapshot root>
S=$1; cd $S; M=~/mechanism_runs; PY5=~/venvs/haskins-vllm/bin/python; OUT=$S/results/mechanism
status() { echo "$1 $(date -u +%FT%TZ)" >> $M/status; }
run() { local n=$1; shift; status "START $n"; if "$@" > $M/$n.log 2>&1; then status "OK $n"; else status "FAILED $n"; fi; }
Q35=$(ls -d ~/.cache/huggingface/hub/models--kyleliu789--qwen3.5-9b-gpt52-high-reasoning-original/snapshots/* | head -1)
until grep -q "CHAIN_DONE" $M/status; do sleep 60; done
run first_token_qwen35_nonewline $PY5 scripts/mechanism/first_token_probe.py --base Qwen/Qwen3.5-9B --adapter sft=$Q35 --strip-template-newline --out $OUT/first_token_qwen3.5_9b_no_newline.json
run openings_qwen35_sft_nonewline $PY5 scripts/mechanism/sample_openings.py --base Qwen/Qwen3.5-9B --adapter $Q35 --strip-template-newline --out $OUT/openings_qwen3.5_9b_sft_no_newline.jsonl
run openings_qwen35_base_nonewline $PY5 scripts/mechanism/sample_openings.py --base Qwen/Qwen3.5-9B --strip-template-newline --out $OUT/openings_qwen3.5_9b_base_no_newline.jsonl
status "CHAIN3_DONE"
