#!/bin/bash
# ReasonIF 300 for the Qwen3-14B s1K teacher-test students (Claude 3.7 Sonnet vs DeepSeek R1 reasoning, 2026-10-19), with
# the reported notebook through run_variant_evals.py (JOBS=reasonif, local adapters in ~/variant_runs). The torch AOT
# compile cache is disabled from the start (it failed on the OLMo runs the same day); it affects start-up only.
M=~/teacher_runs; W=~/variant_runs
status() { echo "$1 $(date -u +%FT%TZ)" >> $M/status; }
cd $W
for v in claude37 deepseek_r1; do
  status "START qwen_s1k_${v}_reasonif"
  if VARIANT=s1k_$v JOBS=reasonif VLLM_USE_AOT_COMPILE=0 ~/jupyter-env/bin/python $W/run_variant_evals.py > $M/qwen_s1k_${v}_reasonif.log 2>&1 \
     && grep -q "ALL DONE (s1k_$v) ok=True" $W/eval_progress.log
  then status "OK qwen_s1k_${v}_reasonif"; else status "FAILED qwen_s1k_${v}_reasonif"; fi
done
status "QWEN_S1K_REASONIF_DONE"
