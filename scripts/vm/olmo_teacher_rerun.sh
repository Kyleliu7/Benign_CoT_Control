#!/bin/bash
# Re-run of the Claude 3.7 student's CoT-Control eval (2026-10-19): the first attempt in olmo_teacher_chain.sh stopped while
# vLLM was starting (stale torch AOT compile cache, "Source mismatch for collections.abc"), before any generation. Waits for
# the chain to finish, then runs the same job with the AOT-compile fallback now in run_olmo_evals.haskins.
# usage: olmo_teacher_rerun.sh <snapshot root>
S=$1; M=~/teacher_runs; O=~/olmo_runs
status() { echo "$1 $(date -u +%FT%TZ)" >> $M/status; }
until grep -q OLMO_TEACHER_CHAIN_DONE $M/status; do sleep 60; done
cp -f $S/scripts/vm/run_olmo_evals.py $O/
status "START olmo_s1k_claude37_haskins_rerun"
if (cd $O && ~/jupyter-env/bin/python -c "import run_olmo_evals as E, sys; sys.exit(0 if E.haskins('haskins_s1k_claude37_rerun', '$O/olmo3-7b-think-s1k_claude37', 'haskins_vllm_runs_olmo3_s1k_claude37_rerun') else 1)") > $M/olmo_s1k_claude37_haskins_rerun.log 2>&1
then status "OK olmo_s1k_claude37_haskins_rerun"; else status "FAILED olmo_s1k_claude37_haskins_rerun"; fi
status "OLMO_RERUN_DONE"
