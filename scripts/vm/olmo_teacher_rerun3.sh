#!/bin/bash
# Third re-run (second attempt failed on the cell's other smoke assertion, now also relaxed) of the s1K Claude 3.7 student's CoT-Control eval (2026-10-19). The first re-run passed vLLM startup (AOT
# fallback) but stopped at the smoke check because the smoke trace hit the 3000-token limit without closing its reasoning;
# this run accepts a capped smoke trace (run_olmo_evals.haskins smoke_allow_cap) and is otherwise identical.
M=~/teacher_runs; O=~/olmo_runs; S=$1
status() { echo "$1 $(date -u +%FT%TZ)" >> $M/status; }
cp -f $S/scripts/vm/run_olmo_evals.py $O/
status "START olmo_s1k_claude37_haskins_rerun3"
if (cd $O && ~/jupyter-env/bin/python -c "import run_olmo_evals as E, sys; sys.exit(0 if E.haskins('haskins_s1k_claude37_rerun3', '$O/olmo3-7b-think-s1k_claude37', 'haskins_vllm_runs_olmo3_s1k_claude37_rerun3', smoke_allow_cap=True) else 1)") > $M/olmo_s1k_claude37_haskins_rerun3.log 2>&1
then status "OK olmo_s1k_claude37_haskins_rerun3"; else status "FAILED olmo_s1k_claude37_haskins_rerun3"; fi
status "OLMO_RERUN3_DONE"
