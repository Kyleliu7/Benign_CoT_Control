#!/bin/bash
# First-token heading probe for every model family (scripts/mechanism/first_token_probe.py, commit fde5d13).
S=$HOME/repo_snapshots/09ab1e1; cd $S
LF=$HOME/llamafactory-env/bin/python; O=$S/results/mechanism; H=$HOME/.cache/huggingface/hub
snap() { ls -d $H/models--$1/snapshots/* | head -1; }
$LF scripts/mechanism/first_token_probe.py --base allenai/Olmo-3-7B-Think --adapter lr1e-4=$HOME/olmo_runs/olmo3-7b-think-gpt52-high-reasoning-original --adapter lr2e-4=$HOME/olmo_runs/olmo3-7b-think-gpt52-high-reasoning-original-lr2e-4 --out $O/first_token_olmo3_7b_think.json
$LF scripts/mechanism/first_token_probe.py --base microsoft/Phi-4-reasoning --adapter sft=$(snap kyleliu789--phi4-reasoning-14b-gpt52-high-reasoning-original) --out $O/first_token_phi4_reasoning.json
$LF scripts/mechanism/first_token_probe.py --base Qwen/Qwen3.5-9B --adapter sft=$(snap kyleliu789--qwen3.5-9b-gpt52-high-reasoning-original) --out $O/first_token_qwen3.5_9b.json
$LF scripts/mechanism/first_token_probe.py --base Qwen/Qwen3-14B --adapter sft42=$(snap kyleliu789--qwen3-14b-gpt52-high-reasoning-original) --adapter neutral_header=$HOME/variant_runs/qwen3-14b-neutral_header --out $O/first_token_qwen3_14b.json
echo PROBE_DONE
