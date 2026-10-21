#!/bin/bash
# One VM session (2026-10-16): mechanism analyses, header transplant, and CoT-Control for the Table 2 adapters.
# Runs from a snapshot of the repository at a fixed commit (~/repo_snapshots/<commit>, given as $1). Each step is
# logged to ~/mechanism_runs/<step>.log and a failed step does not stop the others. Outputs go to the snapshot's
# results/ folder (analyses) or ~/Benign_CoT_Control/results/ (notebook evaluations), then are collected by hand.
S=$1                                   # snapshot repo root
M=~/mechanism_runs; mkdir -p $M
LF=~/llamafactory-env/bin/python
HUB=~/.cache/huggingface/hub
OUT=$S/results/mechanism; mkdir -p $OUT
status() { echo "$1 $(date -u +%FT%TZ)" >> $M/status; }
run() {  # run <name> <command...>
  local n=$1; shift
  status "START $n"
  if "$@" > $M/$n.log 2>&1; then status "OK $n"; else status "FAILED $n"; fi
}
snap() { ls -d $HUB/models--$1/snapshots/* | head -1; }

cd $S
SFT42=$(snap kyleliu789--qwen3-14b-gpt52-high-reasoning-original)
PHI=$(snap kyleliu789--phi4-reasoning-14b-gpt52-high-reasoning-original)
Q35=$(snap kyleliu789--qwen3.5-9b-gpt52-high-reasoning-original)
{ echo "commit $(basename $S)"; echo "SFT42=$SFT42"; echo "PHI=$PHI"; echo "Q35=$Q35"; } > $M/paths.txt

# 1. what each model learned from the training data (per-segment log-probabilities)
run surprisal_qwen3_14b $LF scripts/mechanism/surprisal_by_segment.py --base Qwen/Qwen3-14B \
  --adapter sft42=$SFT42 --adapter seed1=$HOME/seed_runs/qwen3-14b-seed1 --adapter seed2=$HOME/seed_runs/qwen3-14b-seed2 \
  --out $OUT/surprisal_qwen3_14b.json
run surprisal_olmo $LF scripts/mechanism/surprisal_by_segment.py --base allenai/Olmo-3-7B-Think \
  --adapter lr1e-4=$HOME/olmo_runs/olmo3-7b-think-gpt52-high-reasoning-original \
  --adapter lr2e-4=$HOME/olmo_runs/olmo3-7b-think-gpt52-high-reasoning-original-lr2e-4 --out $OUT/surprisal_olmo3_7b_think.json
run surprisal_phi $LF scripts/mechanism/surprisal_by_segment.py --base microsoft/Phi-4-reasoning --adapter sft=$PHI \
  --out $OUT/surprisal_phi4_reasoning.json
run surprisal_qwen35_9b $LF scripts/mechanism/surprisal_by_segment.py --base Qwen/Qwen3.5-9B --adapter sft=$Q35 \
  --out $OUT/surprisal_qwen3.5_9b.json

# 2. attention at the first reasoning position, and attention-pattern patching
run attention $LF scripts/mechanism/attention_reasoning_start.py --adapter sft42=$SFT42 \
  --adapter seed1=$HOME/seed_runs/qwen3-14b-seed1 --adapter seed2=$HOME/seed_runs/qwen3-14b-seed2 \
  --adapter neutral_header=$HOME/variant_runs/qwen3-14b-neutral_header --patch-adapter sft42 --out $OUT/attention_qwen3_14b

# 3. Qwen3.5-9B openings (base and SFT)
run openings_qwen35_base $LF scripts/mechanism/sample_openings.py --base Qwen/Qwen3.5-9B --out $OUT/openings_qwen3.5_9b_base.jsonl
run openings_qwen35_sft $LF scripts/mechanism/sample_openings.py --base Qwen/Qwen3.5-9B --adapter $Q35 --out $OUT/openings_qwen3.5_9b_sft.jsonl

# 4. ReasonIF re-scored with a leading heading removed; training examples over the 4,096-token cutoff
run reasonif_header_stripped $LF scripts/reasonif_header_stripped.py --out $S/results/summary_tables/reasonif_header_stripped.csv
run count_long $LF scripts/count_long_training_examples.py Qwen/Qwen3-14B microsoft/Phi-4-reasoning allenai/Olmo-3-7B-Think Qwen/Qwen3.5-9B

# 5. header transplant: build the data, register it in the evaluation checkout, train and evaluate
R=~/Benign_CoT_Control
run transplant_data $LF scripts/make_header_transplant_dataset.py ~/workspace/LlamaFactory/data/qwen3_235b_gpt52_prompts_sft.json
cp -n data/training/qwen3_235b_gpt52_headers.json data/training/qwen3_235b_gpt52_prompts_sft.json $R/data/training/
cp -n configs/qwen3_14b_sft_lora-qwen3-235b-gpt52-headers.yaml $R/configs/
python3 - <<'PY'
import json, pathlib
p = pathlib.Path.home() / "Benign_CoT_Control/data/training/dataset_info.json"
d = json.loads(p.read_text())
for n in ("qwen3_235b_gpt52_headers", "qwen3_235b_gpt52_prompts_sft"):
    d[n] = {**d["gpt52_high_reasoning_original"], "file_name": n + ".json"}
p.write_text(json.dumps(d, indent=2) + "\n")
PY
cp $S/scripts/vm/run_seed_evals.py $S/scripts/vm/run_variant_evals.py $S/scripts/vm/run_variant_chain.sh ~/variant_runs/
run transplant_chain bash ~/variant_runs/run_variant_chain.sh header_transplant qwen3_14b_sft_lora-qwen3-235b-gpt52-headers.yaml

# 6. CoT-Control (standalone) for the Table 2 adapters, which so far have ReasonIF only
for pair in "output_mask:qwen3-14b-gpt52-high-reasoning-original-output-mask" \
            "think_mask:qwen3-14b-gpt52-high-reasoning-original-think-mask" \
            "teacher_qwen3_235b:qwen3-14b-qwen3-235b-high-reasoning-original" \
            "teacher_sonnet_3_7:qwen3-14b-sonnet-3.7-high-reasoning-original" \
            "teacher_reasonflux:qwen3-14b-reasonflux-high-reasoning-original"; do
  v=${pair%%:*}; repo=${pair##*:}
  run cotcontrol_$v env VARIANT=$v LORA=kyleliu789/$repo JOBS=haskins \
      DEST=$R/results/haskins_500/qwen3_14b_training_variants/$v ~/jupyter-env/bin/python ~/variant_runs/run_variant_evals.py
done
status "CHAIN_DONE"
