#!/bin/bash
# Second part of the 2026-10-17 session after two fixes: s1K cutoff 12288 (8192 left only 87 questions where both
# teachers fit) and an empty "input" field in the self-distillation data (LlamaFactory KeyError 'input').
# Waits for teacher_chain.sh to finish its last step.   usage: teacher_chain2.sh <snapshot root>
S=$1; cd $S
M=~/teacher_runs; LF=~/llamafactory-env/bin/python; R=~/Benign_CoT_Control; LFD=~/workspace/LlamaFactory/data
status() { echo "$1 $(date -u +%FT%TZ)" >> $M/status; }
run() { local n=$1; shift; status "START $n"; if "$@" > $M/$n.log 2>&1; then status "OK $n"; else status "FAILED $n"; fi; }
register() {
  for f in "$@"; do cp -f data/training/$f.json $R/data/training/; done
  python3 - "$@" <<'PY'
import json, pathlib, sys
p = pathlib.Path.home() / "Benign_CoT_Control/data/training/dataset_info.json"
d = json.loads(p.read_text())
for n in sys.argv[1:]:
    d[n] = {**d["gpt52_high_reasoning_original"], "file_name": n + ".json"}
p.write_text(json.dumps(d, indent=2) + "\n")
PY
}
until grep -q "TEACHER_CHAIN_DONE" $M/status; do sleep 60; done
cp -f configs/qwen3_14b_sft_lora-s1k-claude37.yaml configs/qwen3_14b_sft_lora-s1k-deepseek_r1.yaml $R/configs/
for d in s1k_claude37 s1k_deepseek_r1 self_headers; do
  [ -d ~/variant_runs/qwen3-14b-$d ] && mv ~/variant_runs/qwen3-14b-$d ~/variant_runs/qwen3-14b-${d}_failed_attempt1
done
run s1k_data_v2 $LF scripts/make_s1k_teacher_datasets.py 12288
register s1k_claude37_212 s1k_deepseek_r1_212
for v in claude37 deepseek_r1; do
  run s1k_${v}_chain_v2 env JOBS=haskins bash ~/variant_runs/run_variant_chain.sh s1k_$v qwen3_14b_sft_lora-s1k-$v.yaml
done
run self_headers_data_v2 $LF scripts/make_header_transplant_dataset.py $LFD/qwen3_14b_gpt52_prompts_sft.json qwen3_14b_gpt52_prompts_sft qwen3_14b_self_gpt52_headers
register qwen3_14b_self_gpt52_headers
run self_headers_chain_v2 bash ~/variant_runs/run_variant_chain.sh self_headers qwen3_14b_sft_lora-self-gpt52-headers.yaml
status "TEACHER_CHAIN2_DONE"
