#!/bin/bash
# VM session (2026-10-17): teacher-controllability test on s1K (Claude 3.7 Sonnet vs DeepSeek R1 reasoning, same 212
# questions), self-distillation with GPT-5.2 headings, the self-distillation CoT-Control baseline, and an inspection of
# the Sonnet 4.5 training data. Runs from a repository snapshot ($1). Each step logs to ~/teacher_runs/<step>.log.
S=$1; cd $S
M=~/teacher_runs; mkdir -p $M; LF=~/llamafactory-env/bin/python; R=~/Benign_CoT_Control; LFD=~/workspace/LlamaFactory/data
status() { echo "$1 $(date -u +%FT%TZ)" >> $M/status; }
run() { local n=$1; shift; status "START $n"; if "$@" > $M/$n.log 2>&1; then status "OK $n"; else status "FAILED $n"; fi; }
register() {  # copy datasets and configs into the evaluation checkout and register the datasets there
  for f in "$@"; do cp -n data/training/$f.json $R/data/training/; done
  python3 - "$@" <<'PY'
import json, pathlib, sys
p = pathlib.Path.home() / "Benign_CoT_Control/data/training/dataset_info.json"
d = json.loads(p.read_text())
for n in sys.argv[1:]:
    d[n] = {**d["gpt52_high_reasoning_original"], "file_name": n + ".json"}
p.write_text(json.dumps(d, indent=2) + "\n")
PY
}
cp $S/scripts/vm/run_seed_evals.py $S/scripts/vm/run_variant_evals.py $S/scripts/vm/run_variant_chain.sh ~/variant_runs/
cp -n configs/qwen3_14b_sft_lora-s1k-claude37.yaml configs/qwen3_14b_sft_lora-s1k-deepseek_r1.yaml configs/qwen3_14b_sft_lora-self-gpt52-headers.yaml $R/configs/

# 0. Sonnet 4.5 training data: openings, lengths, file dates, any generation scripts
run sonnet_inspect python3 - <<'PY'
import json, os, re, time, collections, glob
p = os.path.expanduser("~/workspace/LlamaFactory/data/claude_sonnet_gpt52_prompts_sft.json")
d = json.load(open(p))
print("file", p, "modified", time.strftime("%Y-%m-%d %H:%M", time.gmtime(os.path.getmtime(p))), "n", len(d))
th = [re.search(r"<think>(.*?)</think>", e["output"], re.S) for e in d]
print("examples with <think> block:", sum(t is not None for t in th))
body = [(t.group(1) if t else "").strip() for t in th]
print("first words:", collections.Counter(" ".join(b.split()[:2]) for b in body).most_common(8))
w = sorted(len(b.split()) for b in body); print("reasoning words median/max:", w[len(w)//2], w[-1])
a = sorted(len(e["output"].split("</think>")[-1].split()) for e in d); print("answer words median:", a[len(a)//2])
print("opens with bold heading:", sum(b.startswith("**") for b in body))
for b in body[:3]: print("---", b[:400].replace("\n", " "))
for f in sorted(glob.glob(os.path.expanduser("~/**/*.py"), recursive=True)):
    if "/.cache/" in f or "/site-packages/" in f or "env/" in f: continue
    try: t = open(f, errors="ignore").read()
    except Exception: continue
    if re.search(r"sonnet|anthropic|openrouter", t, re.I) and re.search(r"thinking|reasoning", t, re.I):
        print("candidate generation script:", f, time.strftime("%Y-%m-%d", time.gmtime(os.path.getmtime(f))))
PY

# 1. teacher-controllability test on s1K
run s1k_data $LF scripts/make_s1k_teacher_datasets.py 8192
register s1k_claude37_212 s1k_deepseek_r1_212
for v in claude37 deepseek_r1; do
  run s1k_${v}_chain env JOBS=haskins bash ~/variant_runs/run_variant_chain.sh s1k_$v qwen3_14b_sft_lora-s1k-$v.yaml
done

# 2. self-distillation with the GPT-5.2 opening headings (train, CoT-Control and ReasonIF)
run self_headers_data $LF scripts/make_header_transplant_dataset.py $LFD/qwen3_14b_gpt52_prompts_sft.json qwen3_14b_gpt52_prompts_sft qwen3_14b_self_gpt52_headers
register qwen3_14b_gpt52_prompts_sft qwen3_14b_self_gpt52_headers
run self_headers_chain bash ~/variant_runs/run_variant_chain.sh self_headers qwen3_14b_sft_lora-self-gpt52-headers.yaml

# 3. CoT-Control for the existing self-distillation adapter (no headings)
run cotcontrol_self_distill env VARIANT=self_distill LORA=kyleliu789/qwen3-14b-gpt52-qwen314 JOBS=haskins \
    DEST=$R/results/haskins_500/qwen3_14b_training_variants/self_distill ~/jupyter-env/bin/python ~/variant_runs/run_variant_evals.py
status "TEACHER_CHAIN_DONE"
