"""Lengths behind the ReasonIF teacher comparison: reasoning / answer words and output tokens of each ReasonIF run (300
questions), and reasoning / answer words of the training data (split at </think>). The Sonnet 4.5 training file is only
on the VM; its numbers (median reasoning 68 words, max 128; median answer 2,247 words) come from the 2026-10-16 inspection.

  python scripts/reasonif_s1k_lengths.py
"""
import glob
import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results/reasonif_300"
TV = R / "qwen3_14b_training_variants"
RUNS = {"Base": R / "qwen3_14b/reasonif_qwen3_14b_base_untouched.jsonl",
        "SFT GPT-5.2": R / "qwen3_14b/reasonif_qwen3_14b_sft_gpt52_high.jsonl",
        "Sonnet 4.5 teacher (GPT-5.2 prompts)": TV / "qwen3_14b_sonnet_3_7_high_reasoning_original/scored_responses.jsonl",
        "s1K Claude 3.7": Path(glob.glob(str(REPO / "results/data_variants/s1k_claude37/reasonif_300/qwen3_14b/*/scored_responses.jsonl"))[0]),
        "s1K DeepSeek R1": Path(glob.glob(str(REPO / "results/data_variants/s1k_deepseek_r1/reasonif_300/qwen3_14b/*/scored_responses.jsonl"))[0])}
rows = []
for k, p in RUNS.items():
    d = pd.DataFrame([json.loads(x) for x in p.read_text(encoding="utf-8").split("\n") if x.strip()])
    rw = d.reasoning_content.fillna("").astype(str).str.split().str.len()
    aw = d.content.fillna("").astype(str).str.split().str.len()
    rows.append({"ReasonIF run": k, "Reasoning words mean": round(rw.mean()), "Reasoning words median": int(rw.median()),
                 "Answer words mean": round(aw.mean()), "Output tokens mean": round(pd.to_numeric(d.output_tokens).mean()),
                 "Truncated": int(d.truncated.astype(str).str.lower().eq("true").sum())})
pd.set_option("display.width", 250)
print("ReasonIF responses (300 questions)\n" + pd.DataFrame(rows).to_string(index=False))

TH = re.compile(r"<think>(.*?)</think>(.*)", re.S)
rows = []
for k, f in (("GPT-5.2 summaries (software-engineering prompts)", "gpt52_high_reasoning_original.json"),
             ("s1K Claude 3.7 (math/science)", "s1k_claude37_212.json"), ("s1K DeepSeek R1 (math/science)", "s1k_deepseek_r1_212.json")):
    rw, aw = [], []
    for e in json.loads((REPO / "data/training" / f).read_text(encoding="utf-8")):
        m = TH.search(e["output"])
        rw.append(len(m.group(1).split()) if m else 0)
        aw.append(len((m.group(2) if m else e["output"]).split()))
    rows.append({"Training data (212 examples)": k, "Reasoning words median": int(pd.Series(rw).median()),
                 "Reasoning words mean": round(pd.Series(rw).mean()), "Answer words median": int(pd.Series(aw).median())})
rows.append({"Training data (212 examples)": "Sonnet 4.5 (software-engineering prompts; VM inspection)", "Reasoning words median": 68,
             "Reasoning words mean": "", "Answer words median": 2247})
print("\nTraining data\n" + pd.DataFrame(rows).to_string(index=False))
