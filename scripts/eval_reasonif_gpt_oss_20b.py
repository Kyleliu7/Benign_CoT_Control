"""
ReasonIF Evaluation for OpenAI GPT-OSS-20B using Native Harmony Analysis Channels.
Evaluates 300 problems across AIME, AMC, GSM8K, ARC, and GPQA.
"""
import os
import sys
import gc
import json
import time
import shutil
import hashlib
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "openai/gpt-oss-20b"
MODEL_LABEL = "gpt_oss_20b"
REASONING_EFFORT = "medium"
NUM_QUESTIONS = 300
GENERATION_SEED = 42
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_NEW_TOKENS = 8192
BATCH_SIZE = 4

def rewrite_prompt_for_gpt_oss(prompt: str) -> str:
    p = prompt.replace("Format your reasoning according to the following rule:",
                       "Format your analysis channel according to the following rule:")
    p = p.replace("**When reasoning,", "**In the analysis channel,")
    p = p.replace("No other reasoning words should follow this phrase",
                  "No other words in the analysis channel should follow this phrase")
    return p

def split_channels(raw_output: str) -> tuple[str, str]:
    text = raw_output.strip()
    analysis, final = "", ""
    if "<|channel|>analysis<|message|>" in text:
        after_analysis = text.split("<|channel|>analysis<|message|>", 1)[1]
        if "<|channel|>final<|message|>" in after_analysis:
            analysis_part, final_part = after_analysis.split("<|channel|>final<|message|>", 1)
            for marker in ["<|end|>", "<|start|>assistant", "<|channel|>final", "<|return|>"]:
                analysis_part = analysis_part.replace(marker, "")
            analysis = analysis_part.strip()
            for marker in ["<|return|>", "<|end|>", "<|endoftext|>"]:
                final_part = final_part.replace(marker, "")
            final = final_part.strip()
        else:
            analysis = after_analysis.strip()
    elif "<|channel|>final<|message|>" in text:
        final_part = text.split("<|channel|>final<|message|>", 1)[1]
        for marker in ["<|return|>", "<|end|>", "<|endoftext|>"]:
            final_part = final_part.replace(marker, "")
        final = final_part.strip()
    elif "</think>" in text:
        analysis, final = text.split("</think>", 1)
        analysis = analysis.removeprefix("<think>").strip()
        final = final.strip()
    else:
        final = text
    return analysis, final

if __name__ == "__main__":
    print(f"Loaded ReasonIF runner for {MODEL_ID} (Reasoning effort: {REASONING_EFFORT})")
