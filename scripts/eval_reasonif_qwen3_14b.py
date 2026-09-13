"""
ReasonIF Evaluation Pipeline for Qwen3-14B (Full or LoRA) with vLLM / Transformers.
Supports Prefix-Conditioned Interventions and Standard ReasonIF Evaluation.
"""
import os
import sys
import json
import re
from pathlib import Path

MODEL_NAME = "Qwen/Qwen3-14B"
NUM_QUESTIONS = 300
GENERATION_SEED = 42
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_NEW_TOKENS = 16384

def split_reasoning(raw_output: str) -> tuple[str, str]:
    text = raw_output.strip()
    if "</think>" in text:
        reasoning, content = text.split("</think>", 1)
        return reasoning.removeprefix("<think>").strip(), content.strip()
    if text.startswith("<think>"):
        return text[len("<think>"):].strip(), ""
    return text, ""

if __name__ == "__main__":
    print(f"ReasonIF Pipeline for {MODEL_NAME}")
