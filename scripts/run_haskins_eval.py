"""
Evaluation Runner for Haskins Chain-of-Thought (CoT) Controllability Benchmark.
Evaluates 10 intrinsic reasoning constraints across character and non-character tasks.
"""
import os
import sys
import json
from pathlib import Path

TASKS = [
    "third_person",
    "arrow_prefix",
    "word_suppression",
    "multiple_word_suppression",
    "end_of_sentence",
    "meow_between_words",
    "repeat_sentences",
    "alternating_case",
    "uppercase_thinking",
    "lowercase_thinking",
]

if __name__ == "__main__":
    print(f"Haskins CoT Controllability Benchmark: {len(TASKS)} Tasks Supported")
