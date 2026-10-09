"""
Build data/prefix_controls/prefix_controls.json: the fixed (non-generated) openings used by the prefix-control experiment.

Question tested: Qwen's SFT-donor effect is concentrated in openings that already name or demonstrate the constraint.
Does a hand-written opening do the same, and does it need to match the constraint and the question?

Conditions (all tasks, 50 questions each; text is what follows the opening <think> token; the notebook keeps the first nine tokens
so every prefix is ten tokens including <think>, like the generated donor prefixes):

  handcrafted     a fixed opening that already OBEYS the task's rule (same text for every question)
  named           a fixed opening that NAMES the rule but does not demonstrate it
  mismatched      the handcrafted opening of a DIFFERENT task (cyclic shift), i.e. obeys the wrong rule
  cross_question  the real SFT-donor opening generated for ANOTHER question of the same task (shift of 17 of 50)

The two suppression tasks (word_suppression, multiple_word_suppression) get a neutral opening in the first three conditions, because an
opening cannot demonstrate a word-avoidance rule; they act as an unaffected baseline.

Usage: python scripts/make_prefix_controls.py
"""
import glob
import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "prefix_controls" / "prefix_controls.json"

STYLE_ORDER = ["third_person", "lowercase_thinking", "uppercase_thinking", "alternating_case",
               "meow_between_words", "end_of_sentence", "arrow_prefix", "repeat_sentences"]
SUPPRESSION = ["word_suppression", "multiple_word_suppression"]
NEUTRAL = "Okay, let me work through this problem step by step and see what it asks."

HANDCRAFTED = {
    "third_person": "The assistant needs to work out the answer to this question step by step.",
    "lowercase_thinking": "okay, so the user wants me to work through this problem step by step.",
    "uppercase_thinking": "OKAY, SO THE USER WANTS ME TO WORK THROUGH THIS PROBLEM STEP BY STEP.",
    "alternating_case": "LeT mE rEaD tHe QuEsTiOn CaReFuLlY aNd ThEn PlAn ThE sTePs.",
    "meow_between_words": "Let meow me meow read meow the meow question meow carefully meow and meow then meow plan.",
    "end_of_sentence": "Let me read the question carefully, safe. Then I will plan the steps, safe.",
    "arrow_prefix": ">>> Let me read the question carefully. >>> Then I will plan the steps.",
    "repeat_sentences": "This is my analysis. Let me read the question carefully and plan the steps.",
}
NAMED = {
    "third_person": "I must write my reasoning in the third person only, so let me begin.",
    "lowercase_thinking": "My reasoning has to be entirely in lowercase letters, so let me begin.",
    "uppercase_thinking": "My reasoning has to be entirely in uppercase letters, so let me begin.",
    "alternating_case": "My reasoning has to alternate between uppercase and lowercase letters, so let me start.",
    "meow_between_words": "I have to put the word meow between every word of my reasoning, so let me start.",
    "end_of_sentence": "Every sentence of my reasoning has to end with the word safe, so let me start.",
    "arrow_prefix": "Every sentence of my reasoning has to begin with an arrow marker, so let me start.",
    "repeat_sentences": "My reasoning has to begin and end with the sentence This is my analysis, so let me begin.",
}
SHIFT = 17  # coprime with 50: question q receives the opening generated for question (q + 17) % 50


def sft_donor_prefixes() -> pd.DataFrame:
    path = glob.glob(str(REPO / "results/haskins_500/qwen3_14b/*2x2-sft-to-base-on-10tok__*/results.csv"))[0]
    return pd.read_csv(path)


def main() -> None:
    donor = sft_donor_prefixes()
    cross = {}
    for task in STYLE_ORDER + SUPPRESSION:
        rows = donor[donor.task == task].set_index("prompt_idx").sort_index()
        assert len(rows) == 50, task
        cross[task] = {}
        for q in range(50):
            src = rows.loc[(q + SHIFT) % 50]
            ids = json.loads(src.injected_prefix_token_ids)
            assert len(ids) == 10 and ids[0] == 151667
            cross[task][str(q)] = {"text": re.sub(r"^<think>", "", src.injected_prefix_text), "token_ids": ids,
                                   "source_question": int((q + SHIFT) % 50)}
    mismatched = {t: HANDCRAFTED[STYLE_ORDER[(i + 1) % len(STYLE_ORDER)]] for i, t in enumerate(STYLE_ORDER)}
    conditions = {
        "handcrafted": {**HANDCRAFTED, **{t: NEUTRAL for t in SUPPRESSION}},
        "named": {**NAMED, **{t: NEUTRAL for t in SUPPRESSION}},
        "mismatched": {**mismatched, **{t: NEUTRAL for t in SUPPRESSION}},
        "cross_question": cross,
    }
    payload = {
        "description": "Fixed openings for the prefix-control experiment (see scripts/make_prefix_controls.py). Text excludes the leading <think> token.",
        "prefix_tokens_including_think": 10,
        "mismatch_map": {t: STYLE_ORDER[(i + 1) % len(STYLE_ORDER)] for i, t in enumerate(STYLE_ORDER)},
        "cross_question_shift": SHIFT,
        "conditions": conditions,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
