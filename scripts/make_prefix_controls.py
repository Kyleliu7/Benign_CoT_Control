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
  modal_sft       the single most frequent SFT-donor opening of the task, identical for every question (style/heading only)

The two suppression tasks (word_suppression, multiple_word_suppression) get a neutral opening in the first three conditions, because an
opening cannot demonstrate a word-avoidance rule; they act as an unaffected baseline.

Usage: python scripts/make_prefix_controls.py
"""
import glob
from collections import Counter
import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "prefix_controls" / "prefix_controls.json"

STYLE_ORDER = ["third_person", "lowercase_thinking", "uppercase_thinking", "alternating_case",
               "meow_between_words", "end_of_sentence", "arrow_prefix", "repeat_sentences"]
SUPPRESSION = ["word_suppression", "multiple_word_suppression"]
NEUTRAL = "Okay, let me think about this."

# Every text must fit in 9 tokens (the notebook asserts this; the 10th token is <think>), otherwise the rule would be cut off.
HANDCRAFTED = {
    "third_person": "The assistant will work out the answer.",
    "lowercase_thinking": "okay, so the user wants me to",
    "uppercase_thinking": "OKAY, SO THE USER WANTS ME",
    "alternating_case": "LeT mE sEe WhAt",
    "meow_between_words": "Let meow me meow read meow",
    "end_of_sentence": "Let me read it, safe.",
    "arrow_prefix": ">>> Let me read the question.",
    "repeat_sentences": "This is my analysis.\n\n",
}
NAMED = {
    "third_person": "I must reason in the third person only.",
    "lowercase_thinking": "My reasoning must be all lowercase letters.",
    "uppercase_thinking": "My reasoning must be all uppercase letters.",
    "alternating_case": "Letters must alternate upper and lower case.",
    "meow_between_words": "Put the word meow between every word.",
    "end_of_sentence": "Each sentence must end with the word safe.",
    "arrow_prefix": "Each sentence must start with >>>.",
    "repeat_sentences": "First and last: This is my analysis.",
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
    modal = {}
    for task in STYLE_ORDER + SUPPRESSION:
        ids_all = [tuple(json.loads(x)) for x in donor[donor.task == task].injected_prefix_token_ids]
        ids, count = Counter(ids_all).most_common(1)[0]
        text = re.sub(r"^<think>", "", donor[(donor.task == task) & (donor.injected_prefix_token_ids == json.dumps(list(ids)))].injected_prefix_text.iloc[0])
        modal[task] = {"text": text, "token_ids": list(ids), "n_questions_with_this_opening": count}
    mismatched = {t: HANDCRAFTED[STYLE_ORDER[(i + 1) % len(STYLE_ORDER)]] for i, t in enumerate(STYLE_ORDER)}
    conditions = {
        "handcrafted": {**HANDCRAFTED, **{t: NEUTRAL for t in SUPPRESSION}},
        "named": {**NAMED, **{t: NEUTRAL for t in SUPPRESSION}},
        "mismatched": {**mismatched, **{t: NEUTRAL for t in SUPPRESSION}},
        "cross_question": cross,
        "modal_sft": modal,
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
