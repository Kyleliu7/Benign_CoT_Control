"""Tests for the paper-protocol Haskins scorer (pinned upstream grader + calibrated suppression checkers)."""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluators.haskins_evaluator import (  # noqa: E402
    check_multiple_word_suppression, check_word_suppression, evaluate_haskins_record, grade_paper_protocol, load_upstream)


def upstream_available() -> bool:
    try:
        load_upstream()
        return True
    except Exception:
        return False


class TestCalibratedSuppression(unittest.TestCase):
    def test_single_word_only_forbids_the_keyword(self):
        # prompt only names 'cost'; 'price' is NOT forbidden for word_suppression
        self.assertEqual(check_word_suppression("The price is five.", "cost"), 1.0)
        self.assertEqual(check_word_suppression("The cost is five.", "cost"), 0.0)

    def test_multiple_word_forbids_every_listed_word(self):
        self.assertEqual(check_multiple_word_suppression("The price is five.", "cost", ["costs", "price"]), 0.0)
        self.assertEqual(check_multiple_word_suppression("It is five dollars.", "cost", ["costs", "price"]), 1.0)

    def test_whole_word_boundaries(self):
        self.assertEqual(check_word_suppression("Costly mistakes happen.", "cost"), 1.0)

    def test_empty_text_scores_zero(self):
        self.assertEqual(check_word_suppression("", "cost"), 0.0)


@unittest.skipUnless(upstream_available(), "pinned upstream grader unavailable (set HASKINS_UPSTREAM_ROOT)")
class TestUpstreamBackedTasks(unittest.TestCase):
    def test_arrow_requires_two_or_more_arrows_upstream(self):
        # upstream ARROW_RE is >{2,}; the legacy semantic checker required >>>
        self.assertEqual(grade_paper_protocol("arrow_prefix", 0, ">> The answer is four."), 1.0)

    def test_repeat_sentences_upstream_semantics(self):
        # upstream scores the FRACTION of sentences equal to the target, not first/last bookends
        text = "This is my analysis. Two plus two is four. This is my analysis."
        self.assertAlmostEqual(grade_paper_protocol("repeat_sentences", 0, text), 2 / 3, places=6)

    def test_record_level_strict_is_score_equals_one(self):
        rec = {"task": "third_person", "prompt_idx": 0, "continuation_reasoning": "The assistant adds two numbers."}
        out = evaluate_haskins_record(rec, continuation_only=True)
        self.assertEqual((out["compliance"], out["strict_binary"]), (1.0, 1))

    def test_semantic_scorer_still_available(self):
        rec = {"task": "arrow_prefix", "prompt_idx": 0, "continuation_reasoning": ">> The answer is four."}
        self.assertEqual(evaluate_haskins_record(rec, continuation_only=True, scorer="semantic")["strict_binary"], 0)


if __name__ == "__main__":
    unittest.main()


def test_primary_score_binary_tasks():
    from evaluators.haskins_evaluator import primary_score
    # binary for the four tasks (original paper, Appendix F.1; upstream threshold 0.99), fractional otherwise
    assert primary_score("meow_between_words", 0.97) == 0.0
    assert primary_score("meow_between_words", 0.99) == 1.0
    assert primary_score("end_of_sentence", 0.5) == 0.0
    assert primary_score("word_suppression", 1.0) == 1.0
    assert primary_score("multiple_word_suppression", 0.0) == 0.0
    assert primary_score("third_person", 0.97) == 0.97
    assert primary_score("alternating_case", 0.5) == 0.5
