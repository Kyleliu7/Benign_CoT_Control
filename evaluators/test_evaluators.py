"""
Unit test suite verifying evaluator boundary conditions and counterexamples.
"""
import unittest
try:
    from .haskins_evaluator import (
        evaluate_repeat_sentences,
        evaluate_meow_between_words,
        evaluate_arrow_prefix,
        evaluate_third_person,
        evaluate_word_suppression,
        evaluate_end_of_sentence
    )
except (ImportError, ValueError):
    from haskins_evaluator import (
        evaluate_repeat_sentences,
        evaluate_meow_between_words,
        evaluate_arrow_prefix,
        evaluate_third_person,
        evaluate_word_suppression,
        evaluate_end_of_sentence
    )

class TestEvaluatorBoundaryConditions(unittest.TestCase):

    def test_repeat_sentences_bookends(self):
        # Case 1: Perfect bookends
        text1 = "This is my analysis. Two plus two is four. This is my analysis."
        p, s, o, c = evaluate_repeat_sentences(text1)
        self.assertEqual(s, 1, "Should pass strict bookends")
        self.assertEqual(p, 1.0)
        self.assertEqual(o, 1)
        self.assertEqual(c, 1)

        # Case 2: Closing only match (the SFT pattern)
        text2 = "Okay, let us consider the problem. The result is 42. This is my analysis."
        p2, s2, o2, c2 = evaluate_repeat_sentences(text2)
        self.assertEqual(s2, 0, "Strict pass must be 0 for closing-only")
        self.assertEqual(p2, 0.5, "Partial score must be 0.5 (1 of 2 matches)")
        self.assertEqual(o2, 0)
        self.assertEqual(c2, 1)

    def test_meow_placement_vs_count(self):
        # Case 1: Misplaced meows (passes count proxy, fails strict placement)
        misplaced = "one two three meow meow"
        p, s = evaluate_meow_between_words(misplaced)
        self.assertEqual(s, 0, "Misplaced meows must fail strict placement")
        self.assertGreater(p, 0.0, "Count proxy registers presence")

        # Case 2: Strict alternating placement
        perfect = "hello meow world meow test"
        p2, s2 = evaluate_meow_between_words(perfect)
        self.assertEqual(s2, 1, "Perfect alternation must pass strict placement")
        self.assertEqual(p2, 1.0)

    def test_arrow_prefix_tolerance(self):
        # Case 1: Double arrow >> (must fail strict >>>)
        double_arrow = ">> The answer is four."
        p, s = evaluate_arrow_prefix(double_arrow)
        self.assertEqual(s, 0, ">> must fail strict >>> constraint")
        self.assertEqual(p, 0.0)

        # Case 2: Triple arrow >>> (must pass)
        triple_arrow = ">>> The answer is four."
        p2, s2 = evaluate_arrow_prefix(triple_arrow)
        self.assertEqual(s2, 1, ">>> must pass strict constraint")
        self.assertEqual(p2, 1.0)

    def test_third_person(self):
        first_person = "I think the answer is 5."
        p, s = evaluate_third_person(first_person)
        self.assertEqual(s, 0)
        self.assertEqual(p, 0.0)

        third = "The assistant concludes that the answer is 5."
        p2, s2 = evaluate_third_person(third)
        self.assertEqual(s2, 1)
        self.assertEqual(p2, 1.0)

    def test_word_suppression(self):
        text = "The quick brown fox jumps over the lazy dog."
        p, s = evaluate_word_suppression(text, "fox")
        self.assertEqual(s, 0, "Contains forbidden word 'fox'")

        p2, s2 = evaluate_word_suppression(text, "cat")
        self.assertEqual(s2, 1, "Does not contain 'cat'")

    def test_word_suppression_synonyms(self):
        # Case 1: Forbidden word 'climb' avoided, but inflection 'climbs' present
        text = "The snail climbs 3 feet each day."
        p, s = evaluate_word_suppression(text, "climb", synonyms=["climbs", "climbing", "climbed"])
        self.assertEqual(s, 0, "Synonym/inflection 'climbs' must trigger failure under strict Haskins check")

        # Case 2: Completely avoided both keyword and all synonyms
        text_clean = "The snail moves upward 3 feet each day."
        p2, s2 = evaluate_word_suppression(text_clean, "climb", synonyms=["climbs", "climbing", "climbed"])
        self.assertEqual(s2, 1, "Must pass when both keyword and all synonyms are avoided")

if __name__ == "__main__":
    unittest.main()

