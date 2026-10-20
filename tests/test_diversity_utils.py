import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.diversity_utils import (  # noqa: E402
    batch_diversity_overlap,
    distinct_n,
    duplicate_stats,
    ngram_set,
)


class TestBatchDiversityOverlap(unittest.TestCase):
    """`batch_diversity_overlap` backs the SCST diversity penalty in
    `modules/modeling.py::_compute_scst_caption_loss` (--scst_diversity_weight).
    These are pure-Python checks that don't need torch, so they can run on a
    machine with no GPU/deep-learning stack installed.
    """

    def test_no_other_captions_returns_zero(self):
        cand = "a man is talking".split()
        self.assertEqual(batch_diversity_overlap(cand, []), 0.0)

    def test_identical_caption_gives_max_overlap(self):
        cand = "a man is talking to the camera".split()
        other = ["a man is talking to the camera".split()]
        self.assertAlmostEqual(batch_diversity_overlap(cand, other), 1.0)

    def test_completely_disjoint_captions_give_zero(self):
        cand = "a man is talking".split()
        other = ["cats sleep quietly outside".split()]  # no shared words at all
        self.assertEqual(batch_diversity_overlap(cand, other), 0.0)

    def test_partial_overlap_is_between_zero_and_one(self):
        cand = "a man is talking about basketball".split()
        other = ["a man is talking about the brain".split()]
        score = batch_diversity_overlap(cand, other)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_uses_the_closest_other_caption_not_the_average(self):
        # One near-duplicate among otherwise unrelated captions should still
        # trigger a high penalty -- mode collapse only needs ONE match
        # elsewhere in the batch, diluting by averaging would hide it.
        cand = "a man is talking to the camera".split()
        others = [
            "a completely unrelated caption here".split(),
            "another unrelated one entirely".split(),
            "a man is talking to the camera".split(),  # near-duplicate
        ]
        self.assertAlmostEqual(batch_diversity_overlap(cand, others), 1.0)

    def test_empty_candidate_tokens_returns_zero(self):
        self.assertEqual(batch_diversity_overlap([], [["a", "b"]]), 0.0)


class TestNgramSetAndDistinctN(unittest.TestCase):
    def test_ngram_set_short_sequence(self):
        self.assertEqual(ngram_set(["a"], 2), set())

    def test_distinct_n_all_unique(self):
        captions = ["a b c", "d e f"]
        self.assertEqual(distinct_n(captions, 1), 1.0)

    def test_duplicate_stats_counts_exact_repeats(self):
        captions = ["a man is talking"] * 3 + ["a woman is cooking"]
        stats = duplicate_stats(captions, top_k=1)
        self.assertEqual(stats["total"], 4)
        self.assertEqual(stats["unique"], 2)
        self.assertAlmostEqual(stats["duplicate_ratio"], 0.5)
        self.assertEqual(stats["top_templates"][0]["caption"], "a man is talking")
        self.assertEqual(stats["top_templates"][0]["count"], 3)


if __name__ == "__main__":
    unittest.main()
