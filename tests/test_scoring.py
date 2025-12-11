"""Tests for the scoring engine."""

import pytest

from resolve_ai.config import MatchConfig
from resolve_ai.matching.scoring import ScoringEngine
from resolve_ai.models import MatchClassification, Record


class TestFuzzyScoring:
    """Tests for fuzzy string matching."""

    def test_identical_strings_score_one(self):
        """Identical strings should have perfect fuzzy scores."""
        engine = ScoringEngine(MatchConfig())
        scores = engine.compute_fuzzy_scores("Apple Inc", "Apple Inc")

        assert scores["levenshtein_ratio"] >= 0.99
        assert scores["jaro_winkler"] >= 0.99
        assert scores["token_sort_ratio"] >= 0.99

    def test_similar_strings_score_high(self):
        """Similar strings should have high fuzzy scores."""
        engine = ScoringEngine(MatchConfig())
        scores = engine.compute_fuzzy_scores("Apple Inc", "Apple Incorporated")

        # These strings share "Apple Inc" prefix so should score reasonably
        assert scores["levenshtein_ratio"] > 0.5
        assert scores["token_sort_ratio"] > 0.5

    def test_dissimilar_strings_score_low(self):
        """Dissimilar strings should have low fuzzy scores."""
        engine = ScoringEngine(MatchConfig())
        scores = engine.compute_fuzzy_scores("Apple Inc", "Microsoft Corporation")

        # Very different strings should have low scores
        assert scores["levenshtein_ratio"] < 0.5

    def test_empty_strings_score_zero(self):
        """Empty strings should have zero scores."""
        engine = ScoringEngine(MatchConfig())
        scores = engine.compute_fuzzy_scores("", "Apple Inc")

        assert scores["levenshtein_ratio"] == 0.0
        assert scores["jaro_winkler"] == 0.0

    def test_typo_tolerance(self):
        """Should handle common typos reasonably."""
        engine = ScoringEngine(MatchConfig())
        scores = engine.compute_fuzzy_scores("Microsoft", "Microsft")

        # Typo with one missing character should still score well
        assert scores["levenshtein_ratio"] > 0.8


class TestCompositeScoring:
    """Tests for composite score calculation."""

    def test_high_scores_yield_high_composite(self):
        """All high component scores should yield high composite."""
        config = MatchConfig(fuzzy_weight=0.4, embedding_weight=0.6, llm_weight=0.0)
        engine = ScoringEngine(config)

        composite = engine.compute_composite_score(
            {"levenshtein_ratio": 0.95, "jaro_winkler": 0.97},
            embedding_similarity=0.92,
            llm_score=None,
        )

        assert composite > 0.9

    def test_mixed_scores_yield_moderate_composite(self):
        """Mixed component scores should yield moderate composite."""
        config = MatchConfig(fuzzy_weight=0.4, embedding_weight=0.6, llm_weight=0.0)
        engine = ScoringEngine(config)

        composite = engine.compute_composite_score(
            {"levenshtein_ratio": 0.5, "jaro_winkler": 0.6},
            embedding_similarity=0.7,
            llm_score=None,
        )

        assert 0.5 < composite < 0.8

    def test_low_scores_yield_low_composite(self):
        """Low component scores should yield low composite."""
        config = MatchConfig(fuzzy_weight=0.4, embedding_weight=0.6, llm_weight=0.0)
        engine = ScoringEngine(config)

        composite = engine.compute_composite_score(
            {"levenshtein_ratio": 0.2, "jaro_winkler": 0.3},
            embedding_similarity=0.25,
            llm_score=None,
        )

        assert composite < 0.4

    def test_weights_affect_composite(self):
        """Different weights should produce different composites."""
        fuzzy_scores = {"levenshtein_ratio": 0.9, "jaro_winkler": 0.9}
        embedding_sim = 0.5

        # Fuzzy-heavy config
        config1 = MatchConfig(fuzzy_weight=0.8, embedding_weight=0.2)
        engine1 = ScoringEngine(config1)
        composite1 = engine1.compute_composite_score(fuzzy_scores, embedding_sim, None)

        # Embedding-heavy config
        config2 = MatchConfig(fuzzy_weight=0.2, embedding_weight=0.8)
        engine2 = ScoringEngine(config2)
        composite2 = engine2.compute_composite_score(fuzzy_scores, embedding_sim, None)

        # Fuzzy-heavy should score higher when fuzzy is high
        assert composite1 > composite2


class TestClassification:
    """Tests for match classification."""

    def test_high_score_classified_as_match(self):
        """High scores should be classified as MATCH."""
        config = MatchConfig(auto_match_threshold=0.9, uncertain_threshold=0.5)
        engine = ScoringEngine(config)

        assert engine.classify(0.95) == MatchClassification.MATCH
        assert engine.classify(0.90) == MatchClassification.MATCH

    def test_medium_score_classified_as_uncertain(self):
        """Medium scores should be classified as UNCERTAIN."""
        config = MatchConfig(auto_match_threshold=0.9, uncertain_threshold=0.5)
        engine = ScoringEngine(config)

        assert engine.classify(0.75) == MatchClassification.UNCERTAIN
        assert engine.classify(0.50) == MatchClassification.UNCERTAIN

    def test_low_score_classified_as_no_match(self):
        """Low scores should be classified as NO_MATCH."""
        config = MatchConfig(auto_match_threshold=0.9, uncertain_threshold=0.5)
        engine = ScoringEngine(config)

        assert engine.classify(0.49) == MatchClassification.NO_MATCH
        assert engine.classify(0.10) == MatchClassification.NO_MATCH


class TestPairScoring:
    """Tests for full pair scoring."""

    def test_score_pair_returns_valid_scores(self, sample_records, match_config):
        """Scoring a pair should return valid PairScores."""
        engine = ScoringEngine(match_config)

        record_a = sample_records[0]  # Apple Inc.
        record_b = sample_records[1]  # Apple Incorporated

        scores = engine.score_pair(record_a, record_b, embedding_similarity=0.9)

        assert 0.0 <= scores.levenshtein_ratio <= 1.0
        assert 0.0 <= scores.jaro_winkler <= 1.0
        assert 0.0 <= scores.composite_score <= 1.0
        assert scores.classification in MatchClassification

    def test_similar_records_score_high(self, sample_records, match_config):
        """Similar records should have high scores."""
        engine = ScoringEngine(match_config)

        record_a = sample_records[0]  # Apple Inc.
        record_b = sample_records[1]  # Apple Incorporated

        scores = engine.score_pair(record_a, record_b, embedding_similarity=0.95)

        assert scores.composite_score > 0.8

    def test_dissimilar_records_score_low(self, sample_records, match_config):
        """Dissimilar records should have low scores."""
        engine = ScoringEngine(match_config)

        record_a = sample_records[0]  # Apple Inc.
        record_b = sample_records[4]  # Amazon.com Inc.

        scores = engine.score_pair(record_a, record_b, embedding_similarity=0.3)

        assert scores.composite_score < 0.5


class TestScorePairs:
    """Tests for batch pair scoring."""

    def test_score_pairs_returns_correct_count(self, sample_records, match_config):
        """Scoring multiple pairs should return correct number of results."""
        engine = ScoringEngine(match_config)

        pairs = [
            (sample_records[0], sample_records[1], 0.9),
            (sample_records[2], sample_records[3], 0.85),
        ]

        results = engine.score_pairs(pairs)

        assert len(results) == 2

    def test_score_pairs_classifies_correctly(self, sample_records, match_config):
        """Batch scoring should classify pairs correctly."""
        engine = ScoringEngine(match_config)

        pairs = [
            (sample_records[0], sample_records[1], 0.95),  # High similarity
            (sample_records[0], sample_records[4], 0.3),  # Low similarity
        ]

        results = engine.score_pairs(pairs)

        # First pair should be match or uncertain (high sim)
        assert results[0].classification in (
            MatchClassification.MATCH,
            MatchClassification.UNCERTAIN,
        )
        # Second pair should be no_match or uncertain (low sim)
        assert results[1].classification in (
            MatchClassification.NO_MATCH,
            MatchClassification.UNCERTAIN,
        )
