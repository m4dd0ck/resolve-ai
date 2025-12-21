"""Scoring engine for entity matching.

FIXME: address scoring seems broken for PO boxes
"""

import structlog
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from resolve_ai.config import MatchConfig
from resolve_ai.models import MatchClassification, PairScores, Record

log = structlog.get_logger()


class ScoringEngine:
    """Compute match scores for candidate pairs."""

    def __init__(self, config: MatchConfig | None = None):
        self.config = config or MatchConfig()

    def compute_fuzzy_scores(self, text_a: str, text_b: str) -> dict[str, float]:
        """Compute traditional fuzzy matching scores.

        Args:
            text_a: First text to compare.
            text_b: Second text to compare.

        Returns:
            Dictionary of score names to values (0.0 to 1.0).
        """
        if not text_a or not text_b:
            return {
                "levenshtein_ratio": 0.0,
                "jaro_winkler": 0.0,
                "token_sort_ratio": 0.0,
                "partial_ratio": 0.0,
            }

        return {
            "levenshtein_ratio": fuzz.ratio(text_a, text_b) / 100.0,
            "jaro_winkler": JaroWinkler.similarity(text_a, text_b),
            "token_sort_ratio": fuzz.token_sort_ratio(text_a, text_b) / 100.0,
            "partial_ratio": fuzz.partial_ratio(text_a, text_b) / 100.0,
        }

    def compute_composite_score(
        self,
        fuzzy_scores: dict[str, float],
        embedding_similarity: float,
        llm_score: float | None = None,
    ) -> float:
        """Compute weighted composite score.

        Args:
            fuzzy_scores: Dictionary of fuzzy match scores.
            embedding_similarity: Cosine similarity from embeddings.
            llm_score: Optional LLM evaluation score.

        Returns:
            Composite score between 0.0 and 1.0.
        """
        best_fuzzy = max(fuzzy_scores.values()) if fuzzy_scores else 0.0

        if llm_score is not None and self.config.llm_weight > 0:
            total_weight = (
                self.config.fuzzy_weight
                + self.config.embedding_weight
                + self.config.llm_weight
            )
            return (
                self.config.fuzzy_weight * best_fuzzy
                + self.config.embedding_weight * embedding_similarity
                + self.config.llm_weight * llm_score
            ) / total_weight
        else:
            total = self.config.fuzzy_weight + self.config.embedding_weight
            if total == 0:
                return 0.0
            return (
                (self.config.fuzzy_weight / total) * best_fuzzy
                + (self.config.embedding_weight / total) * embedding_similarity
            )

    def classify(self, score: float) -> MatchClassification:
        """Classify pair based on composite score.

        Args:
            score: Composite match score.

        Returns:
            Classification (MATCH, NO_MATCH, or UNCERTAIN).
        """
        if score >= self.config.auto_match_threshold:
            return MatchClassification.MATCH
        elif score >= self.config.uncertain_threshold:
            return MatchClassification.UNCERTAIN
        else:
            return MatchClassification.NO_MATCH

    def score_pair(
        self,
        record_a: Record,
        record_b: Record,
        embedding_similarity: float,
        llm_score: float | None = None,
        llm_reasoning: str | None = None,
    ) -> PairScores:
        """Score a single candidate pair.

        Args:
            record_a: First record.
            record_b: Second record.
            embedding_similarity: Cosine similarity between embeddings.
            llm_score: Optional LLM evaluation score.
            llm_reasoning: Optional LLM reasoning text.

        Returns:
            Complete pair scores with classification.
        """
        text_a = record_a.name_normalized or ""
        text_b = record_b.name_normalized or ""

        fuzzy_scores = self.compute_fuzzy_scores(text_a, text_b)

        composite = self.compute_composite_score(
            fuzzy_scores, embedding_similarity, llm_score
        )

        pair_key = tuple(sorted([record_a.record_id, record_b.record_id]))

        return PairScores(
            pair_id=f"{pair_key[0]}_{pair_key[1]}",
            levenshtein_ratio=fuzzy_scores["levenshtein_ratio"],
            jaro_winkler=fuzzy_scores["jaro_winkler"],
            token_sort_ratio=fuzzy_scores["token_sort_ratio"],
            cosine_similarity=embedding_similarity,
            llm_score=llm_score,
            llm_reasoning=llm_reasoning,
            composite_score=composite,
            classification=self.classify(composite),
        )

    def score_pairs(
        self,
        pairs: list[tuple[Record, Record, float]],
    ) -> list[PairScores]:
        """Score multiple candidate pairs.

        Args:
            pairs: List of (record_a, record_b, embedding_similarity) tuples.

        Returns:
            List of pair scores.
        """
        results = []
        for record_a, record_b, embedding_sim in pairs:
            score = self.score_pair(record_a, record_b, embedding_sim)
            results.append(score)

        matches = sum(1 for s in results if s.classification == MatchClassification.MATCH)
        uncertain = sum(
            1 for s in results if s.classification == MatchClassification.UNCERTAIN
        )
        no_match = sum(
            1 for s in results if s.classification == MatchClassification.NO_MATCH
        )

        log.info(
            "scoring_complete",
            total=len(results),
            matches=matches,
            uncertain=uncertain,
            no_match=no_match,
        )

        return results
