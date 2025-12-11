"""
Scoring engine for entity matching.

This is where we actually decide if two records are the same entity or not.
We combine multiple signals:
- Traditional fuzzy string matching (levenshtein, jaro-winkler, etc)
- Embedding similarity (semantic matching)
- Optional LLM evaluation (for the hard cases)

The composite score blends all these together and we classify based on thresholds.

I spent way too long tuning the weights here. The defaults work okay for general
entity matching but you'll probably want to adjust per-dataset. Company names
need different weights than person names for example.

TODO: add LLM scoring for uncertain pairs - right now it's stubbed out
TODO: try adding a phonetic similarity score (soundex/metaphone distance)
TODO: experiment with learned weights instead of hand-tuned
FIXME(2025-01-14): address scoring seems broken for PO boxes, investigate
"""

import structlog
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from resolve_ai.config import MatchConfig
from resolve_ai.models import MatchClassification, PairScores, Record

log = structlog.get_logger()


class ScoringEngine:
    """
    Compute match scores for candidate pairs.

    Each pair gets multiple scores that capture different aspects of similarity.
    String-based methods catch surface-level matches, embeddings catch semantic
    matches (like "General Motors" vs "GM" that strings would miss).
    """

    def __init__(self, config: MatchConfig | None = None):
        self.config = config or MatchConfig()

    def compute_fuzzy_scores(self, text_a: str, text_b: str) -> dict[str, float]:
        """
        Compute traditional fuzzy matching scores.

        Using multiple methods because they each have strengths:
        - levenshtein: edit distance, good baseline
        - jaro-winkler: weights prefix matches higher, great for names where
          "Smith, John" vs "Smith, Jonathan" should score well
        - token_sort: ignores word order, handles "John Smith" vs "Smith John"
        - partial_ratio: finds best substring match, catches "Acme Corp" in
          "Acme Corporation of America"

        Args:
            text_a: First text to compare.
            text_b: Second text to compare.

        Returns:
            Dictionary of score names to values (0.0 to 1.0).
        """
        # empty strings get zeros across the board
        if not text_a or not text_b:
            return {
                "levenshtein_ratio": 0.0,
                "jaro_winkler": 0.0,
                "token_sort_ratio": 0.0,
                "partial_ratio": 0.0,
            }

        # rapidfuzz returns 0-100, normalize to 0-1 for consistency
        # jaro-winkler is already 0-1 from the JaroWinkler class
        return {
            "levenshtein_ratio": fuzz.ratio(text_a, text_b) / 100.0,
            # jaro-winkler is especially good for names - weights early chars higher
            "jaro_winkler": JaroWinkler.similarity(text_a, text_b),
            # handles word reordering - "first last" vs "last, first"
            "token_sort_ratio": fuzz.token_sort_ratio(text_a, text_b) / 100.0,
            # tried using just partial_ratio but it gave too many false positives
            # on short strings - "AI" matching "AI Corp" is fine but also matches
            # "RAIN Corp" which is not great. keeping it as one signal among many.
            "partial_ratio": fuzz.partial_ratio(text_a, text_b) / 100.0,
        }

    def compute_composite_score(
        self,
        fuzzy_scores: dict[str, float],
        embedding_similarity: float,
        llm_score: float | None = None,
    ) -> float:
        """
        Compute weighted composite score.

        This is where we blend all the different signals together. The weights
        are configurable but honestly finding the right values is more art than
        science. I've had decent results with ~0.3 fuzzy, ~0.7 embedding but
        it really depends on your data.

        When LLM scores are available they get heavy weight since they can
        understand context that string/embedding methods miss (like knowing
        "IBM" and "International Business Machines" are the same company).

        Args:
            fuzzy_scores: Dictionary of fuzzy match scores.
            embedding_similarity: Cosine similarity from embeddings.
            llm_score: Optional LLM evaluation score.

        Returns:
            Composite score between 0.0 and 1.0.
        """
        # take the best fuzzy score - different methods excel on different cases
        # so we let the best one "win" rather than averaging (which would dilute
        # a strong signal from one method)
        best_fuzzy = max(fuzzy_scores.values()) if fuzzy_scores else 0.0

        if llm_score is not None and self.config.llm_weight > 0:
            # full weighted average including LLM
            # not 100% sure these weights are optimal, might need tuning per-dataset
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
            # no LLM score - renormalize so fuzzy + embedding weights sum to 1
            # this keeps scores comparable whether or not LLM was used
            total = self.config.fuzzy_weight + self.config.embedding_weight
            if total == 0:
                return 0.0  # edge case, shouldn't happen with sensible config
            return (
                (self.config.fuzzy_weight / total) * best_fuzzy
                + (self.config.embedding_weight / total) * embedding_similarity
            )

    def classify(self, score: float) -> MatchClassification:
        """
        Classify pair based on composite score.

        Three buckets:
        - MATCH: high confidence, can auto-resolve
        - UNCERTAIN: needs human review (or LLM escalation)
        - NO_MATCH: definitely different entities

        The uncertain zone is intentionally wide to avoid false positives.
        In production you'd rather have humans review edge cases than
        silently merge things that shouldn't be merged.

        Args:
            score: Composite match score.

        Returns:
            Classification (MATCH, NO_MATCH, or UNCERTAIN).
        """
        # thresholds are configurable - these are just the decision boundaries
        # in practice you might want tighter thresholds for high-stakes data
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
        """
        Score a single candidate pair.

        This is the main entry point for scoring. Takes two records and all the
        available signals, computes fuzzy scores, blends everything together,
        and returns a classification.

        Args:
            record_a: First record.
            record_b: Second record.
            embedding_similarity: Cosine similarity between embeddings.
            llm_score: Optional LLM evaluation score.
            llm_reasoning: Optional LLM reasoning text.

        Returns:
            Complete pair scores with classification.
        """
        # use normalized names for string comparison
        # normalization should have already lowercased, removed punctuation, etc
        text_a = record_a.name_normalized or ""
        text_b = record_b.name_normalized or ""

        fuzzy_scores = self.compute_fuzzy_scores(text_a, text_b)

        composite = self.compute_composite_score(
            fuzzy_scores, embedding_similarity, llm_score
        )

        # canonical pair ordering - always smaller ID first
        # this ensures pair_id is consistent regardless of argument order
        # important for deduplication and lookup
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
        """
        Score multiple candidate pairs.

        Batch scoring interface - just loops over pairs and scores each one.
        Could potentially be parallelized but fuzzy matching is pretty fast
        and this hasn't been a bottleneck in practice.

        Args:
            pairs: List of (record_a, record_b, embedding_similarity) tuples.

        Returns:
            List of pair scores.
        """
        results = []
        for record_a, record_b, embedding_sim in pairs:
            score = self.score_pair(record_a, record_b, embedding_sim)
            results.append(score)

        # summary stats for monitoring - useful to see how the distribution
        # looks across match/uncertain/no_match buckets
        matches = sum(1 for s in results if s.classification == MatchClassification.MATCH)
        uncertain = sum(
            1 for s in results if s.classification == MatchClassification.UNCERTAIN
        )
        no_match = sum(
            1 for s in results if s.classification == MatchClassification.NO_MATCH
        )

        # if uncertain is really high, thresholds might need adjustment
        # if matches is really low, might have a data quality issue
        # also should probably add
        log.info(
            "scoring_complete",
            total=len(results),
            matches=matches,
            uncertain=uncertain,
            no_match=no_match,
        )

        return results
