"""
Candidate pair generation using ANN search and blocking.

This is the core of the whole matching pipeline - we need to figure out which
record pairs are even worth comparing before we burn compute on scoring them all.

With N records you'd have N*(N-1)/2 possible pairs which blows up fast.
10k records = ~50 million pairs. Not gonna fly.

The trick is combining two approaches:
1. ANN (approximate nearest neighbors) on embeddings - fast semantic similarity
2. Blocking - group records by shared keys and only compare within blocks

Together these get us down to a manageable set of candidates without missing
too many true matches. The tradeoff is always recall vs speed.

TODO: might want to add phonetic blocking (soundex/metaphone) for names
TODO: experiment with locality-sensitive hashing as alternative to ANN
"""

import numpy as np
import structlog

from resolve_ai.config import MatchConfig
from resolve_ai.embeddings.store import EmbeddingStore
from resolve_ai.ingestion.normalizer import TextNormalizer
from resolve_ai.models import CandidatePair, Record

log = structlog.get_logger()


class CandidateGenerator:
    """
    Generate candidate pairs for matching using hybrid blocking + ANN.

    The whole point here is to avoid the O(n^2) comparison problem. We use
    embeddings + ANN to find semantically similar records fast, then blocking
    as a safety net to catch things the embeddings might miss.

    Learned the hard way that relying on just one method leaves gaps -
    ANN misses some obvious string matches, blocking misses semantic similarity.
    """

    def __init__(
        self,
        embedding_store: EmbeddingStore,
        config: MatchConfig | None = None,
        normalizer: TextNormalizer | None = None,
    ):
        self.embedding_store = embedding_store
        self.config = config or MatchConfig()
        self.normalizer = normalizer or TextNormalizer()

    def generate(
        self,
        records: list[Record],
        embeddings: dict[str, np.ndarray],
    ) -> list[CandidatePair]:
        """Generate candidate pairs from records.

        Args:
            records: List of records to find candidates for.
            embeddings: Dictionary mapping record_id to embedding.

        Returns:
            List of candidate pairs.
        """
        candidates = []
        # canonical pair ordering prevents (A,B) and (B,A) duplicates
        # learned this the hard way - was getting double the candidates before
        seen_pairs: set[tuple[str, str]] = set()

        log.info(
            "generating_candidates",
            record_count=len(records),
            top_k=self.config.ann_top_k,
            threshold=self.config.ann_threshold,
        )

        for record in records:
            query_emb = embeddings.get(record.record_id)
            if query_emb is None:
                # shouldn't happen if pipeline is working, but defensive
                continue

            # ANN search for similar records - this is the main workhorse
            # top_k is tunable but 50-100 seems to work well in practice
            # TODO: could make this adaptive based on embedding distribution
            neighbors = self.embedding_store.search(
                query_emb,
                k=self.config.ann_top_k,
                exclude_ids={record.record_id},  # don't match with yourself
            )

            for neighbor_id, similarity in neighbors:
                # filter out low-similarity results early
                # threshold around 0.6-0.7 seems to catch most true matches
                # without drowning in false positives
                if similarity < self.config.ann_threshold:
                    continue

                # canonical ordering - always put smaller ID first
                # this way we never get both (A,B) and (B,A) in our results
                pair_key = tuple(sorted([record.record_id, neighbor_id]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                candidates.append(
                    CandidatePair(
                        pair_id=f"{pair_key[0]}_{pair_key[1]}",
                        record_id_a=pair_key[0],
                        record_id_b=pair_key[1],
                        generation_method="ann",
                        embedding_similarity=similarity,
                    )
                )

        log.info("candidates_generated", count=len(candidates))
        return candidates

    def generate_with_blocking(
        self,
        records: list[Record],
        embeddings: dict[str, np.ndarray],
        blocking_method: str = "first_3_chars",
    ) -> list[CandidatePair]:
        """
        Generate candidate pairs using blocking + ANN.

        Blocking is basically just a performance optimization that groups records
        by shared characteristics. The idea is that true matches probably share
        some obvious features (same first few letters, same zip code, etc).

        We do ANN first then add blocking candidates as a safety net. This catches
        stuff that embeddings might miss - like "IBM" vs "International Business
        Machines" where the strings look nothing alike but blocking on industry
        or location might still group them.

        TODO: the first_3_chars blocking key is pretty basic, could use phonetic
        TODO: experiment with multiple blocking keys and union the results

        Args:
            records: List of records to find candidates for.
            embeddings: Dictionary mapping record_id to embedding.
            blocking_method: Method for generating blocking keys.

        Returns:
            List of candidate pairs.
        """
        # start with ANN candidates - usually gets most of the matches
        candidates = self.generate(records, embeddings)
        seen_pairs = {(c.record_id_a, c.record_id_b) for c in candidates}

        # group records into blocks - records in the same block are compared
        # this catches cases where names are similar strings but embeddings diverged
        # e.g. "Smith & Sons LLC" vs "Smith and Sons" might end up in same block
        blocks: dict[str, list[Record]] = {}
        for record in records:
            name = record.name_normalized or ""
            key = self.normalizer.get_blocking_key(name, blocking_method)
            if key:  # skip empty keys
                blocks.setdefault(key, []).append(record)

        # compare all pairs within each block
        # this is O(k^2) per block but blocks are usually small
        # if a block gets too big we might have a data quality issue
        block_candidates = 0
        for block_records in blocks.values():
            if len(block_records) < 2:
                continue  # need at least 2 records to make a pair

            # nested loop but only upper triangle (i < j) to avoid duplicates
            for i, record_a in enumerate(block_records):
                for record_b in block_records[i + 1 :]:
                    # same canonical ordering trick as above
                    pair_key = tuple(sorted([record_a.record_id, record_b.record_id]))
                    if pair_key in seen_pairs:
                        continue  # already found this pair via ANN
                    seen_pairs.add(pair_key)

                    # still need to check embedding similarity - blocking alone
                    # can produce garbage pairs (everything starting with "A")
                    emb_a = embeddings.get(record_a.record_id)
                    emb_b = embeddings.get(record_b.record_id)
                    if emb_a is not None and emb_b is not None:
                        # dot product works because embeddings are normalized
                        similarity = float(np.dot(emb_a, emb_b))
                        if similarity >= self.config.ann_threshold:
                            candidates.append(
                                CandidatePair(
                                    pair_id=f"{pair_key[0]}_{pair_key[1]}",
                                    record_id_a=pair_key[0],
                                    record_id_b=pair_key[1],
                                    generation_method="blocking",
                                    embedding_similarity=similarity,
                                )
                            )
                            block_candidates += 1

        log.info(
            "blocking_candidates_added",
            block_count=len(blocks),
            additional_candidates=block_candidates,
            total_candidates=len(candidates),
        )

        return candidates
