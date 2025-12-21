"""Candidate pair generation using ANN search and blocking."""

import numpy as np
import structlog

from resolve_ai.config import MatchConfig
from resolve_ai.embeddings.store import EmbeddingStore
from resolve_ai.ingestion.normalizer import TextNormalizer
from resolve_ai.models import CandidatePair, Record

log = structlog.get_logger()


class CandidateGenerator:
    """Generate candidate pairs for matching using hybrid blocking + ANN."""

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
                continue

            neighbors = self.embedding_store.search(
                query_emb,
                k=self.config.ann_top_k,
                exclude_ids={record.record_id},
            )

            for neighbor_id, similarity in neighbors:
                if similarity < self.config.ann_threshold:
                    continue

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
        """Generate candidate pairs using blocking + ANN.

        Args:
            records: List of records to find candidates for.
            embeddings: Dictionary mapping record_id to embedding.
            blocking_method: Method for generating blocking keys.

        Returns:
            List of candidate pairs.
        """
        candidates = self.generate(records, embeddings)
        seen_pairs = {(c.record_id_a, c.record_id_b) for c in candidates}

        blocks: dict[str, list[Record]] = {}
        for record in records:
            name = record.name_normalized or ""
            key = self.normalizer.get_blocking_key(name, blocking_method)
            if key:
                blocks.setdefault(key, []).append(record)

        block_candidates = 0
        for block_records in blocks.values():
            if len(block_records) < 2:
                continue

            for i, record_a in enumerate(block_records):
                for record_b in block_records[i + 1 :]:
                    pair_key = tuple(sorted([record_a.record_id, record_b.record_id]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    emb_a = embeddings.get(record_a.record_id)
                    emb_b = embeddings.get(record_b.record_id)
                    if emb_a is not None and emb_b is not None:
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
