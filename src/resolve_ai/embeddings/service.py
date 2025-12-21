"""Embedding generation service."""

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from resolve_ai.config import MatchConfig
from resolve_ai.models import Record

log = structlog.get_logger()


class EmbeddingService:
    """Generate embeddings for records using sentence-transformers."""

    def __init__(self, config: MatchConfig | None = None):
        self.config = config or MatchConfig()
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            log.info("loading_embedding_model", model=self.config.embedding_model)
            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model

    def prepare_text(self, record: Record, fields: list[str]) -> str:
        """Combine specified fields into single text for embedding.

        Args:
            record: The record to prepare text for.
            fields: List of field names to include.

        Returns:
            Combined text string.
        """
        parts = []

        if "name" in fields and record.name_normalized:
            parts.append(record.name_normalized)
        if "address" in fields and record.address_normalized:
            parts.append(record.address_normalized)

        for field in fields:
            if field not in ("name", "address"):
                value = record.raw_data.get(field, "")
                if value:
                    parts.append(str(value).strip())

        return " | ".join(parts) if parts else ""

    def embed_records(
        self,
        records: list[Record],
        fields: list[str],
        show_progress: bool = True,
    ) -> dict[str, np.ndarray]:
        """Generate embeddings for multiple records with batching.

        Args:
            records: List of records to embed.
            fields: List of field names to use for embedding text.
            show_progress: Whether to show progress bar.

        Returns:
            Dictionary mapping record_id to embedding array.
        """
        if not records:
            return {}

        texts = [self.prepare_text(r, fields) for r in records]

        log.info("generating_embeddings", record_count=len(records), fields=fields)

        embeddings = self.model.encode(
            texts,
            batch_size=self.config.embedding_batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # required for IndexFlatIP cosine similarity
        )

        return {r.record_id: emb for r, emb in zip(records, embeddings)}

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding array.
        """
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding[0]

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.config.embedding_dimension
