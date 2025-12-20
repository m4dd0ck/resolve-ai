"""Embedding generation service.

sentence-transformers makes this way easier than doing it manually - tried using
raw transformers at first and it was a nightmare of tokenization edge cases.
"""


import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from resolve_ai.config import MatchConfig
from resolve_ai.models import Record

log = structlog.get_logger()

# TODO: might want to try different embedding models - all-MiniLM-L6-v2 is fast
# but there might be better options for entity matching specifically. heard good
# things about e5-small-v2 for this kind of thing


class EmbeddingService:
    """Generate embeddings for records using sentence-transformers."""

    def __init__(self, config: MatchConfig | None = None):
        self.config = config or MatchConfig()
        # using lazy loading so we don't load the model until we actually need it
        # saves ~2s on startup which adds up when running tests
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            log.info("loading_embedding_model", model=self.config.embedding_model)
            # TODO: add GPU support if this gets slow - just need to pass device='cuda'
            # but need to handle the case where CUDA isn't available gracefully
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

        # prefer normalized fields when available - the normalization step
        # (lowercasing, removing punctuation, etc) helps embeddings be more consistent
        if "name" in fields and record.name_normalized:
            parts.append(record.name_normalized)
        if "address" in fields and record.address_normalized:
            parts.append(record.address_normalized)

        # fallback to raw data for fields we don't have special normalization for
        for field in fields:
            if field not in ("name", "address"):
                value = record.raw_data.get(field, "")
                if value:
                    parts.append(str(value).strip())

        # pipe separator seemed to work better than just spaces in my testing -
        # helps the model understand these are separate fields, not one blob of text
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

        # batch_size=64 seems to be the sweet spot on my machine, YMMV
        # too small and you lose parallelism, too big and you hit memory issues
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.embedding_batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            # normalize_embeddings=True is crucial - without this cosine similarity
            # doesn't work right when using IndexFlatIP (inner product). spent way
            # too long debugging why my similarities were all over the place before
            # figuring this out
            normalize_embeddings=True,
        )

        return {r.record_id: emb for r, emb in zip(records, embeddings)}

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding array.
        """
        # wrapping in list because encode expects iterable, then grab first result
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding[0]

    @property
    def dimension(self) -> int:
        """Get the embedding dimension.

        TODO: could probably infer this from the model itself instead of config,
        but this is simpler for now
        """
        return self.config.embedding_dimension
