"""FAISS-based embedding storage and search."""

import numpy as np
import structlog

log = structlog.get_logger()


class EmbeddingStore:
    """Store and search embeddings using FAISS."""

    def __init__(self, dimension: int = 384):
        """Initialize the embedding store.

        Args:
            dimension: Embedding vector dimension.
        """
        import faiss

        self.dimension = dimension
        self.id_map: list[str] = []
        self.reverse_map: dict[str, int] = {}
        self._index: faiss.Index | None = None
        self._embeddings: list[np.ndarray] = []

    @property
    def index(self):
        """Get or create the FAISS index."""
        import faiss

        if self._index is None:
            # IndexFlatIP for cosine similarity with normalized vectors
            self._index = faiss.IndexFlatIP(self.dimension)
        return self._index

    def add(self, record_id: str, embedding: np.ndarray) -> None:
        """Add an embedding to the store.

        Args:
            record_id: The record ID.
            embedding: The embedding vector.
        """
        if record_id in self.reverse_map:
            return

        embedding = embedding.astype(np.float32).reshape(1, -1)
        self.index.add(embedding)
        self._embeddings.append(embedding.flatten())
        self.reverse_map[record_id] = len(self.id_map)
        self.id_map.append(record_id)

    def add_batch(self, embeddings: dict[str, np.ndarray]) -> None:
        """Add multiple embeddings at once.

        Args:
            embeddings: Dictionary mapping record_id to embedding.
        """
        if not embeddings:
            return

        new_embeddings = {
            rid: emb for rid, emb in embeddings.items() if rid not in self.reverse_map
        }

        if not new_embeddings:
            return

        record_ids = list(new_embeddings.keys())
        emb_array = np.array(
            [new_embeddings[rid] for rid in record_ids], dtype=np.float32
        )

        self.index.add(emb_array)

        for i, record_id in enumerate(record_ids):
            self._embeddings.append(emb_array[i])
            self.reverse_map[record_id] = len(self.id_map)
            self.id_map.append(record_id)

        log.info("added_embeddings", count=len(record_ids))

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        exclude_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Search for similar embeddings.

        Args:
            query: Query embedding vector.
            k: Number of results to return.
            exclude_ids: Record IDs to exclude from results.

        Returns:
            List of (record_id, similarity) tuples, sorted by similarity descending.
        """
        if self.index.ntotal == 0:
            return []

        exclude_ids = exclude_ids or set()
        query = query.astype(np.float32).reshape(1, -1)

        search_k = min(k + len(exclude_ids) + 1, self.index.ntotal)
        distances, indices = self.index.search(query, search_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue

            record_id = self.id_map[idx]
            if record_id in exclude_ids:
                continue

            results.append((record_id, float(dist)))

            if len(results) >= k:
                break

        return results

    def get_embedding(self, record_id: str) -> np.ndarray | None:
        """Get the embedding for a record.

        Args:
            record_id: The record ID.

        Returns:
            The embedding vector or None if not found.
        """
        idx = self.reverse_map.get(record_id)
        if idx is None:
            return None
        return self._embeddings[idx]

    @property
    def size(self) -> int:
        """Get the number of embeddings in the store."""
        return len(self.id_map)

    def save(self, path: str) -> None:
        """Save the index to disk.

        Args:
            path: Path to save the index (the .meta file will be at path.meta).
        """
        import pickle
        from pathlib import Path

        import faiss

        index_path = Path(path)
        faiss.write_index(self.index, str(index_path))

        meta_path = index_path.with_suffix(".meta")
        with open(meta_path, "wb") as f:
            pickle.dump(
                {
                    "id_map": self.id_map,
                    "reverse_map": self.reverse_map,
                    "embeddings": self._embeddings,
                },
                f,
            )

        log.info("saved_embedding_store", path=str(path), size=self.size)

    def load(self, path: str) -> None:
        """Load the index from disk.

        Args:
            path: Path to the saved index.
        """
        import pickle
        from pathlib import Path

        import faiss

        index_path = Path(path)
        self._index = faiss.read_index(str(index_path))

        meta_path = index_path.with_suffix(".meta")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
            self.id_map = meta["id_map"]
            self.reverse_map = meta["reverse_map"]
            self._embeddings = meta["embeddings"]

        log.info("loaded_embedding_store", path=str(path), size=self.size)
