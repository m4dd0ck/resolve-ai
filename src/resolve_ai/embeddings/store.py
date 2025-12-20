"""FAISS-based embedding storage and search.

FAISS is kinda finicky about dtypes and array shapes but once you get it working
it's really fast. tried chromadb first but it was overkill for this use case -
we just need fast nearest neighbor search, not a whole vector database.

TODO: benchmark against chromadb and hnswlib at some point, curious about the
tradeoffs at different dataset sizes
"""

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
        # importing faiss here instead of top-level because it can be slow to import
        # and not everyone will use this class
        import faiss

        self.dimension = dimension
        # need both directions of mapping - FAISS uses integer indices internally
        # but we want to work with string record IDs
        self.id_map: list[str] = []  # Maps FAISS idx to record_id
        self.reverse_map: dict[str, int] = {}  # Maps record_id to FAISS idx
        self._index: faiss.Index | None = None
        # keeping a copy of embeddings so we can retrieve them later
        # FAISS doesn't let you get vectors back out of an IndexFlatIP easily
        self._embeddings: list[np.ndarray] = []

    @property
    def index(self):
        """Get or create the FAISS index."""
        import faiss

        if self._index is None:
            # IndexFlatIP = inner product, which is equivalent to cosine similarity
            # when vectors are normalized (which they are from our embedding service).
            # this took forever to figure out - if you use IndexFlatL2 with normalized
            # vectors your results are basically backwards
            self._index = faiss.IndexFlatIP(self.dimension)
        return self._index

    def add(self, record_id: str, embedding: np.ndarray) -> None:
        """Add an embedding to the store.

        Args:
            record_id: The record ID.
            embedding: The embedding vector.
        """
        if record_id in self.reverse_map:
            return  # already added, skip silently

        # FAISS is picky about dtypes - everything needs to be float32
        # and shaped as (n, dimension) even for single vectors
        embedding = embedding.astype(np.float32).reshape(1, -1)
        self.index.add(embedding)
        self._embeddings.append(embedding.flatten())
        self.reverse_map[record_id] = len(self.id_map)
        self.id_map.append(record_id)

    def add_batch(self, embeddings: dict[str, np.ndarray]) -> None:
        """Add multiple embeddings at once.

        This is way faster than adding one at a time - FAISS can vectorize
        the operations internally. saw like 10x speedup on larger batches.

        Args:
            embeddings: Dictionary mapping record_id to embedding.
        """
        if not embeddings:
            return

        # filter out duplicates - don't want to add the same record twice
        new_embeddings = {
            rid: emb for rid, emb in embeddings.items() if rid not in self.reverse_map
        }

        if not new_embeddings:
            return

        # prepare batch - need to maintain order between IDs and vectors
        record_ids = list(new_embeddings.keys())
        emb_array = np.array(
            [new_embeddings[rid] for rid in record_ids], dtype=np.float32
        )

        # add to index - this is the fast part
        self.index.add(emb_array)

        # update mappings - have to do this sequentially unfortunately
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

        # fetch extra results to account for exclusions - otherwise we might not
        # get k results back after filtering. the +1 is probably overkill but eh
        search_k = min(k + len(exclude_ids) + 1, self.index.ntotal)
        distances, indices = self.index.search(query, search_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            # FAISS returns -1 for empty slots (shouldn't happen with our setup but just in case)
            if idx == -1:
                continue

            record_id = self.id_map[idx]
            if record_id in exclude_ids:
                continue

            # spent way too long figuring out why similarities were > 1.0 sometimes,
            # turns out floating point precision is fun. they're close enough to 1.0
            # that it doesn't matter for ranking though
            results.append((record_id, float(dist)))

            if len(results) >= k:
                break

        return results

    def get_embedding(self, record_id: str) -> np.ndarray | None:
        """Get the embedding for a record.

        This is why we keep _embeddings around - FAISS IndexFlatIP doesn't have a
        way to reconstruct vectors (unlike IndexFlatL2 which does, weirdly).

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

        Saves two files: the FAISS index itself and a .meta file with our ID mappings.
        Kind of annoying but FAISS doesn't have a way to store metadata with the index.

        Args:
            path: Path to save the index (the .meta file will be at path.meta).
        """
        import pickle
        from pathlib import Path

        import faiss

        index_path = Path(path)
        faiss.write_index(self.index, str(index_path))

        # have to save our mappings separately - FAISS only knows about integer indices
        # TODO: might want to use json instead of pickle for the meta file, would be
        # easier to debug but pickle handles numpy arrays better
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

        # load our ID mappings back
        meta_path = index_path.with_suffix(".meta")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
            self.id_map = meta["id_map"]
            self.reverse_map = meta["reverse_map"]
            self._embeddings = meta["embeddings"]

        log.info("loaded_embedding_store", path=str(path), size=self.size)
