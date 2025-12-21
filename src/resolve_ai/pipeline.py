"""Entity resolution pipeline orchestrator."""

from pathlib import Path

import structlog

from resolve_ai.config import MatchConfig
from resolve_ai.embeddings.service import EmbeddingService
from resolve_ai.embeddings.store import EmbeddingStore
from resolve_ai.ingestion.loader import DataLoader
from resolve_ai.ingestion.normalizer import TextNormalizer
from resolve_ai.matching.candidates import CandidateGenerator
from resolve_ai.matching.scoring import ScoringEngine
from resolve_ai.models import MatchClassification, PairScores, Record
from resolve_ai.storage.database import Database

log = structlog.get_logger()


class ResolutionPipeline:
    """Orchestrate the entity resolution pipeline."""

    def __init__(
        self,
        config: MatchConfig | None = None,
        db_path: Path | str | None = None,
    ):
        """Initialize the pipeline.

        Args:
            config: Matching configuration.
            db_path: Path to database file.
        """
        self.config = config or MatchConfig()
        db_path = db_path or self.config.db_path

        self.db = Database(db_path)
        self.normalizer = TextNormalizer()
        self.loader = DataLoader(self.normalizer)
        self.embedding_service = EmbeddingService(self.config)
        self.embedding_store = EmbeddingStore(self.config.embedding_dimension)
        self.scoring_engine = ScoringEngine(self.config)

    def ingest(
        self,
        file_path: Path | str,
        name_field: str = "name",
        address_field: str | None = "address",
        id_field: str | None = None,
    ) -> list[Record]:
        """Ingest records from a file.

        Args:
            file_path: Path to data file.
            name_field: Column containing entity name.
            address_field: Column containing address (optional).
            id_field: Column containing record ID (optional).

        Returns:
            List of ingested records.
        """
        log.info("ingesting_file", path=str(file_path))

        records = self.loader.load(
            file_path,
            name_field=name_field,
            address_field=address_field,
            id_field=id_field,
        )

        self.db.insert_records(records)
        return records

    def generate_embeddings(
        self,
        records: list[Record],
        fields: list[str] | None = None,
    ) -> dict:
        """Generate embeddings for records.

        Args:
            records: List of records to embed.
            fields: Fields to use for embedding (default: ["name"]).

        Returns:
            Dictionary mapping record_id to embedding.
        """
        fields = fields or ["name"]
        log.info("generating_embeddings", record_count=len(records), fields=fields)

        embeddings = self.embedding_service.embed_records(records, fields)
        self.embedding_store.add_batch(embeddings)

        return embeddings

    def find_candidates(
        self,
        records: list[Record],
        embeddings: dict,
    ) -> list:
        """Find candidate pairs for matching.

        Args:
            records: List of records.
            embeddings: Dictionary of embeddings.

        Returns:
            List of candidate pairs.
        """
        generator = CandidateGenerator(
            self.embedding_store,
            self.config,
            self.normalizer,
        )

        if self.config.use_blocking:
            candidates = generator.generate_with_blocking(
                records, embeddings, self.config.blocking_key
            )
        else:
            candidates = generator.generate(records, embeddings)

        self.db.insert_candidate_pairs(candidates)
        return candidates

    def score_candidates(
        self,
        candidates: list,
        records: list[Record],
    ) -> list[PairScores]:
        """Score candidate pairs.

        Args:
            candidates: List of candidate pairs.
            records: List of records.

        Returns:
            List of pair scores.
        """
        record_map = {r.record_id: r for r in records}
        pairs_to_score = []

        for candidate in candidates:
            record_a = record_map.get(candidate.record_id_a)
            record_b = record_map.get(candidate.record_id_b)
            if record_a and record_b:
                pairs_to_score.append(
                    (record_a, record_b, candidate.embedding_similarity)
                )

        scores = self.scoring_engine.score_pairs(pairs_to_score)
        self.db.insert_pair_scores(scores)

        return scores

    def run(
        self,
        file_path: Path | str,
        name_field: str = "name",
        address_field: str | None = "address",
        id_field: str | None = None,
        match_fields: list[str] | None = None,
    ) -> list[PairScores]:
        """Run the full entity resolution pipeline.

        Args:
            file_path: Path to data file.
            name_field: Column containing entity name.
            address_field: Column containing address (optional).
            id_field: Column containing record ID (optional).
            match_fields: Fields to use for matching.

        Returns:
            List of pair scores.
        """
        log.info("starting_pipeline", file=str(file_path))

        records = self.ingest(
            file_path,
            name_field=name_field,
            address_field=address_field,
            id_field=id_field,
        )

        match_fields = match_fields or ["name"]
        embeddings = self.generate_embeddings(records, match_fields)
        candidates = self.find_candidates(records, embeddings)
        scores = self.score_candidates(candidates, records)

        stats = self.db.get_statistics()
        log.info(
            "pipeline_complete",
            records=stats["total_records"],
            candidates=stats["total_candidate_pairs"],
            matches=stats["matches"],
            uncertain=stats["uncertain"],
            no_match=stats["no_matches"],
        )

        return scores

    def run_on_loaded_records(
        self,
        match_fields: list[str] | None = None,
    ) -> list[PairScores]:
        """Run matching on already-loaded records.

        Args:
            match_fields: Fields to use for matching.

        Returns:
            List of pair scores.
        """
        records = self.db.get_all_records()
        if not records:
            log.warning("no_records_found")
            return []

        match_fields = match_fields or ["name"]
        embeddings = self.generate_embeddings(records, match_fields)
        candidates = self.find_candidates(records, embeddings)
        scores = self.score_candidates(candidates, records)

        return scores

    def get_matches(self) -> list[tuple[PairScores, Record, Record]]:
        """Get all matched pairs."""
        return self.db.get_scores_by_classification(MatchClassification.MATCH)

    def get_uncertain(self) -> list[tuple[PairScores, Record, Record]]:
        """Get all uncertain pairs."""
        return self.db.get_scores_by_classification(MatchClassification.UNCERTAIN)

    def get_statistics(self) -> dict:
        """Get pipeline statistics.

        Returns:
            Dictionary of statistics.
        """
        return self.db.get_statistics()

    def export_matches(self, output_path: Path, format: str = "csv") -> None:
        """Export matches to a file.

        Args:
            output_path: Path to output file.
            format: Output format ('csv' or 'json').
        """
        self.db.export_matches(output_path, format)
