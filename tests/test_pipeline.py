"""Tests for the entity resolution pipeline."""

import pytest
from pathlib import Path

from resolve_ai.config import MatchConfig
from resolve_ai.models import MatchClassification
from resolve_ai.pipeline import ResolutionPipeline


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    def test_pipeline_loads_records(self, sample_csv, tmp_path):
        """Pipeline should load records from CSV."""
        config = MatchConfig(use_llm=False, ann_threshold=0.3)
        db_path = tmp_path / "test.db"
        pipeline = ResolutionPipeline(config, db_path)

        records = pipeline.ingest(sample_csv, name_field="name", address_field="address")

        assert len(records) == 5
        assert records[0].name_normalized is not None

    def test_pipeline_generates_embeddings(self, sample_csv, tmp_path):
        """Pipeline should generate embeddings for records."""
        config = MatchConfig(use_llm=False, ann_threshold=0.3)
        db_path = tmp_path / "test.db"
        pipeline = ResolutionPipeline(config, db_path)

        records = pipeline.ingest(sample_csv, name_field="name")
        embeddings = pipeline.generate_embeddings(records, ["name"])

        assert len(embeddings) == len(records)
        for record_id, embedding in embeddings.items():
            assert embedding.shape[0] == 384  # Default embedding dimension

    def test_pipeline_finds_candidates(self, sample_csv, tmp_path):
        """Pipeline should find candidate pairs."""
        config = MatchConfig(use_llm=False, ann_threshold=0.3, ann_top_k=5)
        db_path = tmp_path / "test.db"
        pipeline = ResolutionPipeline(config, db_path)

        records = pipeline.ingest(sample_csv, name_field="name")
        embeddings = pipeline.generate_embeddings(records, ["name"])
        candidates = pipeline.find_candidates(records, embeddings)

        # Should find some candidates (Apple/Apple, Microsoft/Microsft pairs)
        assert len(candidates) > 0

    def test_full_pipeline_run(self, sample_csv, tmp_path):
        """Full pipeline should run and produce scores."""
        config = MatchConfig(
            use_llm=False,
            ann_threshold=0.3,
            ann_top_k=5,
            auto_match_threshold=0.85,
        )
        db_path = tmp_path / "test.db"
        pipeline = ResolutionPipeline(config, db_path)

        scores = pipeline.run(
            sample_csv,
            name_field="name",
            address_field="address",
            match_fields=["name"],
        )

        assert len(scores) > 0

        # Check statistics
        stats = pipeline.get_statistics()
        assert stats["total_records"] == 5
        assert stats["total_candidate_pairs"] > 0

    def test_pipeline_identifies_duplicates(self, sample_csv, tmp_path):
        """Pipeline should identify obvious duplicates."""
        config = MatchConfig(
            use_llm=False,
            ann_threshold=0.3,
            ann_top_k=5,
            auto_match_threshold=0.80,
            uncertain_threshold=0.50,
        )
        db_path = tmp_path / "test.db"
        pipeline = ResolutionPipeline(config, db_path)

        scores = pipeline.run(
            sample_csv,
            name_field="name",
            match_fields=["name"],
        )

        # Should find at least some matches or uncertain pairs
        high_scores = [s for s in scores if s.composite_score > 0.6]
        assert len(high_scores) > 0

    def test_pipeline_returns_matches(self, sample_csv, tmp_path):
        """Pipeline get_matches should return matched pairs."""
        config = MatchConfig(
            use_llm=False,
            ann_threshold=0.3,
            auto_match_threshold=0.75,
        )
        db_path = tmp_path / "test.db"
        pipeline = ResolutionPipeline(config, db_path)

        pipeline.run(sample_csv, name_field="name", match_fields=["name"])
        matches = pipeline.get_matches()

        # Each match should be a tuple of (scores, record_a, record_b)
        for score, record_a, record_b in matches:
            assert score.classification == MatchClassification.MATCH
            assert record_a.record_id != record_b.record_id


class TestPipelineEdgeCases:
    """Edge case tests for the pipeline."""

    def test_empty_file_handling(self, tmp_path):
        """Pipeline should handle empty files gracefully."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("id,name,address\n")

        config = MatchConfig(use_llm=False)
        db_path = tmp_path / "test.db"
        pipeline = ResolutionPipeline(config, db_path)

        records = pipeline.ingest(empty_csv, name_field="name")
        assert len(records) == 0

    def test_single_record_handling(self, tmp_path):
        """Pipeline should handle single-record files."""
        single_csv = tmp_path / "single.csv"
        single_csv.write_text("id,name,address\n1,Apple Inc.,Cupertino CA\n")

        config = MatchConfig(use_llm=False)
        db_path = tmp_path / "test.db"
        pipeline = ResolutionPipeline(config, db_path)

        scores = pipeline.run(single_csv, name_field="name", match_fields=["name"])

        # Single record can't match with itself
        assert len(scores) == 0

    def test_missing_field_handling(self, tmp_path):
        """Pipeline should handle missing optional fields."""
        csv_no_address = tmp_path / "no_address.csv"
        csv_no_address.write_text("id,name\n1,Apple Inc.\n2,Apple Incorporated\n")

        config = MatchConfig(use_llm=False, ann_threshold=0.3)
        db_path = tmp_path / "test.db"
        pipeline = ResolutionPipeline(config, db_path)

        records = pipeline.ingest(csv_no_address, name_field="name", address_field=None)

        assert len(records) == 2
        assert records[0].address_normalized is None
