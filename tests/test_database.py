"""Tests for reading stored pair scores back from DuckDB."""

import duckdb
import pytest

from resolve_ai.models import CandidatePair, MatchClassification, PairScores, Record
from resolve_ai.storage.database import Database

# pair_scores as created by v0.1.x, with the since-dropped LLM columns in the middle.
OLD_PAIR_SCORES_SQL = """
CREATE TABLE pair_scores (
    pair_id TEXT PRIMARY KEY,
    levenshtein_ratio FLOAT,
    jaro_winkler FLOAT,
    token_sort_ratio FLOAT,
    cosine_similarity FLOAT,
    llm_score FLOAT,
    llm_reasoning TEXT,
    composite_score FLOAT NOT NULL,
    classification TEXT CHECK (classification IN ('match', 'no_match', 'uncertain')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _records(sample_records: list[Record]) -> tuple[Record, Record]:
    return sample_records[0], sample_records[1]


def _pair(record_a: Record, record_b: Record) -> CandidatePair:
    return CandidatePair(
        pair_id="1_2",
        record_id_a=record_a.record_id,
        record_id_b=record_b.record_id,
        generation_method="ann",
        embedding_similarity=0.8,
    )


def _score(classification: MatchClassification) -> PairScores:
    return PairScores(
        pair_id="1_2",
        levenshtein_ratio=0.7,
        jaro_winkler=0.85,
        token_sort_ratio=0.75,
        cosine_similarity=0.8,
        composite_score=0.66,
        classification=classification,
    )


def _assert_same_scores(got: PairScores, expected: PairScores) -> None:
    # Reason: FLOAT columns are 32-bit, so stored scores come back with rounding noise.
    expected_fields = expected.model_dump()
    for name, value in got.model_dump().items():
        if isinstance(value, float):
            assert value == pytest.approx(expected_fields[name], abs=1e-6), name
        else:
            assert value == expected_fields[name], name


def test_fresh_database_round_trips_scored_pairs(sample_records):
    record_a, record_b = _records(sample_records)
    db = Database()
    db.insert_records([record_a, record_b])
    db.insert_candidate_pairs([_pair(record_a, record_b)])
    db.insert_pair_scores([_score(MatchClassification.MATCH)])

    [(score, got_a, got_b)] = db.get_scores_by_classification(MatchClassification.MATCH)

    _assert_same_scores(score, _score(MatchClassification.MATCH))
    assert (got_a.record_id, got_a.raw_data) == (record_a.record_id, record_a.raw_data)
    assert (got_b.record_id, got_b.name_normalized) == (
        record_b.record_id,
        record_b.name_normalized,
    )


def test_old_schema_database_reads_back_by_column_name(sample_records, tmp_path):
    db_path = tmp_path / "old.db"
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(OLD_PAIR_SCORES_SQL)
        conn.execute(
            "INSERT INTO pair_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, DEFAULT)",
            ["1_2", 0.7, 0.85, 0.75, 0.8, 0.99, "old reasoning", 0.66, "uncertain"],
        )

    record_a, record_b = _records(sample_records)
    db = Database(db_path)
    db.insert_records([record_a, record_b])
    db.insert_candidate_pairs([_pair(record_a, record_b)])

    [(score, got_a, got_b)] = db.get_unreviewed_uncertain_pairs()

    _assert_same_scores(score, _score(MatchClassification.UNCERTAIN))
    assert got_a.record_id == record_a.record_id
    assert got_b.record_id == record_b.record_id
    db.close()


def test_old_schema_database_accepts_new_scores(sample_records, tmp_path):
    db_path = tmp_path / "old.db"
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(OLD_PAIR_SCORES_SQL)

    record_a, record_b = _records(sample_records)
    db = Database(db_path)
    db.insert_records([record_a, record_b])
    db.insert_candidate_pairs([_pair(record_a, record_b)])
    db.insert_pair_scores([_score(MatchClassification.MATCH)])

    [(score, _, _)] = db.get_scores_by_classification(MatchClassification.MATCH)

    _assert_same_scores(score, _score(MatchClassification.MATCH))
    db.close()
