"""Pydantic models for entity resolution."""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class MatchClassification(str, Enum):
    """Classification of a candidate pair."""

    MATCH = "match"
    NO_MATCH = "no_match"
    UNCERTAIN = "uncertain"


class Record(BaseModel):
    """A record ingested from a data source."""

    record_id: str
    source_file: str
    source_row: int
    raw_data: dict
    name_normalized: str | None = None
    address_normalized: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)


class CandidatePair(BaseModel):
    """A pair of records that are candidates for matching."""

    pair_id: str
    record_id_a: str
    record_id_b: str
    generation_method: str
    embedding_similarity: float


class PairScores(BaseModel):
    """Detailed scores for a candidate pair."""

    pair_id: str
    levenshtein_ratio: float = Field(ge=0.0, le=1.0)
    jaro_winkler: float = Field(ge=0.0, le=1.0)
    token_sort_ratio: float = Field(ge=0.0, le=1.0)
    cosine_similarity: float
    llm_score: float | None = Field(default=None, ge=0.0, le=1.0)
    llm_reasoning: str | None = None
    composite_score: float
    classification: MatchClassification

    @field_validator("cosine_similarity", "composite_score", mode="before")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """Clamp score to handle floating point precision."""
        return max(0.0, min(1.0, v))


class ReviewDecision(BaseModel):
    """Human review decision for a candidate pair."""

    pair_id: str
    decision: Literal["match", "no_match", "skip"]
    reviewer: str = "cli_user"
    notes: str | None = None
    reviewed_at: datetime = Field(default_factory=datetime.now)


class EntityCluster(BaseModel):
    """A cluster of records representing the same entity.

    TODO: implement transitive closure to build these from pairwise matches.
    """

    cluster_id: str
    record_ids: list[str]
    confidence: float


class LLMMatchResult(BaseModel):
    """Result from LLM match evaluation."""

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
