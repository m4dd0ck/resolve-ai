"""Shared test fixtures."""

import pytest
from pathlib import Path

from resolve_ai.config import MatchConfig
from resolve_ai.models import Record
from resolve_ai.storage.database import Database


@pytest.fixture
def sample_records() -> list[Record]:
    """Sample company records for testing."""
    return [
        Record(
            record_id="1",
            source_file="test.csv",
            source_row=0,
            raw_data={"name": "Apple Inc.", "address": "Cupertino, CA"},
            name_normalized="apple inc",
            address_normalized="cupertino ca",
        ),
        Record(
            record_id="2",
            source_file="test.csv",
            source_row=1,
            raw_data={"name": "Apple Incorporated", "address": "Cupertino, California"},
            name_normalized="apple inc",
            address_normalized="cupertino ca",
        ),
        Record(
            record_id="3",
            source_file="test.csv",
            source_row=2,
            raw_data={"name": "Microsoft Corporation", "address": "Redmond, WA"},
            name_normalized="microsoft corp",
            address_normalized="redmond wa",
        ),
        Record(
            record_id="4",
            source_file="test.csv",
            source_row=3,
            raw_data={"name": "Microsft Corp", "address": "Redmond, Washington"},
            name_normalized="microsft corp",
            address_normalized="redmond wa",
        ),
        Record(
            record_id="5",
            source_file="test.csv",
            source_row=4,
            raw_data={"name": "Amazon.com Inc.", "address": "Seattle, WA"},
            name_normalized="amazon com inc",
            address_normalized="seattle wa",
        ),
    ]


@pytest.fixture
def match_config() -> MatchConfig:
    """Default match configuration for tests."""
    return MatchConfig(
        use_llm=False,
        ann_top_k=5,
        ann_threshold=0.3,
        auto_match_threshold=0.85,
        uncertain_threshold=0.50,
    )


@pytest.fixture
def temp_db(tmp_path) -> Database:
    """Temporary database for tests."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    yield db
    db.close()


@pytest.fixture
def sample_csv(tmp_path) -> Path:
    """Create a sample CSV file for testing."""
    csv_path = tmp_path / "companies.csv"
    csv_path.write_text(
        """id,name,address,industry
1,Apple Inc.,1 Apple Park Way Cupertino CA,Technology
2,Apple Incorporated,One Apple Park Way Cupertino California,Technology
3,Microsoft Corporation,One Microsoft Way Redmond WA,Technology
4,Microsft Corp,1 Microsoft Way Redmond Washington,Technology
5,Amazon.com Inc.,410 Terry Ave N Seattle WA,E-Commerce
"""
    )
    return csv_path
