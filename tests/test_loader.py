"""Tests for file loading."""

import duckdb

from resolve_ai.ingestion.loader import DataLoader


class TestParquetLoading:
    """Parquet files are read through DuckDB without pandas."""

    def test_load_parquet_creates_records(self, tmp_path):
        parquet_path = tmp_path / "companies.parquet"
        duckdb.connect().execute(
            f"""
            COPY (
                SELECT * FROM (VALUES
                    ('1', 'Apple Inc.', '1 Apple Park Way, Cupertino, CA'),
                    ('2', 'Microsoft Corporation', 'One Microsoft Way, Redmond, WA')
                ) AS t(id, name, address)
            ) TO '{parquet_path}' (FORMAT PARQUET)
            """
        )

        records = DataLoader().load(
            parquet_path, name_field="name", address_field="address", id_field="id"
        )

        assert [r.record_id for r in records] == ["1", "2"]
        assert records[0].name_normalized == "apple inc"
        assert records[1].raw_data["address"] == "One Microsoft Way, Redmond, WA"
        assert records[1].source_row == 1
