"""Data loading from various file formats."""

import csv
import json
import uuid
from pathlib import Path

import structlog

from resolve_ai.ingestion.normalizer import TextNormalizer
from resolve_ai.models import Record

log = structlog.get_logger()


class DataLoader:
    """Load records from CSV, JSON, or Parquet files."""

    def __init__(self, normalizer: TextNormalizer | None = None):
        self.normalizer = normalizer or TextNormalizer()

    def load(
        self,
        file_path: Path | str,
        name_field: str = "name",
        address_field: str | None = "address",
        id_field: str | None = None,
    ) -> list[Record]:
        """Load records from a file.

        Args:
            file_path: Path to the data file.
            name_field: Column name containing the entity name.
            address_field: Column name containing the address (optional).
            id_field: Column name for record ID (auto-generated if None).

        Returns:
            List of Record objects.
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            return self._load_csv(file_path, name_field, address_field, id_field)
        elif suffix == ".json":
            return self._load_json(file_path, name_field, address_field, id_field)
        elif suffix == ".parquet":
            return self._load_parquet(file_path, name_field, address_field, id_field)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _load_csv(
        self,
        file_path: Path,
        name_field: str,
        address_field: str | None,
        id_field: str | None,
    ) -> list[Record]:
        """Load records from CSV."""
        records = []
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader):
                record = self._create_record(
                    row,
                    source_file=str(file_path),
                    source_row=row_num,
                    name_field=name_field,
                    address_field=address_field,
                    id_field=id_field,
                )
                records.append(record)

        log.info("loaded_csv", file=str(file_path), record_count=len(records))
        return records

    def _load_json(
        self,
        file_path: Path,
        name_field: str,
        address_field: str | None,
        id_field: str | None,
    ) -> list[Record]:
        """Load records from JSON (expects array of objects)."""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        records = []
        for row_num, row in enumerate(data):
            record = self._create_record(
                row,
                source_file=str(file_path),
                source_row=row_num,
                name_field=name_field,
                address_field=address_field,
                id_field=id_field,
            )
            records.append(record)

        log.info("loaded_json", file=str(file_path), record_count=len(records))
        return records

    def _load_parquet(
        self,
        file_path: Path,
        name_field: str,
        address_field: str | None,
        id_field: str | None,
    ) -> list[Record]:
        """Load records from Parquet using DuckDB."""
        import duckdb

        conn = duckdb.connect()
        df = conn.execute(f"SELECT * FROM '{file_path}'").fetchdf()

        records = []
        for row_num, row in df.iterrows():
            row_dict = row.to_dict()
            record = self._create_record(
                row_dict,
                source_file=str(file_path),
                source_row=row_num,
                name_field=name_field,
                address_field=address_field,
                id_field=id_field,
            )
            records.append(record)

        log.info("loaded_parquet", file=str(file_path), record_count=len(records))
        return records

    def _create_record(
        self,
        row: dict,
        source_file: str,
        source_row: int,
        name_field: str,
        address_field: str | None,
        id_field: str | None,
    ) -> Record:
        """Create a Record from a row of data."""
        if id_field and id_field in row:
            record_id = str(row[id_field])
        else:
            record_id = str(uuid.uuid4())[:8]

        name_raw = row.get(name_field, "")
        name_normalized = self.normalizer.normalize_name(name_raw) if name_raw else None

        address_normalized = None
        if address_field and address_field in row:
            address_raw = row.get(address_field, "")
            address_normalized = (
                self.normalizer.normalize_address(address_raw) if address_raw else None
            )

        return Record(
            record_id=record_id,
            source_file=source_file,
            source_row=source_row,
            raw_data=row,
            name_normalized=name_normalized,
            address_normalized=address_normalized,
        )
