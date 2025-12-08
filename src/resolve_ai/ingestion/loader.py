"""Data loading from various file formats.

this is the main entry point for getting data into the system. learned my lesson
at a previous job about separating loading from transformation.
"""

import csv
import json
import uuid
from pathlib import Path

import structlog

from resolve_ai.ingestion.normalizer import TextNormalizer
from resolve_ai.models import Record

log = structlog.get_logger()

# TODO: might want to add support for xlsx files at some point, but pandas is heavy
# and I'm trying to keep deps minimal for now


class DataLoader:
    """Load records from CSV, JSON, or Parquet files.

    not super fancy but gets the job done. the main thing is that it normalizes
    everything on the way in so we don't have to worry about it later.
    """

    def __init__(self, normalizer: TextNormalizer | None = None):
        # default normalizer is fine for most cases, but you can inject a custom
        # one if you need different behavior (like for international data)
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

        # simple dispatch based on extension - could use a registry pattern but
        # this is more readable and we only have 3 formats anyway
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
        # newline="" is important here or you get weird behavior on windows
        # took me a while to figure out why tests were failing on CI
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

        # handle both single objects and arrays - this is a bit loose but convenient
        # for testing with small files
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
        """Load records from Parquet using DuckDB.

        using duckdb instead of pyarrow because it's faster and handles more edge
        cases out of the box. plus the query syntax is nice if we ever need to
        add filtering.
        """
        # lazy import because duckdb is a big dependency and most users will
        # just be using csv anyway
        import duckdb

        # in-memory connection is fine here, we're just reading a file
        conn = duckdb.connect()
        # this is kinda hacky but duckdb auto-detects parquet files which is nice
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
        """Create a Record from a row of data.

        this is where the magic happens - we store both raw and normalized data
        so we can always trace back to the original if something looks weird.
        """
        # Generate or extract record ID
        # using first 8 chars of uuid is probably fine for most datasets
        # TODO: might want to use a hash of the row content instead to detect duplicates
        if id_field and id_field in row:
            record_id = str(row[id_field])
        else:
            record_id = str(uuid.uuid4())[:8]

        # Normalize name - empty string check avoids unnecessary work
        name_raw = row.get(name_field, "")
        name_normalized = self.normalizer.normalize_name(name_raw) if name_raw else None

        # Normalize address if provided
        # address is optional because some datasets only have names
        address_normalized = None
        if address_field and address_field in row:
            address_raw = row.get(address_field, "")
            address_normalized = (
                self.normalizer.normalize_address(address_raw) if address_raw else None
            )

        # keeping this around in case we need to debug normalization issues
        # print(f"DEBUG: {name_raw!r} -> {name_normalized!r}")

        return Record(
            record_id=record_id,
            source_file=source_file,
            source_row=source_row,
            raw_data=row,
            name_normalized=name_normalized,
            address_normalized=address_normalized,
        )
