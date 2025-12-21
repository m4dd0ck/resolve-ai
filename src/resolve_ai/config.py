"""Configuration for entity resolution."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class MatchConfig(BaseSettings):
    """Configuration for a matching run."""

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 64
    embedding_dimension: int = 384

    ann_top_k: int = 10
    ann_threshold: float = 0.5
    use_blocking: bool = True
    blocking_key: str = "first_3_chars"

    fuzzy_weight: float = 0.4
    embedding_weight: float = 0.6
    llm_weight: float = 0.0

    auto_match_threshold: float = 0.90
    uncertain_threshold: float = 0.50

    use_llm: bool = False
    llm_model: str = "llama3.2:3b"
    llm_base_url: str = "http://localhost:11434"
    llm_only_for_uncertain: bool = True
    max_llm_calls_per_run: int = 100

    db_path: Path = Field(default=Path("resolve.db"))
    cache_dir: Path = Field(default=Path("data/cache"))

    model_config = {"env_prefix": "RESOLVE_"}


class Settings(BaseSettings):
    """Global application settings."""

    log_level: str = "INFO"
    log_format: str = "console"
    show_progress: bool = True

    model_config = {"env_prefix": "RESOLVE_"}
