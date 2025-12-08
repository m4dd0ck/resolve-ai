"""Configuration for entity resolution.

using pydantic-settings so you can override any of these via environment
variables prefixed with RESOLVE_ (e.g., RESOLVE_USE_LLM=true). super handy
for switching between dev and prod settings.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class MatchConfig(BaseSettings):
    """Configuration for a matching run."""

    # embedding settings
    # all-MiniLM-L6-v2 is a good balance of speed and quality for this use case
    # sentence-transformers has bigger models if you need more accuracy
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 64  # tune this based on your GPU memory
    embedding_dimension: int = 384  # must match the model output dim

    # candidate generation
    # blocking + ANN gives us the best of both worlds: blocking catches obvious
    # matches quickly, ANN finds semantically similar pairs that blocking misses
    ann_top_k: int = 10  # how many nearest neighbors to consider per record
    ann_threshold: float = 0.5  # minimum similarity to be a candidate
    use_blocking: bool = True
    blocking_key: str = "first_3_chars"  # TODO: add phonetic blocking (soundex/metaphone)

    # scoring weights (should sum to 1.0 when all are used)
    # fuzzy catches typos, embedding catches semantic similarity
    fuzzy_weight: float = 0.4
    embedding_weight: float = 0.6
    llm_weight: float = 0.0  # disabled by default, adds latency

    # thresholds
    # 0.90 is pretty conservative, might want to lower for messier data
    # tested on a few datasets and this gave good precision without too many false negatives
    auto_match_threshold: float = 0.90
    uncertain_threshold: float = 0.50  # anything between 0.5-0.9 needs human review

    # LLM settings - will add ollama integration later
    # the idea is to use LLM only for uncertain pairs to break ties
    use_llm: bool = False
    llm_model: str = "llama3.2:3b"  # small and fast, runs locally
    llm_base_url: str = "http://localhost:11434"  # ollama default
    llm_only_for_uncertain: bool = True  # don't waste LLM calls on obvious matches
    max_llm_calls_per_run: int = 100  # rate limit to avoid runaway costs

    # paths
    db_path: Path = Field(default=Path("resolve.db"))
    cache_dir: Path = Field(default=Path("data/cache"))  # for embedding cache

    model_config = {"env_prefix": "RESOLVE_"}


class Settings(BaseSettings):
    """Global application settings.

    These are separate from MatchConfig because they're more about operational
    concerns than the matching algorithm itself. Might want to add things like
    metrics endpoint, tracing config, etc. for production use.
    """

    log_level: str = "INFO"  # DEBUG is chatty but useful for troubleshooting
    log_format: str = "console"  # 'console' for dev, 'json' for production (easier to parse)
    show_progress: bool = True  # disable for batch jobs where you just want the final result

    model_config = {"env_prefix": "RESOLVE_"}
