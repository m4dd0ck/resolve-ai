# ResolveAI

Entity resolution system that combines fuzzy matching with semantic embeddings to identify duplicate records across messy real-world datasets.

## The Problem

You have company data from multiple sources:
```
Source A: "Apple Inc.", "1 Apple Park Way, Cupertino CA"
Source B: "Apple Incorporated", "One Apple Park Way, Cupertino, California"
Source C: "AAPL Corp", "Apple Park, Cupertino"
```

Are these the same company? Traditional exact matching says no. ResolveAI says yes.

## How It Works

The system uses a hybrid approach:

1. **Text Normalization** - Standardizes company suffixes (Inc → inc, Corporation → corp), addresses, state names
2. **Semantic Embeddings** - Uses sentence-transformers to capture meaning ("IBM" and "International Business Machines" have similar vectors)
3. **Fuzzy Matching** - Levenshtein, Jaro-Winkler, token sort ratio for character-level similarity
4. **Composite Scoring** - Weighted combination of embedding similarity (60%) and best fuzzy score (40%)
5. **Classification** - High scores auto-match, low scores auto-reject, middle scores flagged for review

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Ingest    │ --> │    Embed    │ --> │ Candidates  │ --> │    Score    │
│  CSV/JSON   │     │   FAISS     │     │  ANN+Block  │     │   Classify  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## Installation

```bash
# requires python 3.11+
git clone https://github.com/m4dd0ck/resolve-ai.git
cd resolve-ai
uv sync
```

## Quick Start

```bash
# run matching on a CSV file
uv run resolve match data/sample/companies.csv --name name --address address

# check results
uv run resolve report

# review uncertain matches interactively
uv run resolve review
```

## Example Output

```
$ uv run resolve report

     Entity Resolution Statistics
┏━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric          ┃ Value ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Total Records   │ 100   │
│ Candidate Pairs │ 614   │
│ Matches         │ 35    │
│ Uncertain       │ 563   │
│ No Match        │ 16    │
└─────────────────┴───────┘

                    Top Matches
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Record A                    ┃ Record B                     ┃ Score ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Lockheed Martin Corporation │ Lockheed Martin Corp         │ 1.000 │
│ Ford Motor Company          │ Ford Motor Co                │ 1.000 │
│ Meta Platforms Inc.         │ Meta Platforms Incorporated  │ 1.000 │
│ Netflix Inc.                │ Netflix Incorporated         │ 1.000 │
└─────────────────────────────┴──────────────────────────────┴───────┘
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `resolve ingest <file>` | Load data into the database |
| `resolve match [file]` | Run matching (on file or existing data) |
| `resolve report` | Show statistics and top matches |
| `resolve review` | Interactive review of uncertain pairs |
| `resolve stats` | Quick statistics summary |

## Configuration

Adjust matching behavior via environment variables or pass to `MatchConfig`:

```python
from resolve_ai.config import MatchConfig
from resolve_ai.pipeline import ResolutionPipeline

config = MatchConfig(
    auto_match_threshold=0.85,  # score above this = auto match
    uncertain_threshold=0.50,   # score below this = no match
    fuzzy_weight=0.4,           # weight for fuzzy scores
    embedding_weight=0.6,       # weight for embedding similarity
    ann_top_k=10,               # candidates per record from ANN
)

pipeline = ResolutionPipeline(config)
scores = pipeline.run("my_data.csv", name_field="company_name")
```

## Tech Stack

- **sentence-transformers** - all-MiniLM-L6-v2 for embeddings
- **FAISS** - approximate nearest neighbor search
- **rapidfuzz** - fast fuzzy string matching
- **DuckDB** - embedded analytics database
- **Typer + Rich** - CLI interface
- **Pydantic** - data validation

## Project Structure

```
src/resolve_ai/
├── ingestion/      # data loading and text normalization
├── embeddings/     # sentence-transformers + FAISS
├── matching/       # candidate generation and scoring
├── storage/        # DuckDB persistence
├── cli/            # command line interface
├── pipeline.py     # orchestrates the full flow
├── models.py       # pydantic data models
└── config.py       # configuration
```

## Future Work

- [ ] LLM integration for disambiguating uncertain pairs (ollama)
- [ ] Streamlit dashboard for bulk review
- [ ] Entity clustering (transitive closure of matches)
- [ ] Learned weights from review feedback

## Background

Built this after losing access to a similar internal tool at a previous job. That version got messy over time with too many special cases. This is a cleaner rewrite.

The hybrid fuzzy + embedding approach catches cases that either method alone would miss. Pure fuzzy matching fails on "IBM" vs "International Business Machines". Pure embeddings might incorrectly match "General Motors" with "General Electric" (both are large industrial companies semantically). Combining both signals gives better results.
