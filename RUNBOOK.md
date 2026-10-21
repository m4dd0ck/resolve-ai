# Runbook

Procedures for resolve-ai. `README.md` describes what each command does; this file says when
to run it and in what order.

## On open

```bash
git status --short
git log -3 --oneline
```

## On close

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
```

## Match a file

1. `uv run resolve match <file> --name <column> --address <column>`; the first run downloads the embedding model.
2. `uv run resolve report` for the statistics and top matches; `--format csv --output <path>` to keep them.
3. `uv run resolve review` to decide the uncertain pairs by hand.
4. `uv run resolve stats` for the quick counts. Delete the `.db` file to start over.

## Release

1. Bump `version` in `pyproject.toml` and `__version__` in `src/resolve_ai/__init__.py` to the same value; `uv lock`.
2. Commit `chore: Release vX.Y.Z`, then `git tag -a vX.Y.Z -m "vX.Y.Z"` and `git push origin main --tags`.
3. `gh release create vX.Y.Z --title "vX.Y.Z" --notes "<what changed>"` with the compare link to the previous tag, marked latest.
4. CI must be green on the tagged commit. Nothing pins this package.

## Restore on a new machine

1. Clone `https://github.com/m4dd0ck/resolve-ai.git`; `uv sync`.
2. Run `uv run pytest` once with network access so the embedding model lands in the cache. No secrets, no data beyond `data/sample/`.

## Validate

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
uv run resolve match data/sample/companies.csv --name name --address address --db /tmp/resolve-check.db
```
