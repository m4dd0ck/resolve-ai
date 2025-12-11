"""CLI interface for ResolveAI entity resolution.

typer + rich is chef's kiss for CLIs - you get argument parsing, help text,
and beautiful terminal output with almost no boilerplate. the progress spinners
and tables make this feel like a real tool.

TODO: add progress bars for long-running jobs - right now we just show a spinner
which isn't super helpful when processing 100k records
"""

from pathlib import Path

import structlog
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from resolve_ai.config import MatchConfig
from resolve_ai.models import MatchClassification, ReviewDecision
from resolve_ai.pipeline import ResolutionPipeline
from resolve_ai.storage.database import Database

# configure structlog for pretty console output during CLI use
# in production you'd probably want JSON format for log aggregation
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
)

log = structlog.get_logger()
console = Console()

# no_args_is_help=True shows help when user runs `resolve` with no subcommand
# way better UX than a cryptic error message
app = typer.Typer(
    name="resolve",
    help="ResolveAI: Entity Resolution with Hybrid Matching",
    no_args_is_help=True,
)


@app.command()
def ingest(
    file_path: Path = typer.Argument(..., help="Path to CSV, JSON, or Parquet file"),
    name_field: str = typer.Option("name", "--name", "-n", help="Column for entity name"),
    address_field: str = typer.Option(
        None, "--address", "-a", help="Column for address (optional)"
    ),
    id_field: str = typer.Option(None, "--id", "-i", help="Column for record ID (optional)"),
    db_path: Path = typer.Option(
        Path("resolve.db"), "--db", "-d", help="Database file path"
    ),
) -> None:
    """Ingest data from a file into the database."""
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)

    config = MatchConfig(db_path=db_path)
    pipeline = ResolutionPipeline(config, db_path)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Loading and normalizing records...", total=None)
        records = pipeline.ingest(
            file_path,
            name_field=name_field,
            address_field=address_field,
            id_field=id_field,
        )

    console.print(f"[green]Ingested {len(records)} records from {file_path}[/green]")


@app.command()
def match(
    file_path: Path = typer.Argument(None, help="Path to data file (optional if already ingested)"),
    name_field: str = typer.Option("name", "--name", "-n", help="Column for entity name"),
    address_field: str = typer.Option(
        None, "--address", "-a", help="Column for address (optional)"
    ),
    id_field: str = typer.Option(None, "--id", "-i", help="Column for record ID (optional)"),
    fields: str = typer.Option("name", "--fields", "-f", help="Comma-separated fields for matching"),
    threshold: float = typer.Option(
        0.5, "--threshold", "-t", help="Similarity threshold for candidates"
    ),
    auto_match: float = typer.Option(
        0.9, "--auto-match", help="Threshold for automatic match classification"
    ),
    db_path: Path = typer.Option(
        Path("resolve.db"), "--db", "-d", help="Database file path"
    ),
) -> None:
    """Run entity matching on data.

    This is the main workhorse command. Can either ingest + match in one go,
    or just run matching on already-ingested records (handy for testing
    different threshold values).
    """
    config = MatchConfig(
        db_path=db_path,
        ann_threshold=threshold,
        auto_match_threshold=auto_match,
    )
    pipeline = ResolutionPipeline(config, db_path)

    match_fields = [f.strip() for f in fields.split(",")]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        if file_path and file_path.exists():
            progress.add_task("Running full pipeline...", total=None)
            scores = pipeline.run(
                file_path,
                name_field=name_field,
                address_field=address_field,
                id_field=id_field,
                match_fields=match_fields,
            )
        else:
            progress.add_task("Running matching on existing records...", total=None)
            scores = pipeline.run_on_loaded_records(match_fields)

    stats = pipeline.get_statistics()

    console.print()
    console.print("[bold]Matching Complete[/bold]")
    console.print(f"  Records: {stats['total_records']}")
    console.print(f"  Candidate pairs: {stats['total_candidate_pairs']}")
    console.print(f"  [green]Matches: {stats['matches']}[/green]")
    console.print(f"  [yellow]Uncertain: {stats['uncertain']}[/yellow]")
    console.print(f"  [dim]No match: {stats['no_matches']}[/dim]")


@app.command()
def report(
    db_path: Path = typer.Option(
        Path("resolve.db"), "--db", "-d", help="Database file path"
    ),
    output: Path = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, csv, json"),
) -> None:
    """Generate a matching report."""
    if not db_path.exists():
        console.print(f"[red]Error: Database not found: {db_path}[/red]")
        raise typer.Exit(1)

    db = Database(db_path)
    stats = db.get_statistics()

    # Print statistics
    table = Table(title="Entity Resolution Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Records", str(stats["total_records"]))
    table.add_row("Candidate Pairs", str(stats["total_candidate_pairs"]))
    table.add_row("Matches", str(stats["matches"]))
    table.add_row("Uncertain", str(stats["uncertain"]))
    table.add_row("No Match", str(stats["no_matches"]))
    table.add_row("Reviewed", str(stats["reviewed"]))

    console.print(table)

    # Get matches
    matches = db.get_scores_by_classification(MatchClassification.MATCH)

    if matches:
        console.print()
        match_table = Table(title=f"Top Matches ({len(matches)} total)")
        match_table.add_column("Record A", style="green")
        match_table.add_column("Record B", style="yellow")
        match_table.add_column("Score", style="cyan")

        for score, record_a, record_b in matches[:10]:
            name_a = record_a.raw_data.get("name", record_a.record_id)[:40]
            name_b = record_b.raw_data.get("name", record_b.record_id)[:40]
            match_table.add_row(name_a, name_b, f"{score.composite_score:.3f}")

        console.print(match_table)

    # Export if requested
    if output:
        export_format = "json" if output.suffix == ".json" else "csv"
        db.export_matches(output, export_format)
        console.print(f"\n[green]Matches exported to {output}[/green]")


@app.command()
def review(
    db_path: Path = typer.Option(
        Path("resolve.db"), "--db", "-d", help="Database file path"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of pairs to review"),
) -> None:
    """Review uncertain matches interactively.

    This is where the human-in-the-loop magic happens. Shows uncertain pairs
    one at a time with all the score details so you can make an informed
    decision. Your decisions are saved and used in the final export.

    TODO: this works but a streamlit UI would be so much better for bulk review
    """
    if not db_path.exists():
        console.print(f"[red]Error: Database not found: {db_path}[/red]")
        raise typer.Exit(1)

    db = Database(db_path)
    pairs = db.get_unreviewed_uncertain_pairs(limit)

    if not pairs:
        console.print("[yellow]No uncertain pairs to review.[/yellow]")
        return

    console.print(f"\n[bold]Reviewing {len(pairs)} uncertain pairs[/bold]")
    console.print("[dim]m=match, n=no match, s=skip, q=quit[/dim]\n")

    reviewed = 0
    for score, record_a, record_b in pairs:
        console.clear()

        # Display pair
        table = Table(title=f"Match Score: {score.composite_score:.3f}")
        table.add_column("Field", style="cyan")
        table.add_column("Record A", style="green")
        table.add_column("Record B", style="yellow")

        all_fields = set(record_a.raw_data.keys()) | set(record_b.raw_data.keys())
        for field in sorted(all_fields):
            val_a = str(record_a.raw_data.get(field, ""))[:50]
            val_b = str(record_b.raw_data.get(field, ""))[:50]
            if val_a != val_b:
                table.add_row(field, val_a, val_b, style="bold")
            else:
                table.add_row(field, val_a, val_b)

        console.print(table)

        # show all the score components so reviewer can understand why it's uncertain
        # in my testing, pairs where fuzzy is high but embedding is low (or vice versa)
        # are usually the trickiest - e.g., "John Smith" vs "Jon Smith" vs "John Smith Jr"
        console.print()
        console.print(f"[dim]Fuzzy (Levenshtein): {score.levenshtein_ratio:.3f}[/dim]")
        console.print(f"[dim]Fuzzy (Jaro-Winkler): {score.jaro_winkler:.3f}[/dim]")
        console.print(f"[dim]Embedding similarity: {score.cosine_similarity:.3f}[/dim]")

        # once we add LLM integration, this will show the model's reasoning
        if score.llm_reasoning:
            console.print(f"\n[bold]LLM Reasoning:[/bold] {score.llm_reasoning}")

        console.print("\n[dim]m=match, n=no match, s=skip, q=quit[/dim]")

        # Get decision
        while True:
            decision = console.input("\n[bold]Decision:[/bold] ").lower().strip()
            if decision in ("m", "n", "s", "q"):
                break
            console.print("[red]Invalid input. Use m, n, s, or q.[/red]")

        if decision == "q":
            break
        elif decision in ("m", "n"):
            notes = console.input("[dim]Notes (optional):[/dim] ").strip() or None
            review_decision = ReviewDecision(
                pair_id=score.pair_id,
                decision="match" if decision == "m" else "no_match",
                notes=notes,
            )
            db.insert_review_decision(review_decision)
            reviewed += 1
            console.print("[green]Decision saved.[/green]")
        else:
            console.print("[yellow]Skipped.[/yellow]")

        # small delay so you can see the "saved" message before screen clears
        # could make this configurable but 0.5s feels right
        import time
        time.sleep(0.5)

    console.print(f"\n[bold]Reviewed {reviewed} pairs.[/bold]")


@app.command()
def stats(
    db_path: Path = typer.Option(
        Path("resolve.db"), "--db", "-d", help="Database file path"
    ),
) -> None:
    """Show quick statistics."""
    if not db_path.exists():
        console.print(f"[red]Error: Database not found: {db_path}[/red]")
        raise typer.Exit(1)

    db = Database(db_path)
    stats = db.get_statistics()

    console.print(f"Records: {stats['total_records']}")
    console.print(f"Candidates: {stats['total_candidate_pairs']}")
    console.print(f"Matches: {stats['matches']}")
    console.print(f"Uncertain: {stats['uncertain']}")
    console.print(f"No match: {stats['no_matches']}")
    console.print(f"Reviewed: {stats['reviewed']}")


if __name__ == "__main__":
    app()
