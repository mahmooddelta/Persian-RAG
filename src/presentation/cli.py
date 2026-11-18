"""
Typer + Rich Interactive CLI for the Persian RAG Question-Answering System.

Commands:
- `ingest`: Ingest a single Persian PDF or a directory of PDFs into Qdrant.
- `ask`: Ask a single question in Persian and display the grounded answer + citations.
- `chat`: Launch an interactive multi-turn Persian Q&A session in the terminal.

All CLI UI labels, headers, and table titles are in English to avoid RTL/LTR
bidirectional rendering glitches in Windows terminals, while also saving the
formatted Persian output to `last_answer.md` for clean RTL viewing in editors.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.domain.entities import QAResponse
from src.domain.exceptions import DomainError
from src.infrastructure.llm_client import (
    PERSIAN_RAG_SYSTEM_PROMPT,
    OpenAICompatibleLLMClient,
)
from src.presentation.api import get_container

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console()
app = typer.Typer(
    name="persian-rag",
    help="Persian RAG Question-Answering System CLI (Clean Architecture).",
    add_completion=False,
)


def _save_answer_markdown(response: QAResponse, llm_model: str) -> Path:
    output_path = Path("last_answer.md")
    lines = [
        "# Grounded RAG Response (`last_answer.md`)",
        "",
        f"- **LLM Model:** `{llm_model}`",
        f"- **Latency:** `{response.latency_ms:.1f} ms`",
        f"- **Query:** {response.normalized_query}",
        "",
        "## Final Grounded Answer",
        "",
        f"> {response.answer}",
        "",
        "## Retrieved Context Chunks (Source Citations)",
        "",
        "| Rank | Source File | Page | Similarity Score | Snippet |",
        "| :---: | :--- | :---: | :---: | :--- |",
    ]
    for idx, cit in enumerate(response.citations, start=1):
        lines.append(
            f"| {idx} | `{cit.source_filename}` | {cit.page_number} | `{cit.score:.4f}` | {cit.snippet} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _render_qa_response(
    response: QAResponse,
    llm_model: str = "",
    show_prompt: bool = False,
) -> None:
    console.print()

    if response.citations:
        table = Table(
            title="[1] Retrieved Context Chunks (Source Citations)",
            show_header=True,
            header_style="bold cyan",
            show_lines=True,
        )
        table.add_column("Rank", justify="center", width=6)
        table.add_column("Source File", style="bold yellow")
        table.add_column("Page", justify="center", width=6)
        table.add_column("Score", justify="center", width=10)
        table.add_column("Persian Snippet", style="white")

        for idx, citation in enumerate(response.citations, start=1):
            table.add_row(
                str(idx),
                citation.source_filename,
                str(citation.page_number),
                f"{citation.score:.4f}",
                citation.snippet,
            )
        console.print(table)
    else:
        console.print("[yellow]No relevant chunks found above score threshold.[/yellow]")

    if show_prompt and response.retrieved_chunks:
        context_block = OpenAICompatibleLLMClient.format_context_block(
            response.retrieved_chunks
        )
        full_prompt = (
            f"[bold magenta][System Prompt][/bold magenta]\n{PERSIAN_RAG_SYSTEM_PROMPT}\n\n"
            f"[bold magenta][User Prompt + Retrieved Context][/bold magenta]\n"
            f"{context_block}\n\n"
            f"Query: {response.normalized_query}"
        )
        console.print()
        console.print(
            Panel(
                full_prompt,
                title="[bold yellow]Augmented Prompt Sent to LLM[/bold yellow]",
                border_style="yellow",
            )
        )

    model_label = f" | Model: {llm_model}" if llm_model else ""
    console.print()
    console.print(
        Panel(
            f"[bold white]{response.answer}[/bold white]",
            title=(
                f"[bold green][2] Final Grounded LLM Answer "
                f"(Latency: {response.latency_ms:.1f} ms{model_label})[/bold green]"
            ),
            subtitle="[dim]Saved clean RTL copy to: ./last_answer.md[/dim]",
            border_style="bold green",
            padding=(1, 2),
        )
    )

    try:
        _save_answer_markdown(response, llm_model=llm_model)
    except Exception:
        pass


@app.command("ingest")
def ingest_command(
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="Path to a Persian PDF file or directory of PDFs (defaults to PDF_DATA_DIR).",
    ),
) -> None:
    container = get_container()
    target = path if path is not None else container.settings.pdf_data_dir

    console.print(
        f"[bold blue]Starting document ingestion from path:[/bold blue] [underline]{target}[/underline]"
    )
    try:
        report = container.ingest_use_case.execute(target)
        table = Table(
            title="Ingestion Summary Report",
            header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold green")
        table.add_row("Documents Processed", str(report.documents_processed))
        table.add_row("Pages Extracted", str(report.pages_extracted))
        table.add_row("Chunks Created", str(report.chunks_created))
        table.add_row("Vectors Stored in Qdrant", str(report.vectors_stored))
        table.add_row("Total Latency", f"{report.latency_ms:.2f} ms")
        table.add_row("Source Files", ", ".join(report.source_files))
        console.print(table)
    except DomainError as exc:
        console.print(f"[bold red]Ingestion Error:[/bold red] {exc}")
        raise typer.Exit(code=1)


@app.command("ask")
def ask_command(
    query: str = typer.Argument(..., help="Your question in Persian."),
    top_k: int = typer.Option(
        4, "--top-k", "-k", min=3, max=20, help="Number of chunks to retrieve (min 3)."
    ),
    source: Optional[str] = typer.Option(
        None, "--source", "-s", help="Filter retrieval by a specific PDF filename."
    ),
    show_prompt: bool = typer.Option(
        False,
        "--show-prompt",
        help="Display the full augmented prompt sent to the LLM.",
    ),
) -> None:
    container = get_container()
    try:
        response = container.query_use_case.execute(
            query=query,
            top_k=top_k,
            source_filter=source,
        )
        _render_qa_response(
            response,
            llm_model=container.settings.llm_model,
            show_prompt=show_prompt,
        )
    except DomainError as exc:
        console.print(f"[bold red]Query Error:[/bold red] {exc}")
        raise typer.Exit(code=1)


@app.command("chat")
def chat_command(
    top_k: int = typer.Option(
        4, "--top-k", "-k", min=3, max=20, help="Number of chunks to retrieve per query."
    ),
) -> None:
    container = get_container()
    console.print(
        Panel(
            "Welcome to the Persian RAG Interactive Q&A CLI!\n"
            "Type [bold yellow]exit[/bold yellow] or [bold yellow]quit[/bold yellow] to leave the session.",
            title="[bold cyan]Persian RAG Interactive Chat[/bold cyan]",
            border_style="cyan",
        )
    )

    while True:
        try:
            user_input = console.input("\n[bold cyan]Question > [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold yellow]Session ended.[/bold yellow]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q", "خروج"}:
            console.print("[bold yellow]Session ended. Goodbye![/bold yellow]")
            break

        try:
            response = container.query_use_case.execute(
                query=user_input,
                top_k=top_k,
            )
            _render_qa_response(
                response,
                llm_model=container.settings.llm_model,
            )
        except DomainError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")


if __name__ == "__main__":
    app()
