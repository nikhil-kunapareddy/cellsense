"""Print mode: answer a single question non-interactively and exit.

Used when CellSense is invoked with -q/--query. Builds context once, runs the
agent for exactly one turn, prints the answer (plus a citation block) and
returns. Suitable for scripting and piping.
"""
from __future__ import annotations

from typing import Callable, Dict

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from src.data import FileData
from src.utils.citations import format_citations
from src.utils.context_builder import build_context

console = Console()


def run_print(
    file_data: Dict[str, FileData],
    ask_fn: Callable,
    agent_label: str,
    question: str,
) -> int:
    """Answer ``question`` once. Returns a process exit code (0 ok, 1 on error)."""
    context = build_context(file_data)
    history: list[dict] = []

    try:
        answer, citations = ask_fn(question, file_data, context, history)
    except Exception as exc:
        console.print(
            Panel(f"[red]Error:[/red] {exc}",
                  title="[bold red]CellSense[/bold red]", border_style="red")
        )
        return 1

    citation_str = format_citations(citations)

    renderables = [Markdown(answer)]
    if citation_str and citation_str not in answer:
        renderables.append(Text())  # blank line
        renderables.append(Text(citation_str, style="dim"))

    console.print(
        Panel(Group(*renderables), title="[bold]CellSense[/bold]",
              border_style="cyan", padding=(1, 2))
    )
    return 0
