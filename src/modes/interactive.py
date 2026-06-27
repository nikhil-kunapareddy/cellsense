"""Interactive REPL mode: banner, prompt, slash commands, rich UI."""
from __future__ import annotations

from typing import Callable, Dict, List

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from src.data import FileData
from src.utils.citations import format_citations
from src.utils.classifier import classify_file
from src.utils.context_builder import build_context

# Enable basic readline history on platforms that support it
try:
    import readline  # noqa: F401
except ImportError:
    pass

console = Console()
VERSION = "0.1.0"

_HELP_MARKUP = """\
[bold]Commands[/bold]

  [bold cyan]/help[/bold cyan]    Show this message
  [bold cyan]/files[/bold cyan]   List loaded files, sheets, and column names
  [bold cyan]/clear[/bold cyan]   Clear the screen and redraw the banner
  [bold cyan]/exit[/bold cyan]    End the session

[bold]Tips[/bold]

  • Ask anything in plain English — no SQL needed.
  • Questions can span multiple files: "Compare revenue in sales.xlsx with targets in budget.xlsx"
  • Use [cyan]/files[/cyan] to remind yourself of available column names.
"""


# ── Public entry point ─────────────────────────────────────────────────────────

def run_repl(file_data: Dict[str, FileData], ask_fn: Callable, agent_label: str) -> None:
    context = build_context(file_data)
    history: List[dict] = []
    questions_asked = 0

    _draw_banner(file_data, agent_label)

    while True:
        try:
            raw = _prompt()
        except (KeyboardInterrupt, EOFError):
            _draw_exit_summary(questions_asked, file_data)
            break

        line = raw.strip()
        if not line:
            continue

        if line.startswith("/"):
            cmd = line.split()[0].lower()
            if cmd == "/exit":
                _draw_exit_summary(questions_asked, file_data)
                break
            elif cmd == "/help":
                console.print(Panel(
                    _HELP_MARKUP,
                    title="[bold cyan]CellSense Help[/bold cyan]",
                    border_style="cyan",
                ))
            elif cmd == "/files":
                _draw_files(file_data)
            elif cmd == "/clear":
                console.clear()
                _draw_banner(file_data, agent_label)
            else:
                console.print(
                    f"[yellow]Unknown command:[/yellow] {cmd}"
                    "  (type [cyan]/help[/cyan] for options)"
                )
            continue

        questions_asked += 1
        _handle_question(line, file_data, context, history, ask_fn)
        context = build_context(file_data)  # rebuild in case find_files loaded new files


# ── Question handling ──────────────────────────────────────────────────────────

def _handle_question(
    question: str,
    file_data: Dict[str, FileData],
    context: str,
    history: List[dict],
    ask_fn: Callable,
) -> None:
    with console.status("[dim]thinking...[/dim]", spinner="dots"):
        try:
            answer, citations = ask_fn(question, file_data, context, history)
        except Exception as exc:
            console.print(
                Panel(f"[red]Error:[/red] {exc}",
                      title="[bold red]CellSense[/bold red]", border_style="red")
            )
            return

    citation_str = format_citations(citations)

    md = Markdown(answer)
    renderables = [md]
    if citation_str and citation_str not in answer:
        renderables.append(Text())  # blank line
        renderables.append(Text(citation_str, style="dim"))

    console.print(
        Panel(Group(*renderables), title="[bold]CellSense[/bold]",
              border_style="cyan", padding=(1, 2))
    )


# ── UI helpers ─────────────────────────────────────────────────────────────────

def _prompt() -> str:
    console.print()
    console.print("[bold green]cellsense[/bold green] [dim]>[/dim] ", end="")
    return input()


def _draw_banner(file_data: Dict[str, FileData], agent_label: str) -> None:
    console.clear()
    console.print()

    title = Text()
    title.append("CellSense", style="bold cyan")
    title.append(f"  v{VERSION}", style="dim")
    console.print(f"  {title}")
    console.print(f"  [dim]AI-powered spreadsheet analysis  ·  {agent_label}[/dim]")
    console.print()
    console.print(Rule(style="cyan dim"))
    console.print()

    total_rows = sum(fd.total_rows for fd in file_data.values())
    console.print(
        f"  [bold]Loaded {len(file_data)} file(s)[/bold]"
        f"  [dim]·  {total_rows:,} total rows[/dim]"
    )
    console.print()

    for filename, fd in file_data.items():
        category = classify_file(fd)
        for sheet_name, df in fd.sheets.items():
            shape = f"[dim]{len(df):,} rows × {len(df.columns)} cols[/dim]"
            cat_badge = f"[dim magenta][{category}][/dim magenta]"
            if fd.file_type == "csv":
                console.print(f"  [green]✓[/green]  [bold]{filename}[/bold]  {shape}  {cat_badge}")
            else:
                console.print(
                    f"  [green]✓[/green]  [bold]{filename}[/bold]"
                    f"  [dim cyan][{sheet_name}][/dim cyan]  {shape}  {cat_badge}"
                )

    console.print()
    console.print(Rule(style="dim"))
    console.print()
    console.print("  Ask a question about your data, or type [cyan]/help[/cyan] for commands.")


def _draw_files(file_data: Dict[str, FileData]) -> None:
    for filename, fd in file_data.items():
        lines: list[str] = []
        for sheet_name, df in fd.sheets.items():
            if fd.file_type == "excel":
                lines.append(
                    f"[bold]Sheet:[/bold] {sheet_name}"
                    f"  ({len(df):,} rows × {len(df.columns)} cols)"
                )
            else:
                lines.append(f"[dim]{len(df):,} rows × {len(df.columns)} columns[/dim]")

            cols = df.columns.tolist()
            preview = ", ".join(cols[:10])
            if len(cols) > 10:
                preview += f" [dim]… +{len(cols) - 10} more[/dim]"
            lines.append(f"  [dim]Columns:[/dim] {preview}")
            lines.append("")

        category = classify_file(fd)
        console.print(
            Panel(
                "\n".join(lines).rstrip(),
                title=f"[bold cyan]{filename}[/bold cyan]  [dim]({fd.file_type})[/dim]  [magenta][{category}][/magenta]",
                border_style="cyan",
                padding=(0, 1),
            )
        )


def _draw_exit_summary(questions: int, file_data: Dict[str, FileData]) -> None:
    console.print()
    console.print(Rule(style="dim"))
    console.print(
        f"  Session ended  ·  "
        f"[bold]{questions}[/bold] question(s) asked  ·  "
        f"[bold]{len(file_data)}[/bold] file(s) analyzed"
    )
    console.print()
