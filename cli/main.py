import sys
from pathlib import Path

# Add project root directory to Python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from config import settings

app = typer.Typer(help="Claude Code Proxy Server CLI Command Tool")
console = Console()


@app.command()
def start(
    host: str = typer.Option("127.0.0.1", help="Host binding address"),
    port: int = typer.Option(8090, help="Port to run proxy server on"),
    reload: bool = typer.Option(False, help="Enable automatic code reload"),
) -> None:
    """Start the Claude Code Proxy FastAPI gateway server."""
    console.print(
        f"[bold green]Starting Uvicorn server on http://{host}:{port}...[/bold green]"
    )
    uvicorn.run("server:app", host=host, port=port, reload=reload)


@app.command()
def doctor() -> None:
    """Run health-check diagnostics for the proxy settings and target providers."""
    console.print("[bold cyan]🩺 Running Diagnostics (Doctor)...[/bold cyan]\n")

    table = Table(
        title="Configuration Checks", show_header=True, header_style="bold magenta"
    )
    table.add_column("Parameter", style="dim")
    table.add_column("Value", style="bold")
    table.add_column("Status")

    # 1. Check NVIDIA NIM Settings
    if settings.NVIDIA_NIM_API_KEY:
        table.add_row(
            "NVIDIA_NIM_API_KEY", "Present (Configured)", "✅ [green]OK[/green]"
        )
    else:
        table.add_row(
            "NVIDIA_NIM_API_KEY",
            "Empty",
            "⚠️ [yellow]Optional (NIM mapping will fail)[/yellow]",
        )

    # 2. Check OpenRouter Settings
    if settings.OPENROUTER_API_KEY:
        table.add_row(
            "OPENROUTER_API_KEY", "Present (Configured)", "✅ [green]OK[/green]"
        )
    else:
        table.add_row(
            "OPENROUTER_API_KEY",
            "Empty",
            "⚠️ [yellow]Optional (OpenRouter mapping will fail)[/yellow]",
        )

    # 3. Model mappings
    table.add_row("MODEL_OPUS", settings.MODEL_OPUS, "✅ [green]Configured[/green]")
    table.add_row("MODEL_SONNET", settings.MODEL_SONNET, "✅ [green]Configured[/green]")
    table.add_row("MODEL_HAIKU", settings.MODEL_HAIKU, "✅ [green]Configured[/green]")
    table.add_row("MODEL", settings.MODEL, "✅ [green]Configured[/green]")

    # 4. Mock settings status
    table.add_row(
        "Fast Prefix Mock", str(settings.FAST_PREFIX_DETECTION), "ℹ️ [blue]Active[/blue]"
    )
    table.add_row(
        "Network Probe Mock",
        str(settings.ENABLE_NETWORK_PROBE_MOCK),
        "ℹ️ [blue]Active[/blue]",
    )
    table.add_row(
        "Filepath Mock",
        str(settings.ENABLE_FILEPATH_EXTRACTION_MOCK),
        "ℹ️ [blue]Active[/blue]",
    )

    console.print(table)
    console.print()

    # 5. Connection Test for local LM Studio
    if "lmstudio" in (
        settings.MODEL_OPUS
        + settings.MODEL_SONNET
        + settings.MODEL_HAIKU
        + settings.MODEL
    ):
        console.print(
            "[bold yellow]Testing local LM Studio/Ollama endpoint...[/bold yellow]"
        )
        url = settings.LM_STUDIO_BASE_URL.rstrip("/")
        try:
            resp = httpx.get(f"{url}/models", timeout=2.0)
            if resp.status_code == 200:
                console.print(
                    f"✅ LM Studio/Ollama endpoint `{url}/models` is responsive.\n"
                )
            else:
                console.print(
                    f"❌ LM Studio/Ollama endpoint returned HTTP status: {resp.status_code}\n"
                )
        except Exception as e:
            console.print(f"❌ Failed to connect to LM Studio/Ollama at `{url}`: {e}\n")

    console.print("[bold green]System check finished.[/bold green]")


if __name__ == "__main__":
    app()
