"""``argus-echo`` command line."""

from __future__ import annotations

import typer

from argus_echo import __version__
from argus_echo.check import check as run_check

app = typer.Typer(help="argus-echo: the Argus suite's speech servers.", no_args_is_help=True)


@app.callback()
def _main() -> None:
    """argus-echo: the Argus suite's speech servers."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def check(
    url: str = typer.Argument(..., help="An engine's base URL, e.g. http://localhost:8140"),
    speak: str | None = typer.Option(
        None, "--speak", help="Also stream this text and check the audio. Loads the model if it is not loaded."
    ),
    speaker: str | None = typer.Option(None, "--speaker", help="Built-in speaker for --speak."),
    timeout: float = typer.Option(30.0, help="Seconds per read-only request."),
) -> None:
    """Check a running engine against the argus-echo contract. Exits 1 if any check fails."""
    report = run_check(url, speak=speak, speaker=speaker, timeout=timeout)
    for f in report.findings:
        mark = typer.style("ok  ", fg="green") if f.ok else typer.style("FAIL", fg="red")
        typer.echo(f"{mark} {f.name}: {f.detail}")
    raise typer.Exit(0 if report.ok else 1)
