"""CLI entry point for non-mutating MVP environment diagnostics."""

from __future__ import annotations

from typing import Annotated

import typer

from bigdb_mvp.config import DoctorSettings
from bigdb_mvp.doctor import derive_doctor_report
from bigdb_mvp.gpu_probe import collect_live_snapshot

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def root() -> None:
    """Group MVP commands without flattening the doctor subcommand."""


@app.command()
def doctor(
    json_output: Annotated[bool, typer.Option("--json")] = False,
    device: Annotated[int, typer.Option("--device", min=0)] = 0,
) -> None:
    """Print current GPU0 training readiness without mutating host state."""
    settings = DoctorSettings(device=device)
    report = derive_doctor_report(collect_live_snapshot(settings), settings)
    if json_output:
        typer.echo(report.model_dump_json())
        return
    typer.echo(
        f"gpu_ready={report.gpu_ready} reason={report.primary_reason_code.value}"
    )


def main() -> None:
    """Run the typed MVP command-line application."""
    app()
