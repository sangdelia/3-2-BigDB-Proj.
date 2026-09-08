"""CLI entry point for non-mutating MVP environment diagnostics."""

from __future__ import annotations

from typing import Annotated

import typer

from bigdb_mvp.config import DoctorSettings
from bigdb_mvp.data.audit import audit_dataset
from bigdb_mvp.data.builder import build_dataset
from bigdb_mvp.doctor import derive_doctor_report
from bigdb_mvp.gpu_probe import collect_live_snapshot

app = typer.Typer(add_completion=False, no_args_is_help=True)
data_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(data_app, name="data")


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


@data_app.command("build")
def data_build(
    seed: Annotated[int, typer.Option("--seed", min=0)] = 42,
) -> None:
    """Build the reproducible supervised bilingual sample."""
    typer.echo(build_dataset(seed=seed).model_dump_json())


@data_app.command("audit")
def data_audit(
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Audit the processed sample without reading upstream source paths."""
    report = audit_dataset()
    if json_output:
        typer.echo(report.model_dump_json())
    else:
        typer.echo(f"valid={report.valid} rows={report.record_count}")
    if not report.valid:
        raise typer.Exit(code=1)


def main() -> None:
    """Run the typed MVP command-line application."""
    app()
