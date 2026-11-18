"""CLI entry point for non-mutating MVP environment diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, assert_never

import typer

from bigdb_mvp.config import DoctorSettings
from bigdb_mvp.doctor import derive_doctor_report
from bigdb_mvp.gpu_probe import collect_live_snapshot

app = typer.Typer(add_completion=False, no_args_is_help=True)
data_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(data_app, name="data")
train_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(train_app, name="train")
DEFAULT_FULL_OUTPUT_ROOT = Path("data/curated")
DEFAULT_XLM_ARTIFACT_DIR = Path("artifacts/xlm")


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
    """Build the reproducible supervised sample."""
    from bigdb_mvp.data.builder import build_dataset

    typer.echo(build_dataset(seed=seed).model_dump_json())


@data_app.command("audit")
def data_audit(
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Audit the processed sample without reading upstream source paths."""
    from bigdb_mvp.data.audit import audit_dataset

    report = audit_dataset()
    if json_output:
        typer.echo(report.model_dump_json())
    else:
        typer.echo(f"valid={report.valid} rows={report.record_count}")
    if not report.valid:
        raise typer.Exit(code=1)


@data_app.command("full-build")
def data_full_build(
    seed: Annotated[int, typer.Option("--seed", min=0)] = 42,
    output_root: Annotated[
        Path, typer.Option("--output-root")
    ] = DEFAULT_FULL_OUTPUT_ROOT,
    fixture: Annotated[bool, typer.Option("--fixture")] = False,
    profile: Annotated[
        Literal["research", "production"], typer.Option("--profile")
    ] = "research",
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    from bigdb_mvp.data.arrow_sources import ArrowSourceError
    from bigdb_mvp.data.canonical_publication import PublicationIntegrityError
    from bigdb_mvp.data.civil_source import CivilAcquisitionError, CivilSourceError
    from bigdb_mvp.data.dedup_models import PublicationLeakageError
    from bigdb_mvp.data.full_canonical_cli import (
        FullBuildError,
        build_full_command,
        failure_report,
    )
    from bigdb_mvp.data.local_sources import LocalSourceError
    from bigdb_mvp.data.remote_integrity import RemoteSourceError
    from bigdb_mvp.data.staging import StagingIntegrityError

    try:
        report = build_full_command(
            output_root=output_root,
            seed=seed,
            fixture=fixture,
            profile=profile,
        )
    except (
        ArrowSourceError,
        CivilAcquisitionError,
        CivilSourceError,
        FullBuildError,
        LocalSourceError,
        PublicationIntegrityError,
        PublicationLeakageError,
        RemoteSourceError,
        StagingIntegrityError,
    ) as error:
        failure = failure_report(error)
        typer.echo(failure.model_dump_json())
        raise typer.Exit(code=2) from None
    typer.echo(
        report.model_dump_json() if json_output else f"rows={report.record_count}"
    )


@data_app.command("full-audit")
def data_full_audit(
    output_root: Annotated[
        Path, typer.Option("--output-root")
    ] = DEFAULT_FULL_OUTPUT_ROOT,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    from bigdb_mvp.data.full_canonical_cli import audit_full_command

    report = audit_full_command(output_root)
    if json_output:
        typer.echo(report.model_dump_json())
    else:
        typer.echo(f"valid={report.valid} rows={report.record_count}")
    if not report.valid:
        raise typer.Exit(code=1)


@train_app.command("baseline")
def train_baseline_command(
    seed: Annotated[int, typer.Option("--seed", min=0)] = 42,
) -> None:
    """Fit and evaluate the digest-bound CPU character TF-IDF baseline."""
    from bigdb_mvp.baseline import train_baseline

    typer.echo(train_baseline(seed=seed).model_dump_json())


@train_app.command("xlm-roberta")
def train_xlm_roberta_command(
    epochs: Annotated[int, typer.Option("--epochs", min=1, max=1)] = 1,
    device: Annotated[Literal["cuda:0"], typer.Option("--device")] = "cuda:0",
    seed: Annotated[int, typer.Option("--seed", min=0)] = 42,
    training_languages: Annotated[
        Literal["ko"], typer.Option("--training-languages")
    ] = "ko",
    artifact_dir: Annotated[
        Path, typer.Option("--artifact-dir")
    ] = DEFAULT_XLM_ARTIFACT_DIR,
) -> None:
    """Gate the one-epoch XLM-R run and record a truthful blocked outcome."""
    from bigdb_mvp.training_contracts import (
        BlockedTrainingReceipt,
        XlmTrainingConfig,
    )
    from bigdb_mvp.xlm_training import (
        GpuTrainingReady,
        XlmRunPaths,
        run_xlm_training,
        start_xlm_training,
    )
    from bigdb_mvp.xlm_transformers import TransformersBackend

    config = XlmTrainingConfig(
        epochs=epochs,
        device=device,
        seed=seed,
        training_languages=training_languages,
    )
    result = start_xlm_training(config)
    match result:
        case BlockedTrainingReceipt():
            evidence_path = Path(".omo/evidence/mvp/task-4-xlm.json")
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            _ = evidence_path.write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )
            typer.echo(result.model_dump_json())
            raise typer.Exit(code=1)
        case GpuTrainingReady(report=report):
            artifact = run_xlm_training(
                report=report,
                backend=TransformersBackend(),
                config=config,
                paths=XlmRunPaths(artifact_dir=artifact_dir),
            )
            evidence_path = Path(".omo/evidence/mvp/task-4-xlm.json")
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            _ = evidence_path.write_text(
                artifact.model_dump_json(indent=2), encoding="utf-8"
            )
            typer.echo(artifact.model_dump_json())
            if not artifact.eligible_for_serving:
                raise typer.Exit(code=1)
        case unreachable:
            assert_never(unreachable)


@app.command("predict")
def predict_command(
    text: Annotated[str, typer.Option("--text")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Predict one in-memory sentence without persisting its content."""
    from bigdb_mvp.predict import PredictionService

    result = PredictionService().predict_text(text)
    if json_output:
        typer.echo(result.model_dump_json())
        return
    typer.echo(f"label={result.label} harmful_probability={result.harmful_probability}")


@app.command("serve")
def serve_command(
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 7861,
) -> None:
    """Serve the local non-persistent API on loopback by default."""
    from bigdb_mvp.api import create_app

    create_app().run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


def main() -> None:
    """Run the typed MVP command-line application."""
    app()
