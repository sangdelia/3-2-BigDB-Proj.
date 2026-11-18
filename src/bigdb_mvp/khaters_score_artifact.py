"""Verify the locally trained rating artifact before loading trusted joblib."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, assert_never

import joblib
from pydantic import BaseModel, ConfigDict

from bigdb_mvp.baseline import BaselineModel
from bigdb_mvp.data.staging import sha256_file
from bigdb_mvp.experiments.classification_contracts import ExperimentError
from bigdb_mvp.experiments.classification_model import Model
from bigdb_mvp.khaters_score import ScoreModel

if TYPE_CHECKING:
    from pathlib import Path


class _Metadata(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")
    model_sha256: str
    dataset_digest: str
    revision: str


class _Selection(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    version: Literal["khaters_score_v1", "khaters_score_v2"]


def load_score_model(root: Path) -> ScoreModel:
    """Require matching model checksum, source revision and dataset provenance."""
    selection = root / "reports/khaters_score_selection.json"
    version = (
        _Selection.model_validate_json(selection.read_bytes()).version
        if selection.exists()
        else "khaters_score_v1"
    )
    metadata = _Metadata.model_validate_json(
        (root / "reports" / version / "metrics.json").read_bytes()
    )
    path = root / "artifacts" / version / "model.joblib"
    if sha256_file(path) != metadata.model_sha256:
        reason = "score_model_checksum_mismatch"
        raise ExperimentError(reason)
    loaded = joblib.load(path, mmap_mode="c")
    match loaded:
        case ScoreModel():
            if (loaded.dataset_digest, loaded.revision) != (
                metadata.dataset_digest,
                metadata.revision,
            ):
                reason = "score_model_provenance_mismatch"
                raise ExperimentError(reason)
            return loaded
        case BaselineModel() | Model():
            reason = "score_model_type_mismatch"
            raise ExperimentError(reason)
        case unreachable:
            assert_never(unreachable)
