from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import polars as pl
import pytest

from bigdb_mvp.baseline import (
    ArtifactIntegrityError,
    extract_positive_ngrams,
    load_baseline,
    train_baseline,
)
from bigdb_mvp.baseline_metrics import select_threshold

if TYPE_CHECKING:
    from pathlib import Path


def _write_dataset(root: Path) -> tuple[Path, Path]:
    rows: list[dict[str, str]] = []
    for language, clean, harmful in (
        ("ko", "평범한 좋은 문장", "모욕적인 나쁜 문장"),
        ("en", "ordinary kind sentence", "abusive hateful sentence"),
    ):
        for split in ("train", "calibration", "test"):
            for label, text in (("clean", clean), ("harmful", harmful)):
                for index in range(4):
                    rows.append(
                        {
                            "text": f"{text} {split} {index}",
                            "label": label,
                            "language": language,
                            "source": "nikl",
                            "source_split": "train",
                            "group_id": f"{language}-{split}-{label}-{index}",
                            "text_sha256": f"{len(rows):064x}",
                            "split": split,
                        }
                    )
    dataset_path = root / "sample.parquet"
    pl.DataFrame(rows).write_parquet(dataset_path)
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    manifest_path = root / "manifest.json"
    _ = manifest_path.write_text(
        '{"schema_version":1,"seed":42,"sources":[],"dataset_sha256":"'
        + digest
        + '","logical_digest":"'
        + "1" * 64
        + '"}',
        encoding="utf-8",
    )
    return dataset_path, manifest_path


def test_select_threshold_prefers_lower_value_after_metric_and_recall_ties() -> None:
    # Given: calibration scores with the same perfect classification across a range.
    labels = (0, 1)
    probabilities = (0.30, 0.70)

    # When: the documented threshold grid and tie-break are applied.
    threshold = select_threshold(labels, probabilities)

    # Then: the lower threshold inside the optimal range is frozen.
    assert threshold == pytest.approx(0.31)


def test_train_baseline_writes_digest_bound_artifact_and_positive_ngrams(
    tmp_path: Path,
) -> None:
    # Given: a small complete bilingual split and a manifest boundary.
    dataset_path, manifest_path = _write_dataset(tmp_path)
    artifact_dir = tmp_path / "artifact"

    # When: the train-only baseline is fit and evaluated once on held-out test data.
    artifact = train_baseline(
        seed=42,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        artifact_dir=artifact_dir,
    )
    loaded = load_baseline(artifact_dir, dataset_path, manifest_path)

    # Then: digests, both labels, and local positive evidence exist.
    assert artifact.dataset_sha256
    assert artifact.model_sha256
    assert {item.label for item in artifact.prediction_distribution} == {
        "clean",
        "harmful",
    }
    assert artifact.score_standard_deviation > 1e-6
    assert extract_positive_ngrams("abusive hateful sentence", loaded.model)


def test_load_baseline_refuses_dataset_digest_mismatch(tmp_path: Path) -> None:
    # Given: a trained artifact bound to one exact processed dataset byte stream.
    dataset_path, manifest_path = _write_dataset(tmp_path)
    artifact_dir = tmp_path / "artifact"
    _ = train_baseline(
        seed=42,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        artifact_dir=artifact_dir,
    )
    _ = dataset_path.write_bytes(b"not the training dataset")

    # When / Then: a changed dataset is rejected instead of silently serving the model.
    with pytest.raises(ArtifactIntegrityError, match="dataset_digest_mismatch"):
        _ = load_baseline(artifact_dir, dataset_path, manifest_path)
