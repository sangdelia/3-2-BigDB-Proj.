"""Train and validate the reproducible CPU character n-gram baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from typing_extensions import override

from bigdb_mvp.baseline_contracts import BaselineArtifact, PredictionCount, SliceMetrics
from bigdb_mvp.baseline_metrics import binary_metrics, select_threshold
from bigdb_mvp.data.builder import DATASET_PATH, MANIFEST_PATH
from bigdb_mvp.data.manifest import Manifest
from bigdb_mvp.data.staging import sha256_file

ARTIFACT_DIR: Final = Path(__file__).resolve().parents[2] / "artifacts/baseline"
MODEL_FILENAME: Final = "baseline_model.joblib"
MANIFEST_FILENAME: Final = "baseline_manifest.json"
HARMFUL_LABEL: Final = "harmful"
CLEAN_LABEL: Final = "clean"
MIN_SCORE_STD: Final = 1e-6


@dataclass(frozen=True, slots=True)
class ArtifactIntegrityError(Exception):
    """Report immutable data/model digest drift at the loading boundary."""

    reason_code: str

    @override
    def __str__(self) -> str:
        return self.reason_code


@dataclass(frozen=True, slots=True)
class NgramContribution:
    """One positive-weighted character n-gram found in an input sentence."""

    ngram: str
    contribution: float


@dataclass(frozen=True, slots=True)
class BaselineModel:
    """Fitted TF-IDF and logistic components used together at inference."""

    vectorizer: TfidfVectorizer
    classifier: LogisticRegression


@dataclass(frozen=True, slots=True)
class LoadedBaseline:
    """Verified persisted model with its immutable provenance contract."""

    artifact: BaselineArtifact
    model: BaselineModel


@dataclass(frozen=True, slots=True)
class DatasetRow:
    """One minimum processed record accepted at the Parquet training boundary."""

    text: str
    label: str
    language: str
    split: str


def _texts(rows: tuple[DatasetRow, ...]) -> tuple[str, ...]:
    return tuple(row.text for row in rows)


def _labels(rows: tuple[DatasetRow, ...]) -> tuple[int, ...]:
    return tuple(int(row.label == HARMFUL_LABEL) for row in rows)


def _slice(
    rows: tuple[DatasetRow, ...], split: str, language: str | None = None
) -> tuple[DatasetRow, ...]:
    return tuple(
        row
        for row in rows
        if row.split == split and (language is None or row.language == language)
    )


def _scores(model: BaselineModel, texts: tuple[str, ...]) -> tuple[float, ...]:
    matrix = model.vectorizer.transform(texts)
    return tuple(float(score) for score in model.classifier.predict_proba(matrix)[:, 1])


def _evaluate(
    model: BaselineModel, rows: tuple[DatasetRow, ...], threshold: float
) -> tuple[SliceMetrics, ...]:
    overall_scores = _scores(model, _texts(rows))
    languages = ("ko", "en")
    parts = tuple(_slice(rows, "test", language) for language in languages)
    return (
        SliceMetrics(
            language="overall",
            metrics=binary_metrics(_labels(rows), overall_scores, threshold),
        ),
        *tuple(
            SliceMetrics(
                language=language,
                metrics=binary_metrics(
                    _labels(part), _scores(model, _texts(part)), threshold
                ),
            )
            for language, part in zip(languages, parts, strict=True)
        ),
    )


def train_baseline(
    *,
    seed: int,
    dataset_path: Path = DATASET_PATH,
    manifest_path: Path = MANIFEST_PATH,
    artifact_dir: Path = ARTIFACT_DIR,
) -> BaselineArtifact:
    """Fit on train, calibrate once, then persist one held-out result."""
    manifest = Manifest.model_validate_json(manifest_path.read_bytes())
    if sha256_file(dataset_path) != manifest.dataset_sha256:
        reason = "dataset_digest_mismatch"
        raise ArtifactIntegrityError(reason)
    table = pq.read_table(dataset_path, columns=["text", "label", "language", "split"])
    rows = tuple(DatasetRow(**row) for row in table.to_pylist())
    train = _slice(rows, "train")
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True)
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=seed, solver="liblinear"
    )
    _ = classifier.fit(vectorizer.fit_transform(_texts(train)), _labels(train))
    model = BaselineModel(vectorizer=vectorizer, classifier=classifier)
    calibration = _slice(rows, "calibration")
    threshold = select_threshold(
        _labels(calibration), _scores(model, _texts(calibration))
    )
    test = _slice(rows, "test")
    scores = _scores(model, _texts(test))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / MODEL_FILENAME
    _ = joblib.dump(model, model_path)
    artifact = BaselineArtifact(
        seed=seed,
        dataset_sha256=manifest.dataset_sha256,
        logical_digest=manifest.logical_digest,
        model_sha256=sha256_file(model_path),
        threshold=threshold,
        score_standard_deviation=float(np.std(np.asarray(scores))),
        metrics=_evaluate(model, test, threshold),
        prediction_distribution=tuple(
            PredictionCount(
                label=label,
                count=sum((score >= threshold) == target for score in scores),
            )
            for label, target in ((CLEAN_LABEL, False), (HARMFUL_LABEL, True))
        ),
    )
    if artifact.score_standard_deviation <= MIN_SCORE_STD:
        reason = "score_variance_ineligible"
        raise ArtifactIntegrityError(reason)
    _ = (artifact_dir / MANIFEST_FILENAME).write_text(
        artifact.model_dump_json(indent=2), encoding="utf-8"
    )
    return artifact


def load_baseline(
    artifact_dir: Path = ARTIFACT_DIR,
    dataset_path: Path = DATASET_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> LoadedBaseline:
    """Load only a model whose artifact and live data digests agree exactly."""
    artifact = BaselineArtifact.model_validate_json(
        (artifact_dir / MANIFEST_FILENAME).read_bytes()
    )
    manifest = Manifest.model_validate_json(manifest_path.read_bytes())
    if (
        sha256_file(dataset_path) != artifact.dataset_sha256
        or manifest.dataset_sha256 != artifact.dataset_sha256
    ):
        reason = "dataset_digest_mismatch"
        raise ArtifactIntegrityError(reason)
    model_path = artifact_dir / artifact.model_file
    if sha256_file(model_path) != artifact.model_sha256:
        reason = "model_digest_mismatch"
        raise ArtifactIntegrityError(reason)
    model: BaselineModel = joblib.load(model_path)
    return LoadedBaseline(artifact=artifact, model=model)


def extract_positive_ngrams(
    text: str, model: BaselineModel, limit: int = 8
) -> tuple[NgramContribution, ...]:
    """Extract positive contributions from the actual TF-IDF feature vector."""
    values = model.vectorizer.transform((text,)).toarray()[0]
    weights = model.classifier.coef_[0]
    names = model.vectorizer.get_feature_names_out()
    contributions = tuple(
        NgramContribution(ngram=ngram, contribution=value * weight)
        for ngram, value, weight in zip(names, values, weights, strict=True)
        if value * weight > 0.0
    )
    return tuple(
        sorted(contributions, key=lambda item: item.contribution, reverse=True)[:limit]
    )
