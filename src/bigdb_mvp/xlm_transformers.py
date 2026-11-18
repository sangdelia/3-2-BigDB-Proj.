"""Pinned Hugging Face/PyTorch backend for the GPU0 XLM-R run."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from bigdb_mvp.training_contracts import (
    BackendOutput,
    TrainingRequest,
    TrainingState,
    XlmBackendError,
    XlmRow,
)

if TYPE_CHECKING:
    from pathlib import Path


def _encode(
    tokenizer: PreTrainedTokenizerBase,
    rows: tuple[XlmRow, ...],
    request: TrainingRequest,
) -> BatchEncoding:
    encoded = tokenizer(
        [row.text for row in rows],
        padding=True,
        truncation=True,
        max_length=request.config.max_length,
        return_tensors="pt",
    )
    return encoded.to(request.config.device)


def _logits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    rows: tuple[XlmRow, ...],
    request: TrainingRequest,
) -> tuple[float, ...]:
    values: list[float] = []
    _ = model.eval()
    _ = model.requires_grad_(requires_grad=False)
    for start in range(0, len(rows), request.config.batch_size):
        encoded = _encode(
            tokenizer, rows[start : start + request.config.batch_size], request
        )
        output = model(**encoded)
        differences = output.logits[:, 1] - output.logits[:, 0]
        values.extend(float(value) for value in differences.cpu().tolist())
    return tuple(values)


class TransformersBackend:
    """Execute one explicit epoch over prepared Korean CUDA0 batches."""

    def train(self, request: TrainingRequest, output_dir: Path) -> BackendOutput:
        """Train once and return logits plus complete temporary directories."""
        _ = torch.manual_seed(request.config.seed)
        torch.cuda.manual_seed_all(request.config.seed)
        tokenizer = AutoTokenizer.from_pretrained(
            request.config.model_id, revision=request.config.revision
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            request.config.model_id,
            revision=request.config.revision,
            num_labels=2,
        ).to(request.config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        _ = model.train()
        try:
            for batch in request.batches:
                encoded = _encode(tokenizer, batch, request)
                labels = torch.tensor(
                    tuple(row.label for row in batch), device=request.config.device
                )
                output = model(**encoded, labels=labels)
                if output.loss is None:
                    raise XlmBackendError(
                        state=TrainingState.CUDA_ERROR,
                        reason_code="missing_training_loss",
                    )
                _ = output.loss.backward()
                _ = optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        except torch.OutOfMemoryError as error:
            raise XlmBackendError(
                state=TrainingState.OOM, reason_code="cuda_out_of_memory"
            ) from error
        except RuntimeError as error:
            if "cuda" not in str(error).lower():
                raise
            raise XlmBackendError(
                state=TrainingState.CUDA_ERROR, reason_code="cuda_runtime_error"
            ) from error
        checkpoint_dir = output_dir / "checkpoint"
        tokenizer_dir = output_dir / "tokenizer"
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        _ = tokenizer.save_pretrained(tokenizer_dir)
        return BackendOutput(
            calibration_logits=_logits(model, tokenizer, request.calibration, request),
            test_logits=_logits(model, tokenizer, request.test, request),
            checkpoint_dir=checkpoint_dir,
            tokenizer_dir=tokenizer_dir,
        )
