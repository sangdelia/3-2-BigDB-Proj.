from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import polars as pl
import pytest

from bigdb_mvp.data import canonical_builder
from bigdb_mvp.data.canonical_audit import audit_full_canonical
from bigdb_mvp.data.canonical_builder import (
    CanonicalInput,
    PublicationIntegrityError,
    build_full_canonical,
)
from bigdb_mvp.data.manifest import SourceReceipt
from bigdb_mvp.data.models import (
    BinaryLabel,
    CanonicalRole,
    Language,
    SourceName,
    SupervisedRecord,
)
from bigdb_mvp.data.staging import sha256_file

if TYPE_CHECKING:
    from pathlib import Path


def _record(index: int) -> SupervisedRecord:
    text = f"synthetic full canonical row {index}"
    return SupervisedRecord(
        text=text,
        label=BinaryLabel.CLEAN,
        language=Language.EN,
        source=SourceName.CIVIL_COMMENTS,
        source_split="train",
        group_id=f"group-{index}",
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def _receipt(count: int) -> SourceReceipt:
    return SourceReceipt(
        source=SourceName.CIVIL_COMMENTS,
        revision="revision-1",
        license="test-license",
        input_sha256="a" * 64,
        source_split="train",
        role=CanonicalRole.RESEARCH_INDEX,
        scanned=count,
        accepted=count,
        excluded=0,
    )


def test_full_builder_retains_more_than_sample_cap(tmp_path: Path) -> None:
    # Given: one accepted source cell larger than the sampled builder cap.
    rows = tuple(_record(index) for index in range(2_501))

    # When: the separate full-canonical builder publishes the fixture.
    manifest = build_full_canonical(
        (CanonicalInput(records=rows, receipt=_receipt(len(rows))),),
        output_root=tmp_path,
        seed=42,
    )

    # Then: every accepted row is present in the distinct cap-free artifact.
    assert manifest.record_count == 2_501
    assert manifest.near_covered_record_count == 2_501
    assert manifest.near_candidate_comparison_count >= 0
    accepted = canonical_builder.load_published_canonical(tmp_path)
    assert pl.read_parquet(accepted.dataset_path).height == 2_501


def test_full_builder_repeated_builds_are_byte_and_logically_deterministic(
    tmp_path: Path,
) -> None:
    # Given: the same records presented in a different input iteration order.
    rows = tuple(_record(index) for index in range(20))
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    # When: both independent output roots are built.
    first = build_full_canonical(
        (CanonicalInput(records=rows, receipt=_receipt(len(rows))),),
        output_root=first_root,
        seed=42,
    )
    second = build_full_canonical(
        (CanonicalInput(records=tuple(reversed(rows)), receipt=_receipt(len(rows))),),
        output_root=second_root,
        seed=42,
    )

    # Then: logical and physical output digests are stable.
    assert first.logical_digest == second.logical_digest
    assert first.dataset_sha256 == second.dataset_sha256
    first_published = canonical_builder.load_published_canonical(first_root)
    second_published = canonical_builder.load_published_canonical(second_root)
    assert first_published.dataset_path.read_bytes() == (
        second_published.dataset_path.read_bytes()
    )


def test_missing_role_blocks_publication(tmp_path: Path) -> None:
    # Given: a source input whose role is absent at the publication boundary.
    rows = (_record(1),)
    receipt = _receipt(1).model_copy(update={"role": None})

    # When / Then: the typed publication boundary rejects the malformed input.
    with pytest.raises(ValueError, match="role"):
        _ = build_full_canonical(
            (CanonicalInput(records=rows, receipt=receipt),),
            output_root=tmp_path,
            seed=42,
        )
    assert not (tmp_path / "full_canonical.parquet").exists()
    assert not (tmp_path / "full_canonical.manifest.json").exists()


def test_interrupted_pair_publication_leaves_no_partial_finals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a publisher that fails while committing the second final file.
    calls = 0
    original_replace = canonical_builder.replace_path

    def fail_second(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            reason = "simulated interruption"
            raise OSError(reason)
        original_replace(source, target)

    monkeypatch.setattr(canonical_builder, "replace_path", fail_second)

    # When / Then: a failed commit cleans up both accepted final paths.
    with pytest.raises(OSError, match="simulated interruption"):
        _ = build_full_canonical(
            (CanonicalInput(records=(_record(1),), receipt=_receipt(1)),),
            output_root=tmp_path,
            seed=42,
        )
    assert not (tmp_path / "full_canonical.parquet").exists()
    assert not (tmp_path / "full_canonical.manifest.json").exists()
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_baseexception_during_pointer_replace_keeps_previous_generation_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a valid published generation and a crash at the pointer commit point.
    old = build_full_canonical(
        (CanonicalInput(records=(_record(1),), receipt=_receipt(1)),),
        output_root=tmp_path,
        seed=42,
    )
    original_replace = canonical_builder.replace_path

    def crash_at_pointer(source: Path, target: Path) -> None:
        if target.name == "full_canonical.current.json":
            reason = "simulated process interruption"
            raise KeyboardInterrupt(reason)
        original_replace(source, target)

    monkeypatch.setattr(canonical_builder, "replace_path", crash_at_pointer)

    # When: a changed generation is built and the sole acceptance pointer crashes.
    with pytest.raises(KeyboardInterrupt, match="simulated process interruption"):
        _ = build_full_canonical(
            (CanonicalInput(records=(_record(2),), receipt=_receipt(1)),),
            output_root=tmp_path,
            seed=42,
        )

    # Then: load-side acceptance still returns the old digest-consistent generation.
    accepted = canonical_builder.load_published_canonical(tmp_path)
    assert accepted.manifest.logical_digest == old.logical_digest
    report = audit_full_canonical(
        tmp_path,
        expected_sources=canonical_builder.expected_source_contract(
            (CanonicalInput(records=(_record(1),), receipt=_receipt(1)),)
        ),
    )
    assert report.valid is True
    assert accepted.dataset_path.exists()
    assert accepted.manifest.dataset_sha256 == sha256_file(accepted.dataset_path)


def test_load_rejects_tampered_current_generation(tmp_path: Path) -> None:
    # Given: a published generation selected by the current pointer.
    _ = build_full_canonical(
        (CanonicalInput(records=(_record(1),), receipt=_receipt(1)),),
        output_root=tmp_path,
        seed=42,
    )
    accepted = canonical_builder.load_published_canonical(tmp_path)
    _ = accepted.dataset_path.write_bytes(b"tampered")

    # When / Then: load-side validation rejects the changed generation.
    with pytest.raises(
        PublicationIntegrityError, match="publication_dataset_digest_mismatch"
    ):
        _ = canonical_builder.load_published_canonical(tmp_path)


@pytest.mark.parametrize(
    "target_kind", ["root", "ancestor", "generations", "pointer", "contract"]
)
def test_publication_rejects_symlinked_output_paths(
    tmp_path: Path, target_kind: str
) -> None:
    # Given: a publication path component that escapes the requested output root.
    real_root = tmp_path / "real"
    real_root.mkdir()
    if target_kind == "root":
        output_root = tmp_path / "root-link"
        output_root.symlink_to(real_root, target_is_directory=True)
    elif target_kind == "ancestor":
        output_root = tmp_path / "ancestor-link" / "nested"
        (tmp_path / "ancestor-link").symlink_to(real_root, target_is_directory=True)
    else:
        output_root = tmp_path / "root"
        output_root.mkdir()
        if target_kind == "generations":
            (output_root / "full_canonical.generations").symlink_to(
                real_root, target_is_directory=True
            )
        else:
            final_target = output_root / (
                "full_canonical.current.json"
                if target_kind == "pointer"
                else "full_canonical.trusted-contract.json"
            )
            outside_target = real_root / final_target.name
            _ = outside_target.write_text("outside", encoding="utf-8")
            final_target.symlink_to(outside_target)

    # When / Then: publication fails closed before writing outside the root.
    with pytest.raises(OSError, match=r"publication|symlink|directory|exists"):
        _ = build_full_canonical(
            (CanonicalInput(records=(_record(1),), receipt=_receipt(1)),),
            output_root=output_root,
            seed=42,
        )
    assert not (real_root / "full_canonical.generations").exists()


def test_standard_load_rejects_tampered_trusted_contract(tmp_path: Path) -> None:
    # Given: a valid publication whose trusted contract bytes are changed in place.
    _ = build_full_canonical(
        (CanonicalInput(records=(_record(1),), receipt=_receipt(1)),),
        output_root=tmp_path,
        seed=42,
    )
    contract_path = tmp_path / "full_canonical.trusted-contract.json"
    _ = contract_path.write_bytes(contract_path.read_bytes() + b"\n")

    # When / Then: the standard load path enforces the pointer-bound contract digest.
    with pytest.raises(PublicationIntegrityError, match="trusted_contract"):
        _ = canonical_builder.load_published_canonical(tmp_path)


@pytest.mark.parametrize(
    "interrupt_target",
    ["generation", "contract", "pointer"],
)
def test_interruption_during_each_publication_stage_keeps_previous_current_loadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt_target: str
) -> None:
    # Given: an accepted current generation and a publisher interrupted at one stage.
    old = build_full_canonical(
        (CanonicalInput(records=(_record(1),), receipt=_receipt(1)),),
        output_root=tmp_path,
        seed=42,
    )
    original_replace = canonical_builder.replace_path

    def interrupt(source: Path, target: Path) -> None:
        is_generation = target.parent.name == "full_canonical.generations"
        is_contract = target.name == "full_canonical.trusted-contract.json"
        is_pointer = target.name == "full_canonical.current.json"
        should_interrupt = (
            (interrupt_target == "generation" and is_generation)
            or (interrupt_target == "contract" and is_contract)
            or (interrupt_target == "pointer" and is_pointer)
        )
        if should_interrupt:
            reason = f"interrupt at {interrupt_target}"
            raise KeyboardInterrupt(reason)
        original_replace(source, target)

    monkeypatch.setattr(canonical_builder, "replace_path", interrupt)

    # When: a changed generation is published.
    with pytest.raises(KeyboardInterrupt, match=f"interrupt at {interrupt_target}"):
        _ = build_full_canonical(
            (CanonicalInput(records=(_record(2),), receipt=_receipt(1)),),
            output_root=tmp_path,
            seed=42,
        )

    # Then: the old pointer remains both loadable and auditable.
    accepted = canonical_builder.load_published_canonical(tmp_path)
    assert accepted.manifest.logical_digest == old.logical_digest
    report = audit_full_canonical(
        tmp_path,
        expected_sources=canonical_builder.expected_source_contract(
            (CanonicalInput(records=(_record(1),), receipt=_receipt(1)),)
        ),
    )
    assert report.valid is True
