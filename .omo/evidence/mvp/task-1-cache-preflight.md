# Task 1 model-cache preflight receipt

## Contract

`bigdb-mvp doctor --json` now reports the resolved cache root and separate
Hugging Face model-cache state for `FacebookAI/xlm-roberta-base`. Root
selection is explicit configuration, `HF_HUB_CACHE`, `HF_HOME/hub`, then the
standard `~/.cache/huggingface/hub` default. The probe uses `lstat`, does not
create files, follows no cache symlink, and scans at most 64 snapshot entries.

## Red and green evidence

- Red: the cache tests initially failed during collection because the old code
  had no `cache_probe` module or model-cache report field.
- Green: `uv run pytest tests/test_doctor.py -q` -> `10 passed`.
- Green: `uv run ruff check ...` -> `All checks passed!`.
- Green: `uv run ruff format --check ...` -> all changed files formatted.
- Green: `uv run basedpyright` -> `0 errors, 0 warnings, 0 notes`.

## Actual live result

Invocation: `uv run bigdb-mvp doctor --json`

- GPU: `gpu_ready=false`, `primary_reason_code=nvml_unavailable`,
  `nvml_status=unavailable`; the bounded diagnostic states that NVIDIA-SMI
  cannot communicate with the NVIDIA driver.
- Cache: default root exists, is a directory, and is writable; free space is
  reported. The exact repository cache and snapshot are absent,
  `model_snapshot_cached=false`, with stable reasons
  `model_repo_cache_missing` and `model_snapshot_missing`.
- Missing model snapshot is reported independently and remains downloadable
  later; it does not alter GPU reason codes or make GPU readiness true.

## Edge coverage

Tests exercise an explicit populated snapshot, missing root, regular-file root,
unwritable directory, `HF_HUB_CACHE` over `HF_HOME`, a symlink root, and a
malformed NUL-byte path. No raw model data, secrets, or arbitrary environment
values are recorded.

## Cleanup

No files were created under the real Hugging Face cache, no cache content was
downloaded, and no tmux session or GPU process was created, killed, or changed.
