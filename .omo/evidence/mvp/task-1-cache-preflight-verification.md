# Task 1 cache preflight direct verification

Verified at `2026-09-08T15:15:21+09:00` in `/home/kyu/BigDB_Proj`.

## Commands and binary observables

- `uv sync --frozen` -> `Checked 20 packages in 0.33ms`.
- `uv run ruff check .` -> `All checks passed!`.
- `uv run ruff format --check .` -> `23 files already formatted`.
- `uv run basedpyright` -> `0 errors, 0 warnings, 0 notes`.
- `uv run pytest -q` -> `13 passed in 0.32s`.
- `uv run pytest tests/test_doctor.py -q` -> `10 passed in 0.10s`.
- `git diff --check` -> no output and exit 0.
- Pure source LOC: `cache_probe.py=171`, `config.py=12`,
  `doctor.py=127`, `gpu_probe.py=152`.

## Live CLI evidence

Command: `uv run bigdb-mvp doctor --json`

- `gpu_ready=false`, `primary_reason_code=nvml_unavailable`.
- `nvml_status=unavailable`, with bounded diagnostic:
  `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.`
- `model_cache.repo_id=FacebookAI/xlm-roberta-base`.
- `model_cache.cache_root=/home/kyu/.cache/huggingface/hub`.
- `model_cache.exists=true`, `is_directory=true`, `writable=true`.
- `model_cache.repo_cache_exists=false`,
  `model_cache.model_snapshot_cached=false`.
- Cache reasons are `model_repo_cache_missing` and
  `model_snapshot_missing`; they are nested separately from GPU
  `reason_codes` and do not change `gpu_ready`.

## Adversarial coverage

The targeted tests directly exercise populated matching snapshot, missing root,
regular-file root, unwritable directory, `HF_HUB_CACHE` precedence over
`HF_HOME`, symlink root rejection, and malformed NUL-byte path handling.
The probe performs no writes and bounds snapshot enumeration at 64 entries.

## Runtime/session and scope facts

`tmux list-sessions` returned no sessions, so no `bigdb_mvp` session was
created, killed, or changed during this verification. `CUDA_VISIBLE_DEVICES`
was unset in the shell invoking the live doctor. Existing staged files remain
untouched; no commit was attempted. Raw data, weights, user input, and secrets
were not written to the evidence.
