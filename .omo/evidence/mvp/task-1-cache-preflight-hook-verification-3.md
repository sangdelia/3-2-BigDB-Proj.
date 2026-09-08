# Task 1 third direct verification receipt

Run timestamp: `2026-09-08T15:18:00+09:00`.

## Filesystem evidence

All implementation files, `task-1-gpu.json`, and prior cache receipts were
non-empty and present. `python -m json.tool .omo/evidence/mvp/task-1-gpu.json`
and `git diff --check` both exited successfully. Pure source LOC stayed under
the 250-line limit: cache probe 171, config 12, doctor 127, GPU probe 152.

`omo ulw-loop status --json` returned `ULW_LOOP_PLAN_MISSING` for the expected
attempt ID, so there is no valid current-attempt directory. This receipt is
therefore under `.omo/evidence/mvp/`, the documented fallback location.

## Fresh quality gates

- `uv sync --frozen` -> `Checked 20 packages in 0.35ms`.
- `uv run ruff check .` -> `All checks passed!`.
- `uv run ruff format --check .` -> `25 files already formatted`.
- `uv run basedpyright` -> `0 errors, 0 warnings, 0 notes`.
- `uv run pytest -q` -> `13 passed in 0.31s`.
- `uv run pytest tests/test_doctor.py -q` -> `10 passed in 0.09s`.

## Fresh live doctor evidence

`uv run bigdb-mvp doctor --json` returned `gpu_ready=false`,
`primary_reason_code=nvml_unavailable`, and `nvml_status=unavailable` with the
bounded NVIDIA-SMI communication diagnostic. Its nested model cache returned:

- repo ID `FacebookAI/xlm-roberta-base`;
- default root `/home/kyu/.cache/huggingface/hub`;
- `exists=true`, `is_directory=true`, `writable=true`;
- `repo_cache_exists=false`, `model_snapshot_cached=false`;
- reasons `model_repo_cache_missing` and `model_snapshot_missing`.

The corrected pipe assertion printed `doctor_schema_ok` after asserting the
typed booleans, pinned repo ID, GPU false state, and disjoint cache/GPU reason
codes.

## Fresh behavior probes

An isolated temporary-cache run printed:

`before=False after=True explicit=explicit env=HF_HUB_CACHE`

The first probe observed no snapshot; after a temporary matching snapshot was
created, the second probe observed one. The same run confirmed explicit config
beats environment selection and `HF_HUB_CACHE` beats `HF_HOME`. Existing tests
cover file, permission, symlink, and malformed-path cases.

## Runtime boundary

`tmux list-sessions` returned no sessions. No tmux session or GPU process was
created, stopped, or modified. No real Hugging Face cache content was written
or downloaded, and no raw data, user input, secrets, or model weights were
recorded.
