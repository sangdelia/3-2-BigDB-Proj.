# Task 1 hook-triggered verification receipt

Verified at `2026-09-08T15:17:07+09:00` from the current worktree.

## Current-state checks

- `omo ulw-loop status --json` reports `ULW_LOOP_PLAN_MISSING`; therefore no
  current attempt directory exists. This receipt is stored under the required
  fallback `.omo/evidence/` tree.
- All implementation files and both cache receipts are present.
- `python -m json.tool .omo/evidence/mvp/task-1-gpu.json` exits 0.
- `git diff --check` exits 0.
- Pure source LOC remains below 250: cache probe 171, config 12, doctor 127,
  GPU probe 152.

## Fresh commands

- `uv sync --frozen` -> `Checked 20 packages in 0.33ms`.
- `uv run ruff check .` -> `All checks passed!`.
- `uv run ruff format --check .` -> `23 files already formatted`.
- `uv run basedpyright` -> `0 errors, 0 warnings, 0 notes`.
- `uv run pytest -q` -> `13 passed in 0.32s`.
- `uv run pytest tests/test_doctor.py -q` -> `10 passed in 0.10s`.

The first schema-pipeline probe had a shell quoting error in its f-string
formatting. It was not counted as a successful check; the corrected command
then produced `doctor_schema_ok` and passed all assertions.

## Fresh runtime assertions

Corrected command: `uv run bigdb-mvp doctor --json` piped through a JSON
schema assertion script produced:

`doctor_schema_ok`

`gpu_ready=False cache_snapshot=False cache_reasons=['model_repo_cache_missing', 'model_snapshot_missing']`

The script asserted the pinned repository ID, typed boolean/path fields,
separate cache reason codes, and `gpu_ready=false`.

A live stale-state probe produced:

`stale_before=False stale_after=True`

after adding a temporary populated snapshot between two reads. Explicit config
precedence produced `explicit_source=explicit`, while the existing pytest
fixture covers `HF_HUB_CACHE` over `HF_HOME`.

No real cache files were changed, no model was downloaded, and no tmux session
or GPU process was started, stopped, or altered. The current host still has no
`bigdb_mvp` tmux session and `CUDA_VISIBLE_DEVICES` is unset in the shell.
