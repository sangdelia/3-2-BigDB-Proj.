# Task 1 GPU owner-query safety fix

## Red-to-green proof

- Red, run twice before the fix:
  `uv run pytest tests/test_doctor.py::test_collect_nvml_probe_returns_unavailable_owner_count_when_owner_lookup_fails -q`
  failed because the injected failed `id -un` query produced
  `foreign_compute_process_count=0` instead of `None`.
- Green after the fix:
  `uv run pytest tests/test_doctor.py tests/test_project_layout.py -q`
  produced `6 passed`.
- The current-user failure regression verifies an otherwise-healthy report is
  not ready and has `primary_reason_code=gpu_process_query_unavailable`.
- The owner failure regression verifies the parsed compute PID invokes
  `ps -o user= -p 4321` and failed owner lookup yields `None`.

## Final verification

- `uv sync --frozen`: `Checked 20 packages`
- ruff check and format: pass
- basedpyright: `0 errors, 0 warnings, 0 notes`
- no-excuse checker: `no violations in 6 file(s)`
- Pure source LOC: config 11, doctor 131, gpu_probe 162, CLI 28.
- Invalid `--device -1` is rejected by the CLI range validator.

## Live diagnostics

- Local CLI reports `gpu_ready=false`, `nvml_status=unavailable`, and retains
  the exact sanitized stdout diagnostic: `NVIDIA-SMI has failed because it
  couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA
  driver is installed and running.`
- The retained `bigdb_mvp` tmux session runs with `CUDA_VISIBLE_DEVICES=0`.
  `task-1-gpu.json` records `gpu_ready=false`,
  `nvml_status=driver_library_mismatch`, and its bounded diagnostic.

## Cleanup receipt

- Debug journal and both temporary nvidia-smi capture files were removed.
- The temporary `.git/info/exclude` entry for `.debug-journal.md` was removed.
- No tmux session or GPU process was terminated.
