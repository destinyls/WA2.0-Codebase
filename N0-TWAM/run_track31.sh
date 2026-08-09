#!/usr/bin/env bash
set -euo pipefail

# Source-tree fallback for the installed `n0-twam` console command.
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -n "${N0_TWAM_PYTHON:-}" ]]; then
  PYTHON_BIN="${N0_TWAM_PYTHON}"
elif [[ -x "${SCRIPT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${SCRIPT_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

cd "${SCRIPT_ROOT}"
exec "${PYTHON_BIN}" -m n0_twam.cli track31 "$@"
