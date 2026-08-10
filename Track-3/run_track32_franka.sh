#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/track32_franka_request.json" >&2
  exit 2
fi

exec python -m n0_twam.cli track32 train --config "$1"
