#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${WORLD_ARENA_ROOT:?Set WORLD_ARENA_ROOT to the official checkout}"
: "${N0_TRACK32_POLICY_CONFIG:?Set N0_TRACK32_POLICY_CONFIG to policy.json}"
: "${HUB_POLICY_URL:?Set HUB_POLICY_URL to the organizer URL ending in /policy}"
: "${POLICY_ID:?Set POLICY_ID to the organizer worker key}"

PYTHON_BIN="${PYTHON:-python}"
export PYTHONPATH="${REPO_ROOT}:${WORLD_ARENA_ROOT}"
export PYTHONNOUSERSITE=1
if [[ -n "${HUB_TOKEN:-}" ]]; then
  export WORLD_ARENA_HUB_TOKEN="${HUB_TOKEN}"
fi

"${PYTHON_BIN}" -m n0_twam.cli track32 bridge-patch \
  --worldarena-root "${WORLD_ARENA_ROOT}"

exec "${PYTHON_BIN}" -m n0_twam.cli track32 worker \
  --worldarena-root "${WORLD_ARENA_ROOT}" \
  --config "${N0_TRACK32_POLICY_CONFIG}" \
  --hub-url "${HUB_POLICY_URL}" \
  --worker-key "${POLICY_ID}" \
  "$@"
