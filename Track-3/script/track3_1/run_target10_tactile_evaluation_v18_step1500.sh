#!/usr/bin/env bash
set -euo pipefail

# Wrapper for the v18 1500-step Target-10 evaluation entry.
# Use a single command; by default it runs with the frozen v18 recipe.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ENTRY="${SCRIPT_DIR}/run_target10_tactile_evaluation_v18_step1500.py"
PYTHON_BIN="${N0_TWAM_PYTHON:-python3}"

# Optional immutable request. Without it, the fixed v18 defaults are used.
REQUEST_JSON="${N0_TWAM_V18_TARGET10_REQUEST_JSON:-}"

if [[ ! -f "${PYTHON_ENTRY}" ]]; then
  echo "找不到入口脚本：${PYTHON_ENTRY}" >&2
  exit 2
fi

if [[ "$#" -eq 0 ]]; then
  if [[ -z "${REQUEST_JSON}" ]]; then
    echo "请先设置 N0_TWAM_V18_TARGET10_REQUEST_JSON 指向已审核的 immutable request。" >&2
    exit 2
  fi
  if [[ ! -f "${REQUEST_JSON}" ]]; then
    echo "请求文件不存在：${REQUEST_JSON}" >&2
    exit 2
  fi
  "${PYTHON_BIN}" "${PYTHON_ENTRY}" --request "${REQUEST_JSON}"
  exit 0
fi

if [[ "$1" == "--print-template" || "$1" == "--print-request-template" ]]; then
  "${PYTHON_BIN}" "${PYTHON_ENTRY}" "$@"
  exit 0
fi

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  "${PYTHON_BIN}" "${PYTHON_ENTRY}" "$@"
  echo
  echo "这个 shell 封装支持单命令复用："
  echo "  ./run_target10_tactile_evaluation_v18_step1500.sh"
  echo "  （必须先设置 N0_TWAM_V18_TARGET10_REQUEST_JSON）"
  echo
  echo "如果你希望把当前默认请求模板落盘，可手动执行："
  echo "  ${PYTHON_BIN} ${PYTHON_ENTRY} --print-template > target10_request.json"
  exit 0
fi

if [[ "$1" == "--request" && "$#" -eq 1 ]]; then
  echo "请提供 --request 的 JSON 路径" >&2
  exit 2
fi

if [[ "$#" -gt 0 ]]; then
  "${PYTHON_BIN}" "${PYTHON_ENTRY}" "$@"
  exit 0
fi
