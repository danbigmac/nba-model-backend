#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

PYTHON_BIN="${REPO_ROOT}/venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/refresh_top_scorers_logs.py" --sleep 2 >> "${LOG_DIR}/refresh_logs.log" 2>&1
