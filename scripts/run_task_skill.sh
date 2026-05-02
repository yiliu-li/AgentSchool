#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
TASKS_ROOT="${PROJECT_ROOT}/skillsbench/tasks"

if [[ $# -lt 1 ]]; then
  echo "Usage:"
  echo "  scripts/run_task_skill.sh <task-name-or-path> [extra agentschool args...]"
  echo
  echo "Example:"
  echo "  scripts/run_task_skill.sh find-topk-similiar-chemicals"
  exit 1
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set."
  echo "Example:"
  echo "  export OPENROUTER_API_KEY=your_key_here"
  exit 1
fi

TASK_INPUT="$1"
shift

if [[ -d "${TASK_INPUT}" ]]; then
  TASK_PATH="$(cd "${TASK_INPUT}" && pwd)"
else
  TASK_PATH="${TASKS_ROOT}/${TASK_INPUT}"
fi

if [[ ! -d "${TASK_PATH}" ]]; then
  echo "Task directory not found: ${TASK_PATH}"
  exit 1
fi

if [[ ! -f "${TASK_PATH}/instruction.md" ]]; then
  echo "instruction.md not found in task directory: ${TASK_PATH}"
  exit 1
fi

cd "${REPO_ROOT}"

if command -v agentschool >/dev/null 2>&1; then
  exec agentschool learn \
    --task "${TASK_PATH}" \
    --root .agentschool/learn \
    "$@"
fi

exec env PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python -m agentschool \
  learn \
  --task "${TASK_PATH}" \
  --root .agentschool/learn \
  "$@"
