#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT="${ROOT}/Scripts/repo_knowledge.py"

usage() {
  cat <<'EOF'
Usage:
  Scripts/repo_knowledge.sh index-local [extra-index-args...]
  Scripts/repo_knowledge.sh index-openai [extra-index-args...]
  Scripts/repo_knowledge.sh map [--limit N]
  Scripts/repo_knowledge.sh ask "your question" [query-args...]
  Scripts/repo_knowledge.sh prompt "your question" [prompt-args...]

Environment overrides:
  OPENAI_API_KEY       Required for index-openai
  OPENAI_EMBED_MODEL   Default: text-embedding-3-small
  OPENAI_EMBED_DIM     Default: 256
  PYTHON_BIN           Default: python3
  ROOT                 Default: current directory
EOF
}

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Script not found: ${SCRIPT}" >&2
  exit 1
fi

cmd="${1:-help}"
shift || true

case "${cmd}" in
  index-local)
    exec "${PYTHON_BIN}" "${SCRIPT}" --root "${ROOT}" index --embedding-provider local "$@"
    ;;
  index-openai)
    model="${OPENAI_EMBED_MODEL:-text-embedding-3-small}"
    dim="${OPENAI_EMBED_DIM:-256}"
    exec "${PYTHON_BIN}" "${SCRIPT}" --root "${ROOT}" index \
      --embedding-provider openai \
      --openai-model "${model}" \
      --openai-dimensions "${dim}" \
      "$@"
    ;;
  map)
    exec "${PYTHON_BIN}" "${SCRIPT}" --root "${ROOT}" map "$@"
    ;;
  ask|query)
    if [[ $# -lt 1 ]]; then
      echo "Missing question for ask/query." >&2
      usage
      exit 1
    fi
    exec "${PYTHON_BIN}" "${SCRIPT}" --root "${ROOT}" query "$@"
    ;;
  prompt)
    if [[ $# -lt 1 ]]; then
      echo "Missing question for prompt." >&2
      usage
      exit 1
    fi
    exec "${PYTHON_BIN}" "${SCRIPT}" --root "${ROOT}" prompt "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage
    exit 1
    ;;
esac
