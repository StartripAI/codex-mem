#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT="${ROOT}/Scripts/codex_mem.py"
MCP_SCRIPT="${ROOT}/Scripts/codex_mem_mcp.py"
WEB_SCRIPT="${ROOT}/Scripts/codex_mem_web.py"
MAKE_GIFS_SCRIPT="${ROOT}/Scripts/make_gifs.sh"
VALIDATE_ASSETS_SCRIPT="${ROOT}/Scripts/validate_assets.py"
LOAD_DEMO_SCRIPT="${ROOT}/Scripts/load_demo_data.py"
REDACT_SCREENSHOT_SCRIPT="${ROOT}/Scripts/redact_screenshot.py"
SOCIAL_PACK_SCRIPT="${ROOT}/Scripts/generate_social_pack.py"
COMPARE_SEARCH_SCRIPT="${ROOT}/Scripts/compare_search_modes.py"
SNAPSHOT_DOCS_SCRIPT="${ROOT}/Scripts/snapshot_docs.sh"
PLACEHOLDER_SCRIPT="${ROOT}/Scripts/generate_placeholder_assets.py"

usage() {
  cat <<'EOF'
Usage:
  Scripts/codex_mem.sh init [--project NAME]
  Scripts/codex_mem.sh session-start <session_id> [--title T]
  Scripts/codex_mem.sh prompt <session_id> "<user prompt>" [--title T]
  Scripts/codex_mem.sh tool <session_id> <tool_name> "<tool output>" [--title T] [--file-path P] [--tag X] [--privacy-tag X] [--compact]
  Scripts/codex_mem.sh stop <session_id> [--title T] [--content C]
  Scripts/codex_mem.sh session-end <session_id> [--skip-summary]
  Scripts/codex_mem.sh search "<query>" [--limit N] [--session-id SID]
  Scripts/codex_mem.sh mem-search "<query>" [--limit N] [--session-id SID]
  Scripts/codex_mem.sh config-get
  Scripts/codex_mem.sh config-set [--channel stable|beta] [--viewer-refresh-sec N] [--beta-endless-mode on|off]
  Scripts/codex_mem.sh export-session <session_id> [--anonymize on|off] [--include-private] [--output PATH]
  Scripts/codex_mem.sh timeline <E123|O45> [--before N] [--after N]
  Scripts/codex_mem.sh get <E123|O45> [more IDs...]
  Scripts/codex_mem.sh ask "<question>" [ask-args...]
  Scripts/codex_mem.sh web [--host 127.0.0.1] [--port 37777] [--project-default NAME]
  Scripts/codex_mem.sh mcp [--project-default NAME]
  Scripts/codex_mem.sh load-demo-data [--reset]
  Scripts/codex_mem.sh make-gifs [--fps N] [--width N]
  Scripts/codex_mem.sh validate-assets [--check-readme] [--strict]
  Scripts/codex_mem.sh redact-screenshot <input> <output>
  Scripts/codex_mem.sh social-pack --version vX.Y.Z
  Scripts/codex_mem.sh compare-search [--project NAME]
  Scripts/codex_mem.sh snapshot-docs <version>
  Scripts/codex_mem.sh generate-placeholders

Environment overrides:
  ROOT         Repository root (default: parent of Scripts)
  PYTHON_BIN   Python executable (default: python3)
EOF
}

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Script not found: ${SCRIPT}" >&2
  exit 1
fi

cmd="${1:-help}"
shift || true

case "${cmd}" in
  init)
    exec "${PYTHON_BIN}" "${SCRIPT}" --root "${ROOT}" init "$@"
    ;;
  session-start)
    exec "${PYTHON_BIN}" "${SCRIPT}" --root "${ROOT}" session-start "$@"
    ;;
  prompt|user-prompt-submit)
    exec "${PYTHON_BIN}" "${SCRIPT}" --root "${ROOT}" user-prompt-submit "$@"
    ;;
  tool|post-tool-use)
    exec "${PYTHON_BIN}" "${SCRIPT}" --root "${ROOT}" post-tool-use "$@"
    ;;
  stop)
    exec "${PYTHON_BIN}" "${SCRIPT}" --root "${ROOT}" stop "$@"
    ;;
  session-end)
    exec "${PYTHON_BIN}" "${SCRIPT}" --root "${ROOT}" session-end "$@"
    ;;
  summarize-session)
    exec "${PYTHON_BIN}" "${SCRIPT}" --root "${ROOT}" summarize-session "$@"
    ;;
  search)
    exec "${PYTHON_BIN}" "${SCRIPT}" --root "${ROOT}" search "$@"
    ;;
  nl-search|mem-search)
    exec "${PYTHON_BIN}" "${SCRIPT}" --root "${ROOT}" nl-search "$@"
    ;;
  config-get)
    exec "${PYTHON_BIN}" "${SCRIPT}" --root "${ROOT}" config-get "$@"
    ;;
  config-set)
    exec "${PYTHON_BIN}" "${SCRIPT}" --root "${ROOT}" config-set "$@"
    ;;
  export-session)
    exec "${PYTHON_BIN}" "${SCRIPT}" --root "${ROOT}" export-session "$@"
    ;;
  timeline)
    exec "${PYTHON_BIN}" "${SCRIPT}" --root "${ROOT}" timeline "$@"
    ;;
  get|get-observations)
    exec "${PYTHON_BIN}" "${SCRIPT}" --root "${ROOT}" get-observations "$@"
    ;;
  ask)
    exec "${PYTHON_BIN}" "${SCRIPT}" --root "${ROOT}" ask "$@"
    ;;
  web)
    if [[ ! -f "${WEB_SCRIPT}" ]]; then
      echo "Web script not found: ${WEB_SCRIPT}" >&2
      exit 1
    fi
    exec "${PYTHON_BIN}" "${WEB_SCRIPT}" --root "${ROOT}" "$@"
    ;;
  mcp)
    if [[ ! -f "${MCP_SCRIPT}" ]]; then
      echo "MCP script not found: ${MCP_SCRIPT}" >&2
      exit 1
    fi
    exec "${PYTHON_BIN}" "${MCP_SCRIPT}" --root "${ROOT}" "$@"
    ;;
  load-demo-data)
    exec "${PYTHON_BIN}" "${LOAD_DEMO_SCRIPT}" --root "${ROOT}" "$@"
    ;;
  make-gifs)
    exec "${MAKE_GIFS_SCRIPT}" "$@"
    ;;
  validate-assets)
    exec "${PYTHON_BIN}" "${VALIDATE_ASSETS_SCRIPT}" --root "${ROOT}" "$@"
    ;;
  redact-screenshot)
    exec "${PYTHON_BIN}" "${REDACT_SCREENSHOT_SCRIPT}" "$@"
    ;;
  social-pack)
    exec "${PYTHON_BIN}" "${SOCIAL_PACK_SCRIPT}" --root "${ROOT}" "$@"
    ;;
  compare-search)
    exec "${PYTHON_BIN}" "${COMPARE_SEARCH_SCRIPT}" --root "${ROOT}" "$@"
    ;;
  snapshot-docs)
    exec "${SNAPSHOT_DOCS_SCRIPT}" "$@"
    ;;
  generate-placeholders)
    exec "${PYTHON_BIN}" "${PLACEHOLDER_SCRIPT}"
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
