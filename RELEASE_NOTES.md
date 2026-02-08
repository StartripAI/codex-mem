# Release Notes

## v0.2.0 - 2026-02-08

### Added
- natural-language retrieval commands:
  - `nl-search`
  - `mem-search` alias
- local web viewer:
  - `Scripts/codex_mem_web.py`
  - endpoints for stream, sessions, NL search, and runtime config
- dual-tag privacy policy support in `post-tool-use`:
  - semantic tags: `--tag`
  - privacy policy tags: `--privacy-tag`
- runtime config APIs:
  - `config-get`
  - `config-set`
  - MCP equivalents: `mem_config_get`, `mem_config_set`
- new MCP retrieval tool:
  - `mem_nl_search`

### Changed
- README redesigned for GitHub first-screen marketing:
  - badges
  - feature GIF placeholders
  - comparison table
  - release notes entry point
- added launch asset production kit:
  - `Documentation/LAUNCH_ASSET_PLAYBOOK.md`
  - `Assets/LaunchKit/` structure
  - GIF shotlist + PRD screenshot copy templates
- docs expanded and aligned with current command surface:
  - operational guide
  - architecture
  - MCP tools
  - troubleshooting
  - installation
- shell wrapper enhanced with:
  - `mem-search`
  - `config-get`
  - `config-set`
  - `web`

### Fixed
- runtime config persistence bug:
  - default config initialization no longer overwrites user-updated values
- redaction regex replacement bug under Python 3.14

### Validation
- passed: `python3 -m py_compile` for all core scripts
- passed: `python3 Scripts/codex_mem_smoketest.py --root .`
- smoke test verifies:
  - lifecycle capture
  - NL search
  - privacy block/private filtering
  - runtime config updates
  - web API flow
  - MCP tool calls

## v0.1.0 - 2026-02-08

### Initial public baseline
- Codex local memory engine (`codex_mem.py`)
- MCP server integration (`codex_mem_mcp.py`)
- progressive retrieval (`search`, `timeline`, `get-observations`)
- fused retrieval (`ask`) with repository context
- skill package and initial docs
