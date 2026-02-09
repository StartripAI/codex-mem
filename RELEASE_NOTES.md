# Release Notes

## v0.3.3 - 2026-02-09

### Added
- English prompt playbook for multi-case usage:
  - `Documentation/PROMPT_PLAYBOOK_EN.md`
- scenario benchmark matrix:
  - `Documentation/benchmarks/scenario_savings_20260209.json`

### Changed
- README now links to the English prompt playbook and scenario-based token savings
- BENCHMARKS now includes cold-start, daily, and forensics savings snapshots

### Validation
- passed: `python3 Scripts/benchmark_marketing_claim.py` across scenario matrix
- passed: `python3 Scripts/codex_mem_smoketest.py --root .`

## v0.3.1 - 2026-02-08

### Removed
- deleted placeholder-only marketing media files from:
  - `Assets/LaunchKit/gif/export/`
  - `Assets/LaunchKit/gif/posters/`
  - `Assets/LaunchKit/screenshots/final/`
- removed placeholder generator script:
  - `Scripts/generate_placeholder_assets.py`
- removed placeholder generation command from shell wrapper:
  - `Scripts/codex_mem.sh generate-placeholders`

### Changed
- README no longer embeds synthetic Feature Tour media
- README now enforces real-media-only policy
- asset validator no longer fails when media is intentionally absent
  - it validates files only when real media exists

### Validation
- passed: `python3 Scripts/validate_assets.py --root . --check-readme --strict`
- passed: `python3 Scripts/codex_mem_smoketest.py --root .`

## v0.3.0 - 2026-02-08

### Added
- launch asset production toolkit:
  - `Assets/LaunchKit/` structured directories
  - GIF shotlist template
  - PRD screenshot copy template
  - launch asset playbook
- media and launch scripts:
  - `Scripts/make_gifs.sh`
  - `Scripts/validate_assets.py`
  - `Scripts/load_demo_data.py`
  - `Scripts/load_demo_data.sh`
  - `Scripts/redact_screenshot.py`
  - `Scripts/generate_social_pack.py`
  - `Scripts/compare_search_modes.py`
  - `Scripts/snapshot_docs.sh`
- MCP export tool: `mem_export_session`
- CLI export command: `export-session`
- CI asset gate workflow: `.github/workflows/asset-gate.yml`
- new docs:
  - `BENCHMARKS.md`
  - `SECURITY.md`
  - `CONTRIBUTING_ASSETS.md`
  - `Documentation/RELEASE_RHYTHM.md`
  - `Documentation/FAILURE_CASE_LIBRARY.md`
  - `Documentation/COMPATIBILITY_MATRIX.md`
  - `Documentation/FIRST_SCREEN_CONVERSION_SPEC.md`
  - `roadmap/public-roadmap.md`

### Changed
- README now uses local GIF assets (no remote placeholder links)
- README adds 30-second value path visual
- web viewer adds:
  - copy PRD caption button
  - recording guide mode with step navigation
- smoke test validates export session path and new MCP tool coverage

### Validation
- passed: `python3 Scripts/codex_mem_smoketest.py --root .`
- passed: `python3 Scripts/validate_assets.py --check-readme --strict`

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
  - feature GIF support
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
