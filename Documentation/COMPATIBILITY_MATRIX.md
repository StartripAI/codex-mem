# Compatibility Matrix

| Component | Supported | Notes |
|---|---|---|
| Python | 3.10+ | Core scripts and MCP server |
| SQLite | FTS5 required | Used for lexical retrieval |
| Codex MCP protocol | 2024-11-05 | `codex_mem_mcp.py` |
| Codex runtime | Current desktop/CLI | via MCP registration |
| OS | macOS/Linux | Windows works with path adjustments |
| Pillow | Optional but recommended | asset validation and redaction helpers |
| pytesseract+tesseract | Optional | required for OCR-based screenshot redaction |
| ffmpeg | Optional | required for `make_gifs.sh` |

## Known Limits

- Web viewer is local, unauthenticated, intended for loopback host.
- Hash-based local vectors are lightweight and deterministic, not model embeddings.
