# Release Rhythm

This project uses a fixed launch bundle for every public release.

## Required Bundle

1. Update `RELEASE_NOTES.md`
2. Publish 3 GIFs:
   - `gif_01_memory-stream_v1.gif`
   - `gif_02_mem-search_v1.gif`
   - `gif_03_privacy-mode_v1.gif`
3. Publish 3 screenshots in `Assets/LaunchKit/screenshots/final/`
4. Refresh `## Comparison Table` in `README.md`

## Process

1. Load sanitized demo data
   - `bash Scripts/load_demo_data.sh --reset`
2. Record and export media
   - `bash Scripts/make_gifs.sh`
3. Validate assets
   - `python Scripts/validate_assets.py --check-readme --strict`
4. Snapshot docs/media for version
   - `bash Scripts/snapshot_docs.sh <version>`
5. Generate social copy pack
   - `python Scripts/generate_social_pack.py --version <version>`

## Gate Rules

- No release if asset validation fails
- No release if README GIF links are remote or missing
- No release if privacy scan fails
