# Launch Asset Kit

This folder standardizes real recording assets for GitHub marketing, launch posts, and docs.

## Directory Structure

```text
Assets/LaunchKit/
  gif/
    source/      # raw screen recordings (.mov/.mp4)
    project/     # editor project files (CapCut/Figma/AE)
    cuts/        # trimmed clips before final encoding
    export/      # final GIF output for README/docs
    webm/        # lightweight web video alternative
    posters/     # poster/thumbnail images
    spec/        # shot list and scene script
  screenshots/
    raw/         # raw captures
    annotated/   # callout/arrow overlays
    final/       # final PNG/JPG used in docs/README
    prd-copy/    # PRD-style screenshot copy templates
  demo-data/     # sanitized seed data for recordings
```

## Naming Convention

- GIF: `gif_<seq>_<feature>_<v>.gif`
  - example: `gif_01_memory-stream_v1.gif`
- Source clip: `src_<seq>_<feature>_<date>.mov`
  - example: `src_01_memory-stream_2026-02-08.mov`
- Screenshot: `ss_<surface>_<state>_<seq>.png`
  - example: `ss_webviewer_search_results_01.png`

## Quality Baseline

- Capture resolution: 1440x900 or 1920x1080
- Export GIF max width: 1200px
- GIF target size: under 8 MB for README
- Keep each GIF 8-18 seconds, one message per GIF

## Recording Rules

- Use clean demo dataset (no personal names, keys, machine paths, or account identifiers)
- Browser/theme and terminal font should be consistent across all clips
- Disable noisy notifications before recording
- Keep cursor movements deliberate and slow enough for playback at 1x

## One-Click Demo Setup

```bash
bash Scripts/load_demo_data.sh --reset
```

## Batch Generation + Validation

```bash
bash Scripts/make_gifs.sh
python Scripts/validate_assets.py --check-readme --strict
```
