# Launch Asset Playbook

This playbook defines exactly how to produce real GIF and screenshot assets for README, release notes, and distribution channels.

## 1) Asset Objectives

- Explain product value in under 20 seconds per visual
- Keep visual narrative aligned with retrieval workflow
- Keep all media privacy-safe and reusable

## 2) Canonical Asset Paths

- Root: `Assets/LaunchKit/`
- Shotlist: `Assets/LaunchKit/gif/spec/SHOTLIST_TEMPLATE.md`
- PRD screenshot copy template: `Assets/LaunchKit/screenshots/prd-copy/PRD_SCREENSHOT_COPY_TEMPLATE.md`

## 3) Required First Batch (v0.x launch)

1. `gif_01_memory-stream_v1.gif`
2. `gif_02_mem-search_v1.gif`
3. `gif_03_privacy-mode_v1.gif`
4. Hero screenshot PNG with callouts
5. mem-search screenshot PNG with copy
6. privacy screenshot PNG with copy

## 4) Recording Workflow

1. Prepare sanitized demo data.
2. Record raw clips into `gif/source/`.
3. Trim into `gif/cuts/`.
4. Export GIF and WEBM into `gif/export/` and `gif/webm/`.
5. Save poster images into `gif/posters/`.
6. Capture screenshots to `screenshots/raw/`, annotate into `screenshots/annotated/`, finalize in `screenshots/final/`.
7. Generate caption/callout text from PRD template.

## 5) Acceptance Checklist

- GIF length 8-18s and under 8 MB
- Key action visible in first 3 seconds
- Title, caption, and CTA match README vocabulary
- No private or personal information in pixels or text
- README links point to real files, not placeholders
