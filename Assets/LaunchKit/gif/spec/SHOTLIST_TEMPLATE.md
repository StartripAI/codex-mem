# GIF Shotlist Template

Use this template for each release batch.

## Batch Meta

- Release: `v0.x.y`
- Owner:
- Record date:
- Product state (stable/beta):

## Shot List

### GIF 01 - Memory Stream + Timeline

- Objective: show progressive retrieval from search to timeline
- Length: 10-14s
- Script:
  1. open local viewer
  2. run mem-search query
  3. open selected record timeline
  4. show compact-to-detail flow
- CTA overlay text: `Find context without context stuffing`
- Output:
  - `Assets/LaunchKit/gif/export/gif_01_memory-stream_v1.gif`
  - `Assets/LaunchKit/gif/posters/gif_01_memory-stream_v1.png`

### GIF 02 - Natural Language mem-search

- Objective: show natural query -> interpreted time filter -> results
- Length: 8-12s
- Script:
  1. type: "what bugs were fixed this week"
  2. submit and show interpreted query block
  3. highlight result list and confidence order
- CTA overlay text: `Ask memory in plain language`
- Output:
  - `Assets/LaunchKit/gif/export/gif_02_mem-search_v1.gif`
  - `Assets/LaunchKit/gif/posters/gif_02_mem-search_v1.png`

### GIF 03 - Dual-Tag Privacy + Runtime Mode

- Objective: show privacy policy and stable/beta config switches
- Length: 12-18s
- Script:
  1. capture tool output with `--privacy-tag private --privacy-tag redact`
  2. show default search excludes private item
  3. switch config to beta endless mode
  4. show compacted event output
- CTA overlay text: `Control memory visibility and cost behavior`
- Output:
  - `Assets/LaunchKit/gif/export/gif_03_privacy-mode_v1.gif`
  - `Assets/LaunchKit/gif/posters/gif_03_privacy-mode_v1.png`
