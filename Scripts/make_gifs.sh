#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${INPUT_DIR:-${ROOT_DIR}/Assets/LaunchKit/gif/source}"
EXPORT_DIR="${EXPORT_DIR:-${ROOT_DIR}/Assets/LaunchKit/gif/export}"
WEBM_DIR="${WEBM_DIR:-${ROOT_DIR}/Assets/LaunchKit/gif/webm}"
POSTER_DIR="${POSTER_DIR:-${ROOT_DIR}/Assets/LaunchKit/gif/posters}"
FPS="${FPS:-12}"
WIDTH="${WIDTH:-1200}"
POSTER_AT="${POSTER_AT:-00:00:01}"
QUALITY_SCALE="${QUALITY_SCALE:-lanczos}"

usage() {
  cat <<USAGE
Usage:
  Scripts/make_gifs.sh [--input DIR] [--export DIR] [--webm DIR] [--posters DIR]
                      [--fps N] [--width N] [--poster-at HH:MM:SS]

Environment overrides:
  INPUT_DIR, EXPORT_DIR, WEBM_DIR, POSTER_DIR, FPS, WIDTH, POSTER_AT, QUALITY_SCALE
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT_DIR="$2"; shift 2 ;;
    --export) EXPORT_DIR="$2"; shift 2 ;;
    --webm) WEBM_DIR="$2"; shift 2 ;;
    --posters) POSTER_DIR="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --poster-at) POSTER_AT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required. Install ffmpeg then rerun." >&2
  exit 1
fi

mkdir -p "$INPUT_DIR" "$EXPORT_DIR" "$WEBM_DIR" "$POSTER_DIR"

shopt -s nullglob
inputs=(
  "$INPUT_DIR"/*.mp4
  "$INPUT_DIR"/*.mov
  "$INPUT_DIR"/*.mkv
  "$INPUT_DIR"/*.webm
)
if [[ ${#inputs[@]} -eq 0 ]]; then
  echo "No source clips found in: $INPUT_DIR" >&2
  exit 1
fi

for src in "${inputs[@]}"; do
  base="$(basename "$src")"
  stem="${base%.*}"
  gif_out="$EXPORT_DIR/${stem}.gif"
  webm_out="$WEBM_DIR/${stem}.webm"
  poster_out="$POSTER_DIR/${stem}.png"
  palette="$(mktemp -t codexmem_palette_XXXXXX.png)"

  echo "[make_gifs] Processing: $base"

  ffmpeg -y -i "$src" \
    -vf "fps=${FPS},scale=${WIDTH}:-1:flags=${QUALITY_SCALE}" \
    -c:v libvpx-vp9 -crf 34 -b:v 0 -an "$webm_out" >/dev/null 2>&1

  ffmpeg -y -i "$src" \
    -vf "fps=${FPS},scale=${WIDTH}:-1:flags=${QUALITY_SCALE},palettegen" \
    "$palette" >/dev/null 2>&1

  ffmpeg -y -i "$src" -i "$palette" \
    -lavfi "fps=${FPS},scale=${WIDTH}:-1:flags=${QUALITY_SCALE}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5" \
    "$gif_out" >/dev/null 2>&1

  ffmpeg -y -ss "$POSTER_AT" -i "$src" -vframes 1 \
    -vf "scale=${WIDTH}:-1:flags=${QUALITY_SCALE}" \
    "$poster_out" >/dev/null 2>&1

  rm -f "$palette"
  echo "[make_gifs] -> $gif_out"
  echo "[make_gifs] -> $webm_out"
  echo "[make_gifs] -> $poster_out"
done

echo "[make_gifs] Completed ${#inputs[@]} clip(s)."
