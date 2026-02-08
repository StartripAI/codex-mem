#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  echo "Usage: Scripts/snapshot_docs.sh <version>" >&2
  exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SNAPSHOT_DIR="${ROOT_DIR}/Snapshots/${VERSION}_${STAMP}"

mkdir -p "$SNAPSHOT_DIR"
mkdir -p "$SNAPSHOT_DIR/assets"

cp "$ROOT_DIR/README.md" "$SNAPSHOT_DIR/README.md"
cp "$ROOT_DIR/RELEASE_NOTES.md" "$SNAPSHOT_DIR/RELEASE_NOTES.md"
cp "$ROOT_DIR/Documentation/LAUNCH_ASSET_PLAYBOOK.md" "$SNAPSHOT_DIR/LAUNCH_ASSET_PLAYBOOK.md"

if [[ -d "$ROOT_DIR/Assets/LaunchKit/gif/export" ]]; then
  cp -R "$ROOT_DIR/Assets/LaunchKit/gif/export" "$SNAPSHOT_DIR/assets/gif_export"
fi
if [[ -d "$ROOT_DIR/Assets/LaunchKit/gif/posters" ]]; then
  cp -R "$ROOT_DIR/Assets/LaunchKit/gif/posters" "$SNAPSHOT_DIR/assets/gif_posters"
fi
if [[ -d "$ROOT_DIR/Assets/LaunchKit/screenshots/final" ]]; then
  cp -R "$ROOT_DIR/Assets/LaunchKit/screenshots/final" "$SNAPSHOT_DIR/assets/screenshots_final"
fi

cat > "$SNAPSHOT_DIR/MANIFEST.json" <<MANIFEST
{
  "version": "$VERSION",
  "captured_at_utc": "$STAMP",
  "source_repo": "codex-mem",
  "included": [
    "README.md",
    "RELEASE_NOTES.md",
    "LAUNCH_ASSET_PLAYBOOK.md",
    "assets/gif_export",
    "assets/gif_posters",
    "assets/screenshots_final"
  ]
}
MANIFEST

echo "[snapshot_docs] Created: $SNAPSHOT_DIR"
