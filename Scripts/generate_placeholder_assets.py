#!/usr/bin/env python3
"""Generate local placeholder media files for README and validation gates."""

from __future__ import annotations

import pathlib
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


def center_text(draw: ImageDraw.ImageDraw, text: str, width: int, height: int) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    return ((width - tw) // 2, (height - th) // 2)


def make_animated_gif(path: pathlib.Path, title: str, color_a: Tuple[int, int, int], color_b: Tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames: List[Image.Image] = []
    for i in range(10):
        ratio = i / 9.0
        r = int(color_a[0] * (1 - ratio) + color_b[0] * ratio)
        g = int(color_a[1] * (1 - ratio) + color_b[1] * ratio)
        b = int(color_a[2] * (1 - ratio) + color_b[2] * ratio)
        frame = Image.new("RGB", (1200, 640), (r, g, b))
        draw = ImageDraw.Draw(frame)
        text = f"{title}\n(placeholder)"
        x, y = center_text(draw, text, 1200, 640)
        draw.text((x, y), text, fill=(255, 255, 255))
        frames.append(frame)
    frames[0].save(path, save_all=True, append_images=frames[1:], optimize=False, duration=180, loop=0)


def make_poster(path: pathlib.Path, title: str, bg: Tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1200, 640), bg)
    draw = ImageDraw.Draw(image)
    x, y = center_text(draw, title, 1200, 640)
    draw.text((x, y), title, fill=(255, 255, 255))
    image.save(path)


def make_screenshot(path: pathlib.Path, title: str, bg: Tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1400, 900), bg)
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 80, 1320, 820), outline=(230, 230, 230), width=2)
    draw.text((120, 120), title, fill=(245, 245, 245))
    draw.text((120, 170), "Placeholder screenshot for launch assets", fill=(210, 210, 210))
    image.save(path)


def make_value_path_svg(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = """<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1200\" height=\"260\" viewBox=\"0 0 1200 260\">
  <rect width=\"1200\" height=\"260\" fill=\"#0f172a\"/>
  <text x=\"24\" y=\"38\" fill=\"#e2e8f0\" font-size=\"24\">30-Second Value Path</text>
  <g font-size=\"18\" fill=\"#f8fafc\" stroke=\"#22c55e\" stroke-width=\"2\">
    <rect x=\"40\" y=\"90\" width=\"220\" height=\"90\" rx=\"12\" fill=\"#1e293b\"/>
    <text x=\"90\" y=\"145\">Install</text>
    <rect x=\"330\" y=\"90\" width=\"220\" height=\"90\" rx=\"12\" fill=\"#1e293b\"/>
    <text x=\"385\" y=\"145\">mem-search</text>
    <rect x=\"620\" y=\"90\" width=\"220\" height=\"90\" rx=\"12\" fill=\"#1e293b\"/>
    <text x=\"690\" y=\"145\">timeline</text>
    <rect x=\"910\" y=\"90\" width=\"220\" height=\"90\" rx=\"12\" fill=\"#1e293b\"/>
    <text x=\"995\" y=\"145\">ask</text>
  </g>
  <g stroke=\"#22c55e\" stroke-width=\"3\" fill=\"none\">
    <line x1=\"260\" y1=\"135\" x2=\"330\" y2=\"135\"/>
    <line x1=\"550\" y1=\"135\" x2=\"620\" y2=\"135\"/>
    <line x1=\"840\" y1=\"135\" x2=\"910\" y2=\"135\"/>
  </g>
</svg>\n"""
    path.write_text(svg, encoding="utf-8")


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    gifs = [
        ("gif_01_memory-stream_v1.gif", "Memory Stream + Timeline", (10, 124, 102), (14, 116, 144)),
        ("gif_02_mem-search_v1.gif", "Natural Language mem-search", (15, 118, 110), (37, 99, 235)),
        ("gif_03_privacy-mode_v1.gif", "Privacy + Runtime Mode", (15, 23, 42), (17, 94, 89)),
    ]
    for filename, title, c1, c2 in gifs:
        gif_path = root / "Assets/LaunchKit/gif/export" / filename
        make_animated_gif(gif_path, title, c1, c2)
        make_poster(root / "Assets/LaunchKit/gif/posters" / f"{gif_path.stem}.png", title, c1)

    make_screenshot(root / "Assets/LaunchKit/screenshots/final/ss_hero_overview_01.png", "Hero Overview", (25, 40, 70))
    make_screenshot(root / "Assets/LaunchKit/screenshots/final/ss_mem-search_results_01.png", "mem-search Results", (15, 94, 89))
    make_screenshot(root / "Assets/LaunchKit/screenshots/final/ss_privacy_policy_01.png", "Dual-Tag Privacy", (30, 58, 95))
    make_value_path_svg(root / "Assets/LaunchKit/screenshots/final/value-path-30s.svg")

    print("{\"ok\": true, \"generated\": \"placeholder launch assets\"}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
