#!/usr/bin/env python3
"""
Redact sensitive text from screenshots using OCR + blur masking.

Requirements:
- pillow
- pytesseract
- tesseract binary installed
"""

from __future__ import annotations

import argparse
import pathlib
import re
from typing import List, Tuple

from PIL import Image, ImageFilter, ImageDraw

SENSITIVE_PATTERNS = [
    re.compile(r"(?i)^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$"),
    re.compile(r"(?i)^(?:github_pat_[a-z0-9_]+|gh[pousr]_[a-z0-9]+|sk-[a-z0-9]{10,})$"),
    re.compile(r"(?i)^(?:AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|AIza[0-9A-Za-z\-_]{10,})$"),
    re.compile(r"(?i)(token|secret|password|apikey|api_key|bearer)"),
    re.compile(r"^(?:[A-Za-z]:\\|/).+"),
]


def load_ocr_boxes(image: Image.Image) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    try:
        import pytesseract  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("pytesseract is required for redact_screenshot.py") from exc

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    out: List[Tuple[str, Tuple[int, int, int, int]]] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        if w <= 0 or h <= 0:
            continue
        out.append((text, (x, y, x + w, y + h)))
    return out


def should_redact(text: str) -> bool:
    token = text.strip()
    for pat in SENSITIVE_PATTERNS:
        if pat.search(token):
            return True
    return False


def expand_box(box: Tuple[int, int, int, int], pad: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


def redact_image(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    blur_radius: int,
    pad: int,
    debug: bool,
    fallback_mode: str,
) -> int:
    image = Image.open(input_path).convert("RGB")
    ocr_available = True
    try:
        boxes = load_ocr_boxes(image)
    except RuntimeError:
        ocr_available = False
        boxes = []

    redactions = [b for t, b in boxes if should_redact(t)]

    if not redactions and not ocr_available:
        width, height = image.size
        if fallback_mode == "none":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            return 0
        if fallback_mode == "full":
            redactions = [(0, 0, width, height)]
        else:
            redactions = [
                (0, 0, width, int(height * 0.24)),
                (0, int(height * 0.82), width, height),
                (0, int(height * 0.24), int(width * 0.22), int(height * 0.82)),
            ]

    if not redactions:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return 0

    width, height = image.size
    edited = image.copy()
    draw = ImageDraw.Draw(edited)

    for box in redactions:
        x1, y1, x2, y2 = expand_box(box, pad, width, height)
        region = edited.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(radius=blur_radius))
        edited.paste(region, (x1, y1, x2, y2))
        if debug:
            draw.rectangle((x1, y1, x2, y2), outline=(255, 64, 64), width=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    edited.save(output_path)
    return len(redactions)


def main() -> int:
    parser = argparse.ArgumentParser(description="Blur sensitive text in screenshots")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--blur-radius", type=int, default=12)
    parser.add_argument("--pad", type=int, default=6)
    parser.add_argument("--debug", action="store_true", help="Draw redaction boxes")
    parser.add_argument(
        "--fallback",
        choices=["bands", "full", "none"],
        default="bands",
        help="Fallback mode when OCR runtime is unavailable.",
    )
    args = parser.parse_args()

    input_path = pathlib.Path(args.input).resolve()
    output_path = pathlib.Path(args.output).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    count = redact_image(input_path, output_path, args.blur_radius, args.pad, args.debug, args.fallback)
    print(f"{{\"ok\": true, \"redactions\": {count}, \"output\": \"{output_path}\"}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
