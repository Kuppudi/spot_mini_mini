#!/usr/bin/env python3
"""Generate the SpotMini development flowchart as PNG and SVG assets."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
PNG_PATH = ROOT / "spotmini_development_flow.png"
SVG_PATH = ROOT / "spotmini_development_flow.svg"

WIDTH = 1800
HEIGHT = 1000

COLORS = {
    "background": "#f8faf7",
    "ink": "#111111",
    "muted": "#5b6267",
    "baseline_fill": "#ffffff",
    "baseline_stroke": "#111111",
    "control_fill": "#d9f7df",
    "control_stroke": "#27864a",
    "perception_fill": "#d9edff",
    "perception_stroke": "#2a6fad",
    "ml_fill": "#ffe9bf",
    "ml_stroke": "#b66d00",
    "ui_fill": "#eadfff",
    "ui_stroke": "#6e46a6",
    "output_fill": "#edf3f5",
    "output_stroke": "#34434d",
}

BOX_STYLE = {
    "baseline": (COLORS["baseline_fill"], COLORS["baseline_stroke"]),
    "control": (COLORS["control_fill"], COLORS["control_stroke"]),
    "perception": (COLORS["perception_fill"], COLORS["perception_stroke"]),
    "ml": (COLORS["ml_fill"], COLORS["ml_stroke"]),
    "ui": (COLORS["ui_fill"], COLORS["ui_stroke"]),
    "output": (COLORS["output_fill"], COLORS["output_stroke"]),
}

NODES = [
    ("A", 45, 95, 240, 78, "Open spot_mini_mini repository", "baseline"),
    ("B", 350, 95, 270, 78, "Create environment and install requirements", "baseline"),
    ("C", 690, 95, 280, 78, "Start SpotMini development run", "baseline"),
    ("D", 1495, 95, 260, 78, "Tune parameters and re-test", "output"),
    ("F", 100, 285, 260, 82, "Load PyBullet world + SpotMini URDF", "baseline"),
    ("G", 100, 430, 260, 82, "Bezier gait creates foot trajectory", "baseline"),
    ("H", 100, 575, 260, 82, "SpotModel IK converts to 12 joint angles", "baseline"),
    ("I", 100, 720, 260, 82, "Apply motor commands in PyBullet", "baseline"),
    ("J", 520, 285, 260, 82, "Enhanced: smoothed keyboard input", "control"),
    ("K", 520, 430, 260, 82, "Enhanced: strafe, turn-in-place, speed control", "control"),
    ("L", 520, 575, 260, 82, "Enhanced: four-phase gait, auto-yaw, auto-reset", "control"),
    ("M", 520, 720, 260, 82, "Tune stride, swing period, clearance", "control"),
    ("N", 940, 285, 260, 82, "Enhanced: robot-mounted RGB/RGB-D camera", "perception"),
    ("O", 940, 430, 260, 82, "Enhanced: rough-terrain height features", "ml"),
    ("P", 940, 575, 260, 82, "Enhanced: per-leg swing-height residuals", "ml"),
    ("Q", 940, 720, 260, 82, "Train or compare ML policy", "ml"),
    ("R", 1360, 285, 260, 82, "Enhanced: launch local Streamlit dashboard", "ui"),
    ("S", 1360, 430, 260, 82, "Stream camera worker and simulation status", "ui"),
    ("T", 1360, 575, 260, 82, "Review plots, snapshots, and compare demos", "ui"),
    ("U", 675, 875, 450, 78, "Improved SpotMini simulation ready for testing and reporting", "output"),
]

CONNECTORS = [
    ("A", "B", "right", "left", ""),
    ("B", "C", "right", "left", ""),
    ("D", "C", "left", "right", "repeat after tuning"),
    ("C", "F", "bottom", "top", ""),
    ("C", "J", "bottom", "top", ""),
    ("C", "N", "bottom", "top", ""),
    ("C", "R", "bottom", "top", ""),
    ("F", "G", "bottom", "top", ""),
    ("G", "H", "bottom", "top", ""),
    ("H", "I", "bottom", "top", ""),
    ("J", "K", "bottom", "top", ""),
    ("K", "L", "bottom", "top", ""),
    ("L", "M", "bottom", "top", ""),
    ("N", "O", "bottom", "top", ""),
    ("O", "P", "bottom", "top", ""),
    ("P", "Q", "bottom", "top", ""),
    ("R", "S", "bottom", "top", ""),
    ("S", "T", "bottom", "top", ""),
    ("I", "U", "bottom", "left", ""),
    ("M", "U", "bottom", "top", ""),
    ("Q", "U", "bottom", "top", ""),
    ("T", "U", "bottom", "right", ""),
]

COLUMN_HEADERS = [
    ("Initial Design Path", 230, 240, "baseline"),
    ("Enhanced Control Path", 650, 240, "control"),
    ("Enhanced Perception + ML", 1070, 240, "ml"),
    ("Enhanced UI / Testing", 1490, 240, "ui"),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE_FONT = font(32, bold=True)
SUBTITLE_FONT = font(18)
LABEL_FONT = font(15, bold=True)
BODY_FONT = font(20)
SMALL_FONT = font(16)


def node_map() -> dict[str, tuple[int, int, int, int, str, str]]:
    return {node[0]: node[1:] for node in NODES}


def anchor(node: tuple[int, int, int, int, str, str], side: str) -> tuple[float, float]:
    x, y, w, h, _, _ = node
    anchors = {
        "left": (x, y + h / 2),
        "right": (x + w, y + h / 2),
        "top": (x + w / 2, y),
        "bottom": (x + w / 2, y + h),
    }
    return anchors[side]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font_obj: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), candidate, font=font_obj)
        if bbox[2] - bbox[0] <= width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = COLORS["ink"],
    width: int = 2,
    label: str = "",
) -> None:
    x1, y1 = start
    x2, y2 = end

    # Keep arrowheads outside the target boxes.
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy) or 1.0
    ux = dx / length
    uy = dy / length
    x2a = x2 - ux * 8
    y2a = y2 - uy * 8

    draw.line((x1, y1, x2a, y2a), fill=color, width=width)

    size = 11
    left_angle = math.atan2(dy, dx) + math.pi * 0.84
    right_angle = math.atan2(dy, dx) - math.pi * 0.84
    p1 = (x2a + size * math.cos(left_angle), y2a + size * math.sin(left_angle))
    p2 = (x2a + size * math.cos(right_angle), y2a + size * math.sin(right_angle))
    draw.polygon([(x2, y2), p1, p2], fill=color)

    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        lines = textwrap.wrap(label, width=24)
        text_h = len(lines) * 17
        text_w = max(draw.textbbox((0, 0), line, font=SMALL_FONT)[2] for line in lines)
        pad = 5
        draw.rounded_rectangle(
            (mx - text_w / 2 - pad, my - text_h / 2 - pad, mx + text_w / 2 + pad, my + text_h / 2 + pad),
            radius=8,
            fill=COLORS["background"],
        )
        for idx, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=SMALL_FONT)
            draw.text((mx - (bbox[2] - bbox[0]) / 2, my - text_h / 2 + idx * 17), line, fill=COLORS["muted"], font=SMALL_FONT)


def draw_node(draw: ImageDraw.ImageDraw, node: tuple[str, int, int, int, int, str, str]) -> None:
    _, x, y, w, h, text, style = node
    fill, stroke = BOX_STYLE[style]
    draw.rectangle((x, y, x + w, y + h), fill=fill, outline=stroke, width=2)

    text_color = COLORS["ink"]
    lines = wrap_text(draw, text, BODY_FONT, w - 28)
    line_height = 23
    total_height = len(lines) * line_height
    top = y + (h - total_height) / 2
    for idx, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=BODY_FONT)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (w - text_w) / 2, top + idx * line_height), line, fill=text_color, font=BODY_FONT)


def draw_legend(draw: ImageDraw.ImageDraw) -> None:
    legend_x = 810
    legend_y = 18
    items = [
        ("Initial design", "baseline"),
        ("Enhanced controls", "control"),
        ("Enhanced sensing", "perception"),
        ("Enhanced ML terrain", "ml"),
        ("Enhanced dashboard", "ui"),
    ]
    x = legend_x
    for label, style in items:
        fill, stroke = BOX_STYLE[style]
        draw.rectangle((x, legend_y + 8, x + 22, legend_y + 30), fill=fill, outline=stroke, width=2)
        draw.text((x + 30, legend_y + 7), label, fill=COLORS["ink"], font=SMALL_FONT)
        bbox = draw.textbbox((0, 0), label, font=SMALL_FONT)
        x += bbox[2] - bbox[0] + 72


def generate_png() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["background"])
    draw = ImageDraw.Draw(image)

    draw.text((45, 22), "SpotMini Developmental Flow Chart", fill=COLORS["ink"], font=TITLE_FONT)
    draw.text((45, 58), "Colored boxes indicate enhancements added beyond the initial SpotMini design.", fill=COLORS["muted"], font=SUBTITLE_FONT)
    draw_legend(draw)

    nodes = node_map()
    for src, dst, src_side, dst_side, label in CONNECTORS:
        draw_arrow(draw, anchor(nodes[src], src_side), anchor(nodes[dst], dst_side), label=label)

    for label, x, y, style in COLUMN_HEADERS:
        _, stroke = BOX_STYLE[style]
        bbox = draw.textbbox((0, 0), label, font=LABEL_FONT)
        draw.text((x - (bbox[2] - bbox[0]) / 2, y), label, fill=stroke, font=LABEL_FONT)

    for node in NODES:
        draw_node(draw, node)

    image.save(PNG_PATH)


def svg_text_lines(text: str, width_chars: int = 22) -> list[str]:
    return textwrap.wrap(text, width=width_chars)


def svg_node(node: tuple[str, int, int, int, int, str, str]) -> str:
    _, x, y, w, h, text, style = node
    fill, stroke = BOX_STYLE[style]
    lines = svg_text_lines(text, max(16, int(w / 11)))
    line_height = 21
    start_y = y + h / 2 - (len(lines) - 1) * line_height / 2 + 7
    tspans = []
    for idx, line in enumerate(lines):
        escaped = (
            line.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        tspans.append(f'<tspan x="{x + w / 2}" y="{start_y + idx * line_height:.1f}">{escaped}</tspan>')
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>\n'
        f'<text class="node-text" text-anchor="middle">{"".join(tspans)}</text>'
    )


def svg_connector(nodes: dict[str, tuple[int, int, int, int, str, str]], connector: tuple[str, str, str, str, str]) -> str:
    src, dst, src_side, dst_side, label = connector
    x1, y1 = anchor(nodes[src], src_side)
    x2, y2 = anchor(nodes[dst], dst_side)
    label_svg = ""
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        lines = svg_text_lines(label, 22)
        rect_w = max(85, max(len(line) for line in lines) * 8 + 14)
        rect_h = len(lines) * 18 + 10
        text_lines = []
        for idx, line in enumerate(lines):
            escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            text_lines.append(f'<tspan x="{mx}" y="{my - rect_h / 2 + 22 + idx * 18:.1f}">{escaped}</tspan>')
        label_svg = (
            f'<rect x="{mx - rect_w / 2:.1f}" y="{my - rect_h / 2:.1f}" width="{rect_w}" height="{rect_h}" rx="8" '
            f'fill="{COLORS["background"]}"/><text class="connector-label" text-anchor="middle">{"".join(text_lines)}</text>'
        )
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="arrow"/>{label_svg}'


def generate_svg() -> None:
    nodes = node_map()
    legend_items = [
        ("Initial design", "baseline"),
        ("Enhanced controls", "control"),
        ("Enhanced sensing", "perception"),
        ("Enhanced ML terrain", "ml"),
        ("Enhanced dashboard", "ui"),
    ]
    legend_parts = []
    lx = 810
    for label, style in legend_items:
        fill, stroke = BOX_STYLE[style]
        legend_parts.append(
            f'<rect x="{lx}" y="26" width="22" height="22" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            f'<text x="{lx + 30}" y="43" class="legend">{label}</text>'
        )
        lx += int(len(label) * 8.5) + 72

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
  <defs>
    <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">
      <path d="M2,2 L10,6 L2,10 Z" fill="{COLORS["ink"]}"/>
    </marker>
    <style>
      .title {{ font: 700 32px Arial, Helvetica, sans-serif; fill: {COLORS["ink"]}; }}
      .subtitle {{ font: 18px Arial, Helvetica, sans-serif; fill: {COLORS["muted"]}; }}
      .node-text {{ font: 20px Arial, Helvetica, sans-serif; fill: {COLORS["ink"]}; }}
      .legend {{ font: 16px Arial, Helvetica, sans-serif; fill: {COLORS["ink"]}; }}
      .column-header {{ font: 700 15px Arial, Helvetica, sans-serif; }}
      .connector-label {{ font: 16px Arial, Helvetica, sans-serif; fill: {COLORS["muted"]}; }}
      .arrow {{ stroke: {COLORS["ink"]}; stroke-width: 2; marker-end: url(#arrow); }}
    </style>
  </defs>
  <rect width="100%" height="100%" fill="{COLORS["background"]}"/>
  <text x="45" y="47" class="title">SpotMini Developmental Flow Chart</text>
  <text x="45" y="76" class="subtitle">Colored boxes indicate enhancements added beyond the initial SpotMini design.</text>
  {''.join(legend_parts)}
  {''.join(svg_connector(nodes, connector) for connector in CONNECTORS)}
  {''.join(f'<text x="{x}" y="{y + 13}" class="column-header" text-anchor="middle" fill="{BOX_STYLE[style][1]}">{label}</text>' for label, x, y, style in COLUMN_HEADERS)}
  {''.join(svg_node(node) for node in NODES)}
</svg>
'''
    SVG_PATH.write_text(svg, encoding="utf-8")


def main() -> None:
    generate_png()
    generate_svg()
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {SVG_PATH}")


if __name__ == "__main__":
    main()
