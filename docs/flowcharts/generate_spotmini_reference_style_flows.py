#!/usr/bin/env python3
"""Generate SpotMini flowcharts in the same simple style as the examples."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent

TECHNICAL_PNG = ROOT / "spotmini_design_flow_technical_reference_style.png"
OPERATIONAL_PNG = ROOT / "spotmini_operational_flow_reference_style.png"

WHITE = "#ffffff"
BLACK = "#000000"
ORANGE = "#ff9b3d"
RED = "#ff130d"


def get_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT = get_font(12)
SMALL_FONT = get_font(11)
KEY_FONT = get_font(12)


def wrapped_lines(draw: ImageDraw.ImageDraw, text: str, width: int, font: ImageFont.ImageFont = FONT) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont = FONT,
    width_pad: int = 8,
) -> None:
    x1, y1, x2, y2 = box
    lines = wrapped_lines(draw, text, x2 - x1 - width_pad * 2, font)
    line_height = 15
    start_y = y1 + ((y2 - y1) - line_height * len(lines)) / 2
    for index, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        draw.text(
            (x1 + ((x2 - x1) - (bbox[2] - bbox[0])) / 2, start_y + index * line_height),
            line,
            fill=BLACK,
            font=font,
        )


def rect(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    text: str,
    fill: str = WHITE,
) -> tuple[int, int, int, int]:
    box = (x, y, x + w, y + h)
    draw.rectangle(box, fill=fill, outline=BLACK, width=1)
    center_text(draw, box, text)
    return box


def round_rect(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    text: str,
    fill: str = WHITE,
) -> tuple[int, int, int, int]:
    box = (x, y, x + w, y + h)
    draw.rounded_rectangle(box, radius=6, fill=fill, outline=BLACK, width=1)
    center_text(draw, box, text)
    return box


def diamond(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    text: str,
    fill: str = WHITE,
) -> tuple[int, int, int, int]:
    points = [(x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h), (x, y + h / 2)]
    draw.polygon(points, fill=fill, outline=BLACK)
    box = (x + 9, y + 12, x + w - 9, y + h - 12)
    center_text(draw, box, text, SMALL_FONT, width_pad=0)
    return (x, y, x + w, y + h)


def anchor(box: tuple[int, int, int, int], side: str) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    if side == "top":
        return ((x1 + x2) / 2, y1)
    if side == "bottom":
        return ((x1 + x2) / 2, y2)
    if side == "left":
        return (x1, (y1 + y2) / 2)
    if side == "right":
        return (x2, (y1 + y2) / 2)
    raise ValueError(side)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str = "",
    dashed: bool = False,
) -> None:
    x1, y1 = start
    x2, y2 = end
    if dashed:
        draw_dashed_line(draw, (x1, y1), (x2, y2))
    else:
        draw.line((x1, y1, x2, y2), fill=BLACK, width=1)

    angle = math.atan2(y2 - y1, x2 - x1)
    size = 7
    p1 = (x2 - size * math.cos(angle - math.pi / 6), y2 - size * math.sin(angle - math.pi / 6))
    p2 = (x2 - size * math.cos(angle + math.pi / 6), y2 - size * math.sin(angle + math.pi / 6))
    draw.polygon([(x2, y2), p1, p2], fill=BLACK)

    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        bbox = draw.textbbox((0, 0), label, font=SMALL_FONT)
        draw.rectangle((mx - 3, my - 11, mx + bbox[2] - bbox[0] + 6, my + 4), fill=WHITE)
        draw.text((mx, my - 10), label, fill=BLACK, font=SMALL_FONT)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    dash_length: int = 3,
    gap: int = 4,
) -> None:
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux = dx / length
    uy = dy / length
    distance = 0.0
    while distance < length:
        start_d = distance
        end_d = min(distance + dash_length, length)
        draw.line(
            (
                x1 + ux * start_d,
                y1 + uy * start_d,
                x1 + ux * end_d,
                y1 + uy * end_d,
            ),
            fill=BLACK,
            width=1,
        )
        distance += dash_length + gap


def key_box(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, include_red: bool = True) -> None:
    draw.rectangle((x + 3, y + 3, x + w + 3, y + h + 3), outline=BLACK, width=1)
    draw.rectangle((x, y, x + w, y + h), fill=WHITE, outline=BLACK, width=1)
    draw.text((x + w / 2 - 12, y + 16), "KEY", fill=BLACK, font=KEY_FONT)
    center_text(draw, (x + 10, y + 38, x + w - 10, y + 78), "ORANGE - Enhanced since initial design", KEY_FONT, width_pad=2)
    if include_red:
        center_text(draw, (x + 10, y + 86, x + w - 10, y + h - 8), "RED - Newly implemented enhancement", KEY_FONT, width_pad=2)


def generate_technical() -> None:
    image = Image.new("RGBA", (646, 1046), WHITE)
    draw = ImageDraw.Draw(image)

    start = round_rect(draw, 5, 2, 120, 58, "SpotMini Simulation Application")
    setup = rect(draw, 5, 80, 120, 62, "Use Python + PyBullet to create simulation application")
    load = rect(draw, 5, 170, 120, 60, "Load SpotMini URDF + PyBullet world")
    loaded = diamond(draw, 26, 245, 100, 86, "Model loaded?")
    deps = rect(draw, 195, 261, 122, 60, "Fix paths and dependency settings", ORANGE)
    init = rect(draw, 16, 371, 120, 60, "Initialize Bezier gait and SpotModel IK")
    command = rect(draw, 16, 470, 120, 60, "Create keyboard command loop")
    smooth = diamond(draw, 16, 555, 120, 92, "Is robot motion smooth?")
    smoothing = rect(draw, 196, 578, 120, 60, "Add command smoothing and input filtering", ORANGE)
    strafe = rect(draw, 336, 578, 120, 60, "Implement strafe and turn-in-place controls", RED)
    gait = diamond(draw, 16, 705, 120, 92, "Is baseline gait sufficient?")
    four_phase = rect(draw, 196, 728, 120, 60, "Add four-phase gait, auto-yaw, auto-reset", RED)
    tune = rect(draw, 366, 728, 120, 60, "Tune stride, swing period, and clearance", ORANGE)
    run = rect(draw, 16, 860, 120, 60, "Run SpotMini in PyBullet simulation")
    output = rect(draw, 16, 962, 120, 80, "Collect motion, stability, and debugging results")

    camera = rect(draw, 335, 405, 126, 60, "Add robot RGB-D camera", RED)
    terrain = rect(draw, 335, 506, 126, 60, "Add rough-terrain height features", ORANGE)
    residual = rect(draw, 335, 650, 126, 70, "Implement per-leg swing-height residual ML env", RED)
    dash = rect(draw, 365, 956, 122, 88, "Design dashboard to show camera, plots, and simulation status", ORANGE)

    key_box(draw, 470, 300, 155, 142)

    arrow(draw, anchor(start, "bottom"), anchor(setup, "top"))
    arrow(draw, anchor(setup, "bottom"), anchor(load, "top"))
    arrow(draw, anchor(load, "bottom"), anchor(loaded, "top"))
    arrow(draw, anchor(loaded, "right"), anchor(deps, "left"), "No")
    draw_dashed_line(draw, anchor(deps, "top"), (255, 200))
    draw_dashed_line(draw, (255, 200), anchor(load, "right"))
    arrow(draw, anchor(loaded, "bottom"), anchor(init, "top"), "Yes")
    arrow(draw, anchor(init, "bottom"), anchor(command, "top"))
    arrow(draw, anchor(command, "bottom"), anchor(smooth, "top"))
    arrow(draw, anchor(smooth, "right"), anchor(smoothing, "left"), "No")
    arrow(draw, anchor(smoothing, "right"), anchor(strafe, "left"))
    arrow(draw, anchor(smooth, "bottom"), anchor(gait, "top"), "Yes")
    arrow(draw, anchor(gait, "right"), anchor(four_phase, "left"), "No")
    arrow(draw, anchor(four_phase, "right"), anchor(tune, "left"))
    arrow(draw, anchor(gait, "bottom"), anchor(run, "top"), "Yes")
    arrow(draw, anchor(run, "bottom"), anchor(output, "top"))

    draw_dashed_line(draw, anchor(command, "right"), (395, 370))
    arrow(draw, (395, 370), anchor(camera, "top"), dashed=False)
    arrow(draw, anchor(camera, "bottom"), anchor(terrain, "top"))
    arrow(draw, anchor(terrain, "bottom"), anchor(residual, "top"))
    arrow(draw, anchor(output, "right"), anchor(dash, "left"))
    draw_dashed_line(draw, anchor(residual, "bottom"), (426, 930))
    arrow(draw, (426, 930), anchor(dash, "top"), dashed=False)

    image.save(TECHNICAL_PNG)


def generate_operational() -> None:
    image = Image.new("RGBA", (831, 461), WHITE)
    draw = ImageDraw.Draw(image)

    nav = rect(draw, 14, 10, 122, 60, "Open SpotMini dashboard")
    enter = rect(draw, 230, 70, 122, 60, 'Click "Start Simulation"')
    back = rect(draw, 700, 60, 130, 80, 'Click "Reset" to return home')

    mode = diamond(draw, 234, 160, 115, 90, "Choose test mode")
    basic = rect(draw, 65, 200, 122, 60, "Select baseline walk / trot mode")
    enhanced = rect(draw, 420, 180, 140, 60, "Enable enhanced camera view", ORANGE)
    controls = rect(draw, 65, 300, 122, 60, "Use W/A/S/D and Q/E controls")
    ml = rect(draw, 420, 300, 140, 60, "Enable rough-terrain residual policy", RED)
    tune = rect(draw, 240, 340, 120, 60, "Adjust speed, stride, and gait settings", ORANGE)
    submit = rect(draw, 580, 330, 120, 60, 'Press "Run" or "Submit"')
    observe = rect(draw, 700, 380, 130, 80, "Observe robot motion, camera feed, and plots")

    draw.text((6, 145), "If user wants to test normal walking", fill=BLACK, font=SMALL_FONT)
    draw.text((420, 145), "If user wants enhanced perception / terrain testing", fill=BLACK, font=SMALL_FONT)

    arrow(draw, anchor(nav, "right"), anchor(enter, "left"))
    arrow(draw, anchor(back, "left"), anchor(enter, "right"))
    arrow(draw, anchor(enter, "bottom"), anchor(mode, "top"))
    arrow(draw, anchor(mode, "left"), anchor(basic, "top"))
    arrow(draw, anchor(mode, "right"), anchor(enhanced, "top"))
    arrow(draw, anchor(basic, "bottom"), anchor(controls, "top"))
    arrow(draw, anchor(enhanced, "bottom"), anchor(ml, "top"))
    arrow(draw, anchor(controls, "right"), anchor(tune, "left"))
    arrow(draw, anchor(tune, "right"), anchor(ml, "left"))
    arrow(draw, anchor(ml, "right"), anchor(submit, "left"))
    arrow(draw, anchor(submit, "right"), anchor(observe, "left"))
    arrow(draw, anchor(observe, "top"), anchor(back, "bottom"))

    draw_dashed_line(draw, anchor(controls, "top"), (250, 280))
    draw_dashed_line(draw, (250, 280), anchor(enhanced, "left"))
    draw.text((236, 263), "Optional", fill=BLACK, font=SMALL_FONT)

    key_box(draw, 475, 1, 175, 120)

    image.save(OPERATIONAL_PNG)


def main() -> None:
    generate_technical()
    generate_operational()
    print(f"Wrote {TECHNICAL_PNG}")
    print(f"Wrote {OPERATIONAL_PNG}")


if __name__ == "__main__":
    main()
