#!/usr/bin/env python3
"""Generate Figure 3 for the SpotMini final report."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
PNG_PATH = ROOT / "spotmini_system_architecture.png"
SVG_PATH = ROOT / "spotmini_system_architecture.svg"

WIDTH = 2100
HEIGHT = 1460

BG = "#f6f4ef"
INK = "#202124"
MUTED = "#6b7280"
LINE = "#444444"
WHITE = "#fffdf8"
SAND = "#f5cf8e"
ROSE = "#f46d61"
STONE = "#e8e3d7"
PALE = "#fff7ea"


def load_font(path: str, size: int, fallback=None):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return fallback or ImageFont.load_default()


TITLE_FONT = load_font("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 42)
SUBTITLE_FONT = load_font("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
CARD_TITLE_FONT = load_font("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
BODY_FONT = load_font("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
SMALL_FONT = load_font("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
KEY_FONT = load_font("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 18)


CARDS = [
    {
        "id": "operator",
        "title": "Operator / Capstone Team",
        "body": [
            "Chooses runs, reviews charts and videos,",
            "adjusts gait and reward settings,",
            "and decides which direction to test next.",
        ],
        "xywh": (620, 120, 860, 150),
        "fill": WHITE,
    },
    {
        "id": "apps",
        "title": "Launchers and Debug Apps",
        "body": [
            "launch_dashboard.py, display.py,",
            "launch_manual_app.py, display_manual.py,",
            "manual_pybullet_worker.py",
        ],
        "xywh": (110, 360, 540, 180),
        "fill": SAND,
    },
    {
        "id": "scripts",
        "title": "Experiment Scripts",
        "body": [
            "spot_train_ml.py",
            "spot_play_ml.py",
            "spot_compare_ml.py",
            "spot_tester.py",
        ],
        "xywh": (720, 360, 660, 180),
        "fill": SAND,
    },
    {
        "id": "artifacts",
        "title": "Training Artifacts and Evaluation",
        "body": [
            "training_runs, checkpoints,",
            "VecNormalize stats, eval plots, GIF previews,",
            "comparison tables, and crash screenshots",
        ],
        "xywh": (1470, 360, 520, 180),
        "fill": SAND,
    },
    {
        "id": "rl",
        "title": "RL Policy and Run Manager",
        "body": [
            "Stable-Baselines3 PPO plus run configs,",
            "warm starts, safer loading, and playback logic",
        ],
        "xywh": (640, 620, 820, 150),
        "fill": SAND,
    },
    {
        "id": "env",
        "title": "Environment Variants",
        "body": [
            "SpotMLWalkEnv, dual-IMU variants,",
            "rough-terrain modules, and height residual envs",
        ],
        "xywh": (300, 880, 620, 160),
        "fill": SAND,
    },
    {
        "id": "teacher",
        "title": "Teacher + Joint Residual Transfer",
        "body": [
            "Uses dual_imu_longwalk_v1 as the walking prior,",
            "then learns bounded rough-terrain corrections",
        ],
        "xywh": (1165, 880, 670, 180),
        "fill": ROSE,
    },
    {
        "id": "control",
        "title": "Gait and Control Core",
        "body": [
            "Bezier gait generation, four-phase scheduling,",
            "IK, posture trim, yaw stabilization,",
            "and per-leg swing shaping",
        ],
        "xywh": (640, 1110, 820, 170),
        "fill": WHITE,
    },
    {
        "id": "sim",
        "title": "Simulation and Robot Layer",
        "body": [
            "PyBullet + SpotMini URDF, terrain probes, rough heightfields,",
            "leveled camera features, dual virtual IMUs, and physics randomization",
        ],
        "xywh": (360, 1295, 1380, 145),
        "fill": WHITE,
    },
]


def wrap_lines(draw: ImageDraw.ImageDraw, lines: Iterable[str], font, width: int) -> list[str]:
    wrapped: list[str] = []
    for raw in lines:
        words = raw.split()
        if not words:
            wrapped.append("")
            continue
        current = words[0]
        for word in words[1:]:
            trial = current + " " + word
            if draw.textlength(trial, font=font) <= width:
                current = trial
            else:
                wrapped.append(current)
                current = word
        wrapped.append(current)
    return wrapped


def get_box(card_id: str) -> dict:
    return next(card for card in CARDS if card["id"] == card_id)


def draw_title(draw: ImageDraw.ImageDraw) -> None:
    title = "SpotMini System Architecture and Training Pipeline"
    subtitle = "Figure 3. Cleaner view of how tools, ML scripts, gait logic, and simulation fit together in the final capstone workflow."
    tw = draw.textlength(title, font=TITLE_FONT)
    sw = draw.textlength(subtitle, font=SUBTITLE_FONT)
    draw.text(((WIDTH - tw) / 2, 24), title, font=TITLE_FONT, fill=INK)
    draw.text(((WIDTH - sw) / 2, 70), subtitle, font=SUBTITLE_FONT, fill=MUTED)


def draw_key(draw: ImageDraw.ImageDraw) -> None:
    x, y, w, h = 1620, 122, 340, 120
    draw.rounded_rectangle((x, y, x + w, y + h), radius=16, outline=LINE, fill=WHITE, width=2)
    draw.text((x + 22, y + 18), "Key", font=KEY_FONT, fill=INK)
    items = [
        (WHITE, "base platform layer"),
        (SAND, "capstone workflow / ML layer"),
        (ROSE, "newest rough-transfer addition"),
    ]
    for idx, (fill, label) in enumerate(items):
        yy = y + 48 + idx * 22
        draw.rounded_rectangle((x + 20, yy, x + 46, yy + 14), radius=4, outline=LINE, fill=fill, width=2)
        draw.text((x + 58, yy - 2), label, font=SMALL_FONT, fill=INK)


def draw_card(draw: ImageDraw.ImageDraw, card: dict) -> None:
    x, y, w, h = card["xywh"]
    draw.rounded_rectangle((x, y, x + w, y + h), radius=22, outline=INK, fill=card["fill"], width=3)
    title = card["title"]
    draw.text((x + 28, y + 20), title, font=CARD_TITLE_FONT, fill=INK)
    lines = wrap_lines(draw, card["body"], BODY_FONT, w - 56)
    yy = y + 68
    for line in lines:
        draw.text((x + 28, yy), line, font=BODY_FONT, fill=INK)
        yy += 27


def center_bottom(card: dict) -> tuple[float, float]:
    x, y, w, h = card["xywh"]
    return x + w / 2, y + h


def center_top(card: dict) -> tuple[float, float]:
    x, y, w, _ = card["xywh"]
    return x + w / 2, y


def middle_left(card: dict) -> tuple[float, float]:
    x, y, _, h = card["xywh"]
    return x, y + h / 2


def middle_right(card: dict) -> tuple[float, float]:
    x, y, w, h = card["xywh"]
    return x + w, y + h / 2


def draw_arrow(draw: ImageDraw.ImageDraw, p1: tuple[float, float], p2: tuple[float, float], width: int = 4) -> None:
    import math

    draw.line((p1[0], p1[1], p2[0], p2[1]), fill=LINE, width=width)
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    size = 16
    wing = math.pi / 7
    left = (p2[0] - size * math.cos(angle - wing), p2[1] - size * math.sin(angle - wing))
    right = (p2[0] - size * math.cos(angle + wing), p2[1] - size * math.sin(angle + wing))
    draw.polygon([p2, left, right], fill=LINE)
def draw_connectors(draw: ImageDraw.ImageDraw) -> None:
    operator = get_box("operator")
    apps = get_box("apps")
    scripts = get_box("scripts")
    artifacts = get_box("artifacts")
    rl = get_box("rl")
    env = get_box("env")
    teacher = get_box("teacher")
    control = get_box("control")
    sim = get_box("sim")

    draw_arrow(draw, center_bottom(operator), center_top(scripts))

    draw_arrow(draw, middle_right(apps), middle_left(scripts))
    draw_arrow(draw, middle_right(scripts), middle_left(artifacts))
    draw_arrow(draw, center_bottom(scripts), center_top(rl))
    draw_arrow(draw, center_bottom(control), center_top(sim))

    rl_bottom = center_bottom(rl)
    split_y = 835
    draw.line((rl_bottom[0], rl_bottom[1], rl_bottom[0], split_y), fill=LINE, width=4)
    draw.line((rl_bottom[0], split_y, center_top(env)[0], split_y), fill=LINE, width=4)
    draw.line((rl_bottom[0], split_y, center_top(teacher)[0], split_y), fill=LINE, width=4)
    draw_arrow(draw, (center_top(env)[0], split_y), center_top(env))
    draw_arrow(draw, (center_top(teacher)[0], split_y), center_top(teacher))

    sim_to = center_top(sim)
    env_bottom = center_bottom(env)
    teacher_bottom = center_bottom(teacher)
    draw.line((env_bottom[0], env_bottom[1], env_bottom[0], sim_to[1] - 24), fill=LINE, width=4)
    draw_arrow(draw, (env_bottom[0], sim_to[1] - 24), (sim_to[0] - 360, sim_to[1]))
    draw.line((teacher_bottom[0], teacher_bottom[1], teacher_bottom[0], sim_to[1] - 24), fill=LINE, width=4)
    draw_arrow(draw, (teacher_bottom[0], sim_to[1] - 24), (sim_to[0] + 360, sim_to[1]))


def draw_png() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    draw_title(draw)
    draw_key(draw)
    for card in CARDS:
        draw_card(draw, card)
    draw_connectors(draw)

    image.save(PNG_PATH)


def svg_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_svg_text(draw: ImageDraw.ImageDraw, x: float, y: float, lines: Iterable[str], font_class: str) -> list[str]:
    parts: list[str] = []
    yy = y
    for line in lines:
        parts.append(f'<text x="{x}" y="{yy}" class="{font_class}">{svg_escape(line)}</text>')
        yy += 27
    return parts


def draw_svg() -> None:
    tmp = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(tmp)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        """
<defs>
  <marker id="arrowhead" markerWidth="12" markerHeight="8" refX="10" refY="4" orient="auto">
    <polygon points="0 0, 12 4, 0 8" fill="#444444"/>
  </marker>
  <style>
    .title { font: bold 42px Arial, sans-serif; fill: #202124; }
    .subtitle { font: 22px Arial, sans-serif; fill: #6b7280; }
    .card-title { font: bold 28px Arial, sans-serif; fill: #202124; }
    .body { font: 20px Arial, sans-serif; fill: #202124; }
    .small { font: 17px Arial, sans-serif; fill: #6b7280; }
    .key { font: bold 18px Arial, sans-serif; fill: #202124; }
    .arrow { stroke: #444444; stroke-width: 4; fill: none; marker-end: url(#arrowhead); }
    .line { stroke: #444444; stroke-width: 4; fill: none; }
  </style>
</defs>
""",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BG}"/>',
    ]

    title = "SpotMini System Architecture and Training Pipeline"
    subtitle = "Figure 3. Cleaner view of how tools, ML scripts, gait logic, and simulation fit together in the final capstone workflow."
    parts.append(f'<text x="{WIDTH / 2}" y="58" text-anchor="middle" class="title">{svg_escape(title)}</text>')
    parts.append(f'<text x="{WIDTH / 2}" y="100" text-anchor="middle" class="subtitle">{svg_escape(subtitle)}</text>')

    parts.append(f'<rect x="1620" y="86" width="340" height="120" rx="16" ry="16" fill="{WHITE}" stroke="{LINE}" stroke-width="2"/>')
    parts.append('<text x="1642" y="114" class="key">Key</text>')
    key_items = [
        (WHITE, "base platform layer"),
        (SAND, "capstone workflow / ML layer"),
        (ROSE, "newest rough-transfer addition"),
    ]
    for idx, (fill, label) in enumerate(key_items):
        yy = 134 + idx * 22
        parts.append(f'<rect x="1640" y="{yy}" width="26" height="14" rx="4" ry="4" fill="{fill}" stroke="{LINE}" stroke-width="2"/>')
        parts.append(f'<text x="1678" y="{yy + 12}" class="small">{svg_escape(label)}</text>')

    for card in CARDS:
        x, y, w, h = card["xywh"]
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" ry="22" fill="{card["fill"]}" stroke="{INK}" stroke-width="3"/>')
        parts.append(f'<text x="{x + 28}" y="{y + 48}" class="card-title">{svg_escape(card["title"])}</text>')
        lines = wrap_lines(draw, card["body"], BODY_FONT, w - 56)
        parts.extend(build_svg_text(draw, x + 28, y + 92, lines, "body"))

    def line(x1: float, y1: float, x2: float, y2: float, arrow: bool = False) -> None:
        cls = "arrow" if arrow else "line"
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}"/>')

    operator = get_box("operator")
    apps = get_box("apps")
    scripts = get_box("scripts")
    artifacts = get_box("artifacts")
    rl = get_box("rl")
    env = get_box("env")
    teacher = get_box("teacher")
    control = get_box("control")
    sim = get_box("sim")

    p1 = center_bottom(operator)
    p2 = center_top(scripts)
    line(*p1, *p2, True)
    line(*middle_right(apps), *middle_left(scripts), True)
    line(*middle_right(scripts), *middle_left(artifacts), True)
    line(*center_bottom(scripts), *center_top(rl), True)
    line(*center_bottom(rl), *center_top(control), True)

    env_start = center_bottom(rl)
    env_end = center_top(env)
    line(env_start[0], env_start[1], env_start[0] - 120, env_start[1], False)
    line(env_start[0] - 120, env_start[1], env_start[0] - 120, env_end[1] - 26, False)
    line(env_start[0] - 120, env_end[1] - 26, env_end[0], env_end[1], True)
    line(*middle_right(rl), *middle_left(teacher), True)
    line(*center_bottom(control), *center_top(sim), True)

    env_to_sim = center_bottom(env)
    sim_top = center_top(sim)
    line(env_to_sim[0], env_to_sim[1], env_to_sim[0], sim_top[1] - 20, False)
    line(env_to_sim[0], sim_top[1] - 20, sim_top[0] - 360, sim_top[1], True)

    teacher_to_sim = center_bottom(teacher)
    line(teacher_to_sim[0], teacher_to_sim[1], teacher_to_sim[0], sim_top[1] - 20, False)
    line(teacher_to_sim[0], sim_top[1] - 20, sim_top[0] + 360, sim_top[1], True)

    line(*center_bottom(control), *center_top(sim), True)

    rl_bottom = center_bottom(rl)
    env_top = center_top(env)
    teacher_top = center_top(teacher)
    split_y = 835
    line(rl_bottom[0], rl_bottom[1], rl_bottom[0], split_y, False)
    line(rl_bottom[0], split_y, env_top[0], split_y, False)
    line(rl_bottom[0], split_y, teacher_top[0], split_y, False)
    line(env_top[0], split_y, env_top[0], env_top[1], True)
    line(teacher_top[0], split_y, teacher_top[0], teacher_top[1], True)

    env_bottom = center_bottom(env)
    teacher_bottom = center_bottom(teacher)
    sim_top = center_top(sim)
    line(env_bottom[0], env_bottom[1], env_bottom[0], sim_top[1] - 24, False)
    line(env_bottom[0], sim_top[1] - 24, sim_top[0] - 360, sim_top[1], True)
    line(teacher_bottom[0], teacher_bottom[1], teacher_bottom[0], sim_top[1] - 24, False)
    line(teacher_bottom[0], sim_top[1] - 24, sim_top[0] + 360, sim_top[1], True)

    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    draw_png()
    draw_svg()
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {SVG_PATH}")


if __name__ == "__main__":
    main()
