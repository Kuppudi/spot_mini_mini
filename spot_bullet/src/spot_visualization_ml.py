from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from camera_sensor import SpotCamera


LEG_NAMES = ("FL", "FR", "BL", "BR")
PREVIEW_WORLD_SIZE = (640, 360)
PREVIEW_CAMERA_SIZE = (280, 140)
PREVIEW_PANEL_WIDTH = 320
PREVIEW_BACKGROUND = (14, 18, 24)
PREVIEW_PANEL_BACKGROUND = (22, 27, 36)
PREVIEW_TEXT = (236, 240, 244)
PREVIEW_MUTED_TEXT = (153, 163, 177)
PREVIEW_ACCENT = (88, 166, 255)
PREVIEW_BAR_FILL = (91, 193, 151)
PREVIEW_BAR_DANGER = (240, 113, 120)


def unwrap_env(env):
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "venv"):
            current = current.venv
            continue
        envs = getattr(current, "envs", None)
        if envs:
            current = envs[0]
            continue
        if hasattr(current, "env"):
            current = current.env
            continue
        break
    return current


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_numpy(value, ndim=None):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if ndim is not None and array.ndim != ndim:
        return None
    return array


def _resize_image(frame, size):
    if frame is None:
        return None
    resampling = getattr(Image, "Resampling", Image)
    return Image.fromarray(frame.astype(np.uint8)).resize(size, resample=resampling.BILINEAR)


def _get_preview_camera(raw_env):
    base_env = getattr(raw_env, "base_env", None)
    if base_env is None:
        return None

    camera = getattr(raw_env, "_preview_camera", None)
    if camera is None:
        try:
            camera = SpotCamera(
                base_env,
                width=PREVIEW_CAMERA_SIZE[0],
                height=PREVIEW_CAMERA_SIZE[1],
                target_distance=1.1,
            )
        except Exception:
            return None
        raw_env._preview_camera = camera
    return camera


def _compute_foothold_error(info):
    foot_body = _to_numpy(info.get("foot_body_positions"), ndim=2)
    target_body = _to_numpy(info.get("target_foot_positions_body"), ndim=2)
    if foot_body is None or target_body is None or foot_body.shape != target_body.shape:
        return None
    return np.linalg.norm(foot_body - target_body, axis=1)


def _draw_metric_bars(draw, title, values, origin_x, origin_y, panel_width, max_value):
    values = _to_numpy(values, ndim=1)
    if values is None or values.size == 0:
        return origin_y

    draw.text((origin_x, origin_y), title, fill=PREVIEW_TEXT)
    y = origin_y + 20
    bar_left = origin_x + 34
    bar_right = origin_x + panel_width - 18
    bar_width = max(10, bar_right - bar_left)

    for idx, value in enumerate(values[: len(LEG_NAMES)]):
        label = LEG_NAMES[idx]
        value = max(0.0, _safe_float(value))
        ratio = min(1.0, value / max(max_value, 1e-6))
        fill_color = PREVIEW_BAR_FILL if ratio < 0.8 else PREVIEW_BAR_DANGER

        draw.text((origin_x, y - 1), label, fill=PREVIEW_MUTED_TEXT)
        draw.rounded_rectangle(
            (bar_left, y, bar_right, y + 12),
            radius=4,
            fill=(38, 46, 58),
        )
        draw.rounded_rectangle(
            (bar_left, y, bar_left + int(bar_width * ratio), y + 12),
            radius=4,
            fill=fill_color,
        )
        draw.text((bar_right - 44, y - 1), f"{value:0.3f}", fill=PREVIEW_TEXT)
        y += 22

    return y + 6


def compose_preview_frame(raw_env, info, *, cumulative_reward, episode_step, title=None, training_timestep=None):
    world_rgb = None
    try:
        world_rgb = raw_env.render()
    except Exception:
        world_rgb = None

    if world_rgb is None or np.asarray(world_rgb).ndim != 3:
        world_rgb = np.zeros((PREVIEW_WORLD_SIZE[1], PREVIEW_WORLD_SIZE[0], 3), dtype=np.uint8)

    world_image = _resize_image(np.asarray(world_rgb, dtype=np.uint8), PREVIEW_WORLD_SIZE)
    canvas_width = PREVIEW_WORLD_SIZE[0] + PREVIEW_PANEL_WIDTH
    canvas_height = PREVIEW_WORLD_SIZE[1]
    canvas = Image.new("RGB", (canvas_width, canvas_height), PREVIEW_BACKGROUND)
    canvas.paste(world_image, (0, 0))

    panel = Image.new("RGB", (PREVIEW_PANEL_WIDTH, canvas_height), PREVIEW_PANEL_BACKGROUND)
    panel_draw = ImageDraw.Draw(panel)

    text_x = 18
    y = 16
    panel_draw.text((text_x, y), title or "SpotMini policy preview", fill=PREVIEW_TEXT)
    y += 20

    terrain_profile = info.get("terrain_profile", "flat")
    status_line = f"{terrain_profile} terrain | step {episode_step}"
    if training_timestep is not None:
        status_line += f" | t={int(training_timestep):,}"
    panel_draw.text((text_x, y), status_line, fill=PREVIEW_MUTED_TEXT)
    y += 22

    camera = _get_preview_camera(raw_env)
    if camera is not None:
        try:
            camera_rgb = camera.get_rgb_frame()
        except Exception:
            camera_rgb = None
        if camera_rgb is not None and np.asarray(camera_rgb).ndim == 3:
            panel_draw.text((text_x, y), "Body camera", fill=PREVIEW_TEXT)
            y += 18
            camera_image = _resize_image(np.asarray(camera_rgb, dtype=np.uint8), PREVIEW_CAMERA_SIZE)
            panel.paste(camera_image, (text_x, y))
            y += PREVIEW_CAMERA_SIZE[1] + 12

    forward_speed = _safe_float(info.get("forward_speed"))
    forward_displacement = _safe_float(info.get("forward_displacement"))
    base_height_error = _safe_float(info.get("base_height_error"))

    panel_draw.text((text_x, y), f"Return: {cumulative_reward:0.2f}", fill=PREVIEW_TEXT)
    y += 18
    panel_draw.text((text_x, y), f"Forward speed: {forward_speed:0.3f} m/s", fill=PREVIEW_TEXT)
    y += 18
    panel_draw.text((text_x, y), f"Distance: {forward_displacement:0.3f} m", fill=PREVIEW_TEXT)
    y += 18
    panel_draw.text((text_x, y), f"Height error: {base_height_error:0.3f} m", fill=PREVIEW_TEXT)
    y += 24

    foot_clearance = info.get("foot_clearance")
    if foot_clearance is not None:
        y = _draw_metric_bars(
            panel_draw,
            "Foot clearance (m)",
            foot_clearance,
            text_x,
            y,
            PREVIEW_PANEL_WIDTH,
            max_value=0.08,
        )

    foothold_error = _compute_foothold_error(info)
    if foothold_error is not None:
        y = _draw_metric_bars(
            panel_draw,
            "Foothold error (m)",
            foothold_error,
            text_x,
            y,
            PREVIEW_PANEL_WIDTH,
            max_value=0.16,
        )

    terrain_forward = _to_numpy(info.get("terrain_heights_forward"), ndim=1)
    if terrain_forward is not None:
        y = _draw_metric_bars(
            panel_draw,
            "Forward probe height (m)",
            np.abs(terrain_forward),
            text_x,
            y,
            PREVIEW_PANEL_WIDTH,
            max_value=0.12,
        )

    panel_draw.text(
        (text_x, canvas_height - 24),
        "Auto-generated preview",
        fill=PREVIEW_ACCENT,
    )
    canvas.paste(panel, (PREVIEW_WORLD_SIZE[0], 0))
    return np.asarray(canvas, dtype=np.uint8)


def record_policy_preview(
    model,
    env,
    output_path,
    *,
    max_steps=240,
    fps=12,
    title=None,
    training_timestep=None,
    still_path=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    obs = env.reset()
    frames = []
    cumulative_reward = 0.0
    done = False
    raw_env = unwrap_env(env)

    for episode_step in range(1, int(max_steps) + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        reward_value = float(np.asarray(rewards).reshape(-1)[0])
        done = bool(np.asarray(dones).reshape(-1)[0])
        info = infos[0] if isinstance(infos, (list, tuple)) else infos

        cumulative_reward += reward_value
        frame = compose_preview_frame(
            raw_env,
            info or {},
            cumulative_reward=cumulative_reward,
            episode_step=episode_step,
            title=title,
            training_timestep=training_timestep,
        )
        frames.append(Image.fromarray(frame))

        if done:
            break

    if not frames:
        return None

    frame_duration_ms = max(1, int(1000 / max(1, int(fps))))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )

    if still_path is not None:
        still_path = Path(still_path)
        still_path.parent.mkdir(parents=True, exist_ok=True)
        frames[-1].save(still_path)

    return {
        "path": str(output_path),
        "frames": len(frames),
        "reward": cumulative_reward,
        "done": done,
    }


def write_eval_progress_plot(evaluations_path, output_path, *, title=None):
    evaluations_path = Path(evaluations_path)
    if not evaluations_path.exists():
        return False

    with np.load(evaluations_path, allow_pickle=True) as data:
        timesteps = np.asarray(data["timesteps"], dtype=np.float32) if "timesteps" in data else np.array([], dtype=np.float32)
        results = np.asarray(data["results"], dtype=np.float32) if "results" in data else np.array([], dtype=np.float32)
        ep_lengths = np.asarray(data["ep_lengths"], dtype=np.float32) if "ep_lengths" in data else np.array([], dtype=np.float32)
    if timesteps.size == 0 or results.size == 0:
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(timesteps, mean_rewards, color="#58A6FF", linewidth=2.0)
    ax.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        color="#58A6FF",
        alpha=0.20,
    )
    ax.set_title(title or "Evaluation reward")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_facecolor("#F7F9FC")

    if ep_lengths.size != 0:
        length_axis = ax.twinx()
        mean_lengths = ep_lengths.mean(axis=1)
        length_axis.plot(timesteps, mean_lengths, color="#7B61FF", linewidth=1.5, alpha=0.7)
        length_axis.set_ylabel("Episode length")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True
