from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
SPOT_BULLET_DIR = APP_DIR.parent
SRC_DIR = SPOT_BULLET_DIR / "src"
PROJECT_ROOT = SPOT_BULLET_DIR.parent
CACHE_DIR = PROJECT_ROOT / ".cache"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("FONTCONFIG_PATH", str(CACHE_DIR / "fontconfig"))

for path in (PROJECT_ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.makedirs(CACHE_DIR / "matplotlib", exist_ok=True)
os.makedirs(CACHE_DIR / "fontconfig", exist_ok=True)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency: {exc.name}\n"
        "Install the ML playback dependencies or point the dashboard to a Python "
        "environment that already has them."
    ) from exc

from camera_sensor import SpotCamera
from spot_play_ml import (
    _spotmini_compat_bit_generator_ctor,
    existing_training_roots,
    load_run_config,
    load_ppo_model,
    make_env,
    patch_numpy_bit_generator_pickle,
    resolve_run_dir,
    resolve_model_path,
)
from spot_visualization_ml import unwrap_env


_spotmini_compat_bit_generator_ctor = _spotmini_compat_bit_generator_ctor
STOP_REQUESTED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SpotMini ML model and export body-camera frames for the dashboard."
    )
    parser.add_argument("--run-dir", default=None, help="Training run directory to load.")
    parser.add_argument("--model", choices=("best", "final", "checkpoint"), default="best")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "frame": output_dir / "latest_camera.jpg",
        "status": output_dir / "status.json",
        "pid": output_dir / "worker.pid",
    }


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp_file:
        json.dump(payload, tmp_file, indent=2)
        tmp_name = tmp_file.name
    os.replace(tmp_name, path)


def write_image_atomic(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", delete=False, dir=path.parent, suffix=".jpg") as tmp_file:
        Image.fromarray(frame).save(tmp_file, format="JPEG", quality=88)
        tmp_name = tmp_file.name
    os.replace(tmp_name, path)


def write_status(path: Path, **payload) -> None:
    payload.setdefault("timestamp", time.time())
    write_json_atomic(path, payload)


def resolve_playable_run_dir(run_dir_arg: str | None) -> Path:
    if run_dir_arg:
        return resolve_run_dir(run_dir_arg)

    candidates = []
    seen = set()
    for training_root in existing_training_roots():
        for path in training_root.iterdir():
            if not path.is_dir():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(resolved)

    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    for run_dir in candidates:
        best_model = run_dir / "best_model" / "best_model.zip"
        if best_model.exists():
            return run_dir

    raise SystemExit(
        "No playable training runs were found. Expected at least "
        "best_model/best_model.zip in a run directory."
    )


def is_stop_requested() -> bool:
    return STOP_REQUESTED


def request_stop(*_args) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def register_signal_handlers() -> None:
    for sig_name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, request_stop)


def make_playback_args(seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        gait_mode=None,
        terrain_profile=None,
        height_field=False,
        terrain_randomization=False,
        target_forward_speed=None,
        episode_steps=None,
        body_height_offset=None,
        body_pitch_bias_deg=None,
        auto_yaw_gain=None,
        gait_geometry=None,
        enable_camera_observation=None,
        enable_imu_yaw=None,
        disable_camera_leveling=False,
        camera_pitch_offset_deg=None,
        render=False,
        bullet_gui=False,
        unsafe_gui=False,
        follow_camera=False,
        seed=seed,
    )


def patch_numpy_module_aliases_safe() -> None:
    numpy_core = getattr(np, "_core", None) or np.core
    sys.modules.setdefault("numpy._core", numpy_core)

    try:
        import numpy.core.numeric as numpy_numeric
    except Exception:
        numpy_numeric = getattr(numpy_core, "numeric", None)
    if numpy_numeric is not None:
        sys.modules.setdefault("numpy._core.numeric", numpy_numeric)

    try:
        import numpy.core.multiarray as numpy_multiarray
    except Exception:
        numpy_multiarray = getattr(numpy_core, "multiarray", None)
    if numpy_multiarray is not None:
        sys.modules.setdefault("numpy._core.multiarray", numpy_multiarray)

    try:
        import numpy.core.umath as numpy_umath
    except Exception:
        numpy_umath = getattr(numpy_core, "umath", None)
    if numpy_umath is not None:
        sys.modules.setdefault("numpy._core.umath", numpy_umath)


def cleanup_pid_file(pid_path: Path) -> None:
    if not pid_path.exists():
        return
    try:
        recorded_pid = int(pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        return
    if recorded_pid == os.getpid():
        pid_path.unlink()


def build_camera(raw_env, run_config: dict, *, width: int, height: int) -> SpotCamera:
    base_env = getattr(raw_env, "base_env", None)
    if base_env is None:
        raise RuntimeError("The selected ML environment does not expose a base_env camera target.")

    terrain_profile = run_config.get("terrain_profile")
    use_stabilized_camera = terrain_profile == "rough"
    return SpotCamera(
        base_env,
        width=width,
        height=height,
        target_distance=1.1,
        stabilize_roll_pitch=run_config.get("camera_stabilize_roll_pitch", use_stabilized_camera),
        pitch_offset_deg=run_config.get("camera_pitch_offset_deg", -10.0 if use_stabilized_camera else 0.0),
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    paths = output_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    register_signal_handlers()
    atexit.register(cleanup_pid_file, paths["pid"])
    patch_numpy_bit_generator_pickle()
    patch_numpy_module_aliases_safe()

    paths["pid"].write_text(str(os.getpid()), encoding="utf-8")

    run_dir = None
    env = None
    try:
        run_dir = resolve_playable_run_dir(args.run_dir)
        run_config = load_run_config(run_dir)
        model_path = resolve_model_path(run_dir, args.model, args.checkpoint_path)
        playback_args = make_playback_args(args.seed)
        base_env = DummyVecEnv([make_env(playback_args, run_config)])
        env = base_env
        vecnormalize_path = run_dir / "vecnormalize" / "vecnormalize.pkl"
        vecnormalize_loaded = False
        vecnormalize_error = None
        if vecnormalize_path.exists():
            try:
                env = VecNormalize.load(str(vecnormalize_path), base_env)
                env.training = False
                env.norm_reward = False
                vecnormalize_loaded = True
            except Exception as exc:
                vecnormalize_error = str(exc)

        model = load_ppo_model(model_path, env, context="dashboard camera feed")

        raw_env = unwrap_env(env)
        camera = build_camera(raw_env, run_config, width=args.width, height=args.height)

        obs = env.reset()
        episode = 1
        episode_step = 0
        episode_reward = 0.0
        last_frame_time = 0.0

        while not is_stop_requested():
            loop_started = time.time()
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            reward_value = float(rewards[0])
            done_value = bool(dones[0])
            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            frame = camera.get_rgb_frame()
            write_image_atomic(paths["frame"], frame)
            last_frame_time = time.time()

            episode_step += 1
            episode_reward += reward_value
            write_status(
                paths["status"],
                state="running",
                running=True,
                run_name=run_dir.name,
                run_dir=str(run_dir),
                model_path=str(model_path),
                terrain_profile=run_config.get("terrain_profile", "flat"),
                gait_mode=run_config.get("gait_mode", "trot"),
                vecnormalize_loaded=vecnormalize_loaded,
                vecnormalize_error=vecnormalize_error,
                episode=episode,
                episode_step=episode_step,
                episode_reward=episode_reward,
                forward_speed=(info or {}).get("forward_speed"),
                frame_width=args.width,
                frame_height=args.height,
                last_frame_time=last_frame_time,
            )

            if done_value or episode_step >= int(args.max_steps):
                obs = env.reset()
                episode += 1
                episode_step = 0
                episode_reward = 0.0

            sleep_time = max(0.0, (1.0 / max(args.fps, 0.1)) - (time.time() - loop_started))
            if sleep_time > 0:
                time.sleep(sleep_time)

        write_status(
            paths["status"],
            state="stopped",
            running=False,
            run_name=run_dir.name,
            run_dir=str(run_dir),
            last_frame_time=last_frame_time,
        )
        if env is not None:
            env.close()
        return 0
    except SystemExit as exc:
        error = str(exc) or "ML camera worker exited during startup."
        write_status(
            paths["status"],
            state="error",
            running=False,
            run_name=run_dir.name if run_dir is not None else "Unavailable",
            run_dir=str(run_dir) if run_dir is not None else None,
            error=error,
        )
        print(f"ML camera worker failed: {error}", file=sys.stderr)
        return 1
    except Exception as exc:
        write_status(
            paths["status"],
            state="error",
            running=False,
            run_name=run_dir.name if run_dir is not None else "Unavailable",
            run_dir=str(run_dir) if run_dir is not None else None,
            error=str(exc),
        )
        traceback.print_exc()
        print(f"ML camera worker failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
