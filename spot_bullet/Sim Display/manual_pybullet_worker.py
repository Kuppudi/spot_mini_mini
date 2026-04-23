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

from spot_ml import SpotMLWalkEnv


STOP_REQUESTED = False
WORKER_VERSION = "manual-trot-side-stride-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a native PyBullet GUI controlled by dashboard button commands."
    )
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--control-source", choices=("manual", "policy"), default="manual")
    parser.add_argument("--terrain-profile", choices=("flat", "rough"), default="flat")
    parser.add_argument("--gait-mode", choices=("trot", "four_phase"), default="trot")
    parser.add_argument("--body-height-offset", type=float, default=0.018)
    parser.add_argument("--body-roll-bias-deg", type=float, default=0.0)
    parser.add_argument("--body-pitch-bias-deg", type=float, default=-0.4)
    parser.add_argument("--front-swing-clearance-scale", type=float, default=1.18)
    parser.add_argument("--rear-swing-clearance-scale", type=float, default=0.82)
    parser.add_argument("--enable-imu-yaw", action="store_true")
    parser.add_argument("--follow-camera", action="store_true")
    parser.add_argument("--policy-run-dir", default=None)
    parser.add_argument("--policy-model", choices=("best", "final", "checkpoint"), default="best")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def output_paths(runtime_dir: Path) -> dict[str, Path]:
    return {
        "pid": runtime_dir / "worker.pid",
        "status": runtime_dir / "status.json",
        "command": runtime_dir / "command.json",
    }


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2)
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def read_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default.copy()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default.copy()


def write_status(path: Path, **payload) -> None:
    payload.setdefault("timestamp", time.time())
    write_json_atomic(path, payload)


def cleanup_pid_file(pid_path: Path) -> None:
    if not pid_path.exists():
        return
    try:
        recorded_pid = int(pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        return
    if recorded_pid == os.getpid():
        pid_path.unlink()


def request_stop(*_args) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def register_signal_handlers() -> None:
    for sig_name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, request_stop)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def command_to_action(command: dict, gait_mode: str) -> np.ndarray:
    name = str(command.get("command", "stop")).strip().lower()
    speed_scale = clamp(float(command.get("speed_scale", 1.0)), 0.10, 1.0)

    if gait_mode == "four_phase":
        base_step = 0.040 * speed_scale
        base_strafe = 0.030 * speed_scale
        base_turn = 0.050 * speed_scale
        base_velocity = 0.24
        back_step = 0.040 * speed_scale
        turn_step = 0.015 * speed_scale
        lateral_amount = 1.0 * speed_scale
        yaw_sign = 1.0
        lateral_sign = 1.0
    else:
        base_step = clamp(float(command.get("trot_step_length", 0.050)), 0.005, 0.060) * speed_scale
        back_step = clamp(float(command.get("trot_back_step_length", 0.040)), 0.005, 0.050) * speed_scale
        turn_step = clamp(float(command.get("trot_turn_step", 0.010)), 0.0, 0.050) * speed_scale
        base_strafe = clamp(float(command.get("trot_strafe_step", 0.030)), 0.0, 0.040) * speed_scale
        base_turn = clamp(float(command.get("trot_turn_rate", 0.90)), 0.05, 1.20)
        base_velocity = clamp(float(command.get("trot_step_velocity", 0.54)), 0.20, 0.90)
        # The Bezier gait treats lateral amount as a direction angle in radians.
        # Map 1.0 to pi/2 so strafe is perpendicular to the body.
        lateral_amount = clamp(float(command.get("trot_lateral_amount", 0.80)), 0.10, 1.0) * (np.pi / 2.0)
        yaw_sign = 1.0
        lateral_sign = 1.0

    actions = {
        "stop": (0.0, 0.0, 0.0, base_velocity),
        "forward": (base_step, 0.0, 0.0, base_velocity),
        "back": (-back_step, 0.0, 0.0, base_velocity),
        "left": (base_strafe, lateral_sign * lateral_amount, 0.0, base_velocity),
        "right": (base_strafe, -lateral_sign * lateral_amount, 0.0, base_velocity),
        "turn_left": (turn_step, 0.0, yaw_sign * base_turn, base_velocity),
        "turn_right": (turn_step, 0.0, -yaw_sign * base_turn, base_velocity),
        "forward_left": (base_step, lateral_sign * 0.20 * speed_scale, -yaw_sign * 0.14 * speed_scale, base_velocity),
        "forward_right": (base_step, -lateral_sign * 0.20 * speed_scale, -yaw_sign * 0.14 * speed_scale, base_velocity),
    }
    return np.array(actions.get(name, actions["stop"]), dtype=np.float32)


def make_env(args: argparse.Namespace) -> SpotMLWalkEnv:
    return SpotMLWalkEnv(
        render=True,
        on_rack=False,
        terrain_profile=args.terrain_profile,
        terrain_randomization=args.terrain_profile == "rough",
        target_forward_speed=0.0,
        gait_mode=args.gait_mode,
        max_episode_steps=1000000,
        body_height_offset=args.body_height_offset,
        body_roll_bias_deg=args.body_roll_bias_deg,
        body_pitch_bias_deg=args.body_pitch_bias_deg,
        auto_yaw_gain=0.0,
        gait_geometry_profile="balanced" if args.gait_mode == "four_phase" else "default",
        enable_imu_yaw=args.enable_imu_yaw,
        allow_turning_commands=True,
        stabilization_profile=None,
        front_swing_clearance_scale=args.front_swing_clearance_scale,
        rear_swing_clearance_scale=args.rear_swing_clearance_scale,
        gui_safe_mode=False,
        follow_gui_camera=args.follow_camera,
    )


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def get_base_yaw(env: SpotMLWalkEnv) -> float:
    try:
        return float(env.base_env.return_yaw())
    except Exception:
        return 0.0


FORWARD_LEFT_STEP_GAIN = 1.00
FORWARD_RIGHT_STEP_GAIN = 2.00


def apply_live_manual_tuning(env: SpotMLWalkEnv, command: dict, command_name: str, action: np.ndarray) -> None:
    if env.gait_mode != "trot":
        return

    curve_height = clamp(
        float(command.get("trot_curve_height", 0.040)),
        0.015,
        0.080,
    )
    env.clearance_height = curve_height

    # Taller foot arcs need slower swing travel to avoid throwing the body.
    curve_speed_scale = clamp(0.035 / max(curve_height, 1e-6), 0.35, 1.0)
    slowed_velocity = float(action[3] * curve_speed_scale)
    action[3] = slowed_velocity
    env.cmd_vel = slowed_velocity

    # For joystick-style manual control, avoid stale smoothed command state
    # bleeding forward motion into strafe or yaw into straight walking.
    if command_name in {"left", "right"}:
        env.cmd_step = float(action[0])
        env.cmd_lat = float(action[1])
        env.cmd_yaw = 0.0
        env.manual_forward_left_step_gain = 1.0
        env.manual_forward_right_step_gain = 1.0
    elif command_name in {"forward", "back"}:
        env.cmd_lat = 0.0
        env.cmd_yaw = 0.0
        if command_name == "forward":
            env.manual_forward_left_step_gain = FORWARD_LEFT_STEP_GAIN
            env.manual_forward_right_step_gain = FORWARD_RIGHT_STEP_GAIN
        else:
            env.manual_forward_left_step_gain = 1.0
            env.manual_forward_right_step_gain = 1.0
    elif command_name in {"turn_left", "turn_right"}:
        env.cmd_lat = 0.0
        env.manual_forward_left_step_gain = 1.0
        env.manual_forward_right_step_gain = 1.0
    env.manual_curve_height = curve_height
    env.manual_curve_speed_scale = curve_speed_scale


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


def make_policy_playback_args(args: argparse.Namespace) -> argparse.Namespace:
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
        render=True,
        bullet_gui=True,
        unsafe_gui=True,
        follow_camera=args.follow_camera,
        seed=args.seed,
    )


def run_policy_worker(args: argparse.Namespace, paths: dict[str, Path]) -> int:
    if not args.policy_run_dir:
        raise RuntimeError("A policy run directory is required for trained-model mode.")

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from spot_play_ml import (
        load_ppo_model,
        load_run_config,
        patch_numpy_bit_generator_pickle,
        resolve_model_path,
        resolve_run_dir,
        make_env as make_policy_env,
    )

    patch_numpy_bit_generator_pickle()
    patch_numpy_module_aliases_safe()

    run_dir = resolve_run_dir(args.policy_run_dir)
    run_config = load_run_config(run_dir)
    model_path = resolve_model_path(run_dir, args.policy_model, args.checkpoint_path)
    playback_args = make_policy_playback_args(args)
    base_env = DummyVecEnv([make_policy_env(playback_args, run_config)])
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

    model = load_ppo_model(model_path, env, context="manual trained-model app")
    obs = env.reset()
    episode = 1
    episode_step = 0
    episode_reward = 0.0

    try:
        while not STOP_REQUESTED:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            reward_value = float(rewards[0])
            done_value = bool(dones[0])
            info = infos[0] if isinstance(infos, (list, tuple)) else infos

            episode_step += 1
            episode_reward += reward_value
            write_status(
                paths["status"],
                state="running",
                running=True,
                control_source="policy",
                command="policy",
                run_name=run_dir.name,
                run_dir=str(run_dir),
                model_path=str(model_path),
                gait_mode=run_config.get("gait_mode", "trot"),
                terrain_profile=run_config.get("terrain_profile", "flat"),
                vecnormalize_loaded=vecnormalize_loaded,
                vecnormalize_error=vecnormalize_error,
                episode=episode,
                episode_step=episode_step,
                episode_reward=episode_reward,
                forward_speed=(info or {}).get("forward_speed"),
                base_height_error=(info or {}).get("base_height_error"),
                is_fallen=(info or {}).get("is_fallen"),
            )

            if done_value or episode_step >= args.max_steps:
                obs = env.reset()
                episode += 1
                episode_step = 0
                episode_reward = 0.0

        write_status(
            paths["status"],
            state="stopped",
            running=False,
            control_source="policy",
            command="policy",
            run_name=run_dir.name,
            run_dir=str(run_dir),
        )
        return 0
    finally:
        env.close()


def main() -> int:
    args = parse_args()
    runtime_dir = Path(args.runtime_dir).expanduser().resolve()
    paths = output_paths(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    register_signal_handlers()
    atexit.register(cleanup_pid_file, paths["pid"])
    paths["pid"].write_text(str(os.getpid()), encoding="utf-8")

    env = None
    try:
        write_status(
            paths["status"],
            state="starting",
            running=False,
            worker_version=WORKER_VERSION,
            command="stop",
            control_source=args.control_source,
            gait_mode=args.gait_mode,
            terrain_profile=args.terrain_profile,
        )
        if args.control_source == "policy":
            return run_policy_worker(args, paths)

        env = make_env(args)
        obs, _ = env.reset(seed=args.seed)
        del obs

        step_count = 0
        command = {"command": "stop", "speed_scale": 1.0}
        last_command_name = "stop"
        last_command_seen = 0.0
        heading_target_yaw = get_base_yaw(env)

        while not STOP_REQUESTED:
            command = read_json(paths["command"], command)
            command_name = str(command.get("command", "stop")).strip().lower()
            current_yaw = get_base_yaw(env)
            yaw_error = 0.0
            if command.get("reset"):
                obs, _ = env.reset()
                del obs
                command = {"command": "stop", "speed_scale": command.get("speed_scale", 1.0)}
                write_json_atomic(paths["command"], command)
                heading_target_yaw = get_base_yaw(env)

            action = np.clip(
                command_to_action(command, args.gait_mode),
                env.action_space.low,
                env.action_space.high,
            )
            yaw_command = float(action[2])
            if command_name != last_command_name:
                last_command_name = command_name
                last_command_seen = time.time()
                heading_target_yaw = current_yaw

            if args.gait_mode == "trot" and command_name in {"forward", "back"}:
                yaw_error = wrap_angle(current_yaw - heading_target_yaw)
            yaw_command = float(action[2])

            apply_live_manual_tuning(env, command, command_name, action)
            _, _, terminated, truncated, info = env.step(action)
            step_count += 1

            write_status(
                paths["status"],
                state="running",
                running=True,
                worker_version=WORKER_VERSION,
                control_source="manual",
                command=command_name,
                raw_command_payload=command,
                speed_scale=float(command.get("speed_scale", 1.0)),
                gait_mode=args.gait_mode,
                terrain_profile=args.terrain_profile,
                curve_height=float(getattr(env, "clearance_height", 0.0)),
                current_yaw=float(current_yaw),
                heading_target_yaw=float(heading_target_yaw),
                heading_yaw_error=float(yaw_error),
                action_yaw_command=float(yaw_command),
                forward_left_step_gain=float(getattr(env, "manual_forward_left_step_gain", 1.0)),
                forward_right_step_gain=float(getattr(env, "manual_forward_right_step_gain", 1.0)),
                step_count=step_count,
                forward_speed=info.get("forward_speed"),
                base_height_error=info.get("base_height_error"),
                is_fallen=info.get("is_fallen"),
                last_command_seen=last_command_seen,
            )

            if terminated or truncated or info.get("is_fallen"):
                obs, _ = env.reset()
                del obs
                write_json_atomic(
                    paths["command"],
                    {"command": "stop", "speed_scale": command.get("speed_scale", 1.0)},
                )

        write_status(
            paths["status"],
            state="stopped",
            running=False,
            worker_version=WORKER_VERSION,
            control_source="manual",
            command=last_command_name,
            gait_mode=args.gait_mode,
            terrain_profile=args.terrain_profile,
            step_count=step_count,
        )
        return 0
    except Exception as exc:
        write_status(paths["status"], state="error", running=False, error=str(exc))
        traceback.print_exc()
        print(f"Manual PyBullet worker failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
