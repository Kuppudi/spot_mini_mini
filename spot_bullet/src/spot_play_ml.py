#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache")

os.makedirs(os.path.join(CACHE_DIR, "matplotlib"), exist_ok=True)
os.makedirs(os.path.join(CACHE_DIR, "fontconfig"), exist_ok=True)

os.environ.setdefault("XDG_CACHE_HOME", CACHE_DIR)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(CACHE_DIR, "matplotlib"))
os.environ.setdefault("FONTCONFIG_PATH", os.path.join(CACHE_DIR, "fontconfig"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def dependency_error(exc: ModuleNotFoundError) -> None:
    missing_module = exc.name or "a required package"
    raise SystemExit(
        f"Missing dependency: {missing_module}\n"
        "Install the same environment you used for training before running playback."
    ) from exc


try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ModuleNotFoundError as exc:
    dependency_error(exc)

try:
    from spot_ml import (
        SpotMLDualIMUStableWalkEnv,
        SpotMLRoughDualIMUStableWalkEnv,
        SpotMLRoughTeacherJointResidualEnv,
        SpotMLWalkEnv,
    )
except ModuleNotFoundError as exc:
    dependency_error(exc)

try:
    from spot_rough_terrain_ml import SpotMLRoughTerrainEnv
except ModuleNotFoundError as exc:
    dependency_error(exc)

try:
    from rough_terrain_height_residuals import SpotMLRoughTerrainHeightResidualEnv
except ModuleNotFoundError as exc:
    dependency_error(exc)

try:
    from spot_visualization_ml import compose_preview_frame, record_policy_preview, unwrap_env
except ModuleNotFoundError as exc:
    dependency_error(exc)


def patch_numpy_bit_generator_pickle() -> None:
    """Allow loading VecNormalize pickles created under newer NumPy builds."""
    try:
        import numpy.random._pickle as numpy_pickle
    except Exception:
        return

    original_ctor = getattr(numpy_pickle, "__bit_generator_ctor", None)
    if original_ctor is None or getattr(original_ctor, "_spotmini_patched", False):
        return

    def compat_bit_generator_ctor(bit_generator_name="MT19937"):
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__
        return original_ctor(bit_generator_name)

    compat_bit_generator_ctor._spotmini_patched = True
    numpy_pickle.__bit_generator_ctor = compat_bit_generator_ctor


def patch_numpy_module_aliases() -> None:
    """Bridge NumPy internal module renames across saved model environments."""
    numpy_core = getattr(np, "_core", None)
    if numpy_core is None:
        numpy_core = np.core
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play back a trained SpotMini walking policy."
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Training run directory. Defaults to the newest folder in spot_bullet/training_runs.",
    )
    parser.add_argument(
        "--model",
        choices=("best", "final", "checkpoint"),
        default="best",
        help="Which saved model to load from the run directory.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Explicit checkpoint zip path used when --model checkpoint.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="How many episodes to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum playback steps per episode before forcing a reset. Defaults to the environment horizon.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Evaluation seed.",
    )
    parser.add_argument(
        "--terrain-profile",
        choices=("flat", "rough"),
        default=None,
        help="Override the terrain preset. 'rough' enables the heightfield and rough-terrain walk defaults.",
    )
    parser.add_argument(
        "--height-field",
        action="store_true",
        help="Deprecated alias for --terrain-profile rough.",
    )
    parser.add_argument(
        "--terrain-randomization",
        action="store_true",
        help="Randomize terrain and physical properties on reset during playback.",
    )
    parser.add_argument(
        "--gait-mode",
        choices=("trot", "four_phase"),
        default=None,
        help="Override the gait mode. Defaults to the value stored in run_config.json when available.",
    )
    parser.add_argument(
        "--target-forward-speed",
        type=float,
        default=None,
        help="Forward speed target for the wrapper reward.",
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=None,
        help="Override the environment's internal episode horizon. Defaults to the training run value.",
    )
    parser.add_argument(
        "--body-height-offset",
        type=float,
        default=None,
        help="Override the body z offset used by IK. Positive values lower the body posture.",
    )
    parser.add_argument(
        "--body-pitch-bias-deg",
        type=float,
        default=None,
        help="Override the fixed torso pitch trim in degrees. Negative values tilt the nose down.",
    )
    parser.add_argument(
        "--auto-yaw-gain",
        type=float,
        default=None,
        help="Override the gait wrapper's yaw stabilization gain.",
    )
    parser.add_argument(
        "--gait-geometry",
        choices=("default", "contact_stable", "balanced", "anti_right_skew", "anti_rock", "wide_stable", "distance"),
        default=None,
        help="Override the named gait geometry profile used by the wrapper.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open a visualization window during playback.",
    )
    parser.add_argument(
        "--bullet-gui",
        action="store_true",
        help="Use the native PyBullet GUI instead of the safer live preview window.",
    )
    parser.add_argument(
        "--unsafe-gui",
        action="store_true",
        help="Disable the macOS-safe PyBullet GUI fallback when --bullet-gui is used.",
    )
    parser.add_argument(
        "--follow-camera",
        action="store_true",
        help="Continuously keep the PyBullet camera centered on the robot during playback.",
    )
    parser.add_argument(
        "--enable-camera-observation",
        dest="enable_camera_observation",
        action="store_true",
        help="Enable rough-terrain camera summary features during playback.",
    )
    parser.add_argument(
        "--disable-camera-observation",
        dest="enable_camera_observation",
        action="store_false",
        help="Disable rough-terrain camera summary features during playback.",
    )
    parser.set_defaults(enable_camera_observation=None)
    parser.add_argument(
        "--disable-camera-leveling",
        action="store_true",
        help="Let the rough-terrain camera pitch/roll with the body instead of stabilizing to the horizon.",
    )
    parser.add_argument(
        "--camera-pitch-offset-deg",
        type=float,
        default=None,
        help="Override the fixed downward pitch used by the stabilized rough-terrain camera view.",
    )
    parser.add_argument(
        "--save-gif",
        default=None,
        help="Optional path for a rendered playback GIF preview.",
    )
    parser.add_argument(
        "--gif-steps",
        type=int,
        default=220,
        help="Maximum rollout steps to include in --save-gif output.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=12,
        help="Frame rate for --save-gif output.",
    )
    return parser.parse_args()


def get_training_root() -> Path:
    return Path(PROJECT_ROOT) / "spot_bullet" / "training_runs"


def resolve_run_dir(run_dir_arg: str | None) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg).expanduser().resolve()
        if not run_dir.exists():
            raise SystemExit(f"Run directory not found: {run_dir}")
        return run_dir

    training_root = get_training_root()
    if not training_root.exists():
        raise SystemExit(f"No training_runs directory found at: {training_root}")

    run_dirs = sorted(
        [path for path in training_root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        raise SystemExit(f"No run directories found in: {training_root}")
    return run_dirs[0]


def resolve_model_path(run_dir: Path, model_kind: str, checkpoint_path: str | None) -> Path:
    if model_kind == "best":
        path = run_dir / "best_model" / "best_model.zip"
    elif model_kind == "final":
        path = run_dir / "models" / "ppo_spot_walk_final.zip"
    else:
        if checkpoint_path is None:
            raise SystemExit("--checkpoint-path is required when --model checkpoint.")
        path = Path(checkpoint_path).expanduser().resolve()

    if not path.exists():
        raise SystemExit(f"Model file not found: {path}")
    return path


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def make_env(args, run_config):
    def _init():
        walk_variant = run_config.get("walk_variant", "default")
        gait_mode = args.gait_mode or run_config.get("gait_mode", "trot")
        terrain_profile = args.terrain_profile
        if terrain_profile is None:
            terrain_profile = run_config.get("terrain_profile")
        if terrain_profile is None:
            terrain_profile = "rough" if (args.height_field or run_config.get("height_field", False)) else "flat"
        height_field = terrain_profile == "rough"
        terrain_randomization = bool(
            args.terrain_randomization
            or run_config.get("terrain_randomization", False)
            or height_field
        )
        target_forward_speed = args.target_forward_speed
        if target_forward_speed is None:
            target_forward_speed = run_config.get("target_forward_speed", target_forward_speed)
        if gait_mode == "four_phase" and target_forward_speed is None:
            if terrain_profile == "rough":
                target_forward_speed = SpotMLWalkEnv.DEFAULT_ROUGH_WALK_TARGET_FORWARD_SPEED
            else:
                target_forward_speed = SpotMLWalkEnv.DEFAULT_WALK_TARGET_FORWARD_SPEED
        elif target_forward_speed is None:
            target_forward_speed = 0.35
        max_episode_steps = args.episode_steps
        if max_episode_steps is None:
            max_episode_steps = run_config.get("episode_steps", 2000)
        if walk_variant in ("dual_imu_stable", "rough_teacher_joint_residual") and max_episode_steps <= 2000:
            max_episode_steps = 6000 if terrain_profile == "flat" else 4000
        body_height_offset = args.body_height_offset
        if body_height_offset is None:
            body_height_offset = run_config.get("body_height_offset")
        if body_height_offset is None and gait_mode == "four_phase":
            if terrain_profile == "rough":
                body_height_offset = SpotMLWalkEnv.DEFAULT_ROUGH_WALK_BODY_HEIGHT_OFFSET
            else:
                body_height_offset = SpotMLWalkEnv.DEFAULT_WALK_BODY_HEIGHT_OFFSET
        body_pitch_bias_deg = args.body_pitch_bias_deg
        if body_pitch_bias_deg is None:
            body_pitch_bias_deg = run_config.get("body_pitch_bias_deg")
        if body_pitch_bias_deg is None and gait_mode == "four_phase":
            if terrain_profile == "rough":
                body_pitch_bias_deg = SpotMLWalkEnv.DEFAULT_ROUGH_WALK_BODY_PITCH_BIAS_DEG
            else:
                body_pitch_bias_deg = SpotMLWalkEnv.DEFAULT_WALK_BODY_PITCH_BIAS_DEG
        elif body_pitch_bias_deg is None:
            body_pitch_bias_deg = 0.0
        auto_yaw_gain = args.auto_yaw_gain
        if auto_yaw_gain is None:
            auto_yaw_gain = run_config.get("auto_yaw_gain")
        if auto_yaw_gain is None and gait_mode == "four_phase":
            if terrain_profile == "rough":
                auto_yaw_gain = SpotMLWalkEnv.DEFAULT_ROUGH_WALK_AUTO_YAW_GAIN
            else:
                auto_yaw_gain = SpotMLWalkEnv.DEFAULT_WALK_AUTO_YAW_GAIN
        gait_geometry = args.gait_geometry
        if gait_geometry is None:
            gait_geometry = run_config.get("gait_geometry")
        if gait_geometry is None and gait_mode == "four_phase":
            if terrain_profile == "rough":
                gait_geometry = SpotMLWalkEnv.DEFAULT_ROUGH_WALK_GEOMETRY_PROFILE
            else:
                gait_geometry = SpotMLWalkEnv.DEFAULT_WALK_GEOMETRY_PROFILE
        elif gait_geometry is None:
            gait_geometry = "default"
        rough_variant = str(run_config.get("rough_env_variant", ""))
        use_residual_rough_env_module = (
            terrain_profile == "rough"
            and rough_variant.startswith("foothold_height_residual")
        )
        use_rough_env_module = (
            terrain_profile == "rough"
            and rough_variant.startswith("foothold_aware")
        )
        enable_camera_observation = args.enable_camera_observation
        if enable_camera_observation is None:
            enable_camera_observation = run_config.get(
                "enable_camera_observation",
                rough_variant != "foothold_aware_v1" and terrain_profile == "rough",
            )
        camera_stabilize_roll_pitch = not args.disable_camera_leveling
        if not args.disable_camera_leveling:
            camera_stabilize_roll_pitch = run_config.get(
                "camera_stabilize_roll_pitch",
                rough_variant != "foothold_aware_v1" and terrain_profile == "rough",
            )
        camera_pitch_offset_deg = args.camera_pitch_offset_deg
        if camera_pitch_offset_deg is None:
            camera_pitch_offset_deg = run_config.get(
                "camera_pitch_offset_deg",
                -10.0 if rough_variant != "foothold_aware_v1" else 0.0,
            )
        use_native_bullet_gui = bool(
            args.render and (args.bullet_gui or sys.platform != "darwin")
        )
        if walk_variant == "rough_teacher_joint_residual":
            env_class = SpotMLRoughTeacherJointResidualEnv
        elif walk_variant == "rough_dual_imu":
            env_class = SpotMLRoughDualIMUStableWalkEnv
        elif walk_variant == "dual_imu_stable":
            env_class = SpotMLDualIMUStableWalkEnv
        elif use_residual_rough_env_module:
            env_class = SpotMLRoughTerrainHeightResidualEnv
        elif use_rough_env_module:
            env_class = SpotMLRoughTerrainEnv
        else:
            env_class = SpotMLWalkEnv
        env_kwargs = dict(
            render=use_native_bullet_gui,
            on_rack=False,
            terrain_profile=terrain_profile,
            terrain_randomization=terrain_randomization,
            draw_foot_path=False,
            env_randomizer=None,
            target_forward_speed=target_forward_speed,
            gait_mode=gait_mode,
            max_episode_steps=max_episode_steps,
            body_height_offset=body_height_offset,
            body_pitch_bias_deg=body_pitch_bias_deg,
            auto_yaw_gain=auto_yaw_gain,
            gait_geometry_profile=gait_geometry,
            gui_safe_mode=False if (use_native_bullet_gui and args.unsafe_gui) else None,
            follow_gui_camera=args.follow_camera if use_native_bullet_gui else None,
        )
        if env_class in (
            SpotMLRoughTerrainEnv,
            SpotMLRoughTerrainHeightResidualEnv,
            SpotMLRoughDualIMUStableWalkEnv,
            SpotMLRoughTeacherJointResidualEnv,
        ):
            env_kwargs.update(
                enable_camera_observation=enable_camera_observation,
                camera_stabilize_roll_pitch=camera_stabilize_roll_pitch,
                camera_pitch_offset_deg=camera_pitch_offset_deg,
            )
        if env_class is SpotMLRoughTeacherJointResidualEnv:
            env_kwargs.update(
                teacher_model_path=run_config.get("teacher_model_path"),
                teacher_vecnormalize_path=run_config.get("teacher_vecnormalize_path"),
            )
        env = env_class(**env_kwargs)
        env.reset(seed=args.seed)
        return Monitor(env)

    return _init


def main():
    args = parse_args()
    patch_numpy_bit_generator_pickle()
    patch_numpy_module_aliases()
    run_dir = resolve_run_dir(args.run_dir)
    run_config = load_run_config(run_dir)
    model_path = resolve_model_path(run_dir, args.model, args.checkpoint_path)
    vecnormalize_path = run_dir / "vecnormalize" / "vecnormalize.pkl"

    print(f"Using run directory: {run_dir}")
    print(f"Loading model: {model_path}")
    print(f"Loading VecNormalize stats: {vecnormalize_path}")
    print(f"Walk variant: {run_config.get('walk_variant', 'default')}")
    print(f"Gait mode: {args.gait_mode or run_config.get('gait_mode', 'trot')}")
    walk_variant = run_config.get("walk_variant", "default")
    effective_episode_steps = args.episode_steps
    if effective_episode_steps is None:
        effective_episode_steps = run_config.get("episode_steps", 2000)
    effective_terrain_profile = args.terrain_profile or run_config.get("terrain_profile")
    if effective_terrain_profile is None:
        effective_terrain_profile = "rough" if (args.height_field or run_config.get("height_field", False)) else "flat"
    if walk_variant in ("dual_imu_stable", "rough_dual_imu", "rough_teacher_joint_residual") and effective_episode_steps <= 2000:
        effective_episode_steps = 6000 if effective_terrain_profile == "flat" else 6000
    effective_body_pitch_bias_deg = args.body_pitch_bias_deg
    if effective_body_pitch_bias_deg is None:
        effective_body_pitch_bias_deg = run_config.get("body_pitch_bias_deg")
    if effective_body_pitch_bias_deg is None:
        effective_body_pitch_bias_deg = (
            SpotMLWalkEnv.DEFAULT_ROUGH_WALK_BODY_PITCH_BIAS_DEG
            if effective_terrain_profile == "rough"
            else SpotMLWalkEnv.DEFAULT_WALK_BODY_PITCH_BIAS_DEG
        )
    effective_max_steps = args.max_steps if args.max_steps is not None else effective_episode_steps
    print(f"Episode horizon: {effective_episode_steps}")
    print(f"Body pitch bias (deg): {effective_body_pitch_bias_deg}")
    print(f"Playback max steps: {effective_max_steps}")
    if (
        str(run_config.get("rough_env_variant", "")).startswith("foothold_aware")
        or str(run_config.get("rough_env_variant", "")).startswith("foothold_height_residual")
    ):
        rough_variant = str(run_config.get("rough_env_variant", ""))
        print(
            "Rough camera config: "
            f"enabled={run_config.get('enable_camera_observation', rough_variant != 'foothold_aware_v1')}, "
            f"stabilized={run_config.get('camera_stabilize_roll_pitch', rough_variant != 'foothold_aware_v1')}, "
            f"pitch_offset_deg={run_config.get('camera_pitch_offset_deg', -10.0 if rough_variant != 'foothold_aware_v1' else 0.0)}"
        )
    use_native_bullet_gui = bool(
        args.render and (args.bullet_gui or sys.platform != "darwin")
    )
    use_live_preview = bool(args.render and not use_native_bullet_gui)
    if use_live_preview:
        print("Render mode: live preview window (macOS fallback)")
    elif use_native_bullet_gui:
        print("Render mode: native PyBullet GUI")

    base_env = DummyVecEnv([make_env(args, run_config)])
    env = base_env
    if vecnormalize_path.exists():
        try:
            env = VecNormalize.load(str(vecnormalize_path), base_env)
            env.training = False
            env.norm_reward = False
            print("VecNormalize stats loaded successfully.")
        except Exception as exc:
            print(
                "Warning: failed to load VecNormalize stats in this environment. "
                "Continuing with the raw environment instead."
            )
            print(f"VecNormalize load error: {exc}")
    else:
        print(
            "Warning: VecNormalize stats were not found for this run. "
            "Continuing with the raw environment instead."
        )

    model = PPO.load(
        str(model_path),
        env=env,
        device="auto",
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        },
    )

    raw_env = unwrap_env(env)
    live_preview_enabled = False
    live_preview_closed = False
    if use_live_preview:
        try:
            import cv2
        except ModuleNotFoundError:
            print("OpenCV is not available in this environment, so live preview is disabled.")
            cv2 = None
        if cv2 is not None:
            live_preview_enabled = True
            print("Live preview controls: press q or Esc to close the preview window.")

    if args.save_gif:
        gif_path = Path(args.save_gif).expanduser().resolve()
        preview_meta = record_policy_preview(
            model,
            env,
            gif_path,
            max_steps=args.gif_steps,
            fps=args.gif_fps,
            title=f"{run_dir.name} ({args.model})",
            training_timestep=None,
            still_path=gif_path.with_suffix(".png"),
        )
        if preview_meta is not None:
            print(
                f"Saved playback GIF: {gif_path} "
                f"({preview_meta['frames']} frames, return={preview_meta['reward']:.2f})"
            )

    obs = env.reset()
    episode_reward = 0.0
    episode_step = 0
    episode_num = 1

    try:
        while episode_num <= args.episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            episode_reward += float(rewards[0])
            episode_step += 1
            info = infos[0] if isinstance(infos, (list, tuple)) else infos

            if live_preview_enabled:
                frame = compose_preview_frame(
                    raw_env,
                    info or {},
                    cumulative_reward=episode_reward,
                    episode_step=episode_step,
                    title=f"{run_dir.name} ({args.model})",
                )
                cv2.imshow("SpotML Playback", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    live_preview_closed = True
                    break

            hit_max_steps = effective_max_steps is not None and episode_step >= effective_max_steps
            if dones[0] or hit_max_steps:
                done_reason = "env_done" if dones[0] else "max_steps"
                print(
                    f"Episode {episode_num}: reward={episode_reward:.2f}, "
                    f"steps={episode_step}, reason={done_reason}"
                )
                obs = env.reset()
                episode_reward = 0.0
                episode_step = 0
                episode_num += 1
    finally:
        if live_preview_enabled:
            cv2.destroyAllWindows()
        if live_preview_closed:
            print("Live preview closed by user.")
        env.close()


if __name__ == "__main__":
    main()
