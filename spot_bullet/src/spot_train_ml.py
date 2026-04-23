#!/usr/bin/env python3

import argparse
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache")
TRAINING_ROOT_NAMES = ("training runs", "training_runs")
DEFAULT_QUICK_ROUGH_WARMSTART_NAME = "rough_walk_height_residual_v3"
DEFAULT_QUICK_ROUGH_DUAL_IMU_WARMSTART_NAME = "dual_imu_stable_forward_v2"
DEFAULT_TEACHER_LONGWALK_V2_NAME = "dual_imu_longwalk_v2"
DEFAULT_TEACHER_LONGWALK_V1_NAME = "dual_imu_longwalk_v1"

os.makedirs(os.path.join(CACHE_DIR, "matplotlib"), exist_ok=True)
os.makedirs(os.path.join(CACHE_DIR, "fontconfig"), exist_ok=True)

os.environ.setdefault("XDG_CACHE_HOME", CACHE_DIR)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(CACHE_DIR, "matplotlib"))
os.environ.setdefault("FONTCONFIG_PATH", os.path.join(CACHE_DIR, "fontconfig"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def dependency_error(exc: ModuleNotFoundError) -> None:
    missing_module = exc.name or "a required package"
    requirements_path = os.path.join(PROJECT_ROOT, "spot_bullet", "requirements.txt")
    raise SystemExit(
        f"Missing dependency: {missing_module}\n"
        f"Install the packages from {requirements_path} before training.\n"
        "Recommended setup on Apple Silicon:\n"
        "  conda create -p ./.conda-spotml -c conda-forge python=3.10 numpy=1.26.4 scipy=1.11.4 matplotlib=3.8.2 pybullet pip -y\n"
        "  ./.conda-spotml/bin/python -m pip install 'setuptools<81' 'torch>=2.0.0' stable-baselines3==2.3.2 gym==0.26.2 gymnasium==0.29.1 filterpy==1.4.5 tensorboard tqdm joblib"
    ) from exc


try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization
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
    from spot_visualization_ml import record_policy_preview, write_eval_progress_plot
except ModuleNotFoundError as exc:
    dependency_error(exc)


def _spotmini_compat_bit_generator_ctor(bit_generator_name="MT19937"):
    """Pickle-safe compatibility wrapper for NumPy bit generator restoration."""
    import numpy.random._pickle as numpy_pickle

    original_ctor = getattr(
        _spotmini_compat_bit_generator_ctor,
        "_spotmini_original_ctor",
        None,
    )
    if original_ctor is None:
        original_ctor = getattr(numpy_pickle, "__bit_generator_ctor", None)
        _spotmini_compat_bit_generator_ctor._spotmini_original_ctor = original_ctor
    if original_ctor is None:
        raise AttributeError("NumPy pickle constructor is unavailable")
    if isinstance(bit_generator_name, type):
        bit_generator_name = bit_generator_name.__name__
    return original_ctor(bit_generator_name)


def patch_numpy_bit_generator_pickle() -> None:
    """Allow loading VecNormalize pickles created under newer NumPy builds."""
    try:
        import numpy.random._pickle as numpy_pickle
    except Exception:
        return

    original_ctor = getattr(numpy_pickle, "__bit_generator_ctor", None)
    compat_ctor = _spotmini_compat_bit_generator_ctor
    if original_ctor is None or getattr(original_ctor, "_spotmini_patched", False):
        return
    compat_ctor._spotmini_original_ctor = original_ctor
    compat_ctor._spotmini_patched = True
    numpy_pickle.__bit_generator_ctor = compat_ctor


def patch_numpy_module_aliases() -> None:
    """Bridge NumPy internal module renames across saved model environments."""
    import numpy as np

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


def training_root_candidates() -> list[Path]:
    spot_bullet_root = Path(PROJECT_ROOT) / "spot_bullet"
    return [spot_bullet_root / root_name for root_name in TRAINING_ROOT_NAMES]


def default_output_root() -> str:
    for candidate in training_root_candidates():
        if candidate.is_dir():
            return str(candidate)
    return str(training_root_candidates()[0])


def resolve_named_run_dir(run_name: str) -> str | None:
    for root in training_root_candidates():
        candidate = root / run_name
        if candidate.is_dir():
            return str(candidate.resolve())
    return None


def resolve_repo_run_dir(run_dir_arg: str | None) -> Path | None:
    if not run_dir_arg:
        return None

    run_dir = Path(run_dir_arg).expanduser()
    if run_dir.exists():
        return run_dir.resolve()

    for root in training_root_candidates():
        candidate = root / run_dir.name
        if candidate.is_dir():
            return candidate.resolve()

    return run_dir.resolve()


class PreviewArtifactsCallback(BaseCallback):
    def __init__(self, args, run_dirs, preview_freq, preview_steps, preview_fps):
        super().__init__()
        self.args = args
        self.run_dirs = run_dirs
        self.preview_freq = int(preview_freq)
        self.preview_steps = int(preview_steps)
        self.preview_fps = int(preview_fps)
        self.preview_env = None
        self.preview_dir = os.path.join(run_dirs["base_dir"], "previews")
        self.latest_gif_path = os.path.join(self.preview_dir, "preview_latest.gif")
        self.latest_png_path = os.path.join(self.preview_dir, "preview_latest.png")
        self.progress_plot_path = os.path.join(self.preview_dir, "eval_progress.png")
        self.last_preview_step = 0

        os.makedirs(self.preview_dir, exist_ok=True)

    def _on_training_start(self):
        if self.preview_freq <= 0:
            return

        base_env = DummyVecEnv([make_env(rank=20_000, args=self.args)])
        self.preview_env = VecNormalize(
            base_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            gamma=0.99,
            training=False,
        )
        self.preview_env.seed(self.args.seed + 20_000)

    def _maybe_write_preview(self):
        if self.preview_env is None:
            return

        try:
            sync_envs_normalization(self.training_env, self.preview_env)
        except Exception as exc:
            print(f"Warning: failed to sync preview normalization stats: {exc}")

        preview_path = os.path.join(
            self.preview_dir,
            f"preview_{self.num_timesteps:09d}.gif",
        )
        preview_meta = record_policy_preview(
            self.model,
            self.preview_env,
            preview_path,
            max_steps=self.preview_steps,
            fps=self.preview_fps,
            title=self.run_dirs["base_dir"].split(os.sep)[-1],
            training_timestep=self.num_timesteps,
            still_path=self.latest_png_path,
        )
        if preview_meta is not None:
            shutil.copyfile(preview_path, self.latest_gif_path)
            print(
                f"Saved visual preview: {preview_path} "
                f"({preview_meta['frames']} frames, return={preview_meta['reward']:.2f})"
            )

        evaluations_path = os.path.join(self.run_dirs["log_dir"], "evaluations.npz")
        if write_eval_progress_plot(
            evaluations_path,
            self.progress_plot_path,
            title=f"{self.run_dirs['base_dir'].split(os.sep)[-1]} eval reward",
        ):
            print(f"Updated evaluation plot: {self.progress_plot_path}")

    def _on_step(self):
        if self.preview_freq <= 0 or self.preview_env is None:
            return True
        if self.num_timesteps - self.last_preview_step < self.preview_freq:
            return True
        self.last_preview_step = self.num_timesteps
        self._maybe_write_preview()
        return True

    def _on_training_end(self):
        try:
            self._maybe_write_preview()
        finally:
            if self.preview_env is not None:
                self.preview_env.close()
                self.preview_env = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a SpotMini walking policy from scratch."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Total PPO timesteps to train from scratch.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel training environments.",
    )
    parser.add_argument(
        "--target-forward-speed",
        type=float,
        default=None,
        help="Forward speed target used in the reward function.",
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=None,
        help="Maximum number of simulator steps per episode before truncation. Defaults depend on the walk variant.",
    )
    parser.add_argument(
        "--body-height-offset",
        type=float,
        default=None,
        help="Body z offset applied through IK. Positive values lower the body posture.",
    )
    parser.add_argument(
        "--body-pitch-bias-deg",
        type=float,
        default=None,
        help="Fixed torso pitch trim in degrees applied before IK. Negative values tilt the nose down.",
    )
    parser.add_argument(
        "--auto-yaw-gain",
        type=float,
        default=None,
        help="Optional proportional yaw correction gain applied inside the gait wrapper.",
    )
    parser.add_argument(
        "--gait-geometry",
        choices=("default", "contact_stable", "balanced", "anti_right_skew", "anti_rock", "wide_stable", "distance"),
        default=None,
        help="Named gait geometry profile controlling default stance footprint.",
    )
    parser.add_argument(
        "--gait-mode",
        choices=("trot", "four_phase"),
        default="four_phase",
        help="Underlying Bezier gait scheduler used by the ML environment.",
    )
    parser.add_argument(
        "--walk-variant",
        choices=("default", "dual_imu_stable", "rough_dual_imu", "rough_teacher_joint_residual"),
        default="default",
        help="Training environment variant. 'dual_imu_stable' adds front/rear virtual IMUs and per-leg stride control, 'rough_dual_imu' combines those sensors with rough-terrain foothold features, and 'rough_teacher_joint_residual' trains bounded joint corrections on top of a longwalk teacher policy.",
    )
    parser.add_argument(
        "--training-preset",
        choices=("none", "quick_rough", "quick_rough_dual_imu", "quick_rough_teacher_joint"),
        default="none",
        help="Optional training preset. 'quick_rough' warm-starts a rough-terrain walker, 'quick_rough_dual_imu' warm-starts a rough-terrain dual-IMU walker, and 'quick_rough_teacher_joint' trains rough-terrain joint residuals on top of a longwalk teacher.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible fresh runs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run folder name. Defaults to a timestamped fresh run.",
    )
    parser.add_argument(
        "--allow-run-overwrite",
        action="store_true",
        help="Allow writing into an existing run folder. By default, existing run folders are protected.",
    )
    parser.add_argument(
        "--init-from-run",
        default=None,
        help="Optional existing run directory to warm-start from before training.",
    )
    parser.add_argument(
        "--init-model",
        choices=("best", "final"),
        default="best",
        help="Which checkpoint to warm-start from when --init-from-run is provided.",
    )
    parser.add_argument(
        "--output-root",
        default=default_output_root(),
        help="Folder where checkpoints, logs, and normalization stats are saved.",
    )
    parser.add_argument(
        "--terrain-profile",
        choices=("flat", "rough"),
        default="flat",
        help="Terrain preset to train on. 'rough' enables the heightfield and terrain randomization defaults.",
    )
    parser.add_argument(
        "--height-field",
        action="store_true",
        help="Deprecated alias for --terrain-profile rough.",
    )
    parser.add_argument(
        "--terrain-randomization",
        action="store_true",
        help="Randomize terrain and physical properties on reset. Enabled automatically for rough terrain.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the first environment while training.",
    )
    parser.add_argument(
        "--enable-camera-observation",
        dest="enable_camera_observation",
        action="store_true",
        help="Enable rough-terrain camera summary features during training.",
    )
    parser.add_argument(
        "--disable-camera-observation",
        dest="enable_camera_observation",
        action="store_false",
        help="Disable rough-terrain camera summary features during training.",
    )
    parser.set_defaults(enable_camera_observation=None)
    parser.add_argument(
        "--disable-camera-leveling",
        action="store_true",
        help="Let the rough-terrain camera pitch/roll with the body instead of stabilizing to the horizon.",
    )
    parser.add_argument(
        "--enable-imu-yaw",
        dest="enable_imu_yaw",
        action="store_true",
        help="Expose yaw-aware IMU features to the policy and enable yaw stabilization in the body controller.",
    )
    parser.add_argument(
        "--disable-imu-yaw",
        dest="enable_imu_yaw",
        action="store_false",
        help="Disable yaw-aware IMU features and yaw stabilization in the body controller.",
    )
    parser.set_defaults(enable_imu_yaw=None)
    parser.add_argument(
        "--camera-pitch-offset-deg",
        type=float,
        default=-10.0,
        help="Fixed downward pitch applied to the stabilized rough-terrain camera view.",
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="Skip SB3's environment checker to speed up startup.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Enable SB3's optional progress bar when tqdm and rich are installed.",
    )
    parser.add_argument(
        "--preview-freq",
        type=int,
        default=10_000,
        help="How often to write a visual training preview GIF and eval plot. Use 0 to disable.",
    )
    parser.add_argument(
        "--preview-steps",
        type=int,
        default=220,
        help="Maximum preview rollout steps per saved training GIF.",
    )
    parser.add_argument(
        "--preview-fps",
        type=int,
        default=12,
        help="Frame rate for saved training preview GIFs.",
    )
    parser.add_argument(
        "--teacher-run-dir",
        default=None,
        help="Run directory containing the teacher policy used by rough_teacher_joint_residual.",
    )
    parser.add_argument(
        "--teacher-model",
        choices=("best", "final"),
        default="best",
        help="Which teacher checkpoint to use when --teacher-run-dir is provided.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO optimizer learning rate.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="PPO entropy coefficient. Lower values reduce exploration noise during fine-tuning.",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO policy clip range.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="PPO minibatch size.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of PPO epochs per rollout batch.",
    )
    return parser.parse_args()


def resolve_walk_defaults(args):
    init_run_hint = ""
    if args.init_from_run:
        init_run_hint = str(Path(args.init_from_run)).lower()
    is_longwalk_transfer = "longwalk" in init_run_hint

    if args.height_field:
        args.terrain_profile = "rough"
    if args.training_preset == "quick_rough":
        args.terrain_profile = "rough"
        args.gait_mode = "four_phase"
        args.walk_variant = "default"
        args.terrain_randomization = True
        if args.timesteps == 50_000:
            args.timesteps = 100_000
        if args.run_name is None:
            args.run_name = "rough_quick_tune_v1"
        if args.init_from_run is None:
            args.init_from_run = resolve_named_run_dir(DEFAULT_QUICK_ROUGH_WARMSTART_NAME)
        if args.preview_freq == 10_000:
            args.preview_freq = 5_000
        if args.preview_steps == 220:
            args.preview_steps = 280
    elif args.training_preset == "quick_rough_dual_imu":
        args.terrain_profile = "rough"
        args.gait_mode = "four_phase"
        args.walk_variant = "rough_dual_imu"
        args.terrain_randomization = True
        if args.timesteps == 50_000:
            args.timesteps = 100_000
        if args.run_name is None:
            args.run_name = "rough_dual_imu_tune_v1"
        if args.init_from_run is None:
            args.init_from_run = resolve_named_run_dir(DEFAULT_QUICK_ROUGH_DUAL_IMU_WARMSTART_NAME)
        if args.preview_freq == 10_000:
            args.preview_freq = 5_000
        if args.preview_steps == 220:
            args.preview_steps = 280
    elif args.training_preset == "quick_rough_teacher_joint":
        args.terrain_profile = "rough"
        args.gait_mode = "four_phase"
        args.walk_variant = "rough_teacher_joint_residual"
        args.terrain_randomization = True
        if args.timesteps == 50_000:
            args.timesteps = 80_000
        if args.run_name is None:
            args.run_name = "rough_teacher_joint_v1"
        if args.preview_freq == 10_000:
            args.preview_freq = 5_000
        if args.preview_steps == 220:
            args.preview_steps = 280
    args.height_field = args.terrain_profile == "rough"
    args.terrain_randomization = bool(args.terrain_randomization or args.height_field)
    if args.enable_camera_observation is None:
        args.enable_camera_observation = args.terrain_profile == "rough"
    if args.enable_imu_yaw is None:
        args.enable_imu_yaw = True
    args.camera_stabilize_roll_pitch = not args.disable_camera_leveling

    if args.walk_variant in ("dual_imu_stable", "rough_dual_imu", "rough_teacher_joint_residual"):
        if args.episode_steps is None:
            if args.training_preset in ("quick_rough_dual_imu", "quick_rough_teacher_joint") and args.terrain_profile == "rough":
                args.episode_steps = 6000
            else:
                args.episode_steps = 6000 if args.terrain_profile == "flat" else 4000
        if args.target_forward_speed is None:
            if args.training_preset == "quick_rough_teacher_joint" and args.terrain_profile == "rough":
                args.target_forward_speed = 0.070
            elif args.training_preset == "quick_rough_dual_imu" and args.terrain_profile == "rough":
                args.target_forward_speed = 0.070 if is_longwalk_transfer else 0.055
            else:
                args.target_forward_speed = (
                    SpotMLDualIMUStableWalkEnv.DEFAULT_ROUGH_STABLE_TARGET_FORWARD_SPEED
                    if args.terrain_profile == "rough"
                    else SpotMLDualIMUStableWalkEnv.DEFAULT_STABLE_TARGET_FORWARD_SPEED
                )
        if args.auto_yaw_gain is None:
            if args.training_preset == "quick_rough_teacher_joint" and args.terrain_profile == "rough":
                args.auto_yaw_gain = 0.45
            elif args.training_preset == "quick_rough_dual_imu" and args.terrain_profile == "rough":
                args.auto_yaw_gain = 0.45 if is_longwalk_transfer else 0.50
            else:
                args.auto_yaw_gain = (
                    SpotMLDualIMUStableWalkEnv.DEFAULT_ROUGH_STABLE_AUTO_YAW_GAIN
                    if args.terrain_profile == "rough"
                    else SpotMLDualIMUStableWalkEnv.DEFAULT_STABLE_AUTO_YAW_GAIN
                )
        if args.gait_geometry is None:
            if args.training_preset == "quick_rough_teacher_joint" and args.terrain_profile == "rough":
                args.gait_geometry = "anti_rock"
            elif args.training_preset == "quick_rough_dual_imu" and args.terrain_profile == "rough":
                args.gait_geometry = "anti_rock"
            else:
                args.gait_geometry = (
                    SpotMLDualIMUStableWalkEnv.DEFAULT_ROUGH_STABLE_GEOMETRY_PROFILE
                    if args.terrain_profile == "rough"
                    else SpotMLDualIMUStableWalkEnv.DEFAULT_STABLE_GEOMETRY_PROFILE
                )
        if args.body_height_offset is None:
            args.body_height_offset = (
                SpotMLDualIMUStableWalkEnv.DEFAULT_ROUGH_WALK_BODY_HEIGHT_OFFSET
                if args.terrain_profile == "rough"
                else SpotMLDualIMUStableWalkEnv.DEFAULT_WALK_BODY_HEIGHT_OFFSET
            )
        if args.body_pitch_bias_deg is None:
            if args.training_preset == "quick_rough_teacher_joint" and args.terrain_profile == "rough":
                args.body_pitch_bias_deg = -0.2
            elif args.training_preset == "quick_rough_dual_imu" and args.terrain_profile == "rough":
                args.body_pitch_bias_deg = -0.2 if is_longwalk_transfer else -0.6
            else:
                args.body_pitch_bias_deg = (
                    SpotMLDualIMUStableWalkEnv.DEFAULT_ROUGH_WALK_BODY_PITCH_BIAS_DEG
                    if args.terrain_profile == "rough"
                    else SpotMLDualIMUStableWalkEnv.DEFAULT_WALK_BODY_PITCH_BIAS_DEG
                )
        return args

    if args.episode_steps is None:
        args.episode_steps = 2000

    if args.gait_mode == "four_phase":
        if args.terrain_profile == "rough":
            if args.training_preset == "quick_rough" and args.episode_steps == 2000:
                args.episode_steps = 3000
            if args.target_forward_speed is None:
                args.target_forward_speed = (
                    0.050 if args.training_preset == "quick_rough"
                    else SpotMLWalkEnv.DEFAULT_ROUGH_WALK_TARGET_FORWARD_SPEED
                )
            if args.body_height_offset is None:
                args.body_height_offset = SpotMLWalkEnv.DEFAULT_ROUGH_WALK_BODY_HEIGHT_OFFSET
            if args.body_pitch_bias_deg is None:
                args.body_pitch_bias_deg = (
                    -1.0 if args.training_preset == "quick_rough"
                    else SpotMLWalkEnv.DEFAULT_ROUGH_WALK_BODY_PITCH_BIAS_DEG
                )
            if args.auto_yaw_gain is None:
                args.auto_yaw_gain = (
                    0.58 if args.training_preset == "quick_rough"
                    else SpotMLWalkEnv.DEFAULT_ROUGH_WALK_AUTO_YAW_GAIN
                )
            if args.gait_geometry is None:
                args.gait_geometry = (
                    "anti_rock" if args.training_preset == "quick_rough"
                    else SpotMLWalkEnv.DEFAULT_ROUGH_WALK_GEOMETRY_PROFILE
                )
        else:
            if args.target_forward_speed is None:
                args.target_forward_speed = SpotMLWalkEnv.DEFAULT_WALK_TARGET_FORWARD_SPEED
            if args.body_height_offset is None:
                args.body_height_offset = SpotMLWalkEnv.DEFAULT_WALK_BODY_HEIGHT_OFFSET
            if args.body_pitch_bias_deg is None:
                args.body_pitch_bias_deg = SpotMLWalkEnv.DEFAULT_WALK_BODY_PITCH_BIAS_DEG
            if args.auto_yaw_gain is None:
                args.auto_yaw_gain = SpotMLWalkEnv.DEFAULT_WALK_AUTO_YAW_GAIN
            if args.gait_geometry is None:
                args.gait_geometry = SpotMLWalkEnv.DEFAULT_WALK_GEOMETRY_PROFILE
    else:
        if args.target_forward_speed is None:
            args.target_forward_speed = 0.35
        if args.body_pitch_bias_deg is None:
            args.body_pitch_bias_deg = 0.0
        if args.gait_geometry is None:
            args.gait_geometry = "default"
    return args


def get_env_class(terrain_profile: str, walk_variant: str):
    if walk_variant == "rough_teacher_joint_residual":
        return SpotMLRoughTeacherJointResidualEnv
    if walk_variant == "rough_dual_imu":
        return SpotMLRoughDualIMUStableWalkEnv
    if walk_variant == "dual_imu_stable":
        return SpotMLDualIMUStableWalkEnv
    if terrain_profile == "rough":
        return SpotMLRoughTerrainHeightResidualEnv
    return SpotMLWalkEnv


def resolve_init_model_path(run_dir_arg: str | None, model_kind: str) -> str | None:
    if not run_dir_arg:
        return None

    run_dir = resolve_repo_run_dir(run_dir_arg)
    if run_dir is None:
        return None
    if not run_dir.exists():
        raise SystemExit(f"Warm-start run directory not found: {run_dir}")

    if model_kind == "best":
        model_path = run_dir / "best_model" / "best_model.zip"
    else:
        model_path = run_dir / "models" / "ppo_spot_walk_final.zip"

    if not model_path.exists():
        raise SystemExit(f"Warm-start model file not found: {model_path}")

    return str(model_path)


def default_teacher_run_dir() -> str | None:
    for candidate_name in (
        DEFAULT_TEACHER_LONGWALK_V2_NAME,
        DEFAULT_TEACHER_LONGWALK_V1_NAME,
        DEFAULT_QUICK_ROUGH_DUAL_IMU_WARMSTART_NAME,
    ):
        candidate = resolve_named_run_dir(candidate_name)
        if candidate is not None:
            return candidate
    return None


def resolve_teacher_paths(args) -> tuple[str | None, str | None]:
    if args.walk_variant != "rough_teacher_joint_residual":
        return None, None

    teacher_run_dir = args.teacher_run_dir or default_teacher_run_dir()
    if teacher_run_dir is None:
        raise SystemExit(
            "No teacher run directory was found for rough_teacher_joint_residual. "
            "Provide --teacher-run-dir explicitly."
        )

    teacher_model_path = resolve_init_model_path(teacher_run_dir, args.teacher_model)
    teacher_vecnormalize_path = (
        Path(teacher_run_dir).expanduser().resolve() / "vecnormalize" / "vecnormalize.pkl"
    )
    if not teacher_vecnormalize_path.exists():
        teacher_vecnormalize_path = None

    args.teacher_run_dir = str(Path(teacher_run_dir).expanduser().resolve())
    args.teacher_model_path = teacher_model_path
    args.teacher_vecnormalize_path = (
        None if teacher_vecnormalize_path is None else str(teacher_vecnormalize_path)
    )
    return args.teacher_model_path, args.teacher_vecnormalize_path


def warm_start_policy(model: PPO, init_model_path: str) -> tuple[int, int]:
    try:
        source_model = PPO.load(init_model_path, device="auto")
    except Exception:
        source_model = PPO.load(
            init_model_path,
            device="auto",
            custom_objects={
                "observation_space": model.observation_space,
                "action_space": model.action_space,
            },
        )
    try:
        target_state = model.policy.state_dict()
        source_state = source_model.policy.state_dict()
        compatible_items = {}
        exact_match_count = 0
        expanded_input_count = 0
        for key, target_value in target_state.items():
            source_value = source_state.get(key)
            if source_value is None:
                continue
            if target_value.shape == source_value.shape:
                compatible_items[key] = source_value
                exact_match_count += 1
                continue

            # Allow transfer from a smaller observation space into a larger one by
            # copying the shared input columns of linear layers and keeping the
            # new terrain-specific inputs at their fresh initialization.
            if (
                target_value.ndim == 2
                and source_value.ndim == 2
                and target_value.shape[0] == source_value.shape[0]
                and target_value.shape[1] >= source_value.shape[1]
            ):
                expanded_value = target_value.clone()
                expanded_value[:, : source_value.shape[1]] = source_value
                compatible_items[key] = expanded_value
                expanded_input_count += 1
        target_state.update(compatible_items)
        model.policy.load_state_dict(target_state, strict=False)
        return exact_match_count + expanded_input_count, len(target_state)
    finally:
        del source_model


def make_env(rank: int, args):
    def _init():
        env_class = get_env_class(args.terrain_profile, args.walk_variant)
        env_kwargs = dict(
            render=args.render and rank == 0,
            on_rack=False,
            terrain_profile=args.terrain_profile,
            terrain_randomization=args.terrain_randomization,
            draw_foot_path=False,
            env_randomizer=None,
            target_forward_speed=args.target_forward_speed,
            gait_mode=args.gait_mode,
            max_episode_steps=args.episode_steps,
            body_height_offset=args.body_height_offset,
            body_pitch_bias_deg=args.body_pitch_bias_deg,
            auto_yaw_gain=args.auto_yaw_gain,
            gait_geometry_profile=args.gait_geometry,
            enable_imu_yaw=args.enable_imu_yaw,
        )
        if env_class in (
            SpotMLRoughTerrainHeightResidualEnv,
            SpotMLRoughDualIMUStableWalkEnv,
            SpotMLRoughTeacherJointResidualEnv,
        ):
            env_kwargs.update(
                enable_camera_observation=args.enable_camera_observation,
                camera_stabilize_roll_pitch=args.camera_stabilize_roll_pitch,
                camera_pitch_offset_deg=args.camera_pitch_offset_deg,
            )
        if env_class is SpotMLRoughTeacherJointResidualEnv:
            env_kwargs.update(
                teacher_model_path=args.teacher_model_path,
                teacher_vecnormalize_path=args.teacher_vecnormalize_path,
            )
        env = env_class(**env_kwargs)
        env.reset(seed=args.seed + rank)
        return Monitor(env)

    return _init


def build_run_dirs(args):
    run_name = args.run_name or datetime.now().strftime("spot_ml_walk_%Y%m%d_%H%M%S")
    base_dir = os.path.join(os.path.abspath(args.output_root), run_name)
    if os.path.isdir(base_dir) and os.listdir(base_dir) and not args.allow_run_overwrite:
        raise SystemExit(
            f"Training run already exists and is not empty: {base_dir}\n"
            "Choose a new --run-name or pass --allow-run-overwrite intentionally."
        )
    model_dir = os.path.join(base_dir, "models")
    best_model_dir = os.path.join(base_dir, "best_model")
    log_dir = os.path.join(base_dir, "logs")
    preview_dir = os.path.join(base_dir, "previews")
    vecnorm_dir = os.path.join(base_dir, "vecnormalize")

    for directory in (model_dir, best_model_dir, log_dir, preview_dir, vecnorm_dir):
        os.makedirs(directory, exist_ok=True)

    config_path = os.path.join(base_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(vars(args), config_file, indent=2, sort_keys=True)

    return {
        "base_dir": base_dir,
        "model_dir": model_dir,
        "best_model_dir": best_model_dir,
        "log_dir": log_dir,
        "vecnorm_dir": vecnorm_dir,
        "config_path": config_path,
    }


def maybe_check_env(args):
    if args.skip_env_check:
        print("Skipping environment check.")
        return

    print("Checking environment compatibility...")
    env_class = get_env_class(args.terrain_profile, args.walk_variant)
    env_kwargs = dict(
        render=False,
        on_rack=False,
        terrain_profile=args.terrain_profile,
        terrain_randomization=args.terrain_randomization,
        draw_foot_path=False,
        env_randomizer=None,
        target_forward_speed=args.target_forward_speed,
        gait_mode=args.gait_mode,
        max_episode_steps=args.episode_steps,
        body_height_offset=args.body_height_offset,
        body_pitch_bias_deg=args.body_pitch_bias_deg,
        auto_yaw_gain=args.auto_yaw_gain,
        gait_geometry_profile=args.gait_geometry,
        enable_imu_yaw=args.enable_imu_yaw,
    )
    if env_class in (
        SpotMLRoughTerrainHeightResidualEnv,
        SpotMLRoughDualIMUStableWalkEnv,
        SpotMLRoughTeacherJointResidualEnv,
    ):
        env_kwargs.update(
            enable_camera_observation=args.enable_camera_observation,
            camera_stabilize_roll_pitch=args.camera_stabilize_roll_pitch,
            camera_pitch_offset_deg=args.camera_pitch_offset_deg,
        )
    if env_class is SpotMLRoughTeacherJointResidualEnv:
        env_kwargs.update(
            teacher_model_path=args.teacher_model_path,
            teacher_vecnormalize_path=args.teacher_vecnormalize_path,
        )
    test_env = env_class(**env_kwargs)
    try:
        check_env(test_env, warn=True)
    finally:
        test_env.close()
    print("Environment check finished.")


def main():
    patch_numpy_bit_generator_pickle()
    patch_numpy_module_aliases()
    args = resolve_walk_defaults(parse_args())
    resolve_teacher_paths(args)
    if args.walk_variant == "rough_teacher_joint_residual":
        args.rough_env_variant = "rough_teacher_joint_residual_v1"
    elif args.walk_variant == "rough_dual_imu":
        args.rough_env_variant = "rough_dual_imu_stable_v1"
    elif args.walk_variant == "dual_imu_stable":
        args.rough_env_variant = "dual_imu_stable_v1" if args.terrain_profile == "rough" else None
    else:
        args.rough_env_variant = "foothold_height_residual_v3" if args.terrain_profile == "rough" else None
    run_dirs = build_run_dirs(args)
    has_tensorboard = importlib.util.find_spec("tensorboard") is not None
    init_model_path = resolve_init_model_path(args.init_from_run, args.init_model)

    print(f"Fresh training run: {run_dirs['base_dir']}")
    print(f"Seed: {args.seed}")
    print(f"Parallel envs: {args.n_envs}")
    print(f"Walk variant: {args.walk_variant}")
    if args.training_preset != "none":
        print(f"Training preset: {args.training_preset}")
    print(f"Gait mode: {args.gait_mode}")
    print(f"Terrain profile: {args.terrain_profile}")
    print(f"Gait geometry: {args.gait_geometry}")
    print(f"Body height offset: {args.body_height_offset}")
    print(f"Body pitch bias (deg): {args.body_pitch_bias_deg}")
    print(f"Auto yaw gain: {args.auto_yaw_gain}")
    print(f"IMU yaw stabilization enabled: {args.enable_imu_yaw}")
    print(f"PPO learning rate: {args.learning_rate}")
    print(f"PPO entropy coef: {args.ent_coef}")
    print(f"PPO clip range: {args.clip_range}")
    print(f"PPO batch size: {args.batch_size}")
    print(f"PPO epochs: {args.n_epochs}")
    if args.terrain_profile == "rough":
        print(f"Camera observation enabled: {args.enable_camera_observation}")
        print(f"Camera roll/pitch stabilized: {args.camera_stabilize_roll_pitch}")
        print(f"Camera pitch offset (deg): {args.camera_pitch_offset_deg}")
    if init_model_path is not None:
        print(f"Warm start: {init_model_path}")
    if not has_tensorboard:
        print("TensorBoard is not installed in this environment. Training will continue without TensorBoard logs.")

    maybe_check_env(args)

    train_env = None
    eval_env = None

    try:
        train_env = DummyVecEnv([make_env(rank=i, args=args) for i in range(args.n_envs)])
        eval_env = DummyVecEnv([make_env(rank=10_000, args=args)])

        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            gamma=0.99,
        )
        train_env.seed(args.seed)

        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            gamma=0.99,
            training=False,
        )
        eval_env.seed(args.seed)

        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, 25_000 // args.n_envs),
            save_path=run_dirs["model_dir"],
            name_prefix="ppo_spot_walk",
            save_vecnormalize=True,
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=run_dirs["best_model_dir"],
            log_path=run_dirs["log_dir"],
            eval_freq=max(1, 10_000 // args.n_envs),
            deterministic=True,
            render=False,
            n_eval_episodes=5,
        )
        preview_callback = PreviewArtifactsCallback(
            args=args,
            run_dirs=run_dirs,
            preview_freq=args.preview_freq,
            preview_steps=args.preview_steps,
            preview_fps=args.preview_fps,
        )

        policy_kwargs = {
            "net_arch": {
                "pi": [256, 256],
                "vf": [256, 256],
            }
        }

        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=args.learning_rate,
            n_steps=max(64, 2048 // args.n_envs),
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=run_dirs["log_dir"] if has_tensorboard else None,
            device="auto",
            seed=args.seed,
        )

        if init_model_path is not None:
            print("Loading warm-start weights...")
            loaded_count, total_count = warm_start_policy(model, init_model_path)
            print(f"Warm-started {loaded_count}/{total_count} compatible policy tensors.")

        can_show_progress = (
            args.progress_bar
            and importlib.util.find_spec("tqdm") is not None
            and importlib.util.find_spec("rich") is not None
        )
        if args.progress_bar and not can_show_progress:
            print("Progress bar requested, but tqdm/rich are not both installed. Continuing without it.")

        if init_model_path is not None:
            print(f"Starting PPO fine-tuning for {args.timesteps:,} timesteps...")
        else:
            print(f"Starting fresh PPO training for {args.timesteps:,} timesteps...")
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList([checkpoint_callback, eval_callback, preview_callback]),
            progress_bar=can_show_progress,
        )

        final_model_path = os.path.join(run_dirs["model_dir"], "ppo_spot_walk_final")
        vecnorm_path = os.path.join(run_dirs["vecnorm_dir"], "vecnormalize.pkl")

        model.save(final_model_path)
        train_env.save(vecnorm_path)

        print("Training complete.")
        print(f"Final model saved to: {final_model_path}.zip")
        print(f"Best model directory: {run_dirs['best_model_dir']}")
        print(f"VecNormalize stats saved to: {vecnorm_path}")
        print(f"Run configuration saved to: {run_dirs['config_path']}")
        print(f"Visual previews saved to: {run_dirs['base_dir']}/previews")
    finally:
        if eval_env is not None:
            eval_env.close()
        if train_env is not None:
            train_env.close()


if __name__ == "__main__":
    main()
