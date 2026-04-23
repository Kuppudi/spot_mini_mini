#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spot_play_ml import (
    DummyVecEnv,
    PPO,
    VecNormalize,
    _spotmini_compat_bit_generator_ctor,
    load_run_config,
    load_ppo_model,
    make_env,
    patch_numpy_bit_generator_pickle,
    patch_numpy_module_aliases,
    resolve_model_path,
    resolve_run_dir,
)

# Keep the pickle compatibility constructor available from this module too,
# because older VecNormalize pickles may refer to __main__.<name>.
_spotmini_compat_bit_generator_ctor = _spotmini_compat_bit_generator_ctor


@dataclass
class EpisodeStats:
    reward: float
    steps: int
    done_reason: str
    final_forward_speed: float
    final_forward_displacement: float
    final_base_height_error: float
    fell: bool
    timed_out: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple trained SpotML runs headlessly."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more run directories to compare.",
    )
    parser.add_argument(
        "--model",
        choices=("best", "final"),
        default="best",
        help="Which saved checkpoint to evaluate for each run.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="How many evaluation episodes to run per model.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional hard cap on playback steps per episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base evaluation seed.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("reward", "distance", "speed", "fall_rate"),
        default="reward",
        help="Metric used to sort the comparison table.",
    )
    return parser.parse_args()


def _safe_float(value, default=0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def load_eval_env(run_dir: Path, run_config: dict, args: argparse.Namespace):
    env_args = argparse.Namespace(
        run_dir=str(run_dir),
        model=args.model,
        checkpoint_path=None,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        terrain_profile=None,
        height_field=False,
        terrain_randomization=False,
        gait_mode=None,
        target_forward_speed=None,
        episode_steps=None,
        body_height_offset=None,
        body_pitch_bias_deg=None,
        auto_yaw_gain=None,
        gait_geometry=None,
        render=False,
        bullet_gui=False,
        unsafe_gui=False,
        follow_camera=False,
        enable_camera_observation=None,
        disable_camera_leveling=False,
        enable_imu_yaw=None,
        camera_pitch_offset_deg=None,
        save_gif=None,
        gif_steps=220,
        gif_fps=12,
    )

    base_env = DummyVecEnv([make_env(env_args, run_config)])
    vecnormalize_path = run_dir / "vecnormalize" / "vecnormalize.pkl"
    env = base_env
    vecnormalize_loaded = False

    if vecnormalize_path.exists():
        try:
            env = VecNormalize.load(str(vecnormalize_path), base_env)
            env.training = False
            env.norm_reward = False
            vecnormalize_loaded = True
        except Exception as exc:
            print(f"Warning: VecNormalize load failed for {run_dir.name}: {exc}")

    model_path = resolve_model_path(run_dir, args.model, None)
    model = load_ppo_model(model_path, env, context=f"comparison for {run_dir.name}")
    return env, model, vecnormalize_loaded


def run_episode(env, model, *, max_steps: int | None) -> EpisodeStats:
    obs = env.reset()
    episode_reward = 0.0
    episode_step = 0
    last_info = {}
    fell = False
    timed_out = False
    done_reason = "unknown"

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        episode_reward += float(rewards[0])
        episode_step += 1
        last_info = infos[0] if isinstance(infos, (list, tuple)) else infos
        fell = fell or bool(last_info.get("is_fallen", False))

        hit_max_steps = max_steps is not None and episode_step >= max_steps
        if bool(dones[0]) or hit_max_steps:
            timed_out = bool(last_info.get("time_limit_reached", False)) or hit_max_steps
            done_reason = "max_steps" if hit_max_steps and not bool(dones[0]) else "env_done"
            return EpisodeStats(
                reward=episode_reward,
                steps=episode_step,
                done_reason=done_reason,
                final_forward_speed=_safe_float(last_info.get("forward_speed")),
                final_forward_displacement=_safe_float(last_info.get("forward_displacement")),
                final_base_height_error=_safe_float(last_info.get("base_height_error")),
                fell=fell,
                timed_out=timed_out,
            )


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def compare_run(run_dir: Path, args: argparse.Namespace) -> dict:
    run_config = load_run_config(run_dir)
    env, model, vecnormalize_loaded = load_eval_env(run_dir, run_config, args)
    episodes: list[EpisodeStats] = []
    try:
        for episode_index in range(args.episodes):
            episode_seed = args.seed + episode_index
            try:
                env.seed(episode_seed)
            except Exception:
                pass
            episodes.append(run_episode(env, model, max_steps=args.max_steps))
    finally:
        env.close()

    fall_rate = sum(1 for episode in episodes if episode.fell) / max(1, len(episodes))
    timeout_rate = sum(1 for episode in episodes if episode.timed_out) / max(1, len(episodes))
    return {
        "run_name": run_dir.name,
        "status": "ok",
        "error": "",
        "model": args.model,
        "walk_variant": run_config.get("walk_variant", "default"),
        "terrain": run_config.get("terrain_profile", "unknown"),
        "imu_yaw": bool(run_config.get("enable_imu_yaw", False)),
        "vecnormalize": vecnormalize_loaded,
        "reward_mean": mean([episode.reward for episode in episodes]),
        "steps_mean": mean([float(episode.steps) for episode in episodes]),
        "speed_mean": mean([episode.final_forward_speed for episode in episodes]),
        "distance_mean": mean([episode.final_forward_displacement for episode in episodes]),
        "height_err_mean": mean([episode.final_base_height_error for episode in episodes]),
        "fall_rate": fall_rate,
        "timeout_rate": timeout_rate,
    }


def failed_row(run_dir_arg: str, exc: BaseException) -> dict:
    run_name = Path(run_dir_arg).name or run_dir_arg
    return {
        "run_name": run_name,
        "status": "failed",
        "error": str(exc).splitlines()[0],
        "model": "",
        "walk_variant": "n/a",
        "terrain": "n/a",
        "imu_yaw": False,
        "vecnormalize": False,
        "reward_mean": -math.inf,
        "steps_mean": math.nan,
        "speed_mean": math.nan,
        "distance_mean": math.nan,
        "height_err_mean": math.nan,
        "fall_rate": 1.0,
        "timeout_rate": 0.0,
    }


def sort_rows(rows: list[dict], sort_by: str) -> list[dict]:
    key_map = {
        "reward": ("reward_mean", True),
        "distance": ("distance_mean", True),
        "speed": ("speed_mean", True),
        "fall_rate": ("fall_rate", False),
    }
    key_name, descending = key_map[sort_by]
    return sorted(rows, key=lambda row: row[key_name], reverse=descending)


def format_bool(value: bool) -> str:
    return "yes" if value else "no"


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'run':30} {'variant':24} {'terrain':8} {'yaw':5} {'vecnorm':8} "
        f"{'reward':>10} {'speed':>8} {'distance':>10} {'height_err':>11} {'falls':>7} {'timeouts':>9} status"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['run_name'][:30]:30} "
            f"{row['walk_variant'][:24]:24} "
            f"{row['terrain'][:8]:8} "
            f"{format_bool(row['imu_yaw']):5} "
            f"{format_bool(row['vecnormalize']):8} "
            f"{row['reward_mean']:10.2f} "
            f"{row['speed_mean']:8.3f} "
            f"{row['distance_mean']:10.3f} "
            f"{row['height_err_mean']:11.4f} "
            f"{row['fall_rate'] * 100:6.1f}% "
            f"{row['timeout_rate'] * 100:8.1f}% "
            f"{row['status']}"
        )
        if row["status"] != "ok":
            print(f"  reason: {row['error']}")


def main() -> None:
    args = parse_args()
    patch_numpy_bit_generator_pickle()
    patch_numpy_module_aliases()

    rows = []
    for run_dir_arg in args.run_dirs:
        try:
            run_dir = resolve_run_dir(run_dir_arg)
            print(f"Evaluating {run_dir} ...")
            rows.append(compare_run(run_dir, args))
        except SystemExit as exc:
            print(f"Skipping failed candidate {run_dir_arg}: {exc}")
            rows.append(failed_row(run_dir_arg, exc))
        except Exception as exc:
            print(f"Skipping failed candidate {run_dir_arg}: {exc}")
            rows.append(failed_row(run_dir_arg, exc))

    rows = sort_rows(rows, args.sort_by)
    print()
    print_table(rows)


if __name__ == "__main__":
    main()
