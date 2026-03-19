#!/usr/bin/env python3

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datetime import datetime
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env

from spot_ml import SpotMLWalkEnv


def make_env(render: bool = False, height_field: bool = False):
    """
    Factory for creating a single ML walking environment instance.
    """
    def _init():
        env = SpotMLWalkEnv(
            render=render,
            on_rack=False,
            height_field=height_field,
            draw_foot_path=False,
            env_randomizer=None,
            target_forward_speed=0.35,
        )
        env = Monitor(env)
        return env

    return _init


def main():
    # -----------------------------
    # Output folders
    # -----------------------------
    run_name = datetime.now().strftime("spot_ml_walk_%Y%m%d_%H%M%S")
    base_dir = os.path.join("training_runs", run_name)
    model_dir = os.path.join(base_dir, "models")
    best_model_dir = os.path.join(base_dir, "best_model")
    log_dir = os.path.join(base_dir, "logs")
    vecnorm_dir = os.path.join(base_dir, "vecnormalize")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(vecnorm_dir, exist_ok=True)

    # -----------------------------
    # Quick environment check
    # -----------------------------
    # This is useful once before training.
    # If this fails, fix the environment API first.
    print("Checking environment...")
    test_env = SpotMLWalkEnv(
        render=False,
        on_rack=False,
        height_field=False,
        draw_foot_path=False,
        env_randomizer=None,
        target_forward_speed=0.35,
    )
    check_env(test_env, warn=True)
    test_env.close()
    print("Environment check finished.")

    # -----------------------------
    # Training / evaluation envs
    # -----------------------------
    n_envs = 4

    train_env = DummyVecEnv([make_env(render=False, height_field=False) for _ in range(n_envs)])
    eval_env = DummyVecEnv([make_env(render=False, height_field=False)])

    # Normalize observations and rewards for smoother PPO training
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.99,
    )

    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
        training=False,
    )

    # -----------------------------
    # Callbacks
    # -----------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=25_000 // n_envs,
        save_path=model_dir,
        name_prefix="ppo_spot_walk"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=10_000 // n_envs,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # -----------------------------
    # PPO model
    # -----------------------------
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048 // n_envs,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        device="auto",
    )

    # -----------------------------
    # Train
    # -----------------------------
    total_timesteps = 1_000_000

    print(f"Starting training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )

    # -----------------------------
    # Save final artifacts
    # -----------------------------
    final_model_path = os.path.join(model_dir, "ppo_spot_walk_final")
    model.save(final_model_path)

    vecnorm_path = os.path.join(vecnorm_dir, "vecnormalize.pkl")
    train_env.save(vecnorm_path)

    print(f"Training complete.")
    print(f"Final model saved to: {final_model_path}.zip")
    print(f"VecNormalize stats saved to: {vecnorm_path}")


if __name__ == "__main__":
    main()