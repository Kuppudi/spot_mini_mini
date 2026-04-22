#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"

(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("FONTCONFIG_PATH", str(CACHE_DIR / "fontconfig"))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def dependency_error(exc: ModuleNotFoundError) -> None:
    missing_module = exc.name or "a required package"
    raise SystemExit(
        f"Missing dependency: {missing_module}\n"
        "Install the same environment you used for training before running the ML tester."
    ) from exc


try:
    import tkinter as tk
    from tkinter import ttk
except ModuleNotFoundError:
    tk = None
    ttk = None

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ModuleNotFoundError as exc:
    dependency_error(exc)

try:
    from spot_ml import (
        SpotMLDualIMUStableWalkEnv,
        SpotMLRoughDualIMUStableWalkEnv,
        SpotMLRoughTeacherJointResidualEnv,
    )
except ModuleNotFoundError as exc:
    dependency_error(exc)

try:
    from spot_play_ml import (
        load_run_config,
        make_env,
        patch_numpy_bit_generator_pickle,
        patch_numpy_module_aliases,
        resolve_model_path,
    )
except ModuleNotFoundError as exc:
    dependency_error(exc)

try:
    from spot_visualization_ml import unwrap_env
except ModuleNotFoundError as exc:
    dependency_error(exc)


RUN_ROOT_CANDIDATES = (
    PROJECT_ROOT / "spot_bullet" / "training runs",
    PROJECT_ROOT / "spot_bullet" / "training_runs",
)
DEFAULT_TURN_RESIDUAL = 0.006
DEFAULT_IDLE_SCALE = 0.15
DEFAULT_FORWARD_SCALE = 1.0
DEFAULT_SPEED_SCALE = 1.0
TURN_PATTERN = np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class RunEntry:
    run_dir: Path
    run_name: str
    root_label: str
    run_config: dict
    has_best: bool
    has_final: bool
    has_vecnormalize: bool

    @property
    def display_name(self) -> str:
        terrain = self.run_config.get("terrain_profile", "unknown")
        walk_variant = self.run_config.get("walk_variant", "default")
        gait_mode = self.run_config.get("gait_mode", "trot")
        return f"{self.run_name} | {terrain} | {walk_variant} | {gait_mode}"


@dataclass
class LoadedSession:
    run_entry: RunEntry
    model_kind: str
    run_config: dict
    env: object
    raw_env: object
    model: PPO
    obs: np.ndarray
    vecnormalize_loaded: bool

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


class ControlState:
    def __init__(self) -> None:
        self.forward_scale = DEFAULT_FORWARD_SCALE
        self.speed_scale = DEFAULT_SPEED_SCALE
        self.pending_reset = False
        self.pending_reload = False
        self.last_pressed: set[int] = set()


class ModelSelectorUI:
    def __init__(self, run_entries: list[RunEntry], initial_display_name: str, initial_model_kind: str):
        if tk is None or ttk is None:
            raise RuntimeError("tkinter is not available in this Python environment.")

        self.root = tk.Tk()
        self.root.title("Spot ML Model Selector")
        self.root.geometry("760x250")
        self.root.resizable(False, False)

        self.run_entries = {entry.display_name: entry for entry in run_entries}
        self.pending_reload = False
        self.closed = False

        self.run_var = tk.StringVar(value=initial_display_name)
        self.model_kind_var = tk.StringVar(value=initial_model_kind)
        self.status_var = tk.StringVar(value="Ready.")

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Walk Model").grid(row=0, column=0, sticky="w")
        self.run_combo = ttk.Combobox(
            main,
            textvariable=self.run_var,
            values=list(self.run_entries.keys()),
            state="readonly",
            width=88,
        )
        self.run_combo.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 10))
        self.run_combo.bind("<<ComboboxSelected>>", self._queue_reload)

        ttk.Label(main, text="Saved Checkpoint").grid(row=2, column=0, sticky="w")
        self.model_kind_combo = ttk.Combobox(
            main,
            textvariable=self.model_kind_var,
            values=("best", "final"),
            state="readonly",
            width=12,
        )
        self.model_kind_combo.grid(row=3, column=0, sticky="w", pady=(4, 10))
        self.model_kind_combo.bind("<<ComboboxSelected>>", self._queue_reload)

        reload_button = ttk.Button(main, text="Load Selected Model", command=self._queue_reload)
        reload_button.grid(row=3, column=1, sticky="w", padx=(12, 0))

        ttk.Label(
            main,
            text=(
                "Controls: hold W to walk, S to idle, Q/E to steer, Z/X to change speed, "
                "R to reset, L to reload the selected model."
            ),
            wraplength=720,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 10))

        ttk.Label(main, textvariable=self.status_var, wraplength=720).grid(
            row=5,
            column=0,
            columnspan=3,
            sticky="w",
        )

        main.columnconfigure(0, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _queue_reload(self, _event=None) -> None:
        self.pending_reload = True

    def _on_close(self) -> None:
        self.closed = True
        self.root.destroy()

    def pump(self) -> None:
        if self.closed:
            return
        self.root.update_idletasks()
        self.root.update()

    def consume_reload_request(self) -> bool:
        if self.pending_reload:
            self.pending_reload = False
            return True
        return False

    def get_selection(self) -> tuple[RunEntry, str]:
        return self.run_entries[self.run_var.get()], self.model_kind_var.get()

    def set_status(self, text: str) -> None:
        self.status_var.set(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained walk model in the PyBullet environment with keyboard controls."
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional run directory to load immediately. If omitted, the newest run is selected.",
    )
    parser.add_argument(
        "--model",
        choices=("best", "final"),
        default="best",
        help="Which saved checkpoint to load for the selected run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reset seed for the loaded environment.",
    )
    parser.add_argument(
        "--idle-scale",
        type=float,
        default=DEFAULT_IDLE_SCALE,
        help="How much of the model action to keep when the walk key is not held.",
    )
    parser.add_argument(
        "--turn-residual",
        type=float,
        default=DEFAULT_TURN_RESIDUAL,
        help="Per-leg residual delta used for keyboard steering on stable-walk models.",
    )
    parser.add_argument(
        "--no-selector",
        action="store_true",
        help="Skip the Tk dropdown window and just use the command-line run selection.",
    )
    return parser.parse_args()


def existing_run_roots() -> list[Path]:
    return [root for root in RUN_ROOT_CANDIDATES if root.exists()]


def list_available_runs() -> list[RunEntry]:
    run_entries: list[RunEntry] = []
    seen: set[Path] = set()

    for root in existing_run_roots():
        for run_dir in sorted(root.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
            if not run_dir.is_dir() or run_dir in seen:
                continue
            config_path = run_dir / "run_config.json"
            best_path = run_dir / "best_model" / "best_model.zip"
            final_path = run_dir / "models" / "ppo_spot_walk_final.zip"
            if not config_path.exists() and not best_path.exists() and not final_path.exists():
                continue

            seen.add(run_dir)
            run_config = load_run_config(run_dir)
            run_entries.append(
                RunEntry(
                    run_dir=run_dir,
                    run_name=run_dir.name,
                    root_label=root.name,
                    run_config=run_config,
                    has_best=best_path.exists(),
                    has_final=final_path.exists(),
                    has_vecnormalize=(run_dir / "vecnormalize" / "vecnormalize.pkl").exists(),
                )
            )

    if not run_entries:
        searched = ", ".join(str(path) for path in RUN_ROOT_CANDIDATES)
        raise SystemExit(f"No trained walk runs were found. Searched: {searched}")

    return run_entries


def select_initial_run(run_entries: list[RunEntry], run_dir_arg: str | None) -> RunEntry:
    if run_dir_arg is None:
        return run_entries[0]

    requested = Path(run_dir_arg).expanduser().resolve()
    for entry in run_entries:
        if entry.run_dir.resolve() == requested:
            return entry

    raise SystemExit(f"Run directory not found in discovered runs: {requested}")


def ensure_model_kind_for_run(run_entry: RunEntry, model_kind: str) -> str:
    if model_kind == "best" and run_entry.has_best:
        return "best"
    if model_kind == "final" and run_entry.has_final:
        return "final"
    if run_entry.has_best:
        return "best"
    if run_entry.has_final:
        return "final"
    raise SystemExit(f"No saved model was found in: {run_entry.run_dir}")


def build_env_args(seed: int) -> SimpleNamespace:
    return SimpleNamespace(
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
        render=True,
        bullet_gui=True,
        unsafe_gui=True,
        follow_camera=True,
        enable_camera_observation=None,
        disable_camera_leveling=False,
        camera_pitch_offset_deg=None,
        seed=seed,
    )


def load_session(run_entry: RunEntry, model_kind: str, seed: int) -> LoadedSession:
    model_kind = ensure_model_kind_for_run(run_entry, model_kind)
    run_config = load_run_config(run_entry.run_dir)
    model_path = resolve_model_path(run_entry.run_dir, model_kind, None)
    vecnormalize_path = run_entry.run_dir / "vecnormalize" / "vecnormalize.pkl"

    env_args = build_env_args(seed)
    base_env = DummyVecEnv([make_env(env_args, run_config)])
    env = base_env
    vecnormalize_loaded = False

    if vecnormalize_path.exists():
        try:
            env = VecNormalize.load(str(vecnormalize_path), base_env)
            env.training = False
            env.norm_reward = False
            vecnormalize_loaded = True
        except Exception as exc:
            print(
                "Warning: failed to load VecNormalize stats for "
                f"{run_entry.run_name}. Using the raw environment instead."
            )
            print(f"VecNormalize load error: {exc}")

    model = PPO.load(
        str(model_path),
        env=env,
        device="auto",
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        },
    )
    obs = env.reset()
    raw_env = unwrap_env(env)

    print(
        f"Loaded {run_entry.run_name} ({model_kind}) | "
        f"terrain={run_config.get('terrain_profile', 'unknown')} | "
        f"walk_variant={run_config.get('walk_variant', 'default')} | "
        f"gait={run_config.get('gait_mode', 'trot')}"
    )
    if vecnormalize_loaded:
        print("VecNormalize stats loaded successfully.")
    else:
        print("VecNormalize stats were not loaded; using raw environment observations.")

    return LoadedSession(
        run_entry=run_entry,
        model_kind=model_kind,
        run_config=run_config,
        env=env,
        raw_env=raw_env,
        model=model,
        obs=obs,
        vecnormalize_loaded=vecnormalize_loaded,
    )


def key_down(keys: dict[int, int], pybullet_client, char: str) -> bool:
    key_code = ord(char)
    return key_code in keys and (keys[key_code] & pybullet_client.KEY_IS_DOWN)


def key_triggered(keys: dict[int, int], pybullet_client, char: str, control_state: ControlState) -> bool:
    key_code = ord(char)
    is_pressed = key_code in keys and (keys[key_code] & pybullet_client.KEY_WAS_TRIGGERED)
    if is_pressed and key_code not in control_state.last_pressed:
        return True
    return is_pressed


def update_last_pressed(keys: dict[int, int], pybullet_client, control_state: ControlState) -> None:
    control_state.last_pressed = {
        key_code
        for key_code, key_state in keys.items()
        if key_state & pybullet_client.KEY_IS_DOWN
    }


def clip_action(action: np.ndarray, raw_env) -> np.ndarray:
    low = np.asarray(raw_env.action_space.low, dtype=np.float32)
    high = np.asarray(raw_env.action_space.high, dtype=np.float32)
    return np.clip(action, low, high).astype(np.float32)


def apply_keyboard_to_default_model(
    action: np.ndarray,
    raw_env,
    keys: dict[int, int],
    pybullet_client,
    control_state: ControlState,
    idle_scale: float,
) -> np.ndarray:
    low = np.asarray(raw_env.action_space.low, dtype=np.float32)
    high = np.asarray(raw_env.action_space.high, dtype=np.float32)
    adjusted = action.astype(np.float32).copy()

    walk_pressed = key_down(keys, pybullet_client, "w") or key_down(keys, pybullet_client, "W")
    idle_pressed = key_down(keys, pybullet_client, "s") or key_down(keys, pybullet_client, "S")
    turn_left = key_down(keys, pybullet_client, "q") or key_down(keys, pybullet_client, "Q")
    turn_right = key_down(keys, pybullet_client, "e") or key_down(keys, pybullet_client, "E")
    strafe_left = key_down(keys, pybullet_client, "a") or key_down(keys, pybullet_client, "A")
    strafe_right = key_down(keys, pybullet_client, "d") or key_down(keys, pybullet_client, "D")

    forward_target = abs(float(adjusted[0]))
    if walk_pressed:
        adjusted[0] = np.clip(forward_target * control_state.forward_scale, low[0], high[0])
    elif idle_pressed:
        adjusted[0] = max(0.0, low[0])
    else:
        adjusted[0] = np.clip(forward_target * idle_scale, low[0], high[0])

    adjusted[1] = 0.0
    if strafe_left:
        adjusted[1] = np.clip(0.45, low[1], high[1])
    elif strafe_right:
        adjusted[1] = np.clip(-0.45, low[1], high[1])

    adjusted[2] = 0.0
    if turn_left:
        adjusted[2] = np.clip(0.55, low[2], high[2])
    elif turn_right:
        adjusted[2] = np.clip(-0.55, low[2], high[2])

    adjusted[3] = np.clip(
        max(float(low[3]), float(adjusted[3]) * control_state.speed_scale),
        low[3],
        high[3],
    )
    return adjusted


def apply_keyboard_to_stable_model(
    action: np.ndarray,
    raw_env,
    keys: dict[int, int],
    pybullet_client,
    control_state: ControlState,
    turn_residual: float,
    idle_scale: float,
) -> np.ndarray:
    low = np.asarray(raw_env.action_space.low, dtype=np.float32)
    high = np.asarray(raw_env.action_space.high, dtype=np.float32)
    adjusted = action.astype(np.float32).copy()

    walk_pressed = key_down(keys, pybullet_client, "w") or key_down(keys, pybullet_client, "W")
    idle_pressed = key_down(keys, pybullet_client, "s") or key_down(keys, pybullet_client, "S")
    turn_left = key_down(keys, pybullet_client, "q") or key_down(keys, pybullet_client, "Q")
    turn_right = key_down(keys, pybullet_client, "e") or key_down(keys, pybullet_client, "E")

    if walk_pressed:
        adjusted[0] = np.clip(float(adjusted[0]) * control_state.forward_scale, low[0], high[0])
        adjusted[1] = np.clip(float(adjusted[1]) * control_state.speed_scale, low[1], high[1])
    elif idle_pressed:
        adjusted[0] = low[0]
        adjusted[1] = low[1]
    else:
        adjusted[0] = np.clip(max(low[0], float(adjusted[0]) * idle_scale), low[0], high[0])
        adjusted[1] = np.clip(max(low[1], float(adjusted[1]) * idle_scale), low[1], high[1])

    if adjusted.shape[0] >= 6:
        residual_delta = np.zeros(4, dtype=np.float32)
        if turn_left:
            residual_delta = TURN_PATTERN * float(turn_residual)
        elif turn_right:
            residual_delta = -TURN_PATTERN * float(turn_residual)
        adjusted[2:6] = np.clip(adjusted[2:6] + residual_delta, low[2:6], high[2:6])

    return adjusted


def apply_keyboard_controls(
    session: LoadedSession,
    control_state: ControlState,
    turn_residual: float,
    idle_scale: float,
) -> np.ndarray:
    pybullet_client = session.raw_env.base_env._pybullet_client
    keys = pybullet_client.getKeyboardEvents()

    if key_triggered(keys, pybullet_client, "r", control_state) or key_triggered(keys, pybullet_client, "R", control_state):
        control_state.pending_reset = True
    if key_triggered(keys, pybullet_client, "l", control_state) or key_triggered(keys, pybullet_client, "L", control_state):
        control_state.pending_reload = True
    if key_triggered(keys, pybullet_client, "z", control_state) or key_triggered(keys, pybullet_client, "Z", control_state):
        control_state.speed_scale = max(0.55, control_state.speed_scale - 0.10)
        print(f"Speed scale: {control_state.speed_scale:.2f}")
    if key_triggered(keys, pybullet_client, "x", control_state) or key_triggered(keys, pybullet_client, "X", control_state):
        control_state.speed_scale = min(1.75, control_state.speed_scale + 0.10)
        print(f"Speed scale: {control_state.speed_scale:.2f}")

    action, _ = session.model.predict(session.obs, deterministic=True)
    action = np.asarray(action, dtype=np.float32).reshape(-1)

    if isinstance(
        session.raw_env,
        (SpotMLDualIMUStableWalkEnv, SpotMLRoughDualIMUStableWalkEnv, SpotMLRoughTeacherJointResidualEnv),
    ):
        adjusted = apply_keyboard_to_stable_model(
            action,
            session.raw_env,
            keys,
            pybullet_client,
            control_state,
            turn_residual=turn_residual,
            idle_scale=idle_scale,
        )
    else:
        adjusted = apply_keyboard_to_default_model(
            action,
            session.raw_env,
            keys,
            pybullet_client,
            control_state,
            idle_scale=idle_scale,
        )

    update_last_pressed(keys, pybullet_client, control_state)
    return clip_action(adjusted, session.raw_env)


def reset_session(session: LoadedSession, seed: int) -> None:
    session.obs = session.env.reset()
    print(f"Environment reset for {session.run_entry.run_name} (seed={seed}).")


def build_selector(
    run_entries: list[RunEntry],
    initial_entry: RunEntry,
    initial_model_kind: str,
    disabled: bool,
):
    if disabled:
        return None
    if tk is None or ttk is None:
        print("tkinter is not available, so the dropdown selector is disabled.")
        return None
    return ModelSelectorUI(run_entries, initial_entry.display_name, initial_model_kind)


def main() -> None:
    args = parse_args()
    patch_numpy_bit_generator_pickle()
    patch_numpy_module_aliases()

    run_entries = list_available_runs()
    initial_entry = select_initial_run(run_entries, args.run_dir)
    args.model = ensure_model_kind_for_run(initial_entry, args.model)
    selector = build_selector(run_entries, initial_entry, args.model, args.no_selector)
    control_state = ControlState()

    if selector is not None:
        selector.set_status(f"Loading {initial_entry.run_name} ({args.model})...")

    session = load_session(initial_entry, args.model, args.seed)
    print("Keyboard controls:")
    print("  W -> walk using the loaded model")
    print("  S -> idle / slow to minimum gait")
    print("  Q / E -> steer left / right")
    print("  Z / X -> decrease / increase speed scale")
    print("  R -> reset the episode")
    print("  L -> reload the currently selected model")
    print("Close the PyBullet window or the selector window to exit.")

    if selector is not None:
        selector.set_status(
            f"Loaded {session.run_entry.run_name} ({session.model_kind}) | "
            f"VecNormalize: {'yes' if session.vecnormalize_loaded else 'no'}"
        )

    try:
        while True:
            if selector is not None:
                try:
                    selector.pump()
                except tk.TclError:
                    break

                if selector.closed:
                    break

                if selector.consume_reload_request():
                    selected_run, selected_model_kind = selector.get_selection()
                    try:
                        selected_model_kind = ensure_model_kind_for_run(selected_run, selected_model_kind)
                        selector.set_status(f"Loading {selected_run.run_name} ({selected_model_kind})...")
                        session.close()
                        session = load_session(selected_run, selected_model_kind, args.seed)
                        control_state = ControlState()
                        selector.model_kind_var.set(selected_model_kind)
                        selector.set_status(
                            f"Loaded {session.run_entry.run_name} ({session.model_kind}) | "
                            f"VecNormalize: {'yes' if session.vecnormalize_loaded else 'no'}"
                        )
                    except SystemExit as exc:
                        selector.set_status(str(exc))

            if control_state.pending_reload:
                control_state.pending_reload = False
                selected_run = session.run_entry
                selected_model_kind = session.model_kind
                if selector is not None:
                    selected_run, selected_model_kind = selector.get_selection()
                try:
                    selected_model_kind = ensure_model_kind_for_run(selected_run, selected_model_kind)
                    if selector is not None:
                        selector.set_status(f"Reloading {selected_run.run_name} ({selected_model_kind})...")
                    session.close()
                    session = load_session(selected_run, selected_model_kind, args.seed)
                    control_state = ControlState()
                    if selector is not None:
                        selector.model_kind_var.set(selected_model_kind)
                        selector.set_status(
                            f"Loaded {session.run_entry.run_name} ({session.model_kind}) | "
                            f"VecNormalize: {'yes' if session.vecnormalize_loaded else 'no'}"
                        )
                except SystemExit as exc:
                    if selector is not None:
                        selector.set_status(str(exc))
                    else:
                        raise

            action = apply_keyboard_controls(
                session,
                control_state,
                turn_residual=args.turn_residual,
                idle_scale=args.idle_scale,
            )

            if control_state.pending_reset:
                control_state.pending_reset = False
                reset_session(session, args.seed)
                continue

            obs, rewards, dones, infos = session.env.step(action)
            session.obs = obs

            done_flag = bool(np.asarray(dones).reshape(-1)[0])
            info = infos[0] if isinstance(infos, (list, tuple)) else infos

            if done_flag or bool(info.get("time_limit_reached", False)):
                session.obs = session.env.reset()
                print(
                    f"Episode reset | run={session.run_entry.run_name} | "
                    f"forward_speed={info.get('forward_speed', 0.0):.3f}"
                )
    finally:
        session.close()


if __name__ == "__main__":
    main()
