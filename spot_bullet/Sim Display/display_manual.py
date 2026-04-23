from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = APP_DIR / ".runtime" / "manual_pybullet"
STATUS_PATH = RUNTIME_DIR / "status.json"
COMMAND_PATH = RUNTIME_DIR / "command.json"
PID_PATH = RUNTIME_DIR / "worker.pid"
LOG_PATH = RUNTIME_DIR / "worker.log"
WORKER_PATH = APP_DIR / "manual_pybullet_worker.py"
TRAINING_RUN_ROOT_NAMES = ("training runs", "training_runs")


st.set_page_config(
    page_title="SpotMini Manual Control",
    page_icon="SM",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 12% 15%, rgba(255, 193, 7, 0.18), transparent 28%),
            radial-gradient(circle at 82% 8%, rgba(34, 197, 94, 0.12), transparent 26%),
            linear-gradient(135deg, #0b0f14 0%, #111827 54%, #15110a 100%);
        color: #f8fafc;
    }
    .hero-card, .control-card {
        background: rgba(15, 23, 42, 0.74);
        border: 1px solid rgba(250, 204, 21, 0.18);
        border-radius: 24px;
        padding: 1.1rem 1.35rem;
        box-shadow: 0 20px 70px rgba(0,0,0,0.32);
    }
    .title-kicker {
        color: #facc15;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.22rem;
    }
    .status-pill {
        display:inline-block;
        padding:0.35rem 0.75rem;
        border-radius:999px;
        border:1px solid rgba(132, 204, 22, 0.38);
        color:#bef264;
        background:rgba(132, 204, 22, 0.10);
        font-weight:800;
        font-size:0.82rem;
    }
    .metric {
        display:flex;
        justify-content:space-between;
        gap:1rem;
        padding:0.75rem 0.9rem;
        margin-bottom:0.55rem;
        border-radius:14px;
        background:rgba(255,255,255,0.055);
        border:1px solid rgba(255,255,255,0.07);
    }
    .metric strong { color:#a3e635; }
    .hint {
        color:#cbd5e1;
        font-size:0.92rem;
        line-height:1.45;
    }
    div.stButton > button {
        min-height: 3rem;
        border-radius: 16px;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    baseline_version = 7
    defaults = {
        "manual_baseline_version": baseline_version,
        "control_mode": "Auto",
        "last_action": "Ready for manual GUI test",
        "manual_control_source": "Manual Mode",
        "manual_gait_mode": "trot",
        "manual_terrain_profile": "flat",
        "manual_speed_scale": 0.45,
        "manual_body_height_offset": 0.0,
        "manual_body_roll_bias_deg": 0.0,
        "manual_body_pitch_bias_deg": 0.0,
        "manual_front_swing_clearance_scale": 1.0,
        "manual_rear_swing_clearance_scale": 1.0,
        "manual_trot_step_length": 0.050,
        "manual_trot_back_step_length": 0.040,
        "manual_trot_turn_step": 0.010,
        "manual_trot_turn_rate": 0.90,
        "manual_trot_strafe_step": 0.030,
        "manual_trot_lateral_amount": 0.80,
        "manual_trot_step_velocity": 0.54,
        "manual_trot_curve_height": 0.040,
        "manual_policy_model": "best",
        "manual_enable_imu_yaw": False,
        "manual_follow_camera": True,
        "manual_notifications": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    legacy_sources = {
        "Manual Buttons": "Manual Mode",
        "Trained PPO Model": "Test PPO Model",
    }
    current_source = st.session_state.get("manual_control_source")
    if current_source in legacy_sources:
        st.session_state.manual_control_source = legacy_sources[current_source]

    if st.session_state.get("manual_baseline_version") != baseline_version:
        for key in (
            "manual_speed_scale",
            "manual_body_height_offset",
            "manual_body_roll_bias_deg",
            "manual_body_pitch_bias_deg",
            "manual_front_swing_clearance_scale",
            "manual_rear_swing_clearance_scale",
            "manual_trot_step_length",
            "manual_trot_back_step_length",
            "manual_trot_turn_step",
            "manual_trot_turn_rate",
            "manual_trot_strafe_step",
            "manual_trot_lateral_amount",
            "manual_trot_step_velocity",
            "manual_trot_curve_height",
            "manual_enable_imu_yaw",
        ):
            st.session_state[key] = defaults[key]
        st.session_state.manual_baseline_version = baseline_version


def notify(title: str, message: str) -> None:
    now = datetime.now().strftime("%I:%M %p").lstrip("0")
    st.session_state.manual_notifications.insert(0, (title, message, now))
    st.session_state.manual_notifications = st.session_state.manual_notifications[:6]


def read_json(path: Path, default: dict | None = None) -> dict:
    if default is None:
        default = {}
    if not path.exists():
        return default.copy()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default.copy()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def training_run_roots() -> list[Path]:
    roots = []
    seen = set()
    for root_name in TRAINING_RUN_ROOT_NAMES:
        root = APP_DIR.parent / root_name
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(root)
    return roots


def discover_playable_runs() -> list[dict]:
    runs = []
    seen = set()
    candidates = []
    for training_root in training_run_roots():
        if not training_root.exists():
            continue
        candidates.extend(path for path in training_root.iterdir() if path.is_dir())

    for run_dir in sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True):
        resolved = run_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)

        best_model = run_dir / "best_model" / "best_model.zip"
        final_model = run_dir / "models" / "ppo_spot_walk_final.zip"
        model_options = []
        if best_model.exists():
            model_options.append("best")
        if final_model.exists():
            model_options.append("final")
        if not model_options:
            continue

        terrain_profile = "flat"
        gait_mode = "trot"
        walk_variant = "default"
        config_path = run_dir / "run_config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                terrain_profile = config.get("terrain_profile") or terrain_profile
                gait_mode = config.get("gait_mode") or gait_mode
                walk_variant = config.get("walk_variant") or walk_variant
            except Exception:
                pass

        has_vecnormalize = (run_dir / "vecnormalize" / "vecnormalize.pkl").exists()
        vecnormalize_label = "VecNormalize" if has_vecnormalize else "raw env"
        runs.append(
            {
                "label": f"{run_dir.name} ({terrain_profile}, {gait_mode}, {vecnormalize_label})",
                "path": str(run_dir),
                "name": run_dir.name,
                "terrain_profile": terrain_profile,
                "gait_mode": gait_mode,
                "walk_variant": walk_variant,
                "model_options": model_options,
                "has_vecnormalize": has_vecnormalize,
            }
        )
    return runs


def read_status() -> dict:
    return read_json(STATUS_PATH)


def read_log_tail(max_lines: int = 14) -> str:
    if not LOG_PATH.exists():
        return ""
    try:
        return "\n".join(LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:])
    except Exception:
        return ""


def read_worker_pid() -> int | None:
    if not PID_PATH.exists():
        return None
    try:
        return int(PID_PATH.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False

    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
    except Exception:
        return True

    state = result.stdout.strip()
    if result.returncode != 0 or not state:
        return False
    return not state.startswith("Z")


def worker_is_running() -> bool:
    pid = read_worker_pid()
    if pid is None:
        return False
    running = pid_is_alive(pid)
    if not running:
        try:
            PID_PATH.unlink()
        except OSError:
            pass
    return running


def discover_worker_python() -> str:
    env_override = os.environ.get("SPOTMINI_ML_PYTHON")
    candidates = []
    if env_override:
        candidates.append(Path(env_override).expanduser())
    candidates.extend(
        [
            APP_DIR.parent / "src" / ".conda-spotml" / "bin" / "python",
            APP_DIR.parent / "src" / ".conda-spotml" / "bin" / "python3.10",
            APP_DIR.parent / "spotmini-env" / "bin" / "python",
            APP_DIR.parent / "src" / ".venv" / "bin" / "python",
            Path(sys.executable),
        ]
    )
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve() if candidate.exists() else candidate
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if candidate.exists():
            return candidate_str
    return str(Path(sys.executable))


def start_worker() -> bool:
    if worker_is_running():
        return False
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    control_source = st.session_state.manual_control_source
    write_json(COMMAND_PATH, build_command_payload("stop"))
    worker_python = discover_worker_python()
    command = [
        worker_python,
        str(WORKER_PATH),
        "--runtime-dir",
        str(RUNTIME_DIR),
        "--control-source",
        "policy" if control_source == "Test PPO Model" else "manual",
        "--terrain-profile",
        st.session_state.manual_terrain_profile,
        "--gait-mode",
        st.session_state.manual_gait_mode,
        "--body-height-offset",
        str(st.session_state.manual_body_height_offset),
        "--body-roll-bias-deg",
        str(st.session_state.manual_body_roll_bias_deg),
        "--body-pitch-bias-deg",
        str(st.session_state.manual_body_pitch_bias_deg),
        "--front-swing-clearance-scale",
        str(st.session_state.manual_front_swing_clearance_scale),
        "--rear-swing-clearance-scale",
        str(st.session_state.manual_rear_swing_clearance_scale),
    ]
    if control_source == "Test PPO Model":
        selected_run = st.session_state.get("manual_policy_run")
        if not selected_run:
            notify("Manual Mode", "Select a trained run before launching model playback.")
            return False
        command.extend(
            [
                "--policy-run-dir",
                selected_run,
                "--policy-model",
                st.session_state.manual_policy_model,
            ]
        )
    # Keep the new baseline manual mode raw: no IMU yaw/body stabilization.
    if st.session_state.manual_follow_camera:
        command.append("--follow-camera")

    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(
            f"\n[{datetime.now().isoformat(timespec='seconds')}] Starting manual PyBullet worker with {worker_python}\n"
        )
        process = subprocess.Popen(
            command,
            cwd=str(APP_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=os.name != "nt",
        )
    PID_PATH.write_text(str(process.pid), encoding="utf-8")
    return True


def wait_for_worker_exit(pid: int, timeout_seconds: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not PID_PATH.exists() or not pid_is_alive(pid):
            return True
        time.sleep(0.1)
    return False


def stop_worker(wait: bool = False) -> bool:
    pid = read_worker_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        if wait:
            return wait_for_worker_exit(pid)
        return True
    except OSError:
        return False


def build_command_payload(command_name: str, *, reset: bool = False) -> dict:
    return {
        "command": command_name,
        "speed_scale": float(st.session_state.manual_speed_scale),
        "trot_step_length": float(st.session_state.manual_trot_step_length),
        "trot_back_step_length": float(st.session_state.manual_trot_back_step_length),
        "trot_turn_step": float(st.session_state.manual_trot_turn_step),
        "trot_turn_rate": float(st.session_state.manual_trot_turn_rate),
        "trot_strafe_step": float(st.session_state.manual_trot_strafe_step),
            "trot_lateral_amount": float(st.session_state.manual_trot_lateral_amount),
            "trot_step_velocity": float(st.session_state.manual_trot_step_velocity),
            "trot_curve_height": float(st.session_state.manual_trot_curve_height),
        "reset": reset,
        "timestamp": time.time(),
    }


def send_command(command_name: str, *, reset: bool = False, refresh: bool = True) -> None:
    write_json(COMMAND_PATH, build_command_payload(command_name, reset=reset))
    st.session_state.last_action = f"Manual command: {command_name.replace('_', ' ')}"
    if refresh:
        time.sleep(0.15)
        st.rerun()


def sync_live_trot_tuning() -> None:
    current_command = read_json(COMMAND_PATH, {"command": "stop"}).get("command", "stop")
    if current_command == "stop":
        return
    write_json(COMMAND_PATH, build_command_payload(str(current_command)))
    st.session_state.last_action = f"Tuning applied to {str(current_command).replace('_', ' ')}"


def restart_worker() -> bool:
    if worker_is_running():
        stop_worker(wait=True)
    return start_worker()


def render_metric(label: str, value) -> None:
    st.markdown(
        f"<div class='metric'><span>{label}</span><strong>{value}</strong></div>",
        unsafe_allow_html=True,
    )


def button_choice(label: str, key: str, options: list[tuple[str, str]], *, disabled: bool = False) -> None:
    st.caption(label)
    columns = st.columns(len(options))
    for column, (value, display) in zip(columns, options):
        selected = st.session_state.get(key) == value
        button_label = f"[x] {display}" if selected else display
        if column.button(button_label, use_container_width=True, disabled=disabled or selected):
            st.session_state[key] = value
            st.rerun()


def reset_trot_baseline() -> None:
    st.session_state.manual_speed_scale = 0.45
    st.session_state.manual_body_height_offset = 0.0
    st.session_state.manual_body_roll_bias_deg = 0.0
    st.session_state.manual_body_pitch_bias_deg = 0.0
    st.session_state.manual_front_swing_clearance_scale = 1.0
    st.session_state.manual_rear_swing_clearance_scale = 1.0
    st.session_state.manual_trot_step_length = 0.050
    st.session_state.manual_trot_back_step_length = 0.040
    st.session_state.manual_trot_turn_step = 0.010
    st.session_state.manual_trot_turn_rate = 0.90
    st.session_state.manual_trot_strafe_step = 0.030
    st.session_state.manual_trot_lateral_amount = 0.80
    st.session_state.manual_trot_step_velocity = 0.54
    st.session_state.manual_trot_curve_height = 0.040
    st.session_state.manual_enable_imu_yaw = False


init_state()
status = read_status()
running = worker_is_running()
command_payload = read_json(COMMAND_PATH, {"command": "stop"})
playable_runs = discover_playable_runs()
run_by_path = {run["path"]: run for run in playable_runs}
if playable_runs and st.session_state.get("manual_policy_run") not in run_by_path:
    st.session_state.manual_policy_run = playable_runs[0]["path"]

left, right = st.columns([3, 1.2])
with left:
    st.markdown(
        """
        <div class='hero-card'>
            <div class='title-kicker'>SPOTMINI MANUAL LAB</div>
            <div style='font-size:2.35rem;font-weight:900;margin-top:0.35rem;'>Manual PyBullet Control</div>
            <div class='hint' style='margin-top:0.75rem;max-width:900px;'>
                This is a separate dashboard file for manual testing. Enter Manual Mode to launch a native PyBullet GUI,
                then drive the robot with the on-screen movement buttons.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    state_text = "Manual GUI Live" if running else "GUI Offline"
    st.markdown(
        f"""
        <div class='hero-card' style='text-align:center;'>
            <div class='title-kicker'>CONTROL STATE</div>
            <div style='font-size:1.35rem;font-weight:900;margin-top:0.35rem;'>{state_text}</div>
            <div style='margin-top:0.85rem;'><span class='status-pill'>{st.session_state.control_mode}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

setup_col, controls_col, status_col = st.columns([1.05, 1.25, 1.0])

with setup_col:
    st.markdown("<div class='control-card'>", unsafe_allow_html=True)
    st.subheader("Manual Setup")
    button_choice(
        "Control Source",
        "manual_control_source",
        [("Manual Mode", "Manual Mode"), ("Test PPO Model", "Test PPO Model")],
        disabled=running,
    )
    using_policy = st.session_state.manual_control_source == "Test PPO Model"
    if using_policy:
        if playable_runs:
            st.selectbox(
                "Trained Run",
                [run["path"] for run in playable_runs],
                key="manual_policy_run",
                format_func=lambda path: run_by_path.get(path, {}).get("label", Path(path).name),
                disabled=running,
            )
            selected_run = run_by_path.get(st.session_state.manual_policy_run)
            model_options = selected_run["model_options"] if selected_run else ["best"]
            if st.session_state.manual_policy_model not in model_options:
                st.session_state.manual_policy_model = model_options[0]
            st.selectbox(
                "Model",
                model_options,
                key="manual_policy_model",
                disabled=running,
            )
            if selected_run:
                st.caption(
                    f"{selected_run['walk_variant']} | {selected_run['terrain_profile']} | {selected_run['gait_mode']}"
                )
        else:
            st.warning("No trained PPO runs with best/final models were found.")

    if not using_policy:
        button_choice(
            "Gait",
            "manual_gait_mode",
            [("trot", "Trot"), ("four_phase", "4 Phase")],
            disabled=running,
        )
        button_choice(
            "Terrain",
            "manual_terrain_profile",
            [("flat", "Flat"), ("rough", "Rough")],
            disabled=running,
        )
    st.slider("Speed Scale", 0.10, 1.00, key="manual_speed_scale", step=0.05)
    st.slider("Body Lowering Offset", 0.0, 0.05, key="manual_body_height_offset", step=0.002, disabled=running or using_policy)
    st.slider("Body Roll Trim", -4.0, 4.0, key="manual_body_roll_bias_deg", step=0.1, disabled=running or using_policy)
    st.slider("Body Pitch Bias", -6.0, 3.0, key="manual_body_pitch_bias_deg", step=0.2, disabled=running or using_policy)
    st.slider("Front Swing Height Scale", 0.85, 1.45, key="manual_front_swing_clearance_scale", step=0.01, disabled=running or using_policy)
    st.slider("Rear Swing Height Scale", 0.55, 1.05, key="manual_rear_swing_clearance_scale", step=0.01, disabled=running or using_policy)
    st.checkbox("Enable IMU Yaw", key="manual_enable_imu_yaw", disabled=True)
    if not using_policy and st.session_state.manual_gait_mode == "trot":
        if running and status.get("control_source") != "policy":
            sync_live_trot_tuning()
    st.checkbox("Follow PyBullet Camera", key="manual_follow_camera", disabled=running)

    launch_label = (
        "Launch Selected Trained Model"
        if using_policy
        else "Launch Manual Mode"
    )
    launch_disabled = running or (using_policy and not playable_runs)
    if st.button(launch_label, use_container_width=True, disabled=launch_disabled):
        if start_worker():
            st.session_state.control_mode = "Model" if using_policy else "Manual"
            st.session_state.last_action = (
                "Trained PPO model launched" if using_policy else "Manual PyBullet GUI launched"
            )
            notify(
                "Manual Mode",
                "Native PyBullet GUI worker started with selected model."
                if using_policy
                else "Native PyBullet GUI worker started.",
            )
        else:
            notify("Manual Mode", "Manual worker is already running.")
        st.rerun()

    if st.button("Exit Manual Mode + Stop GUI", use_container_width=True, disabled=not running):
        if stop_worker(wait=True):
            st.session_state.control_mode = "Auto"
            st.session_state.last_action = "Manual PyBullet GUI stopped"
            notify("Manual Mode", "Stop signal sent to PyBullet GUI worker.")
        st.rerun()

    if st.button("Force Restart Manual Worker", use_container_width=True, disabled=using_policy):
        if restart_worker():
            st.session_state.control_mode = "Manual"
            st.session_state.last_action = "Manual worker force restarted"
            notify("Manual Mode", "Worker force restarted with the latest code.")
        else:
            notify("Manual Mode", "Could not restart worker.")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

with controls_col:
    st.markdown("<div class='control-card'>", unsafe_allow_html=True)
    st.subheader("On-Screen Controls")
    st.caption("Commands hold until you press another direction or Stop.")

    policy_running = status.get("control_source") == "policy"
    disabled = not running or policy_running
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Forward Left", use_container_width=True, disabled=disabled):
            send_command("forward_left")
    with c2:
        if st.button("Forward", use_container_width=True, disabled=disabled):
            send_command("forward")
    with c3:
        if st.button("Forward Right", use_container_width=True, disabled=disabled):
            send_command("forward_right")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Turn Left", use_container_width=True, disabled=disabled):
            send_command("turn_left")
    with c2:
        if st.button("Stop", use_container_width=True, disabled=disabled):
            send_command("stop")
    with c3:
        if st.button("Turn Right", use_container_width=True, disabled=disabled):
            send_command("turn_right")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Strafe Left", use_container_width=True, disabled=disabled):
            send_command("left")
    with c2:
        if st.button("Back", use_container_width=True, disabled=disabled):
            send_command("back")
    with c3:
        if st.button("Strafe Right", use_container_width=True, disabled=disabled):
            send_command("right")

    if st.button("Reset Robot Pose", use_container_width=True, disabled=disabled):
        send_command("stop", reset=True)
        notify("Manual Control", "Reset command sent.")

    st.markdown(
        "<div class='hint'>Tip: four-phase turning is now enabled for manual mode. Use Stop before switching directions if a gait starts arcing.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with status_col:
    st.markdown("<div class='control-card'>", unsafe_allow_html=True)
    st.subheader("Live Status")
    render_metric("Worker", "Running" if running else status.get("state", "stopped").title())
    render_metric("Worker Version", status.get("worker_version", "not started"))
    render_metric("Command", status.get("command", "stop"))
    render_metric("Command File", command_payload.get("command", "stop"))
    render_metric("Source", status.get("control_source", "manual"))
    if status.get("run_name"):
        render_metric("Run", status.get("run_name"))
    render_metric("Gait", status.get("gait_mode", st.session_state.manual_gait_mode))
    render_metric("Terrain", status.get("terrain_profile", st.session_state.manual_terrain_profile))
    curve_height = status.get("curve_height")
    render_metric("Curve Height", f"{float(curve_height):0.3f}" if curve_height is not None else "Awaiting data")
    left_gain = status.get("forward_left_step_gain")
    right_gain = status.get("forward_right_step_gain")
    render_metric(
        "Stride Gain",
        (
            f"L {float(left_gain):0.2f} / R {float(right_gain):0.2f}"
            if left_gain is not None and right_gain is not None
            else "Awaiting data"
        ),
    )
    speed = status.get("forward_speed")
    render_metric("Forward Speed", f"{float(speed):0.2f} m/s" if speed is not None else "Awaiting data")
    render_metric("Fallen", "Yes" if status.get("is_fallen") else "No")
    render_metric("Last Action", st.session_state.last_action)

    if status.get("state") == "error":
        st.error(status.get("error", "Manual worker failed."))
        log_tail = read_log_tail()
        if log_tail:
            st.code(log_tail, language="text")

    st.caption(f"Worker Python: `{discover_worker_python()}`")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

st.subheader("Manual Event Log")
if st.session_state.manual_notifications:
    for title, message, when in st.session_state.manual_notifications:
        st.markdown(f"**{title}** - {when}  \n{message}")
else:
    st.caption("No manual events yet.")
