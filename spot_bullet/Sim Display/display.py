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
RUNTIME_DIR = APP_DIR / ".runtime" / "ml_camera"
FRAME_PATH = RUNTIME_DIR / "latest_camera.jpg"
STATUS_PATH = RUNTIME_DIR / "status.json"
PID_PATH = RUNTIME_DIR / "worker.pid"
LOG_PATH = RUNTIME_DIR / "worker.log"
WORKER_PATH = APP_DIR / "ml_camera_worker.py"
TRAINING_RUN_ROOT_NAMES = ("training runs", "training_runs")

st.set_page_config(
    page_title="SpotMini Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme / styles ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #09090b 0%, #0f172a 50%, #111827 100%);
        color: white;
    }
    .hero-card, .panel-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 22px;
        padding: 1rem 1.25rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.25);
    }
    .phone-shell {
        background: linear-gradient(180deg, #0f172a 0%, #09090b 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 32px;
        padding: 1rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        min-height: 640px;
    }
    .screen-title { font-size: 1.6rem; font-weight: 700; color: white; margin-bottom: 0.2rem; }
    .screen-subtitle { color: #a1a1aa; font-size: 0.95rem; }
    .badge {
        display: inline-block;
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
        border: 1px solid rgba(250, 204, 21, 0.35);
        background: rgba(250, 204, 21, 0.10);
        color: #fde68a;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .robot-banner {
        border-radius: 20px;
        padding: 1rem;
        background: linear-gradient(135deg, #27272a 0%, #09090b 100%);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .robot-thumb {
        width: 100%;
        height: 80px;
        border-radius: 16px;
        background: linear-gradient(135deg, #fde047 0%, #f59e0b 100%);
    }
    .metric {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.65rem 0.9rem;
        border-radius: 14px;
        background: rgba(255,255,255,0.05);
        margin-bottom: 0.5rem;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .good { color: #84cc16; font-weight: 700; }
    .warn { color: #facc15; font-weight: 700; }
    .neutral { color: white; font-weight: 700; }
    .waypoint {
        padding: 0.65rem 0.8rem;
        border-radius: 12px;
        background: rgba(255,255,255,0.05);
        margin-bottom: 0.45rem;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .waypoint.active {
        background: rgba(250, 204, 21, 0.14);
        color: #fde68a;
        border-color: rgba(250, 204, 21, 0.25);
    }
    .notif {
        padding: 0.85rem 1rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 0.75rem;
    }
    .camera-box {
        border-radius: 22px;
        border: 1px solid rgba(255,255,255,0.08);
        height: 220px;
        display: flex;
        justify-content: center;
        align-items: center;
        background:
            radial-gradient(circle at center, rgba(250,204,21,0.18), transparent 30%),
            linear-gradient(135deg, #3f3f46, #18181b);
        color: rgba(255,255,255,0.85);
        font-weight: 600;
    }
    .planner-grid {
        position: relative;
        border-radius: 22px;
        height: 220px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
        background:
            linear-gradient(135deg, #3f3f46, #18181b);
    }
    .planner-grid::before {
        content: "";
        position: absolute;
        inset: 0;
        background-image:
            linear-gradient(to right, rgba(255,255,255,0.12) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(255,255,255,0.12) 1px, transparent 1px);
        background-size: 24px 24px;
        opacity: 0.25;
    }
    .bottom-nav {
        margin-top: 1rem;
        background: rgba(0,0,0,0.22);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 0.55rem;
    }
    .tiny { color:#a1a1aa; font-size:0.8rem; }
    .lime { color:#84cc16; }
    .yellow { color:#facc15; }
    .red { color:#f87171; }
    .rose { color:#fb7185; }

/* Hide any unused placeholder panels (the large empty boxes) */
.panel-card:empty {
    display: none !important;}

/* Also collapse empty Streamlit column blocks if they contain no widgets */
div[data-testid="stVerticalBlock"]:has(> div:empty) {
    min-height: 0 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- State ----------
def init_state():
    defaults = {
        "mode": "Trot",
        "control_mode": "Auto",
        "recording": False,
        "lights": False,
        "zoom": False,
        "speak": False,
        "mission_status": "Idle",
        "selected_mission": "Patrol Route",
        "waypoint_idx": 2,
        "battery": 85,
        "system_temp": 47,
        "connection": "Strong",
        "last_action": "Ready for demo",
        "notification_log": [
            ("Mission Completed", "Patrol route finished successfully.", "9:26 AM", "lime"),
            ("Obstacle Detected", "Object ahead, mission paused.", "8:57 AM", "yellow"),
            ("Temperature Warning", "System temperature reached 55°C.", "8:45 AM", "red"),
            ("Connection Lost", "Robot disconnected briefly.", "8:37 AM", "rose"),
        ],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

# ---------- Helpers ----------
def add_notification(title, message, color="lime"):
    now = datetime.now().strftime("%I:%M %p").lstrip("0")
    st.session_state.notification_log.insert(0, (title, message, now, color))
    st.session_state.notification_log = st.session_state.notification_log[:6]


def mission_select(label):
    st.session_state.selected_mission = label
    st.session_state.last_action = f"Selected mission: {label}"


def set_control_mode(mode):
    if mode not in {"Auto", "Manual"}:
        return
    if st.session_state.control_mode == mode:
        return

    st.session_state.control_mode = mode
    st.session_state.last_action = f"Switched control to {mode.lower()} mode"
    if mode == "Manual":
        st.session_state.mission_status = "Manual Control"
        add_notification("Control Mode", "Robot switched to manual movement mode.", "yellow")
    else:
        if st.session_state.mission_status == "Manual Control":
            st.session_state.mission_status = "Idle"
        add_notification("Control Mode", "Robot switched to auto movement mode.", "lime")


def training_run_roots():
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


def discover_playable_runs():
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
        if not best_model.exists():
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
                "root": str(run_dir.parent),
                "terrain_profile": terrain_profile,
                "gait_mode": gait_mode,
                "walk_variant": walk_variant,
                "has_vecnormalize": has_vecnormalize,
            }
        )
    return runs


def discover_worker_python():
    env_override = os.environ.get("SPOTMINI_ML_PYTHON")
    candidates = []
    if env_override:
        candidates.append(Path(env_override).expanduser())

    candidates.extend(
        [
            APP_DIR.parent / "src" / ".conda-spotml" / "bin" / "python",
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


def read_status():
    if not STATUS_PATH.exists():
        return {}
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_status(path, **payload):
    payload.setdefault("timestamp", time.time())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_log_tail(max_lines=12):
    if not LOG_PATH.exists():
        return ""
    try:
        lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-max_lines:])


def read_worker_pid():
    if not PID_PATH.exists():
        return None
    try:
        return int(PID_PATH.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def pid_is_alive(pid):
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


def worker_is_running():
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


def same_path(left, right):
    if not left or not right:
        return False
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except Exception:
        return str(left) == str(right)


def start_ml_camera_worker(run_dir):
    if worker_is_running():
        return False

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    write_status(
        STATUS_PATH,
        state="starting",
        running=False,
        run_name=Path(run_dir).name,
        run_dir=run_dir,
    )
    worker_python = discover_worker_python()
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(
            f"\n[{datetime.now().isoformat(timespec='seconds')}] Starting ML camera worker for {run_dir} with {worker_python}\n"
        )

    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [
                worker_python,
                str(WORKER_PATH),
                "--run-dir",
                run_dir,
                "--output-dir",
                str(RUNTIME_DIR),
            ],
            cwd=str(APP_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=os.name != "nt",
        )
    PID_PATH.write_text(str(process.pid), encoding="utf-8")
    return True


def wait_for_worker_exit(pid, timeout_seconds=10.0):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not PID_PATH.exists():
            return True
        if not pid_is_alive(pid):
            return True
        time.sleep(0.1)
    return False


def stop_ml_camera_worker(wait=False):
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


def switch_ml_camera_worker(run_dir):
    if worker_is_running():
        stop_ml_camera_worker(wait=True)
        if worker_is_running():
            return False
    return start_ml_camera_worker(run_dir)


def render_camera_placeholder():
    st.markdown(
        "<div class='camera-box'>ML CAMERA FEED OFFLINE</div>",
        unsafe_allow_html=True,
    )


AVAILABLE_RUNS = discover_playable_runs()
AVAILABLE_RUN_PATHS = [run["path"] for run in AVAILABLE_RUNS]
DEFAULT_RUN_PATH = AVAILABLE_RUNS[0]["path"] if AVAILABLE_RUNS else None
if "ml_run_path" not in st.session_state or (
    AVAILABLE_RUN_PATHS and st.session_state.ml_run_path not in AVAILABLE_RUN_PATHS
):
    st.session_state.ml_run_path = DEFAULT_RUN_PATH
elif not AVAILABLE_RUN_PATHS:
    st.session_state.ml_run_path = None

AUTO_MODE_ENABLED = st.session_state.control_mode == "Auto"


# ---------- Header ----------
left, right = st.columns([4, 1.4])
with left:
    st.markdown(
        """
            <div class='hero-card'>
            <div style='letter-spacing:4px;color:#facc15;font-size:0.78rem;font-weight:700;'>SPOTMINI LOCAL UI</div>
            <div style='font-size:2.2rem;font-weight:800;margin-top:0.35rem;'>SpotMini Dashboard</div>
            <div class='screen-subtitle' style='margin-top:0.8rem;max-width:900px;'>
                A local control dashboard for the SpotMini project. This Streamlit version keeps the same core sections from your mockup: control, missions, live camera, diagnostics, mission planning, and notifications.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        """
        <div class='hero-card' style='text-align:center;padding-top:1.1rem;padding-bottom:1.1rem;'>
            <div class='tiny'>Launch Mode</div>
            <div style='font-size:1.5rem;font-weight:800;margin-top:0.25rem;'>Local Dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ---------- Top Panels ----------
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    top_l, top_r = st.columns([4, 1])
    with top_l:
        st.markdown("<div class='screen-title'>Control</div>", unsafe_allow_html=True)
        st.markdown("<div class='screen-subtitle'>CONNECTED • Battery 85%</div>", unsafe_allow_html=True)
    with top_r:
        st.markdown("<div style='height:12px'></div><div class='badge'>Live Demo</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    banner_l, banner_r = st.columns([2.2, 1])
    with banner_l:
        st.markdown(
            f"""
            <div class='robot-banner'>
                <div style='letter-spacing:3px;color:#a3e635;font-size:0.76rem;font-weight:700;'>ROBOT ONLINE</div>
                <div style='font-size:1.55rem;font-weight:800;margin-top:0.4rem;'>SpotMini Simulator</div>
                <div class='screen-subtitle' style='margin-top:0.35rem;'>Mode: {st.session_state.mode} · Last action: {st.session_state.last_action}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with banner_r:
        st.markdown("<div class='robot-thumb'></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'><span>Connection</span><span class='good'>{st.session_state.connection}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'><span>Battery</span><span class='warn'>{st.session_state.battery}%</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'><span>System Temp</span><span class='neutral'>{st.session_state.system_temp}°C</span></div>", unsafe_allow_html=True)
    control_mode_class = "good" if AUTO_MODE_ENABLED else "warn"
    st.markdown(
        f"<div class='metric'><span>Control Mode</span><span class='{control_mode_class}'>{st.session_state.control_mode}</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    selected_mode = st.selectbox("Movement Mode", ["Trot", "Walk", "Standby", "Inspect"], index=["Trot", "Walk", "Standby", "Inspect"].index(st.session_state.mode), key="mode_select")
    if selected_mode != st.session_state.mode:
        st.session_state.mode = selected_mode
        st.session_state.last_action = f"Switched mode to {selected_mode}"
        add_notification("Mode Updated", f"Robot changed to {selected_mode} mode.", "lime")

    toggle_label = "Switch To Manual" if AUTO_MODE_ENABLED else "Switch To Auto"
    if st.button(toggle_label, use_container_width=True):
        set_control_mode("Manual" if AUTO_MODE_ENABLED else "Auto")
        st.rerun()

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Toggle Lights", use_container_width=True):
            st.session_state.lights = not st.session_state.lights
            st.session_state.last_action = "Lights enabled" if st.session_state.lights else "Lights disabled"
            add_notification("Lights", st.session_state.last_action + ".", "yellow")
    with b2:
        if st.button("Emergency Stop", use_container_width=True):
            st.session_state.mission_status = "Stopped"
            st.session_state.last_action = "Emergency stop activated"
            add_notification("Emergency Stop", "Mission halted immediately.", "red")

    b3, b4 = st.columns(2)
    with b3:
        if st.button("Start Patrol", use_container_width=True, disabled=not AUTO_MODE_ENABLED):
            st.session_state.mission_status = "Running"
            st.session_state.selected_mission = "Patrol Route"
            st.session_state.last_action = "Patrol started"
            add_notification("Mission Started", "Patrol Route is now running.", "lime")
    with b4:
        if st.button("Return Home", use_container_width=True, disabled=not AUTO_MODE_ENABLED):
            st.session_state.mission_status = "Returning"
            st.session_state.last_action = "Robot returning to dock"
            add_notification("Return Home", "Robot is navigating back to dock.", "yellow")

    control_hint = "Auto missions enabled" if AUTO_MODE_ENABLED else "Manual movement enabled · auto missions locked"
    st.markdown(f"<div class='bottom-nav'><div class='tiny'>{control_hint}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    top_l, top_r = st.columns([4, 1.2])
    with top_l:
        st.markdown("<div class='screen-title'>Missions</div>", unsafe_allow_html=True)
        st.markdown("<div class='screen-subtitle'>4 automation presets</div>", unsafe_allow_html=True)
    with top_r:
        st.markdown("<div style='height:12px'></div><div class='badge'>Mission Hub</div>", unsafe_allow_html=True)

    missions = ["Patrol Route", "Inspect Site", "Delivery Run", "Perimeter Scan"]
    for mission in missions:
        is_selected = st.session_state.selected_mission == mission
        if st.button(("● " if is_selected else "○ ") + mission, use_container_width=True, key=f"mission_{mission}", disabled=not AUTO_MODE_ENABLED):
            mission_select(mission)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'><span>Status</span><span class='neutral'>{st.session_state.mission_status}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'><span>Active Mission</span><span class='good'>{st.session_state.selected_mission}</span></div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:10px;font-weight:700;'>Waypoint Progress</div>", unsafe_allow_html=True)
    waypoints = ["Dock", "Corridor A", "Inspection Zone", "Storage Rack", "Return Path"]
    for i, wp in enumerate(waypoints):
        cls = "waypoint active" if i == st.session_state.waypoint_idx else "waypoint"
        st.markdown(f"<div class='{cls}'>{i+1}. {wp}</div>", unsafe_allow_html=True)

    n1, n2 = st.columns(2)
    with n1:
        if st.button("Next Waypoint", use_container_width=True, disabled=not AUTO_MODE_ENABLED):
            st.session_state.waypoint_idx = (st.session_state.waypoint_idx + 1) % len(waypoints)
            st.session_state.last_action = f"Moved to {waypoints[st.session_state.waypoint_idx]}"
            add_notification("Waypoint Updated", st.session_state.last_action + ".", "lime")
    with n2:
        if st.button("Pause Mission", use_container_width=True, disabled=not AUTO_MODE_ENABLED):
            st.session_state.mission_status = "Paused"
            st.session_state.last_action = "Mission paused"
            add_notification("Mission Paused", "Awaiting operator input.", "yellow")

    mission_hint = "Preset routes · Waypoint navigation" if AUTO_MODE_ENABLED else "Mission routing disabled in manual mode"
    st.markdown(f"<div class='bottom-nav'><div class='tiny'>{mission_hint}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    top_l, top_r = st.columns([4, 1])
    with top_l:
        st.markdown("<div class='screen-title'>Live Camera</div>", unsafe_allow_html=True)
        st.markdown("<div class='screen-subtitle'>ML body camera • PyBullet live feed</div>", unsafe_allow_html=True)
    with top_r:
        st.markdown("<div style='height:12px'></div><div class='badge'>Camera</div>", unsafe_allow_html=True)

    if AVAILABLE_RUNS:
        st.selectbox(
            "ML Run",
            options=AVAILABLE_RUN_PATHS,
            format_func=lambda path: next((run["label"] for run in AVAILABLE_RUNS if run["path"] == path), path),
            key="ml_run_path",
        )
        selected_run = next((run for run in AVAILABLE_RUNS if run["path"] == st.session_state.ml_run_path), None)
        if selected_run:
            st.caption(
                "Loaded from "
                f"`{selected_run['root']}` · "
                f"variant `{selected_run['walk_variant']}` · "
                f"{'normalization stats found' if selected_run['has_vecnormalize'] else 'normalization stats missing; worker will try raw env'}"
            )
    else:
        scanned_roots = ", ".join(f"`{root}`" for root in training_run_roots())
        st.warning(
            "No playable ML runs found. A playable run needs "
            "`best_model/best_model.zip` inside one of the scanned training folders."
        )
        st.caption(f"Scanned: {scanned_roots}")

    @st.fragment(run_every=1.0)
    def live_camera_fragment():
        status = read_status()
        running = worker_is_running()
        selected_run_path = st.session_state.ml_run_path
        active_run_path = status.get("run_dir")
        selected_is_active = running and same_path(selected_run_path, active_run_path)
        switch_pending = running and bool(selected_run_path) and not selected_is_active
        cached_frame_available = FRAME_PATH.exists()
        frame_available = running and cached_frame_available and not switch_pending

        if frame_available:
            st.image(str(FRAME_PATH), use_container_width=True)
        else:
            render_camera_placeholder()
            if switch_pending:
                selected_name = next(
                    (run["name"] for run in AVAILABLE_RUNS if run["path"] == selected_run_path),
                    "the selected run",
                )
                active_name = status.get("run_name", "the current worker")
                st.caption(
                    f"Selected `{selected_name}`, but the active worker is still `{active_name}`. "
                    "Click Switch ML Feed to restart the camera worker with the selected run."
                )
            elif cached_frame_available:
                st.caption("A cached frame exists, but the ML worker is not running, so the live feed is marked offline.")

        c1, c2 = st.columns(2)
        with c1:
            start_label = "Switch ML Feed" if switch_pending else "Start ML Feed"
            start_disabled = not AVAILABLE_RUNS or (running and not switch_pending)
            if st.button(start_label, use_container_width=True, disabled=start_disabled):
                if switch_pending:
                    started = switch_ml_camera_worker(selected_run_path)
                    if started:
                        st.session_state.last_action = "Switched ML body camera feed"
                        add_notification("ML Camera", "Camera worker restarted with the selected run.", "lime")
                    else:
                        add_notification("ML Camera", "Could not stop the previous camera worker yet.", "red")
                else:
                    started = start_ml_camera_worker(selected_run_path)
                    if started:
                        st.session_state.last_action = "Started ML body camera feed"
                        add_notification("ML Camera", "Live body camera feed started.", "lime")
                    else:
                        add_notification("ML Camera", "Camera feed is already running.", "yellow")
                st.rerun()
        with c2:
            if st.button("Stop ML Feed", use_container_width=True, disabled=not running):
                stopped = stop_ml_camera_worker()
                if stopped:
                    st.session_state.last_action = "Stopped ML body camera feed"
                    add_notification("ML Camera", "Live body camera feed stopped.", "yellow")
                st.rerun()

        run_name = status.get("run_name", "Unavailable")
        feed_state = status.get("state", "idle")
        feed_speed = status.get("forward_speed")
        stale_running_status = not running and feed_state == "running"
        if running:
            status_text = "Running"
        elif stale_running_status:
            status_text = "Stopped (stale status)"
        else:
            status_text = feed_state.title()
        signal_text = "Live" if running and frame_available else "Idle"

        st.markdown(
            f"<div class='metric'><span>Feed State</span><span class='neutral'>{status_text}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='metric'><span>Active Run</span><span class='good'>{run_name}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='metric'><span>Signal</span><span class='good'>{signal_text}</span></div>",
            unsafe_allow_html=True,
        )
        speed_value = f"{float(feed_speed):0.2f} m/s" if feed_speed is not None else "Awaiting data"
        st.markdown(
            f"<div class='metric'><span>Forward Speed</span><span class='neutral'>{speed_value}</span></div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Worker Python: `{discover_worker_python()}`")
        if status.get("vecnormalize_loaded") is False and status.get("vecnormalize_error"):
            st.caption("Normalization fallback active: using raw environment because saved VecNormalize stats did not load cleanly.")
        if stale_running_status:
            st.caption("Previous worker status was left behind. Start the ML feed again to refresh the camera state.")

        if status.get("state") == "error":
            st.error(status.get("error", "ML camera worker failed."))
            log_tail = read_log_tail()
            if log_tail:
                st.code(log_tail, language="text")

    live_camera_fragment()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Snapshot Marker", use_container_width=True):
            st.session_state.last_action = "Snapshot marker dropped"
            add_notification("Camera", "Snapshot marker saved for the live feed.", "lime")
    with c2:
        if st.button("Zoom View", use_container_width=True):
            st.session_state.zoom = not st.session_state.zoom
            st.session_state.last_action = "Zoom enabled" if st.session_state.zoom else "Zoom disabled"
            add_notification("Camera Zoom", st.session_state.last_action + ".", "lime")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'><span>Lens Mode</span><span class='neutral'>{'Zoomed' if st.session_state.zoom else 'Standard'}</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='metric'><span>Stabilization</span><span class='good'>Model Camera Ready</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='bottom-nav'><div class='tiny'>Vision system · ML playback body camera</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ---------- Bottom Panels ----------
b1, b2 = st.columns([1.5, 1])
with b1:
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("<div class='screen-title'>Mission Planner</div>", unsafe_allow_html=True)
    st.markdown("<div class='screen-subtitle'>Interactive route grid for mock mission mapping</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='planner-grid'></div>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("Add Waypoint", use_container_width=True):
            add_notification("Planner", "New waypoint drafted on route grid.", "lime")
    with p2:
        if st.button("Optimize Route", use_container_width=True):
            add_notification("Planner", "Route optimized for shortest safe path.", "yellow")
    with p3:
        if st.button("Deploy Mission", use_container_width=True, disabled=not AUTO_MODE_ENABLED):
            st.session_state.mission_status = "Running"
            add_notification("Deployment", f"{st.session_state.selected_mission} deployed from planner.", "lime")
    st.markdown("</div>", unsafe_allow_html=True)

with b2:
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("<div class='screen-title'>Notifications</div>", unsafe_allow_html=True)
    st.markdown("<div class='screen-subtitle'>Recent robot events and alerts</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    for title, msg, ts, color in st.session_state.notification_log:
        st.markdown(
            f"""
            <div class='notif'>
                <div style='display:flex;justify-content:space-between;gap:12px;'>
                    <div style='font-weight:800;' class='{color}'>{title}</div>
                    <div class='tiny'>{ts}</div>
                </div>
                <div style='margin-top:0.35rem;color:#d4d4d8;'>{msg}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
