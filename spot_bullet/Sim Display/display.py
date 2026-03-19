import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Quadruped Robot Control UI",
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


# ---------- Header ----------
left, right = st.columns([4, 1.4])
with left:
    st.markdown(
        """
        <div class='hero-card'>
            <div style='letter-spacing:4px;color:#facc15;font-size:0.78rem;font-weight:700;'>DEMO PLATFORM</div>
            <div style='font-size:2.2rem;font-weight:800;margin-top:0.35rem;'>Quadruped Robot Control UI</div>
            <div class='screen-subtitle' style='margin-top:0.8rem;max-width:900px;'>
                A browser-based demo inspired by your mockups. This Python version is built in Streamlit and keeps the same overall sections: control, missions, live camera, diagnostics, mission planning, and notifications.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        """
        <div class='hero-card' style='text-align:center;padding-top:1.1rem;padding-bottom:1.1rem;'>
            <div class='tiny'>Built with</div>
            <div style='font-size:1.5rem;font-weight:800;margin-top:0.25rem;'>Python + Streamlit</div>
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
                <div style='font-size:1.55rem;font-weight:800;margin-top:0.4rem;'>Quadruped Robo-1</div>
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

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    selected_mode = st.selectbox("Movement Mode", ["Trot", "Walk", "Standby", "Inspect"], index=["Trot", "Walk", "Standby", "Inspect"].index(st.session_state.mode), key="mode_select")
    if selected_mode != st.session_state.mode:
        st.session_state.mode = selected_mode
        st.session_state.last_action = f"Switched mode to {selected_mode}"
        add_notification("Mode Updated", f"Robot changed to {selected_mode} mode.", "lime")

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
        if st.button("Start Patrol", use_container_width=True):
            st.session_state.mission_status = "Running"
            st.session_state.selected_mission = "Patrol Route"
            st.session_state.last_action = "Patrol started"
            add_notification("Mission Started", "Patrol Route is now running.", "lime")
    with b4:
        if st.button("Return Home", use_container_width=True):
            st.session_state.mission_status = "Returning"
            st.session_state.last_action = "Robot returning to dock"
            add_notification("Return Home", "Robot is navigating back to dock.", "yellow")

    st.markdown("<div class='bottom-nav'><div class='tiny'>Quick controls active · Demo responsive</div></div>", unsafe_allow_html=True)
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
        if st.button(("● " if is_selected else "○ ") + mission, use_container_width=True, key=f"mission_{mission}"):
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
        if st.button("Next Waypoint", use_container_width=True):
            st.session_state.waypoint_idx = (st.session_state.waypoint_idx + 1) % len(waypoints)
            st.session_state.last_action = f"Moved to {waypoints[st.session_state.waypoint_idx]}"
            add_notification("Waypoint Updated", st.session_state.last_action + ".", "lime")
    with n2:
        if st.button("Pause Mission", use_container_width=True):
            st.session_state.mission_status = "Paused"
            st.session_state.last_action = "Mission paused"
            add_notification("Mission Paused", "Awaiting operator input.", "yellow")

    st.markdown("<div class='bottom-nav'><div class='tiny'>Preset routes · Waypoint navigation</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    top_l, top_r = st.columns([4, 1])
    with top_l:
        st.markdown("<div class='screen-title'>Live Camera</div>", unsafe_allow_html=True)
        st.markdown("<div class='screen-subtitle'>Warehouse feed • Recording ready</div>", unsafe_allow_html=True)
    with top_r:
        st.markdown("<div style='height:12px'></div><div class='badge'>Camera</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class='camera-box'>
            {"● REC  LIVE FEED SIMULATION" if st.session_state.recording else "LIVE FEED SIMULATION"}
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start Recording" if not st.session_state.recording else "Stop Recording", use_container_width=True):
            st.session_state.recording = not st.session_state.recording
            st.session_state.last_action = "Recording started" if st.session_state.recording else "Recording stopped"
            add_notification("Camera", st.session_state.last_action + ".", "red" if st.session_state.recording else "yellow")
    with c2:
        if st.button("Zoom View", use_container_width=True):
            st.session_state.zoom = not st.session_state.zoom
            st.session_state.last_action = "Zoom enabled" if st.session_state.zoom else "Zoom disabled"
            add_notification("Camera Zoom", st.session_state.last_action + ".", "lime")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'><span>Lens Mode</span><span class='neutral'>{'Zoomed' if st.session_state.zoom else 'Standard'}</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='metric'><span>Stabilization</span><span class='good'>Enabled</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='metric'><span>Signal</span><span class='good'>1080p · Low Latency</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='bottom-nav'><div class='tiny'>Vision system · Remote operator view</div></div>", unsafe_allow_html=True)
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
        if st.button("Deploy Mission", use_container_width=True):
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