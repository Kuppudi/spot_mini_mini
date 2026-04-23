from __future__ import annotations

import atexit
import contextlib
import importlib.util
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
DASHBOARD_FILE = APP_DIR / "display_manual.py"
HOST = "127.0.0.1"
PORT = int(os.environ.get("SPOTMINI_MANUAL_APP_PORT", "8503"))


def require_app_dependencies() -> None:
    missing: list[str] = []
    if importlib.util.find_spec("streamlit") is None:
        missing.append("streamlit")
    if importlib.util.find_spec("webview") is None:
        missing.append("pywebview")

    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"Missing app dependency: {names}\n"
            "Install the manual app dependencies in this active environment with:\n"
            f"{sys.executable} -m pip install streamlit pywebview\n"
            "Then run this launcher again."
        )


def stop_streamlit(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return

    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)


def start_streamlit() -> subprocess.Popen[str]:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD_FILE),
        "--server.headless",
        "true",
        "--server.address",
        HOST,
        "--server.port",
        str(PORT),
        "--browser.gatherUsageStats",
        "false",
    ]

    return subprocess.Popen(
        command,
        cwd=str(APP_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=os.name != "nt",
    )


def wait_for_server(process: subprocess.Popen[str], timeout_seconds: int = 25) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise RuntimeError(
                "Streamlit exited before the manual app became available.\n"
                f"{output.strip()}"
            )

        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((HOST, PORT)) == 0:
                return

        time.sleep(0.25)

    raise TimeoutError(f"Timed out waiting for the manual app at http://{HOST}:{PORT}")


def open_native_window(url: str) -> None:
    import webview

    window = webview.create_window(
        "SpotMini Manual Control",
        url,
        width=1500,
        height=950,
        min_size=(1180, 760),
        resizable=True,
    )
    del window
    webview.start()


def main() -> int:
    try:
        require_app_dependencies()
    except Exception as exc:
        print(f"Failed to launch SpotMini Manual Control app: {exc}", file=sys.stderr)
        return 1

    process = start_streamlit()
    atexit.register(stop_streamlit, process)

    try:
        wait_for_server(process)
        open_native_window(f"http://{HOST}:{PORT}")
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Failed to launch SpotMini Manual Control app: {exc}", file=sys.stderr)
        return 1
    finally:
        stop_streamlit(process)


if __name__ == "__main__":
    raise SystemExit(main())
