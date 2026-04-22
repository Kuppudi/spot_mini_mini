from __future__ import annotations

import atexit
import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
DASHBOARD_FILE = APP_DIR / "display.py"
HOST = "127.0.0.1"
PORT = int(os.environ.get("SPOTMINI_DASHBOARD_PORT", "8502"))


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


def wait_for_server(process: subprocess.Popen[str], timeout_seconds: int = 20) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise RuntimeError(
                "Streamlit exited before the dashboard became available.\n"
                f"{output.strip()}"
            )

        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((HOST, PORT)) == 0:
                return

        time.sleep(0.25)

    raise TimeoutError(
        f"Timed out waiting for the dashboard at http://{HOST}:{PORT}"
    )


def open_native_window(url: str) -> bool:
    try:
        import webview
    except ImportError:
        return False

    webview.create_window(
        "SpotMini Dashboard",
        url,
        width=1500,
        height=950,
        min_size=(1100, 720),
    )
    webview.start()
    return True


def main() -> int:
    process = start_streamlit()
    atexit.register(stop_streamlit, process)

    try:
        wait_for_server(process)
        url = f"http://{HOST}:{PORT}"

        if not open_native_window(url):
            print("pywebview is not installed. Opening the dashboard in your default browser instead.")
            print("Install it with: pip install pywebview")
            webbrowser.open(url)
            print(f"Dashboard available at {url}. Press Ctrl+C to stop it.")
            while process.poll() is None:
                time.sleep(1)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Failed to launch dashboard: {exc}", file=sys.stderr)
        return 1
    finally:
        stop_streamlit(process)


if __name__ == "__main__":
    raise SystemExit(main())
