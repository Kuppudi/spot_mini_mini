# SpotMini Local Dashboard

This folder contains the Streamlit dashboard template for the SpotMini project and a small launcher that can open it like a local app window.

## What to run

From the repository root:

```bash
python3 "spot_mini_mini/spot_bullet/Sim Display/launch_dashboard.py"
```

## Native window mode

The launcher starts Streamlit on a local port and then:

- opens a native desktop window if `pywebview` is installed
- falls back to your default browser if `pywebview` is missing

Install the optional window package with:

```bash
pip install pywebview
```

## Notes

- The dashboard app itself lives in `display.py`.
- The default local address is `http://127.0.0.1:8502`.
- You can change the port by setting `SPOTMINI_DASHBOARD_PORT`.
