# SpotMini Local Dashboard

This folder contains the Streamlit dashboard template for the SpotMini project and a small launcher that can open it like a local app window.

## What to run

From the repository root:

```bash
python3 "spot_bullet/Sim Display/launch_dashboard.py"
```

## Native window mode

The launcher starts Streamlit on a local port and then:

- opens a native desktop window if `pywebview` is installed
- falls back to your default browser if `pywebview` is missing

Install the optional window package with:

```bash
pip install pywebview
```

## Manual PyBullet control

The regular dashboard remains in `display.py`. Manual control runs through a separate native app launcher:

```bash
python3 "spot_bullet/Sim Display/launch_manual_app.py"
```

This opens `display_manual.py` in a desktop window and does not fall back to a browser. If it reports that `streamlit` or `pywebview` is missing, install the app UI dependencies in the same `spotmini` environment:

```bash
python3 -m pip install streamlit pywebview
```

Once the app window opens, click `Manual Mode`, choose the gait and terrain buttons, then click `Launch Manual Mode` to open the native PyBullet GUI.

To test a trained model from the same app, click `Test PPO Model`, choose a run from the dropdown, choose `best` or `final`, then launch the worker.

For development only, you can still run the manual dashboard in a browser:

```bash
python3 -m streamlit run "spot_bullet/Sim Display/display_manual.py" --server.address 127.0.0.1 --server.port 8503
```

## Notes

- The dashboard app itself lives in `display.py`.
- The manual-control app launcher lives in `launch_manual_app.py`.
- The manual-control UI lives in `display_manual.py` and uses `manual_pybullet_worker.py`.
- Manual mode uses on-screen movement buttons; PPO test mode uses the selected trained run.
- The default local address is `http://127.0.0.1:8502`.
- You can change the port by setting `SPOTMINI_DASHBOARD_PORT`.
- You can change the manual app port by setting `SPOTMINI_MANUAL_APP_PORT`.
