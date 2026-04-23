# SpotMini Mini PyBullet Platform

This repository contains the SpotMini Mini simulation, local dashboard tools, manual PyBullet control app, and reinforcement-learning training/playback scripts used for the capstone project.

The current workflow is designed to run from the repository root:

```bash
cd /Users/Diviprakash/PSU/spot_mini_mini
conda activate spotmini
```

## Environment Setup

For the current macOS Apple Silicon setup:

```bash
conda create -n spotmini python=3.10 -y
conda activate spotmini
conda install -c conda-forge numpy scipy matplotlib opencv pybullet -y
python -m pip install --upgrade pip wheel
python -m pip install "setuptools<82"
python -m pip install -r spot_bullet/src/requirements.txt
python -m pip install streamlit pywebview
```

The `setuptools<82` pin avoids `pkg_resources` breakage in the older Gym/PyBullet environment. `streamlit` and `pywebview` are needed for the local app windows.

## Manual PyBullet App

Use this when you want an app-style control panel plus the native PyBullet GUI:

```bash
python3 "spot_bullet/Sim Display/launch_manual_app.py"
```

Inside the app:

- Click `Manual Mode` to control the robot with on-screen buttons.
- Click `Test PPO Model` to choose a trained run and model checkpoint.
- Use the gait buttons to switch between `trot` and `four_phase`.
- Use the terrain buttons to switch between `flat` and `rough`.
- Click `Launch Manual Mode` to start the PyBullet GUI.
- Use `Force Restart Manual Worker` if the PyBullet worker is still running with an older version.

Manual trot mode is currently a raw no-IMU baseline. The tuning values are intentionally fixed in the app while we tune gait behavior: forward step `0.050`, backward step `0.040`, turn assist `0.010`, turn rate `0.90`, strafe step `0.030`, strafe angle `0.80`, step velocity `0.54`, and curve height `0.040`.

## Main Dashboard

The original dashboard remains separate from the manual app:

```bash
python3 "spot_bullet/Sim Display/launch_dashboard.py"
```

This starts the Streamlit dashboard on a local port. If `pywebview` is installed, it opens in a desktop window; otherwise it can fall back to a browser.

## Train A PPO Model

Train a short smoke-test run:

```bash
python3 spot_bullet/src/spot_train_ml.py \
  --timesteps 2048 \
  --n-envs 1 \
  --skip-env-check \
  --preview-freq 0 \
  --run-name yaw_smoke_test \
  --enable-imu-yaw
```

Train a rough-terrain run from a previous model:

```bash
python3 spot_bullet/src/spot_train_ml.py \
  --training-preset quick_rough \
  --run-name rough_longwalk_yaw_transfer_v1 \
  --enable-imu-yaw
```

Training runs are saved under:

```text
spot_bullet/training runs/<run-name>
```

The code also recognizes the older underscore folder name:

```text
spot_bullet/training_runs
```

## Play A PPO Model

Run the newest available training run headlessly:

```bash
python3 spot_bullet/src/spot_play_ml.py --episodes 1
```

Run a specific model in the native PyBullet GUI:

```bash
python3 spot_bullet/src/spot_play_ml.py \
  --run-dir "spot_bullet/training runs/yaw_smoke_test" \
  --model final \
  --episodes 1 \
  --render \
  --bullet-gui
```

Use `--model best` to load `best_model/best_model.zip`, or `--model final` to load `models/ppo_spot_walk_final.zip`.

## Compare Models

Compare multiple trained models without opening the GUI:

```bash
python3 spot_bullet/src/spot_compare_ml.py \
  "spot_bullet/training runs/no_yaw_test" \
  "spot_bullet/training runs/with_yaw_test" \
  --model final \
  --episodes 5 \
  --sort-by reward
```

The comparison report includes reward, speed, distance, height error, fall rate, and timeout rate.

## Testing And Crash Evidence

The crash-test assignment report and commands are documented here:

```text
spot_bullet/CRASH_TEST_REPORT.md
```

Use those commands to capture screenshots before and after the implemented protections.

## Important Files

- `spot_bullet/Sim Display/display.py`: main dashboard UI.
- `spot_bullet/Sim Display/launch_dashboard.py`: starts the main dashboard.
- `spot_bullet/Sim Display/display_manual.py`: manual-control app UI.
- `spot_bullet/Sim Display/launch_manual_app.py`: starts the native manual-control app.
- `spot_bullet/Sim Display/manual_pybullet_worker.py`: PyBullet worker used by manual mode and PPO model testing.
- `spot_bullet/src/spot_train_ml.py`: PPO training entrypoint.
- `spot_bullet/src/spot_play_ml.py`: PPO playback entrypoint.
- `spot_bullet/src/spot_compare_ml.py`: PPO comparison entrypoint.
- `spot_bullet/src/spot_ml.py`: reinforcement-learning environment wrapper.
- `spotmicro/GaitGenerator/Bezier.py`: Bezier gait trajectory generator.

## Citing Spot Mini Mini

```bibtex
@software{spotminimini2020github,
  author = {Maurice Rahme and Ian Abraham and Matthew Elwin and Todd Murphey},
  title = {SpotMiniMini: Pybullet Gym Environment for Gait Modulation with Bezier Curves},
  url = {https://github.com/moribots/spot_mini_mini},
  version = {2.1.0},
  year = {2020},
}
```

## Credits

Original Spot Design and CAD files: [Spot Micro AI Community](https://spotmicroai.readthedocs.io/en/latest/)

Collaborator on `OpenQuadruped` design, including mechanical parts, custom PCB, and Teensy interface: [Adham Elarabawy](https://github.com/adham-elarabawy/OpenQuadruped)

OpenAI Gym and Heightfield Interface: [Minitaur Environment](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/minitaur.py)

Deprecated URDF for earlier development: [Rex Gym](https://github.com/nicrusso7/rex-gym)
