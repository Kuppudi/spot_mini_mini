# SpotMini Platform Crash-Test Report

This report documents deliberate attempts to crash or break the SpotMini simulation and reinforcement-learning platform. Each case includes the attack, the expected failure mode, the prevention implemented in the project, and what screenshot evidence should be captured.

## Platform Under Test

- Project: SpotMini Mini PyBullet reinforcement-learning platform
- Main scripts:
  - `spot_bullet/src/spot_train_ml.py`
  - `spot_bullet/src/spot_play_ml.py`
  - `spot_bullet/src/spot_compare_ml.py`
- Simulator: PyBullet
- ML stack: Stable-Baselines3 PPO, VecNormalize
- Primary GUI: native PyBullet GUI on macOS when `--render --bullet-gui` is used

## Case 1: Platform / GUI Dependency Attack

### Attack Goal

Break visualization by running the simulator on macOS where native PyBullet GUI behavior can differ from headless or OpenCV preview behavior.

### Attack Commands

Run playback without visualization:

```bash
python3 spot_bullet/src/spot_play_ml.py \
  --run-dir "spot_bullet/training runs/dual_imu_longwalk_yaw_lowbody_v1" \
  --model best \
  --episodes 1
```

Then run native PyBullet GUI:

```bash
python3 spot_bullet/src/spot_play_ml.py \
  --run-dir "spot_bullet/training runs/dual_imu_longwalk_yaw_lowbody_v1" \
  --model best \
  --episodes 1 \
  --render --bullet-gui
```

Optional fallback test:

```bash
python3 spot_bullet/src/spot_play_ml.py \
  --run-dir "spot_bullet/training runs/dual_imu_longwalk_yaw_lowbody_v1" \
  --model best \
  --episodes 1 \
  --render
```

### Expected Failure Before Prevention

Users could think the platform was frozen because the simulation runs headlessly unless `--render` is passed. On some macOS setups, native GUI launch can also be fragile.

### Prevention Implemented

- `spot_play_ml.py` explicitly supports native PyBullet GUI with `--render --bullet-gui`.
- On macOS, plain `--render` uses a safer live-preview fallback instead of forcing native GUI.
- The script prints the selected render mode, such as `Render mode: native PyBullet GUI` or `Render mode: live preview window (macOS fallback)`.

### Result

The simulator runs headlessly when intended, and opens the native PyBullet GUI when `--render --bullet-gui` is used. This prevents false failure reports and gives a reliable visualization path.

### Screenshot Evidence

- Screenshot 1: Terminal showing headless playback completing an episode.
- Screenshot 2: Terminal showing `Render mode: native PyBullet GUI`.
- Screenshot 3: Native PyBullet GUI window with the SpotMini robot visible.

## Case 2: Model / Environment Mismatch Attack

### Attack Goal

Crash playback by forcing an old checkpoint to load with a different observation space. This is easy to trigger when a model trained without yaw-aware IMU features is played with yaw features forced on.

### Attack Command

```bash
python3 spot_bullet/src/spot_play_ml.py \
  --run-dir "spot_bullet/training runs/dual_imu_stable_forward_v2" \
  --model best \
  --episodes 1 \
  --enable-imu-yaw
```

### Expected Failure Before Prevention

Stable-Baselines3 can throw a raw PyTorch stack trace similar to:

```text
RuntimeError: size mismatch for mlp_extractor.policy_net.0.weight
```

This is confusing for users and looks like a platform crash.

### Prevention Implemented

- `spot_play_ml.py` now loads PPO models through `load_ppo_model(...)`.
- If a size mismatch is detected, the platform exits with a clear explanation:
  - The checkpoint and environment feature set do not match.
  - Remove overrides such as `--enable-imu-yaw`, or retrain with the same flags.
- The raw stack trace is avoided for this known crash case.

### Repeat Attack After Prevention

Run the same command again. The platform should stop safely with a readable error message instead of a long stack trace.

### Screenshot Evidence

- Screenshot 1: Terminal command that forces `--enable-imu-yaw` on an older model.
- Screenshot 2: Friendly error explaining model/environment observation mismatch.
- Screenshot 3: Successful playback after removing the bad override:

```bash
python3 spot_bullet/src/spot_play_ml.py \
  --run-dir "spot_bullet/training runs/dual_imu_stable_forward_v2" \
  --model best \
  --episodes 1
```

## Case 3: Batch Evaluation / Bad Candidate Attack

### Attack Goal

Crash a multi-model comparison by mixing valid runs with a missing or incompatible run. This tests whether one bad candidate can bring down the whole evaluation platform.

### Attack Command

```bash
python3 spot_bullet/src/spot_compare_ml.py \
  "spot_bullet/training runs/with_yaw_test" \
  "spot_bullet/training runs/does_not_exist" \
  "spot_bullet/training runs/no_yaw_test" \
  --model final \
  --episodes 3 \
  --sort-by reward
```

### Expected Failure Before Prevention

The comparison script could abort when it reached the missing run, preventing all remaining valid runs from being scored.

### Prevention Implemented

- `spot_compare_ml.py` now catches per-run failures.
- A failed candidate is reported as a `failed` row instead of aborting the entire comparison.
- Valid candidates continue to run and are still included in the final table.

### Repeat Attack After Prevention

Run the same command again. The output table should include:

- `with_yaw_test` scored normally
- `no_yaw_test` scored normally
- `does_not_exist` marked as `failed`
- A one-line reason explaining the failure

### Screenshot Evidence

- Screenshot 1: Terminal command with one intentionally bad run directory.
- Screenshot 2: Final comparison table showing successful rows and one `failed` row.
- Screenshot 3: A clean comparison with only valid runs:

```bash
python3 spot_bullet/src/spot_compare_ml.py \
  "spot_bullet/training runs/with_yaw_test" \
  "spot_bullet/training runs/no_yaw_test" \
  --model final \
  --episodes 3 \
  --sort-by reward
```

## Case 4: Training Data Integrity Attack

### Attack Goal

Accidentally overwrite an existing training run by reusing the same `--run-name`. This can destroy reproducibility and make experiment results unreliable.

### Attack Command

Run a training job with a run name that already exists:

```bash
python3 spot_bullet/src/spot_train_ml.py \
  --walk-variant dual_imu_stable \
  --terrain-profile flat \
  --gait-mode four_phase \
  --enable-imu-yaw \
  --run-name with_yaw_test \
  --timesteps 2048 \
  --n-envs 1 \
  --skip-env-check \
  --preview-freq 0
```

### Expected Failure Before Prevention

The script could write into an existing run folder, replacing configuration files or adding checkpoints to an old experiment.

### Prevention Implemented

- `spot_train_ml.py` now checks whether the target run folder already exists and is non-empty.
- By default, it refuses to write into that folder.
- If overwriting is intentional, the user must pass `--allow-run-overwrite`.

### Repeat Attack After Prevention

Run the same command again. The platform should stop with:

```text
Training run already exists and is not empty
```

Then run with a new name:

```bash
python3 spot_bullet/src/spot_train_ml.py \
  --walk-variant dual_imu_stable \
  --terrain-profile flat \
  --gait-mode four_phase \
  --enable-imu-yaw \
  --run-name with_yaw_test_v2 \
  --timesteps 2048 \
  --n-envs 1 \
  --skip-env-check \
  --preview-freq 0
```

### Screenshot Evidence

- Screenshot 1: Attempt to train into an existing run folder.
- Screenshot 2: Safe refusal message.
- Screenshot 3: Successful training with a new run name.

## Summary Of Preventive Measures

- Native PyBullet GUI is explicitly selected with `--render --bullet-gui`.
- macOS fallback rendering avoids GUI dependency crashes.
- Model/environment observation mismatches now produce clear, actionable errors.
- Batch comparison continues even when one model candidate is bad.
- Existing training runs are protected from accidental overwrite.
- VecNormalize compatibility shims are used in playback and comparison so old saved normalization files still load.

## Final Notes

The most important screenshots for submission are the terminal outputs showing:

- The attack command.
- The safe failure or graceful recovery.
- The repeated attack after prevention.
- The successful normal operation after using the correct command or settings.
