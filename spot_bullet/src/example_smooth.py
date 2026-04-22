#!/usr/bin/env python3

import sys
import time
import copy
import numpy as np

sys.path.append('../../')

from spotmicro.GymEnvs.spot_bezier_env import spotBezierEnv
from spotmicro.Kinematics.SpotKinematics import SpotModel
from spotmicro.GaitGenerator.Bezier import BezierGait
from spotmicro.OpenLoopSM.SpotOL import BezierStepper


def get_command_profile(mode_name, t):
    """
    Returns:
        step_length, lateral_fraction, yaw_rate, step_velocity
    """

    # -----------------------------
    # 1) BASELINE WALK
    # -----------------------------
    if mode_name == "baseline":
        if t < 150:
            return 0.0, 0.0, 0.0, 0.8
        elif t < 1400:
            return 0.055, 0.0, 0.0, 1.0
        elif t < 2000:
            return 0.045, 0.0, 0.45, 0.95
        elif t < 3400:
            return 0.055, 0.0, 0.0, 1.0
        else:
            return 0.0, 0.0, 0.0, 0.8

    # -----------------------------
    # 2) SLOW / SMOOTH WALK
    # -----------------------------
    elif mode_name == "smooth":
        if t < 150:
            return 0.0, 0.0, 0.0, 0.8
        elif t < 1400:
            return 0.045, 0.0, 0.0, 0.85
        elif t < 2000:
            return 0.040, 0.0, 0.30, 0.85
        elif t < 3400:
            return 0.045, 0.0, 0.0, 0.85
        else:
            return 0.0, 0.0, 0.0, 0.8

    # -----------------------------
    # 3) STRONGER / FASTER WALK
    # -----------------------------
    elif mode_name == "fast":
        if t < 150:
            return 0.0, 0.0, 0.0, 0.8
        elif t < 1400:
            return 0.065, 0.0, 0.0, 1.10
        elif t < 2000:
            return 0.055, 0.0, 0.55, 1.00
        elif t < 3400:
            return 0.065, 0.0, 0.0, 1.10
        else:
            return 0.0, 0.0, 0.0, 0.8

    return 0.0, 0.0, 0.0, 0.8


def set_swing_period(bzg, step_length, lateral_fraction, yaw_rate):
    if abs(step_length) > 1e-4 and abs(yaw_rate) > 1e-4:
        bzg.Tswing = 0.42
    elif abs(yaw_rate) > 1e-4:
        bzg.Tswing = 0.40
    elif abs(lateral_fraction) > 1e-4:
        bzg.Tswing = 0.40
    else:
        bzg.Tswing = 0.28


def run_one_demo(env, mode_name, total_steps=4000):
    print(f"\n==============================")
    print(f"Starting demo: {mode_name}")
    print(f"==============================")

    spot = SpotModel()
    T_bf0 = copy.deepcopy(spot.WorldToFoot)
    T_bf = copy.deepcopy(T_bf0)

    bzg = BezierGait(dt=env._time_step)
    bz_step = BezierStepper(dt=env._time_step, mode=0)

    state = env.reset()

    for t in range(total_steps):
        bz_step.ramp_up()

        step_length, lateral_fraction, yaw_rate, step_velocity = get_command_profile(mode_name, t)

        set_swing_period(bzg, step_length, lateral_fraction, yaw_rate)

        bz_step.StepLength = step_length
        bz_step.LateralFraction = lateral_fraction
        bz_step.YawRate = yaw_rate
        bz_step.StepVelocity = step_velocity

        contacts = state[-4:]

        T_bf = bzg.GenerateTrajectory(
            step_length,
            lateral_fraction,
            yaw_rate,
            step_velocity,
            T_bf0,
            T_bf,
            clearance_height=0.05,
            penetration_depth=0.01,
            contacts=contacts
        )

        pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        orn = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        joint_angles = spot.IK(orn, pos, T_bf)
        env.pass_joint_angles(joint_angles.reshape(-1))

        env.spot.GetExternalObservations(bzg, bz_step)

        state, reward, done, _ = env.step(np.zeros(4, dtype=np.float32))

        if t % 200 == 0:
            base_pos = env.spot.GetBasePosition()
            print(
                f"[{mode_name}] t={t:4d} | "
                f"base_x={base_pos[0]: .3f} | "
                f"step={step_length: .3f} | "
                f"yaw={yaw_rate: .3f} | "
                f"vel={step_velocity: .2f}"
            )

        if done:
            print(f"[{mode_name}] Robot fell or episode ended. Resetting...")
            state = env.reset()
            T_bf = copy.deepcopy(T_bf0)

        if t >= 2000:
            print(f"[{mode_name}] Reached t=4000.")
            break

        time.sleep(env._time_step)

    print(f"Finished demo: {mode_name}")


def main():
    print("Starting 3-run SpotMini comparison demo...")

    env = spotBezierEnv(
        render=True,
        on_rack=False,
        height_field=False,
        draw_foot_path=False,
        env_randomizer=None
    )

    demo_modes = ["baseline", "smooth", "fast"]

    for mode_name in demo_modes:
        run_one_demo(env, mode_name, total_steps=4000)

        print(f"Pausing before next demo...\n")
        time.sleep(1.5)

    env.close()
    print("All 3 demos finished.")


if __name__ == "__main__":
    main()