#!/usr/bin/env python3

import os
import sys
import time
import copy
import numpy as np

sys.path.append('../../')

from spotmicro.GymEnvs.spot_bezier_env import spotBezierEnv
from spotmicro.Kinematics.SpotKinematics import SpotModel
from spotmicro.GaitGenerator.Bezier import BezierGait
from spotmicro.OpenLoopSM.SpotOL import BezierStepper


def main():
    print("Starting baseline SpotMini visual demo...")

    env = spotBezierEnv(
        render=True,
        on_rack=False,
        height_field=False,
        draw_foot_path=False,
        env_randomizer=None
    )

    spot = SpotModel()
    T_bf0 = copy.deepcopy(spot.WorldToFoot)
    T_bf = copy.deepcopy(T_bf0)

    bzg = BezierGait(dt=env._time_step)
    bz_step = BezierStepper(dt=env._time_step, mode=0)

    state = env.reset()

    max_steps = 4500

    for t in range(max_steps):
        bz_step.ramp_up()

        # -------- baseline command profile --------
        # First: stand still briefly
        if t < 150:
            step_length = 0.0
            lateral_fraction = 0.0
            yaw_rate = 0.0
            step_velocity = 0.8

        # Then: walk forward
        elif t < 1400:
            step_length = 0.055
            lateral_fraction = 0.0
            yaw_rate = 0.0
            step_velocity = 1.0

        # Then: slight turn
        elif t < 2000:
            step_length = 0.045
            lateral_fraction = 0.0
            yaw_rate = 0.45
            step_velocity = 0.95

        # Then: walk forward again
        elif t < 3400:
            step_length = 0.055
            lateral_fraction = 0.0
            yaw_rate = 0.0
            step_velocity = 1.0

        # Then: stop
        else:
            step_length = 0.0
            lateral_fraction = 0.0
            yaw_rate = 0.0
            step_velocity = 0.8

        # Swing timing similar to your tester / ML env logic
        if abs(step_length) > 1e-4 and abs(yaw_rate) > 1e-4:
            bzg.Tswing = 0.42
        elif abs(yaw_rate) > 1e-4:
            bzg.Tswing = 0.40
        elif abs(lateral_fraction) > 1e-4:
            bzg.Tswing = 0.40
        else:
            bzg.Tswing = 0.28

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
                f"t={t:4d} | base_x={base_pos[0]: .3f} | "
                f"step={step_length: .3f} yaw={yaw_rate: .3f} vel={step_velocity: .2f}"
            )

        if done:
            print("Robot fell or episode ended. Resetting...")
            state = env.reset()
            T_bf = copy.deepcopy(T_bf0)

        if t >= 2000:
            print("Reached t=4000, closing demo.")
            break

        time.sleep(env._time_step)

    env.close()
    print("Baseline demo finished.")


if __name__ == "__main__":
    main()