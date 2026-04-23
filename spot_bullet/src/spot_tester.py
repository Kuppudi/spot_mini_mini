#spot_tester.py
#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import time
import os
import pybullet as pb

import argparse
try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: opencv-python\n"
        "Install OpenCV in your SpotMini environment, then rerun:\n"
        "  conda install -c conda-forge opencv -y\n"
        "or:\n"
        "  python -m pip install opencv-python==4.9.0.80"
    ) from exc

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from spotmicro.GymEnvs.spot_bezier_env import spotBezierEnv
try:
    from spotmicro.util.gui import GUI, IndividualLegGUI
except ImportError:
    from spotmicro.util.gui import GUI
    IndividualLegGUI = GUI
from spotmicro.Kinematics.SpotKinematics import SpotModel
from spotmicro.Kinematics.LieAlgebra import RPY
from spotmicro.GaitGenerator.Bezier import BezierGait
from spotmicro.GaitGenerator.gait_generator_4_phase import BezierGait4Phase
from spotmicro.spot_env_randomizer import SpotEnvRandomizer
from imu_controller import IMUController
from camera_sensor import SpotCamera

# TESTING
from spotmicro.OpenLoopSM.SpotOL import BezierStepper

# ARGUMENTS
descr = "Spot Mini Mini Environment Tester (No Joystick)."
parser = argparse.ArgumentParser(description=descr)
parser.add_argument("-hf",
                    "--HeightField",
                    help="Use HeightField",
                    action='store_true')
parser.add_argument("-r",
                    "--DebugRack",
                    help="Put Spot on an Elevated Rack",
                    action='store_true')
parser.add_argument("-p",
                    "--DebugPath",
                    help="Draw Spot's Foot Path",
                    action='store_true')
parser.add_argument("-ay",
                    "--AutoYaw",
                    help="Automatically Adjust Spot's Yaw",
                    action='store_true')
parser.add_argument("-ar",
                    "--AutoReset",
                    help="Automatically Reset Environment When Spot Falls",
                    action='store_true')
parser.add_argument("-dr",
                    "--DontRandomize",
                    help="Do NOT Randomize State and Environment.",
                    action='store_true')
parser.add_argument("-il",
                    "--IndividualLegs",
                    help="Control each leg with individual XYZ sliders instead of gait commands.",
                    action='store_true')
parser.add_argument("-g4",
                    "--FourPhaseGait",
                    help="Use a one-leg-at-a-time four-phase Bezier gait instead of the default trot scheduler.",
                    action='store_true')
ARGS = parser.parse_args()

# Smoothed command state
cmd_step = 0.0
cmd_lat = 0.0
cmd_yaw = 0.0
cmd_vel = 1.0


def redirect_forward_step_to_sideways(T_bf0, T_bf, side_sign=1.0, keep_z=True):
    """
    Redirect the Bezier-generated forward step into sideways motion.

    Works for dict-style foot transforms like:
        {'FL': ..., 'FR': ..., 'BL': ..., 'BR': ...}
    and also list-style storage.

    side_sign:
        +1.0 -> left
        -1.0 -> right

    keep_z:
        True  -> preserve stepping lift
        False -> flatten vertical stepping
    """
    T_new = copy.deepcopy(T_bf)

    # Case 1: dict of leg transforms
    if isinstance(T_bf, dict):
        for leg in T_bf.keys():
            x0 = T_bf0[leg][0, 3]
            y0 = T_bf0[leg][1, 3]
            z0 = T_bf0[leg][2, 3]

            x = T_bf[leg][0, 3]
            z = T_bf[leg][2, 3]

            dx = x - x0
            dz = z - z0

            # Remove forward/backward motion
            T_new[leg][0, 3] = x0

            # Redirect forward stride into lateral stride
            T_new[leg][1, 3] = y0 + side_sign * dx

            # Keep or flatten vertical lift
            if keep_z:
                T_new[leg][2, 3] = z0 + dz
            else:
                T_new[leg][2, 3] = z0

    # Case 2: list/tuple of transforms
    else:
        for i in range(len(T_bf)):
            x0 = T_bf0[i][0, 3]
            y0 = T_bf0[i][1, 3]
            z0 = T_bf0[i][2, 3]

            x = T_bf[i][0, 3]
            z = T_bf[i][2, 3]

            dx = x - x0
            dz = z - z0

            T_new[i][0, 3] = x0
            T_new[i][1, 3] = y0 + side_sign * dx

            if keep_z:
                T_new[i][2, 3] = z0 + dz
            else:
                T_new[i][2, 3] = z0

    return T_new


def get_keyboard_command(env):
    """
    Read keyboard input directly from the PyBullet window.
    Uses smoothing so commands do not jump instantly.
    Keeps turn-in-place responsive while allowing smoother walk+turn.
    """
    global cmd_step, cmd_lat, cmd_yaw, cmd_vel

    keys = env._pybullet_client.getKeyboardEvents()
    p = env._pybullet_client

    pos = np.array([0.0, 0.0, 0.0])
    orn = np.array([0.0, 0.0, 0.0])

    ClearanceHeight = 0.05
    PenetrationDepth = 0.01

    def key_down(ch):
        return ord(ch) in keys and (keys[ord(ch)] & p.KEY_IS_DOWN)

    target_step = 0.0
    target_lat = 0.0
    target_yaw = 0.0
    target_vel = 1.0
    SwingPeriod = 0.25

    # Forward / backward
    if key_down('w') or key_down('W'):
        target_step += 0.05
    if key_down('s') or key_down('S'):
        target_step -= 0.05

    # Turn left / right
    if key_down('q') or key_down('Q'):
        target_yaw += 0.95
    if key_down('e') or key_down('E'):
        target_yaw -= 0.95

    # Left / right strafe
    if key_down('a') or key_down('A'):
        target_lat += 0.9
    if key_down('d') or key_down('D'):
        target_lat -= 0.9

    # Faster / slower gait
    if key_down('z') or key_down('Z'):   # slower
        target_vel = 0.8
    if key_down('x') or key_down('X'):   # faster
        target_vel = 1.2

    # Turn in place still needs a small support step
    if target_yaw != 0.0 and target_step == 0.0:
        target_step = 0.035

    # Strafe still needs Bezier stride generation,
    # but we will remap that stride sideways later
    if target_lat != 0.0 and target_step == 0.0:
        target_step = 0.05

    # Only slightly reduce forward step during combined walk + turn
    if target_step != 0.0 and target_yaw != 0.0:
        target_step *= 0.75
        SwingPeriod = 0.42
    elif target_yaw != 0.0:
        SwingPeriod = 0.40
    elif target_lat != 0.0:
        SwingPeriod = 0.32
    else:
        SwingPeriod = 0.27

    # Smooth commands
    alpha_step = 0.10
    alpha_lat = 0.10
    alpha_yaw = 0.09
    alpha_vel = 0.07

    cmd_step += alpha_step * (target_step - cmd_step)
    cmd_lat += alpha_lat * (target_lat - cmd_lat)
    cmd_yaw += alpha_yaw * (target_yaw - cmd_yaw)
    cmd_vel += alpha_vel * (target_vel - cmd_vel)

    StepLength = cmd_step
    LateralFraction = cmd_lat
    YawRate = cmd_yaw
    StepVelocity = cmd_vel

    return pos, orn, StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, PenetrationDepth, SwingPeriod


def main():
    """ The main() function. """

    print("STARTING SPOT TEST ENV")
    print("Keyboard Controls:")
    print("  R/F -> forward/backward")
    print("  D/H -> strafe left/right")
    print("  E/T -> turn left/right")
    print("  Q/A -> faster/slower gait")
    if ARGS.IndividualLegs:
        print("Individual leg mode enabled:")
        print("  Use FL/FR/BL/BR dx/dy/dz sliders to move each foot independently.")
        print("  Use body_x/body_y/body_z and body_roll/body_pitch/body_yaw to move the body.")
    elif ARGS.FourPhaseGait:
        print("Four-phase gait mode enabled:")
        print("  Each leg swings in its own quarter of the gait cycle using Bezier foot arcs.")

    seed = 0
    max_timesteps = 4e6

    # Find abs path to this file
    my_path = os.path.abspath(os.path.dirname(__file__))
    results_path = os.path.join(my_path, "../results")
    models_path = os.path.join(my_path, "../models")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    if ARGS.DebugRack:
        on_rack = True
    else:
        on_rack = False

    if ARGS.DebugPath:
        draw_foot_path = True
    else:
        draw_foot_path = False

    if ARGS.HeightField:
        height_field = True
    else:
        height_field = False

    if ARGS.DontRandomize:
        env_randomizer = None
    else:
        env_randomizer = SpotEnvRandomizer()

    env = spotBezierEnv(render=True,
                        on_rack=on_rack,
                        height_field=height_field,
                        draw_foot_path=draw_foot_path,
                        env_randomizer=env_randomizer)

    robot_camera = SpotCamera(
        env,
        width=320,
        height=240,
        fov=75.0,
        near=0.02,
        far=5.0,
        camera_offset=(0.18, 0.0, 0.08),
        target_distance=1.0
    )

    imu_controller = IMUController()

    # Set seeds
    env.seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    print("STATE DIM: {}".format(state_dim))
    action_dim = env.action_space.shape[0]
    print("ACTION DIM: {}".format(action_dim))
    max_action = float(env.action_space.high[0])

    state = env.reset()
    imu_controller.reset()

    # Keeping this in case you want it later
    if ARGS.IndividualLegs:
        g_u_i = IndividualLegGUI(env.spot.quadruped)
    else:
        g_u_i = GUI(env.spot.quadruped)

    spot = SpotModel()
    T_bf0 = spot.WorldToFoot
    T_bf = copy.deepcopy(T_bf0)

    if ARGS.FourPhaseGait:
        bzg = BezierGait4Phase(dt=env._time_step)
    else:
        bzg = BezierGait(dt=env._time_step)
    bz_step = BezierStepper(dt=env._time_step, mode=0)

    # Use zero action so random action does not interfere with keyboard gait control
    action = np.zeros(action_dim)

    FL_phases = []
    FR_phases = []
    BL_phases = []
    BR_phases = []

    FL_Elbow = []

    yaw = 0.0

    print("STARTED SPOT TEST ENV")
    t = 0
    while t < int(max_timesteps):

        bz_step.ramp_up()

        yaw = env.return_yaw()
        P_yaw = 5.0

        kb_pos, kb_orn, StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, PenetrationDepth, SwingPeriod = get_keyboard_command(env)

        imu_pos, imu_orn = imu_controller.compute(state)

        # Read GUI slider values
        try:
            gui_data = g_u_i.UserInput()
        except pb.error:
            if ARGS.IndividualLegs:
                g_u_i = IndividualLegGUI(env.spot.quadruped)
            else:
                g_u_i = GUI(env.spot.quadruped)
            gui_data = g_u_i.UserInput()

        if ARGS.IndividualLegs:
            gui_pos = np.array(gui_data[0])
            gui_orn = np.array(gui_data[1])
            leg_offsets = gui_data[2]
            pos = gui_pos + imu_pos
            orn = gui_orn + imu_orn
            StepLength = 0.0
            LateralFraction = 0.0
            YawRate = 0.0
            StepVelocity = 0.0
            ClearanceHeight = 0.0
            PenetrationDepth = 0.0
            SwingPeriod = 0.2
        else:
            gui_pos = np.array(gui_data[0])
            gui_orn = np.array(gui_data[1])
            pos = kb_pos + imu_pos
            orn = kb_orn + imu_orn

            if StepLength > 0.01:
                pos[2] = gui_pos[2] + imu_pos[2]

        

        if t % 100 == 0:
            print("GUI DATA:", gui_data)
            
        if t % 200 == 0:
            print("roll={:.3f}, pitch={:.3f}, gx={:.3f}, gy={:.3f}, corr_roll={:.3f}, corr_pitch={:.3f}".format(
                state[0], state[1], state[2], state[3], orn[0], orn[1]))

        # Update Swing Period
        bzg.Tswing = SwingPeriod

        if ARGS.AutoYaw:
            YawRate += -yaw * P_yaw

        # Update stepper state
        bz_step.StepLength = StepLength
        bz_step.LateralFraction = LateralFraction
        bz_step.YawRate = YawRate
        bz_step.StepVelocity = StepVelocity

        contacts = state[-4:]

        FL_phases.append(env.spot.LegPhases[0])
        FR_phases.append(env.spot.LegPhases[1])
        BL_phases.append(env.spot.LegPhases[2])
        BR_phases.append(env.spot.LegPhases[3])

        # Get desired foot poses
        if ARGS.IndividualLegs:
            T_bf = copy.deepcopy(T_bf0)
            for leg, offset in leg_offsets.items():
                T_bf[leg][:3, 3] += offset
        else:
            T_bf = bzg.GenerateTrajectory(StepLength,
                                          LateralFraction,
                                          YawRate,
                                          StepVelocity,
                                          T_bf0,
                                          T_bf,
                                          ClearanceHeight,
                                          PenetrationDepth,
                                          contacts)

            # Redirect forward Bezier stepping into pure sideways stepping
            if abs(LateralFraction) > 0.01:
                side_sign = 1.0 if LateralFraction > 0.0 else -1.0
                T_bf = redirect_forward_step_to_sideways(
                    T_bf0,
                    T_bf,
                    side_sign=side_sign,
                    keep_z=True
                )

        joint_angles = spot.IK(orn, pos, T_bf)

        FL_Elbow.append(np.degrees(joint_angles[0][-1]))

        env.pass_joint_angles(joint_angles.reshape(-1))

        # Get external observations
        env.spot.GetExternalObservations(bzg, bz_step)

        # Step simulation
        state, reward, done, _ = env.step(action)

        frame = robot_camera.get_rgb_frame()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Spot Camera", frame_bgr)
        cv2.waitKey(1)

        if done:
            print("DONE")
            if ARGS.AutoReset:
                state = env.reset()
                imu_controller.reset()
                if ARGS.IndividualLegs:
                    g_u_i = IndividualLegGUI(env.spot.quadruped)
                else:
                    g_u_i = GUI(env.spot.quadruped)
                T_bf = copy.deepcopy(T_bf0)
                bzg.reset()
                bz_step = BezierStepper(dt=env._time_step, mode=0)
                continue
            print("Episode ended. Rerun with --AutoReset to restart automatically after a fall.")
            break

        t += 1

    env.close()
    cv2.destroyAllWindows()
    print(joint_angles)


if __name__ == '__main__':
    main()
