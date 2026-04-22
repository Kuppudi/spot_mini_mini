#!/usr/bin/env python3

import pybullet as pb
import time
import numpy as np
import copy
import sys

sys.path.append('../../')

from spotmicro.util import pybullet_data as spot_pybullet_data
from spotmicro.Kinematics.SpotKinematics import SpotModel
from spotmicro.GaitGenerator.Bezier import BezierGait
from spotmicro.OpenLoopSM.SpotOL import BezierStepper


TIME_STEP = 1.0 / 240.0


class Dog:
    def __init__(self, client, base_position, mode_name):
        self.client = client
        self.mode = mode_name

        self.robot = self.client.loadURDF(
            spot_pybullet_data.getDataPath() + "/assets/urdf/spot.urdf",
            base_position,
            [0, 0, 0, 1]
        )

        print(f"[{self.mode}] loaded robot id = {self.robot}")

        self.spot_model = SpotModel()
        self.T_bf0 = copy.deepcopy(self.spot_model.WorldToFoot)
        self.T_bf = copy.deepcopy(self.T_bf0)

        self.bzg = BezierGait(dt=TIME_STEP)
        self.bz_step = BezierStepper(dt=TIME_STEP, mode=0)

        self.t = 0

        self.motor_joint_names = [
            "motor_front_left_hip",
            "motor_front_left_upper_leg",
            "motor_front_left_lower_leg",
            "motor_front_right_hip",
            "motor_front_right_upper_leg",
            "motor_front_right_lower_leg",
            "motor_back_left_hip",
            "motor_back_left_upper_leg",
            "motor_back_left_lower_leg",
            "motor_back_right_hip",
            "motor_back_right_upper_leg",
            "motor_back_right_lower_leg",
        ]

        self.motor_joint_ids = self._find_motor_joints()

        self._disable_default_motor_behavior()
        self._print_joint_map()

    def _find_motor_joints(self):
        joint_ids = {}

        num_joints = self.client.getNumJoints(self.robot)
        for j in range(num_joints):
            info = self.client.getJointInfo(self.robot, j)
            joint_name = info[1].decode("utf-8")
            if joint_name in self.motor_joint_names:
                joint_ids[joint_name] = j

        missing = [name for name in self.motor_joint_names if name not in joint_ids]
        if missing:
            raise ValueError(
                f"[{self.mode}] Could not find these motor joints in URDF: {missing}"
            )

        return joint_ids

    def _disable_default_motor_behavior(self):
        for joint_name in self.motor_joint_names:
            joint_id = self.motor_joint_ids[joint_name]
            self.client.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=joint_id,
                controlMode=self.client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0
            )

    def _print_joint_map(self):
        print(f"[{self.mode}] motor joint map:")
        for name in self.motor_joint_names:
            print(f"    {name} -> {self.motor_joint_ids[name]}")

    def get_command(self):
        if self.mode == "baseline":
            return 0.035, 0.0, 0.0, 0.55
        elif self.mode == "smooth":
            return 0.028, 0.0, 0.0, 0.45
        #elif self.mode == "fast":
            #return 0.065, 0.0, 0.0, 1.10
        return 0.0, 0.0, 0.0, 0.80

    def step(self):
        self.bz_step.ramp_up()

        step_length, lateral, yaw, velocity = self.get_command()

        if abs(step_length) > 1e-4 and abs(yaw) > 1e-4:
            self.bzg.Tswing = 0.42
        elif abs(yaw) > 1e-4:
            self.bzg.Tswing = 0.40
        elif abs(lateral) > 1e-4:
            self.bzg.Tswing = 0.40
        else:
            self.bzg.Tswing = 0.28

        self.bz_step.StepLength = step_length
        self.bz_step.LateralFraction = lateral
        self.bz_step.YawRate = yaw
        self.bz_step.StepVelocity = velocity

        contacts = [1, 1, 1, 1]

        self.T_bf = self.bzg.GenerateTrajectory(
            step_length,
            lateral,
            yaw,
            velocity,
            self.T_bf0,
            self.T_bf,
            0.035,
            0.005,
            contacts
        )

        pos = np.array([0.0, 0.0, 0.0])
        orn = np.array([0.0, 0.0, 0.0])

        joint_angles = self.spot_model.IK(orn, pos, self.T_bf).reshape(-1)

        joint_targets = {
            "motor_front_left_hip": joint_angles[0],
            "motor_front_left_upper_leg": joint_angles[1],
            "motor_front_left_lower_leg": joint_angles[2],
            "motor_front_right_hip": joint_angles[3],
            "motor_front_right_upper_leg": joint_angles[4],
            "motor_front_right_lower_leg": joint_angles[5],
            "motor_back_left_hip": joint_angles[6],
            "motor_back_left_upper_leg": joint_angles[7],
            "motor_back_left_lower_leg": joint_angles[8],
            "motor_back_right_hip": joint_angles[9],
            "motor_back_right_upper_leg": joint_angles[10],
            "motor_back_right_lower_leg": joint_angles[11],
        }

        for joint_name, target in joint_targets.items():
            self.client.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.motor_joint_ids[joint_name],
                controlMode=self.client.POSITION_CONTROL,
                targetPosition=float(target),
                force=8
            )

        self.t += 1

        if self.t % 120 == 0:
            base_pos, base_orn = self.client.getBasePositionAndOrientation(self.robot)
            rpy = self.client.getEulerFromQuaternion(base_orn)
            print(f"[{self.mode}] t={self.t} base_pos={base_pos} rpy={rpy}")


def print_all_joints(robot_id):
    print("\n--- Joint Info ---")
    for j in range(pb.getNumJoints(robot_id)):
        info = pb.getJointInfo(robot_id, j)
        print(j, info[1].decode("utf-8"), "type =", info[2])
    print("------------------\n")


def main():
    print("Starting 2-dog demo...")

    pb.connect(pb.GUI)
    pb.setGravity(0, 0, -9.8)
    pb.setTimeStep(TIME_STEP)
    pb.setAdditionalSearchPath(spot_pybullet_data.getDataPath())

    pb.loadURDF(spot_pybullet_data.getDataPath() + "/plane.urdf")

    dogs = [
        Dog(pb, [0.0, -0.9, 0.25], "baseline"),
        Dog(pb, [0.0,  0.9, 0.25], "smooth"),
    ]

    print_all_joints(dogs[0].robot)

    pb.resetDebugVisualizerCamera(
        cameraDistance=4.2,
        cameraYaw=55,
        cameraPitch=-24,
        cameraTargetPosition=[1.0, 0.0, 0.15]
    )

    for _ in range(4000):
        for dog in dogs:
            dog.step()

        pb.stepSimulation()
        time.sleep(TIME_STEP)

    pb.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()