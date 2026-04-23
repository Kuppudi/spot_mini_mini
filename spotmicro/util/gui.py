#!/usr/bin/env python

import pybullet as pb
import time
import numpy as np
import sys


class GUI:
    def __init__(self, quadruped):

        time.sleep(0.5)

        self.cyaw = 0
        self.cpitch = -7
        self.cdist = 0.66

        self.xId = pb.addUserDebugParameter("x", -0.10, 0.10, 0.)
        self.yId = pb.addUserDebugParameter("y", -0.10, 0.10, 0.)
        self.zId = pb.addUserDebugParameter("z", -0.055, 0.17, 0.)
        self.rollId = pb.addUserDebugParameter("roll", -np.pi / 4, np.pi / 4,
                                               0.)
        self.pitchId = pb.addUserDebugParameter("pitch", -np.pi / 4, np.pi / 4,
                                                0.)
        self.yawId = pb.addUserDebugParameter("yaw", -np.pi / 4, np.pi / 4, 0.)
        self.StepLengthID = pb.addUserDebugParameter("Step Length", -0.1, 0.1,
                                                     0.0)
        self.YawRateId = pb.addUserDebugParameter("Yaw Rate", -2.0, 2.0, 0.)
        self.LateralFractionId = pb.addUserDebugParameter(
            "Lateral Fraction", -np.pi / 2.0, np.pi / 2.0, 0.)
        self.StepVelocityId = pb.addUserDebugParameter("Step Velocity", 0.001,
                                                       3., 0.001)
        self.SwingPeriodId = pb.addUserDebugParameter("Swing Period", 0.1, 0.4,
                                                      0.2)

        self.ClearanceHeightId = pb.addUserDebugParameter(
            "Clearance Height", 0.0, 0.1, 0.045)
        self.PenetrationDepthId = pb.addUserDebugParameter(
            "Penetration Depth", 0.0, 0.05, 0.003)

        self.quadruped = quadruped

    def UserInput(self):

        quadruped_pos, _ = pb.getBasePositionAndOrientation(self.quadruped)
        pb.resetDebugVisualizerCamera(cameraDistance=self.cdist,
                                      cameraYaw=self.cyaw,
                                      cameraPitch=self.cpitch,
                                      cameraTargetPosition=quadruped_pos)
        keys = pb.getKeyboardEvents()
        # Keys to change camera
        if keys.get(ord('u')):  # U -> yaw right
            self.cyaw += 1
        if keys.get(ord('h')):  # H -> yaw left
            self.cyaw -= 1
        if keys.get(ord('j')):  # J -> pitch up
            self.cpitch += 1
        if keys.get(ord('k')):  # K -> pitch down
            self.cpitch -= 1
        if keys.get(ord('y')):  # Y -> zoom out
            self.cdist += 0.01
        if keys.get(ord('i')):  # I -> zoom in
            self.cdist -= 0.01
        if keys.get(27):  # ESC
            pb.disconnect()
            sys.exit()

        # Read Robot Transform from GUI
        pos = np.array([
            pb.readUserDebugParameter(self.xId),
            pb.readUserDebugParameter(self.yId),
            pb.readUserDebugParameter(self.zId)
        ])
        orn = np.array([
            pb.readUserDebugParameter(self.rollId),
            pb.readUserDebugParameter(self.pitchId),
            pb.readUserDebugParameter(self.yawId)
        ])
        StepLength = pb.readUserDebugParameter(self.StepLengthID)
        YawRate = pb.readUserDebugParameter(self.YawRateId)
        LateralFraction = pb.readUserDebugParameter(self.LateralFractionId)
        StepVelocity = pb.readUserDebugParameter(self.StepVelocityId)
        ClearanceHeight = pb.readUserDebugParameter(self.ClearanceHeightId)
        PenetrationDepth = pb.readUserDebugParameter(self.PenetrationDepthId)
        SwingPeriod = pb.readUserDebugParameter(self.SwingPeriodId)

        return pos, orn, StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, PenetrationDepth, SwingPeriod


class IndividualLegGUI:
    def __init__(self, quadruped):

        time.sleep(0.5)

        self.cyaw = 0
        self.cpitch = -7
        self.cdist = 0.66

        self.xId = pb.addUserDebugParameter("body_x", -0.10, 0.10, 0.0)
        self.yId = pb.addUserDebugParameter("body_y", -0.10, 0.10, 0.0)
        self.zId = pb.addUserDebugParameter("body_z", -0.055, 0.17, 0.0)
        self.rollId = pb.addUserDebugParameter("body_roll", -np.pi / 4, np.pi / 4, 0.0)
        self.pitchId = pb.addUserDebugParameter("body_pitch", -np.pi / 4, np.pi / 4, 0.0)
        self.yawId = pb.addUserDebugParameter("body_yaw", -np.pi / 4, np.pi / 4, 0.0)

        self.leg_slider_ids = {}
        for leg in ("FL", "FR", "BL", "BR"):
            self.leg_slider_ids[leg] = {
                "x": pb.addUserDebugParameter(f"{leg}_dx", -0.08, 0.08, 0.0),
                "y": pb.addUserDebugParameter(f"{leg}_dy", -0.08, 0.08, 0.0),
                "z": pb.addUserDebugParameter(f"{leg}_dz", -0.12, 0.08, 0.0),
            }

        self.quadruped = quadruped

    def _update_camera_and_keyboard(self):
        quadruped_pos, _ = pb.getBasePositionAndOrientation(self.quadruped)
        pb.resetDebugVisualizerCamera(
            cameraDistance=self.cdist,
            cameraYaw=self.cyaw,
            cameraPitch=self.cpitch,
            cameraTargetPosition=quadruped_pos,
        )
        keys = pb.getKeyboardEvents()
        if keys.get(ord('u')):
            self.cyaw += 1
        if keys.get(ord('h')):
            self.cyaw -= 1
        if keys.get(ord('j')):
            self.cpitch += 1
        if keys.get(ord('k')):
            self.cpitch -= 1
        if keys.get(ord('y')):
            self.cdist += 0.01
        if keys.get(ord('i')):
            self.cdist -= 0.01
        if keys.get(27):
            pb.disconnect()
            sys.exit()

    def UserInput(self):
        self._update_camera_and_keyboard()

        pos = np.array([
            pb.readUserDebugParameter(self.xId),
            pb.readUserDebugParameter(self.yId),
            pb.readUserDebugParameter(self.zId)
        ])
        orn = np.array([
            pb.readUserDebugParameter(self.rollId),
            pb.readUserDebugParameter(self.pitchId),
            pb.readUserDebugParameter(self.yawId)
        ])

        leg_offsets = {}
        for leg, slider_ids in self.leg_slider_ids.items():
            leg_offsets[leg] = np.array([
                pb.readUserDebugParameter(slider_ids["x"]),
                pb.readUserDebugParameter(slider_ids["y"]),
                pb.readUserDebugParameter(slider_ids["z"]),
            ])

        return pos, orn, leg_offsets
