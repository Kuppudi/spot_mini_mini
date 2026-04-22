import numpy as np

class IMUController:
    """
    Simple IMU stabilization controller for the Spot robot.

    Uses roll/pitch and gyro data to generate body orientation corrections.
    """

    def __init__(self):
        # smoothed corrections
        self.x_corr = 0.0
        self.y_corr = 0.0
        self.roll_corr = 0.0
        self.pitch_corr = 0.0

        # control gains
        self.Kp_roll = 0.35
        self.Kp_pitch = 0.35
        self.Kd_roll = 0.02
        self.Kd_pitch = 0.02
        self.Kp_pos_roll = 0.0
        self.Kp_pos_pitch = 0.0
        self.Kd_pos_roll = 0.0
        self.Kd_pos_pitch = 0.0

        # smoothing
        self.alpha = 0.12
        self.alpha_pos = 0.10

        # safety clamp
        self.max_corr = 0.20
        self.max_pos_corr = 0.015

        # ignore tiny noise near zero
        self.deadband = 0.01

    def reset(self):
        """Reset smoothed controller state."""
        self.x_corr = 0.0
        self.y_corr = 0.0
        self.roll_corr = 0.0
        self.pitch_corr = 0.0

    def compute(self, state):
        """
        Generate stabilized body pose from IMU values in state.

        Expected state format:
        0 roll
        1 pitch
        2 gx
        3 gy
        4 gz
        5 ax
        6 ay
        7 az
        """
        if len(state) < 8:
            raise ValueError("IMUController expected state with at least 8 values")

        roll = state[0]
        pitch = state[1]
        gx = state[2]
        gy = state[3]

        # deadband to reduce tiny jitter corrections
        if abs(roll) < self.deadband:
            roll = 0.0
        if abs(pitch) < self.deadband:
            pitch = 0.0

        target_roll = -(self.Kp_roll * roll + self.Kd_roll * gx)
        target_pitch = -(self.Kp_pitch * pitch + self.Kd_pitch * gy)
        target_x = -(self.Kp_pos_pitch * pitch + self.Kd_pos_pitch * gy)
        target_y = -(self.Kp_pos_roll * roll + self.Kd_pos_roll * gx)

        target_roll = np.clip(target_roll, -self.max_corr, self.max_corr)
        target_pitch = np.clip(target_pitch, -self.max_corr, self.max_corr)
        target_x = np.clip(target_x, -self.max_pos_corr, self.max_pos_corr)
        target_y = np.clip(target_y, -self.max_pos_corr, self.max_pos_corr)

        # smoothing
        self.x_corr += self.alpha_pos * (target_x - self.x_corr)
        self.y_corr += self.alpha_pos * (target_y - self.y_corr)
        self.roll_corr += self.alpha * (target_roll - self.roll_corr)
        self.pitch_corr += self.alpha * (target_pitch - self.pitch_corr)

        pos = np.array([self.x_corr, self.y_corr, 0.0])

        # yaw correction intentionally not used yet
        orn = np.array([self.roll_corr, self.pitch_corr, 0.0])

        return pos, orn
