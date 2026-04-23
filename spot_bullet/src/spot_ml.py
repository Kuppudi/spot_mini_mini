#spot_ml.py

import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from camera_sensor import SpotCamera
from spotmicro.GymEnvs.spot_bezier_env import spotBezierEnv
from spotmicro.Kinematics.SpotKinematics import SpotModel
from spotmicro.GaitGenerator.Bezier import BezierGait
from spotmicro.GaitGenerator.gait_generator_4_phase import BezierGait4Phase
from spotmicro.OpenLoopSM.SpotOL import BezierStepper
from spotmicro.spot_env_randomizer import SpotEnvRandomizer
from imu_controller import IMUController


class SpotMLWalkEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    IMU_ORIENTATION_FEATURE_DIM = 4

    DEFAULT_WALK_TARGET_FORWARD_SPEED = 0.06
    DEFAULT_WALK_BODY_HEIGHT_OFFSET = 0.018
    DEFAULT_WALK_BODY_PITCH_BIAS_DEG = -0.4
    DEFAULT_WALK_AUTO_YAW_GAIN = 0.55
    DEFAULT_WALK_GEOMETRY_PROFILE = "balanced"
    DEFAULT_ROUGH_WALK_TARGET_FORWARD_SPEED = 0.045
    DEFAULT_ROUGH_WALK_BODY_HEIGHT_OFFSET = 0.014
    DEFAULT_ROUGH_WALK_BODY_PITCH_BIAS_DEG = -0.8
    DEFAULT_ROUGH_WALK_AUTO_YAW_GAIN = 0.65
    DEFAULT_ROUGH_WALK_GEOMETRY_PROFILE = "contact_stable"
    DEFAULT_FRONT_SWING_CLEARANCE_SCALE = 1.0
    DEFAULT_REAR_SWING_CLEARANCE_SCALE = 1.0

    def __init__(self,
                 render=False,
                 on_rack=False,
                 height_field=False,
                 terrain_profile=None,
                 terrain_randomization=None,
                 draw_foot_path=False,
                 env_randomizer=None,
                 target_forward_speed=0.35,
                 gait_mode="trot",
                 max_episode_steps=2000,
                 body_height_offset=None,
                 body_roll_bias_deg=None,
                 body_pitch_bias_deg=None,
                 auto_yaw_gain=None,
                 gait_geometry_profile=None,
                 enable_imu_yaw=False,
                 allow_turning_commands=False,
                 stabilization_profile=None,
                 front_swing_clearance_scale=None,
                 rear_swing_clearance_scale=None,
                 gui_safe_mode=None,
                 follow_gui_camera=None):

        super().__init__()

        if terrain_profile is None:
            terrain_profile = "rough" if height_field else "flat"
        if terrain_profile not in ("flat", "rough"):
            raise ValueError(
                f"Unsupported terrain_profile={terrain_profile!r}. "
                "Expected 'flat' or 'rough'."
            )

        self.terrain_profile = terrain_profile
        self.height_field = terrain_profile == "rough"
        self.terrain_randomization = (
            self.height_field
            if terrain_randomization is None
            else bool(terrain_randomization)
        )
        self.requested_env_randomizer = env_randomizer

        self.base_env = spotBezierEnv(
            render=render,
            on_rack=on_rack,
            height_field=self.height_field,
            reflection=not self.height_field,
            draw_foot_path=draw_foot_path,
            env_randomizer=self._make_env_randomizer(),
            gui_safe_mode=gui_safe_mode,
            follow_gui_camera=follow_gui_camera,
        )

        self.gait_mode = gait_mode
        self.target_forward_speed = target_forward_speed
        self.max_episode_steps = int(max_episode_steps)
        self.requested_body_height_offset = body_height_offset
        self.requested_body_roll_bias_deg = body_roll_bias_deg
        self.requested_body_pitch_bias_deg = body_pitch_bias_deg
        self.requested_auto_yaw_gain = auto_yaw_gain
        self.requested_gait_geometry_profile = gait_geometry_profile
        self.enable_imu_yaw = bool(enable_imu_yaw)
        self.allow_turning_commands = bool(allow_turning_commands)
        self.stabilization_profile = stabilization_profile
        self.front_swing_clearance_scale = (
            self.DEFAULT_FRONT_SWING_CLEARANCE_SCALE
            if front_swing_clearance_scale is None
            else float(front_swing_clearance_scale)
        )
        self.rear_swing_clearance_scale = (
            self.DEFAULT_REAR_SWING_CLEARANCE_SCALE
            if rear_swing_clearance_scale is None
            else float(rear_swing_clearance_scale)
        )
        self.imu_orientation_feature_dim = (
            self.IMU_ORIENTATION_FEATURE_DIM if self.enable_imu_yaw else 0
        )

        self.spot_model = SpotModel()
        self.T_bf0 = copy.deepcopy(self.spot_model.WorldToFoot)
        self.T_bf = copy.deepcopy(self.T_bf0)
        self.bzg = self._make_gait_generator()
        self.bz_step = BezierStepper(dt=self.base_env._time_step, mode=0)
        self.imu_controller = IMUController()

        self.cmd_step = 0.0
        self.cmd_lat = 0.0
        self.cmd_yaw = 0.0
        self.cmd_vel = 0.8
        self.clearance_height = 0.035
        self.penetration_depth = 0.008
        self.nominal_base_height = None
        self.fall_penalty = 25.0
        self.body_height_offset = 0.0
        self.body_roll_bias_deg = 0.0
        self.body_pitch_bias_deg = 0.0
        self.auto_yaw_gain = 0.0
        self.last_stabilized_yaw_rate = 0.0
        self.body_position = np.zeros(3, dtype=np.float32)
        self.body_orientation_bias = np.zeros(3, dtype=np.float32)
        self.manual_curve_height = None
        self.manual_curve_speed_scale = 1.0
        self.manual_forward_left_step_gain = 1.0
        self.manual_forward_right_step_gain = 1.0
        self.gait_geometry_profile = "default"
        self.footprint_scale_x = 1.0
        self.footprint_scale_y = 1.0
        self.contact_offsets = {
            "FL": np.zeros(3, dtype=np.float32),
            "FR": np.zeros(3, dtype=np.float32),
            "BL": np.zeros(3, dtype=np.float32),
            "BR": np.zeros(3, dtype=np.float32),
        }

        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_base_pos = np.zeros(3, dtype=np.float32)
        self.episode_start_pos = np.zeros(3, dtype=np.float32)
        self.episode_step_count = 0
        self.last_forward_speed = 0.0

        self.action_space = self._make_action_space()

        obs_dim = 16 + 4 + 4 + 1 + 8 + self.imu_orientation_feature_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        self._configure_gait_mode()

    def _make_env_randomizer(self):
        if self.requested_env_randomizer is not None:
            return self.requested_env_randomizer
        if self.terrain_randomization:
            return SpotEnvRandomizer()
        return None

    def seed(self, seed=None):
        self.base_env.seed(seed)
        return [seed]

    def _make_gait_generator(self):
        if self.gait_mode == "trot":
            return BezierGait(dt=self.base_env._time_step)
        if self.gait_mode == "four_phase":
            if self.height_field:
                return BezierGait4Phase(
                    dt=self.base_env._time_step,
                    Tswing=0.18,
                    max_swing_ratio=0.17,
                    inter_swing_stance_ratio=0.10,
                    preload_ratio=0.16,
                    preload_step_scale=0.18,
                    preload_clearance_scale=0.05,
                )
            return BezierGait4Phase(
                dt=self.base_env._time_step,
                Tswing=0.16,
                max_swing_ratio=0.18,
                inter_swing_stance_ratio=0.08,
                preload_ratio=0.18,
                preload_step_scale=0.26,
                preload_clearance_scale=0.08,
            )
        raise ValueError(
            f"Unsupported gait_mode={self.gait_mode!r}. Expected 'trot' or 'four_phase'."
        )

    def _make_action_space(self):
        if self.gait_mode == "four_phase":
            low = np.array([0.0, -0.05, -0.05, 0.15], dtype=np.float32)
            if self.height_field:
                high = np.array([0.055, 0.05, 0.05, 0.42], dtype=np.float32)
            else:
                high = np.array([0.06, 0.05, 0.05, 0.50], dtype=np.float32)
        else:
            lateral_limit = np.pi / 2.0 if self.allow_turning_commands else 1.0
            low = np.array([-0.08, -lateral_limit, -1.2, 0.4], dtype=np.float32)
            high = np.array([0.08, lateral_limit, 1.2, 1.4], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _get_four_phase_geometry_config(self, profile_name):
        profiles = {
            "default": {
                "footprint_scale_x": 1.00,
                "footprint_scale_y": 1.00,
                "contact_offsets": {
                    "FL": np.zeros(3, dtype=np.float32),
                    "FR": np.zeros(3, dtype=np.float32),
                    "BL": np.zeros(3, dtype=np.float32),
                    "BR": np.zeros(3, dtype=np.float32),
                },
            },
            "contact_stable": {
                "footprint_scale_x": 1.00,
                "footprint_scale_y": 1.00,
                "contact_offsets": {
                    "FL": np.array([0.010, 0.014, 0.0], dtype=np.float32),
                    "FR": np.array([0.010, -0.014, 0.0], dtype=np.float32),
                    "BL": np.array([-0.010, 0.014, 0.0], dtype=np.float32),
                    "BR": np.array([-0.010, -0.014, 0.0], dtype=np.float32),
                },
            },
            "balanced": {
                "footprint_scale_x": 1.04,
                "footprint_scale_y": 1.05,
                "contact_offsets": {
                    "FL": np.array([0.010, 0.012, 0.0], dtype=np.float32),
                    "FR": np.array([0.010, -0.012, 0.0], dtype=np.float32),
                    "BL": np.array([-0.010, 0.012, 0.0], dtype=np.float32),
                    "BR": np.array([-0.010, -0.012, 0.0], dtype=np.float32),
                },
            },
            "anti_right_skew": {
                # Keep the balanced crawl footprint, but bias support
                # slightly left to counter the persistent mild right drift.
                "footprint_scale_x": 1.04,
                "footprint_scale_y": 1.05,
                "contact_offsets": {
                    "FL": np.array([0.010, 0.016, 0.0], dtype=np.float32),
                    "FR": np.array([0.010, -0.009, 0.0], dtype=np.float32),
                    "BL": np.array([-0.010, 0.016, 0.0], dtype=np.float32),
                    "BR": np.array([-0.010, -0.009, 0.0], dtype=np.float32),
                },
            },
            "anti_rock": {
                # Increase both wheelbase and stance width so each single-leg
                # swing leaves a larger support polygon under the torso.
                "footprint_scale_x": 1.08,
                "footprint_scale_y": 1.10,
                "contact_offsets": {
                    "FL": np.array([0.014, 0.016, 0.0], dtype=np.float32),
                    "FR": np.array([0.014, -0.016, 0.0], dtype=np.float32),
                    "BL": np.array([-0.014, 0.016, 0.0], dtype=np.float32),
                    "BR": np.array([-0.014, -0.016, 0.0], dtype=np.float32),
                },
            },
            "wide_stable": {
                "footprint_scale_x": 1.02,
                "footprint_scale_y": 1.10,
                "contact_offsets": {
                    "FL": np.array([0.008, 0.018, 0.0], dtype=np.float32),
                    "FR": np.array([0.008, -0.018, 0.0], dtype=np.float32),
                    "BL": np.array([-0.008, 0.018, 0.0], dtype=np.float32),
                    "BR": np.array([-0.008, -0.018, 0.0], dtype=np.float32),
                },
            },
            "distance": {
                "footprint_scale_x": 1.08,
                "footprint_scale_y": 1.03,
                "contact_offsets": {
                    "FL": np.array([0.014, 0.010, 0.0], dtype=np.float32),
                    "FR": np.array([0.014, -0.010, 0.0], dtype=np.float32),
                    "BL": np.array([-0.014, 0.010, 0.0], dtype=np.float32),
                    "BR": np.array([-0.014, -0.010, 0.0], dtype=np.float32),
                },
            },
        }
        if profile_name not in profiles:
            raise ValueError(
                f"Unsupported gait_geometry_profile={profile_name!r}. "
                "Expected one of: default, contact_stable, balanced, anti_right_skew, anti_rock, wide_stable, distance."
            )
        return profiles[profile_name]

    def _configure_gait_mode(self):
        if self.gait_mode == "four_phase":
            if self.height_field:
                self.cmd_vel = 0.18
                self.clearance_height = 0.040
                self.penetration_depth = 0.007
                self.target_forward_speed = min(
                    self.target_forward_speed,
                    self.DEFAULT_ROUGH_WALK_TARGET_FORWARD_SPEED,
                )
                self.fall_penalty = 45.0
                self.body_height_offset = (
                    self.DEFAULT_ROUGH_WALK_BODY_HEIGHT_OFFSET
                    if self.requested_body_height_offset is None
                    else float(self.requested_body_height_offset)
                )
                self.body_roll_bias_deg = (
                    0.0 if self.requested_body_roll_bias_deg is None
                    else float(self.requested_body_roll_bias_deg)
                )
                self.body_pitch_bias_deg = (
                    self.DEFAULT_ROUGH_WALK_BODY_PITCH_BIAS_DEG
                    if self.requested_body_pitch_bias_deg is None
                    else float(self.requested_body_pitch_bias_deg)
                )
                self.auto_yaw_gain = (
                    self.DEFAULT_ROUGH_WALK_AUTO_YAW_GAIN
                    if self.requested_auto_yaw_gain is None
                    else float(self.requested_auto_yaw_gain)
                )
                self.imu_controller.Kp_roll = 0.86
                self.imu_controller.Kp_pitch = 0.72
                self.imu_controller.Kd_roll = 0.10
                self.imu_controller.Kd_pitch = 0.08
                self.imu_controller.alpha = 0.26
                self.imu_controller.Kp_pos_roll = 0.010
                self.imu_controller.Kp_pos_pitch = 0.008
                self.imu_controller.Kd_pos_roll = 0.003
                self.imu_controller.Kd_pos_pitch = 0.003
                self.imu_controller.alpha_pos = 0.12
                self.imu_controller.max_corr = 0.25
                self.imu_controller.max_pos_corr = 0.014
                self.imu_controller.deadband = 0.005
            else:
                self.cmd_vel = 0.20
                self.clearance_height = 0.025
                self.penetration_depth = 0.005
                self.target_forward_speed = min(
                    self.target_forward_speed,
                    self.DEFAULT_WALK_TARGET_FORWARD_SPEED,
                )
                self.fall_penalty = 35.0
                self.body_height_offset = (
                    self.DEFAULT_WALK_BODY_HEIGHT_OFFSET
                    if self.requested_body_height_offset is None
                    else float(self.requested_body_height_offset)
                )
                self.body_roll_bias_deg = (
                    0.0 if self.requested_body_roll_bias_deg is None
                    else float(self.requested_body_roll_bias_deg)
                )
                self.body_pitch_bias_deg = (
                    self.DEFAULT_WALK_BODY_PITCH_BIAS_DEG
                    if self.requested_body_pitch_bias_deg is None
                    else float(self.requested_body_pitch_bias_deg)
                )
                self.auto_yaw_gain = (
                    self.DEFAULT_WALK_AUTO_YAW_GAIN
                    if self.requested_auto_yaw_gain is None
                    else float(self.requested_auto_yaw_gain)
                )
                # Favor stronger angular damping while keeping body translation
                # corrections subtle so the torso settles instead of rocking.
                self.imu_controller.Kp_roll = 0.74
                self.imu_controller.Kp_pitch = 0.60
                self.imu_controller.Kd_roll = 0.08
                self.imu_controller.Kd_pitch = 0.06
                self.imu_controller.alpha = 0.24
                self.imu_controller.Kp_pos_roll = 0.008
                self.imu_controller.Kp_pos_pitch = 0.006
                self.imu_controller.Kd_pos_roll = 0.002
                self.imu_controller.Kd_pos_pitch = 0.002
                self.imu_controller.alpha_pos = 0.10
                self.imu_controller.max_corr = 0.22
                self.imu_controller.max_pos_corr = 0.010
                self.imu_controller.deadband = 0.006
            self.gait_geometry_profile = (
                (
                    self.DEFAULT_ROUGH_WALK_GEOMETRY_PROFILE
                    if self.height_field
                    else self.DEFAULT_WALK_GEOMETRY_PROFILE
                )
                if self.requested_gait_geometry_profile is None
                else self.requested_gait_geometry_profile
            )
            geometry = self._get_four_phase_geometry_config(self.gait_geometry_profile)
            self.footprint_scale_x = geometry["footprint_scale_x"]
            self.footprint_scale_y = geometry["footprint_scale_y"]
            self.contact_offsets = geometry["contact_offsets"]
        else:
            self.cmd_vel = 0.8
            self.clearance_height = 0.035
            self.penetration_depth = 0.008
            self.fall_penalty = 25.0
            self.body_height_offset = (
                0.0 if self.requested_body_height_offset is None
                else float(self.requested_body_height_offset)
            )
            self.body_roll_bias_deg = (
                0.0 if self.requested_body_roll_bias_deg is None
                else float(self.requested_body_roll_bias_deg)
            )
            self.body_pitch_bias_deg = (
                0.0 if self.requested_body_pitch_bias_deg is None
                else float(self.requested_body_pitch_bias_deg)
            )
            self.auto_yaw_gain = (
                0.0 if self.requested_auto_yaw_gain is None
                else float(self.requested_auto_yaw_gain)
            )
            self.imu_controller.Kp_roll = 0.35
            self.imu_controller.Kp_pitch = 0.35
            self.imu_controller.Kd_roll = 0.02
            self.imu_controller.Kd_pitch = 0.02
            self.imu_controller.alpha = 0.12
            self.imu_controller.Kp_pos_roll = 0.0
            self.imu_controller.Kp_pos_pitch = 0.0
            self.imu_controller.Kd_pos_roll = 0.0
            self.imu_controller.Kd_pos_pitch = 0.0
            self.imu_controller.alpha_pos = 0.10
            self.imu_controller.max_corr = 0.20
            self.imu_controller.max_pos_corr = 0.015
            self.imu_controller.deadband = 0.01
            self.gait_geometry_profile = (
                "default"
                if self.requested_gait_geometry_profile is None
                else self.requested_gait_geometry_profile
            )
            self.footprint_scale_x = 1.0
            self.footprint_scale_y = 1.0
            self.contact_offsets = {
                "FL": np.zeros(3, dtype=np.float32),
                "FR": np.zeros(3, dtype=np.float32),
                "BL": np.zeros(3, dtype=np.float32),
                "BR": np.zeros(3, dtype=np.float32),
            }
        # Positive body_height_offset should lower the body posture.
        self.body_position = np.array([0.0, 0.0, self.body_height_offset], dtype=np.float32)
        self.body_orientation_bias = np.array(
            [
                np.deg2rad(self.body_roll_bias_deg),
                np.deg2rad(self.body_pitch_bias_deg),
                0.0,
            ],
            dtype=np.float32,
        )
        self._configure_imu_yaw_controller()
        self._apply_stabilization_profile()
        self._apply_swing_clearance_tuning()

    def _apply_swing_clearance_tuning(self):
        if not hasattr(self.bzg, "swing_clearance_scales"):
            return
        front_scale = float(np.clip(self.front_swing_clearance_scale, 0.75, 1.45))
        rear_scale = float(np.clip(self.rear_swing_clearance_scale, 0.55, 1.15))
        self.bzg.swing_clearance_scales.update(
            {
                "FL": front_scale,
                "FR": front_scale,
                "BL": rear_scale,
                "BR": rear_scale,
            }
        )

    def _apply_stabilization_profile(self):
        if self.stabilization_profile != "manual":
            return

        # Manual testing includes mouse perturbations, so use quicker,
        # stronger damping than the conservative training defaults.
        if self.gait_mode == "four_phase":
            self.imu_controller.Kp_roll *= 1.28
            self.imu_controller.Kp_pitch *= 1.35
            self.imu_controller.Kd_roll *= 1.55
            self.imu_controller.Kd_pitch *= 1.60
            self.imu_controller.alpha = max(self.imu_controller.alpha, 0.32)
            self.imu_controller.Kp_pos_roll *= 1.45
            self.imu_controller.Kp_pos_pitch *= 1.55
            self.imu_controller.Kd_pos_roll *= 1.45
            self.imu_controller.Kd_pos_pitch *= 1.55
            self.imu_controller.max_corr = max(self.imu_controller.max_corr, 0.30)
            self.imu_controller.max_pos_corr = max(self.imu_controller.max_pos_corr, 0.018)
            self.imu_controller.deadband = min(self.imu_controller.deadband, 0.0035)
        else:
            self.imu_controller.Kp_roll = max(self.imu_controller.Kp_roll, 0.62)
            self.imu_controller.Kp_pitch = max(self.imu_controller.Kp_pitch, 0.66)
            self.imu_controller.Kd_roll = max(self.imu_controller.Kd_roll, 0.055)
            self.imu_controller.Kd_pitch = max(self.imu_controller.Kd_pitch, 0.060)
            self.imu_controller.alpha = max(self.imu_controller.alpha, 0.22)
            self.imu_controller.Kp_pos_roll = max(self.imu_controller.Kp_pos_roll, 0.006)
            self.imu_controller.Kp_pos_pitch = max(self.imu_controller.Kp_pos_pitch, 0.008)
            self.imu_controller.Kd_pos_roll = max(self.imu_controller.Kd_pos_roll, 0.002)
            self.imu_controller.Kd_pos_pitch = max(self.imu_controller.Kd_pos_pitch, 0.0025)
            self.imu_controller.max_corr = max(self.imu_controller.max_corr, 0.26)
            self.imu_controller.max_pos_corr = max(self.imu_controller.max_pos_corr, 0.016)
            self.imu_controller.deadband = min(self.imu_controller.deadband, 0.004)

        if self.enable_imu_yaw:
            self.imu_controller.Kp_yaw *= 1.35
            self.imu_controller.Kd_yaw *= 1.45
            self.imu_controller.alpha_yaw = max(self.imu_controller.alpha_yaw, 0.20)
            self.imu_controller.max_yaw_corr = max(self.imu_controller.max_yaw_corr, 0.10)
            self.imu_controller.deadband_yaw = min(self.imu_controller.deadband_yaw, 0.006)

    def _configure_imu_yaw_controller(self):
        if not self.enable_imu_yaw:
            self.imu_controller.Kp_yaw = 0.0
            self.imu_controller.Kd_yaw = 0.0
            self.imu_controller.alpha_yaw = self.imu_controller.alpha
            self.imu_controller.max_yaw_corr = 0.0
            self.imu_controller.deadband_yaw = self.imu_controller.deadband
            return

        if self.gait_mode == "four_phase":
            if self.height_field:
                self.imu_controller.Kp_yaw = 0.16
                self.imu_controller.Kd_yaw = 0.035
                self.imu_controller.alpha_yaw = 0.18
                self.imu_controller.max_yaw_corr = 0.11
            else:
                self.imu_controller.Kp_yaw = 0.12
                self.imu_controller.Kd_yaw = 0.025
                self.imu_controller.alpha_yaw = 0.16
                self.imu_controller.max_yaw_corr = 0.08
        else:
            self.imu_controller.Kp_yaw = 0.08
            self.imu_controller.Kd_yaw = 0.020
            self.imu_controller.alpha_yaw = 0.12
            self.imu_controller.max_yaw_corr = 0.06
        self.imu_controller.deadband_yaw = 0.01

    def _compose_body_pose(self, base_state):
        yaw = float(self.base_env.return_yaw()) if self.enable_imu_yaw else None
        imu_pos, imu_orn = self.imu_controller.compute(base_state, yaw=yaw)
        pos = self.body_position.copy() + imu_pos
        orn = self.body_orientation_bias.copy() + imu_orn
        return pos.astype(np.float32), orn.astype(np.float32)

    def _get_orientation_features(self):
        if not self.enable_imu_yaw:
            return np.zeros(0, dtype=np.float32)

        yaw = float(self.base_env.return_yaw())
        return np.array(
            [
                yaw,
                np.sin(yaw),
                np.cos(yaw),
                float(self.imu_controller.yaw_corr),
            ],
            dtype=np.float32,
        )

    def _apply_contact_offsets(self):
        for leg_name, offset in self.contact_offsets.items():
            self.T_bf0[leg_name][0, 3] *= self.footprint_scale_x
            self.T_bf0[leg_name][1, 3] *= self.footprint_scale_y
            self.T_bf[leg_name][0, 3] *= self.footprint_scale_x
            self.T_bf[leg_name][1, 3] *= self.footprint_scale_y
            self.T_bf0[leg_name][0, 3] += float(offset[0])
            self.T_bf0[leg_name][1, 3] += float(offset[1])
            self.T_bf0[leg_name][2, 3] += float(offset[2])
            self.T_bf[leg_name][0, 3] += float(offset[0])
            self.T_bf[leg_name][1, 3] += float(offset[1])
            self.T_bf[leg_name][2, 3] += float(offset[2])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.seed(seed)

        base_obs = self.base_env.reset()

        self.bzg = self._make_gait_generator()
        self.bzg.reset()
        self.bz_step = BezierStepper(dt=self.base_env._time_step, mode=0)
        self.imu_controller.reset()

        self._configure_gait_mode()
        self.T_bf0 = copy.deepcopy(self.spot_model.WorldToFoot)
        self.T_bf = copy.deepcopy(self.T_bf0)
        self._apply_contact_offsets()
        self.cmd_step = 0.0
        self.cmd_lat = 0.0
        self.cmd_yaw = 0.0
        self.cmd_vel = 0.20 if self.gait_mode == "four_phase" else 0.8

        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_base_pos = np.array(self.base_env.spot.GetBasePosition(), dtype=np.float32)
        self.episode_start_pos = self.prev_base_pos.copy()
        self.nominal_base_height = float(self.prev_base_pos[2])
        self.episode_step_count = 0
        self.last_forward_speed = 0.0
        self.last_stabilized_yaw_rate = 0.0

        obs = self._get_ml_observation(base_obs)
        info = {}
        return obs, info

    def _smooth_command(self, action):
        if self.gait_mode == "four_phase":
            alpha_step = 0.05
            alpha_lat = 0.02
            alpha_yaw = 0.02
            alpha_vel = 0.04
        else:
            alpha_step = 0.10
            alpha_lat = 0.12
            alpha_yaw = 0.10
            alpha_vel = 0.08

        if self.gait_mode == "four_phase" and not self.allow_turning_commands:
            # Keep existing trained crawl policies compatible: their lateral
            # and yaw actions were ignored during training.
            target_step = max(0.0, float(action[0]))
            target_lat = 0.0
            target_yaw = 0.0
        elif self.gait_mode == "four_phase":
            target_step = max(0.0, float(action[0]))
            target_lat = float(action[1])
            target_yaw = float(action[2])
        else:
            target_step = float(action[0])
            target_lat = float(action[1])
            target_yaw = float(action[2])
        target_vel = float(action[3])

        #if abs(target_lat) > 1e-4 and abs(target_step) < 1e-4:
            #target_step = 0.06

        self.cmd_step += alpha_step * (target_step - self.cmd_step)
        self.cmd_lat += alpha_lat * (target_lat - self.cmd_lat)
        self.cmd_yaw += alpha_yaw * (target_yaw - self.cmd_yaw)
        self.cmd_vel += alpha_vel * (target_vel - self.cmd_vel)

        return self.cmd_step, self.cmd_lat, self.cmd_yaw, self.cmd_vel

    def _command_to_joint_angles(self, action):
        self.bz_step.ramp_up()

        step_length, lateral_fraction, yaw_rate, step_velocity = self._smooth_command(action)
        base_state = self.base_env.return_state()
        stabilized_yaw_rate = yaw_rate
        if abs(self.auto_yaw_gain) > 1e-8:
            stabilized_yaw_rate += -self.base_env.return_yaw() * self.auto_yaw_gain
        stabilized_yaw_rate = float(np.clip(
            stabilized_yaw_rate,
            self.action_space.low[2],
            self.action_space.high[2],
        ))
        self.last_stabilized_yaw_rate = stabilized_yaw_rate

        if self.gait_mode == "four_phase":
            if self.height_field:
                if abs(step_length) > 1e-4 and abs(stabilized_yaw_rate) > 1e-4:
                    self.bzg.Tswing = 0.20
                elif abs(stabilized_yaw_rate) > 1e-4:
                    self.bzg.Tswing = 0.19
                elif abs(lateral_fraction) > 1e-4:
                    self.bzg.Tswing = 0.19
                else:
                    self.bzg.Tswing = 0.18
            else:
                if abs(step_length) > 1e-4 and abs(stabilized_yaw_rate) > 1e-4:
                    self.bzg.Tswing = 0.18
                elif abs(stabilized_yaw_rate) > 1e-4:
                    self.bzg.Tswing = 0.17
                elif abs(lateral_fraction) > 1e-4:
                    self.bzg.Tswing = 0.17
                else:
                    self.bzg.Tswing = 0.16
        else:
            if abs(step_length) > 1e-4 and abs(stabilized_yaw_rate) > 1e-4:
                self.bzg.Tswing = 0.42
            elif abs(stabilized_yaw_rate) > 1e-4:
                self.bzg.Tswing = 0.40
            elif abs(lateral_fraction) > 1e-4:
                self.bzg.Tswing = 0.40
            else:
                self.bzg.Tswing = 0.28
            if self.allow_turning_commands and self.manual_curve_height is not None:
                speed_scale = float(np.clip(self.manual_curve_speed_scale, 0.35, 1.0))
                step_velocity *= speed_scale
                self.cmd_vel = step_velocity
                self.bzg.Tswing /= max(speed_scale, 1e-6)

        self.bz_step.StepLength = step_length
        self.bz_step.LateralFraction = lateral_fraction
        self.bz_step.YawRate = stabilized_yaw_rate
        self.bz_step.StepVelocity = step_velocity

        contacts = base_state[-4:]

        step_command = step_length
        if (
            self.gait_mode == "trot"
            and self.allow_turning_commands
            and abs(step_length) > 1e-4
            and abs(lateral_fraction) < 1e-4
            and abs(stabilized_yaw_rate) < 1e-4
            and (
                abs(self.manual_forward_left_step_gain - 1.0) > 1e-4
                or abs(self.manual_forward_right_step_gain - 1.0) > 1e-4
            )
        ):
            step_command = {
                "FL": step_length * self.manual_forward_left_step_gain,
                "BL": step_length * self.manual_forward_left_step_gain,
                "FR": step_length * self.manual_forward_right_step_gain,
                "BR": step_length * self.manual_forward_right_step_gain,
            }

        self.T_bf = self.bzg.GenerateTrajectory(
            step_command,
            lateral_fraction,
            stabilized_yaw_rate,
            step_velocity,
            self.T_bf0,
            self.T_bf,
            clearance_height=self.clearance_height,
            penetration_depth=self.penetration_depth,
            contacts=contacts
        )

        pos, orn = self._compose_body_pose(base_state)
        joint_angles = self.spot_model.IK(orn, pos, self.T_bf)
        self.base_env.spot.GetExternalObservations(self.bzg, self.bz_step)

        return joint_angles.reshape(-1)

    def _get_ml_observation(self, base_obs=None, include_orientation_features=None):
        if base_obs is None:
            base_obs = self.base_env.return_state()
        if include_orientation_features is None:
            include_orientation_features = self.enable_imu_yaw

        base_obs = np.array(base_obs, dtype=np.float32)
        base_height = np.array([self.base_env.spot.GetBasePosition()[2]], dtype=np.float32)

        cmd_state = np.array([
            self.cmd_step,
            self.cmd_lat,
            self.cmd_yaw,
            self.cmd_vel
        ], dtype=np.float32)

        leg_phases = np.array(self.base_env.spot.LegPhases, dtype=np.float32)

        phase_sin = np.sin(2.0 * np.pi * leg_phases).astype(np.float32)
        phase_cos = np.cos(2.0 * np.pi * leg_phases).astype(np.float32)

        obs = np.concatenate([
            base_obs,
            self.prev_action.astype(np.float32),
            cmd_state,
            base_height,
            phase_sin,
            phase_cos
        ]).astype(np.float32)

        if include_orientation_features:
            obs = np.concatenate([obs, self._get_orientation_features()]).astype(np.float32)

        return obs

    def _get_base_height_error(self):
        if self.nominal_base_height is None:
            return 0.0
        current_height = float(self.base_env.spot.GetBasePosition()[2])
        return abs(current_height - self.nominal_base_height)

    def _compute_reward(self, done=False):
        obs = np.array(self.base_env.return_state(), dtype=np.float32)

        roll = obs[0]
        pitch = obs[1]
        gx = obs[2]
        gy = obs[3]
        gz = obs[4]

        base_pos = np.array(self.base_env.spot.GetBasePosition(), dtype=np.float32)
        dt = self.base_env.control_time_step if self.base_env.control_time_step > 0 else self.base_env._time_step
        forward_speed = (base_pos[0] - self.prev_base_pos[0]) / max(dt, 1e-6)
        self.last_forward_speed = float(forward_speed)
        forward_displacement = float(base_pos[0] - self.episode_start_pos[0])
        lateral_drift = abs(base_pos[1])
        base_height_error = self._get_base_height_error()
        contacts = self.base_env.return_state()[-4:]
        contact_count = int(np.sum(np.asarray(contacts) > 0.5))
        airborne_legs = max(0, 2 - int(np.sum(np.asarray(contacts) > 0.5)))

        vel_reward = np.exp(-((forward_speed - self.target_forward_speed) ** 2) / 0.02)
        heading_penalty = (0.65 if self.gait_mode == "four_phase" else 0.08) * abs(self.base_env.return_yaw())
        if self.gait_mode == "four_phase":
            forward_progress_reward = (7.0 if self.height_field else 8.0) * max(0.0, forward_speed)
            distance_bonus = (0.24 if self.height_field else 0.30) * max(0.0, forward_displacement)
            backward_penalty = 8.0 * max(0.0, -forward_speed)
            support_penalty = (0.26 if self.height_field else 0.18) * max(0, 3 - contact_count)
            alive_bonus = 0.05 if forward_speed > 0.015 else 0.0
        else:
            forward_progress_reward = 3.0 * forward_speed
            distance_bonus = 0.0
            backward_penalty = 2.5 * max(0.0, -forward_speed)
            support_penalty = 0.0
            alive_bonus = 0.03 if forward_speed > 0.01 else 0.0

        if self.gait_mode == "four_phase":
            if self.height_field:
                upright_penalty = 4.4 * abs(roll) + 3.2 * abs(pitch)
            else:
                upright_penalty = 3.6 * abs(roll) + 2.6 * abs(pitch)
        else:
            upright_penalty = 0.8 * (abs(roll) + abs(pitch))
        ang_penalty = (
            (0.18 if self.height_field else 0.14) * (abs(gx) + abs(gy) + 0.8 * abs(gz))
            if self.gait_mode == "four_phase"
            else 0.05 * (abs(gx) + abs(gy) + 0.5 * abs(gz))
        )
        height_penalty = (
            (2.0 if self.height_field else 2.5)
            if self.gait_mode == "four_phase"
            else 1.5
        ) * base_height_error
        airborne_penalty = (
            (0.42 if self.height_field else 0.30)
            if self.gait_mode == "four_phase"
            else 0.12
        ) * airborne_legs

        action_delta_penalty = (0.30 if self.gait_mode == "four_phase" else 0.20) * np.linalg.norm(
            np.array([self.cmd_step, self.cmd_lat, self.cmd_yaw, self.cmd_vel]) - self.prev_action
        )

        energy_penalty = 0.002 * np.abs(
            np.dot(self.base_env.spot.GetMotorTorques(), self.base_env.spot.GetMotorVelocities())
        )

        drift_penalty = (
            (0.40 if self.height_field else 0.55)
            if self.gait_mode == "four_phase"
            else 0.10
        ) * lateral_drift

        reward = (
            2.0 * vel_reward
            + forward_progress_reward
            + distance_bonus
            - upright_penalty
            - ang_penalty
            - height_penalty
            - airborne_penalty
            - support_penalty
            - action_delta_penalty
            - energy_penalty
            - drift_penalty
            - heading_penalty
            - backward_penalty
        )

        if not self.base_env.is_fallen():
            reward += alive_bonus
        elif done:
            reward -= self.fall_penalty

        self.prev_base_pos = base_pos.copy()
        return float(reward)

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        joint_angles = self._command_to_joint_angles(action)
        self.base_env.pass_joint_angles(joint_angles)

        base_obs, _, done, info = self.base_env.step(np.zeros(4, dtype=np.float32))

        self.prev_action = action.copy()
        ml_obs = self._get_ml_observation(base_obs)
        reward = self._compute_reward(done=done)
        base_height_error = self._get_base_height_error()
        current_base_pos = np.array(self.base_env.spot.GetBasePosition(), dtype=np.float32)
        self.episode_step_count += 1

        terminated = done
        truncated = self.episode_step_count >= self.max_episode_steps and not terminated

        info.update({
            "forward_speed_target": self.target_forward_speed,
            "gait_mode": self.gait_mode,
            "terrain_profile": self.terrain_profile,
            "height_field": bool(self.height_field),
            "terrain_randomization": bool(self.terrain_randomization),
            "imu_yaw_enabled": bool(self.enable_imu_yaw),
            "imu_yaw_correction": float(self.imu_controller.yaw_corr),
            "is_fallen": bool(self.base_env.is_fallen()),
            "base_position": current_base_pos,
            "forward_speed": float(self.last_forward_speed),
            "forward_displacement": float(current_base_pos[0] - self.episode_start_pos[0]),
            "base_height_error": float(base_height_error),
            "body_height_offset": float(self.body_height_offset),
            "body_pitch_bias_deg": float(self.body_pitch_bias_deg),
            "auto_yaw_gain": float(self.auto_yaw_gain),
            "gait_geometry_profile": self.gait_geometry_profile,
            "footprint_scale_x": float(self.footprint_scale_x),
            "footprint_scale_y": float(self.footprint_scale_y),
            "contact_offsets": {
                leg_name: offset.tolist()
                for leg_name, offset in self.contact_offsets.items()
            },
            "episode_step_count": int(self.episode_step_count),
            "time_limit_reached": bool(truncated),
            "command": {
                "step_length": float(self.cmd_step),
                "lateral_fraction": float(self.cmd_lat),
                "yaw_rate": float(self.cmd_yaw),
                "stabilized_yaw_rate": float(self.last_stabilized_yaw_rate),
                "step_velocity": float(self.cmd_vel),
            }
        })

        return ml_obs, reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        if hasattr(self.base_env, "close"):
            return self.base_env.close()

        pybullet_client = getattr(self.base_env, "_pybullet_client", None)
        if pybullet_client is not None:
            try:
                pybullet_client.disconnect()
            except Exception:
                pass

        return None


class SpotMLDualIMUStableWalkEnv(SpotMLWalkEnv):
    """Stable walk variant with dual virtual IMUs and per-leg stride control.

    The policy controls a shared forward step length and velocity plus four
    leg-specific stride residuals. Two virtual IMU probes are attached to the
    front and rear of the torso so the reward can directly penalize rocking,
    pitching, yaw drift, and uneven body height while still rewarding forward
    progress.
    """

    LEG_NAMES = ("FL", "FR", "BL", "BR")
    FRONT_SENSOR_OFFSET = np.array([0.22, 0.0, 0.05], dtype=np.float32)
    REAR_SENSOR_OFFSET = np.array([-0.22, 0.0, 0.05], dtype=np.float32)

    BASE_COMMAND_DIM = 2
    LEG_RESIDUAL_DIM = 4
    ACTION_DIM = BASE_COMMAND_DIM + LEG_RESIDUAL_DIM
    COMMAND_STATE_DIM = 5
    DUAL_IMU_FEATURE_DIM = 18

    DEFAULT_STABLE_TARGET_FORWARD_SPEED = 0.085
    DEFAULT_ROUGH_STABLE_TARGET_FORWARD_SPEED = 0.060
    DEFAULT_STABLE_AUTO_YAW_GAIN = 0.45
    DEFAULT_ROUGH_STABLE_AUTO_YAW_GAIN = 0.60
    DEFAULT_STABLE_GEOMETRY_PROFILE = "anti_rock"
    DEFAULT_ROUGH_STABLE_GEOMETRY_PROFILE = "contact_stable"
    DEFAULT_LEG_STEP_RESIDUAL_MAX = 0.020

    def __init__(
        self,
        *args,
        leg_step_residual_max=DEFAULT_LEG_STEP_RESIDUAL_MAX,
        **kwargs,
    ):
        if kwargs.get("gait_mode", "four_phase") != "four_phase":
            raise ValueError("SpotMLDualIMUStableWalkEnv currently requires gait_mode='four_phase'.")

        super().__init__(*args, **kwargs)

        self.leg_step_residual_max = float(leg_step_residual_max)
        self._stable_step_alpha = 0.12
        self._stable_velocity_alpha = 0.10
        self._stable_leg_alpha = 0.10

        self.prev_action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        self.prev_leg_step_residual_action = np.zeros(self.LEG_RESIDUAL_DIM, dtype=np.float32)
        self.leg_step_residuals = np.zeros(self.LEG_RESIDUAL_DIM, dtype=np.float32)
        self.last_leg_step_lengths = np.zeros(self.LEG_RESIDUAL_DIM, dtype=np.float32)
        self.last_dual_imu_obs = np.zeros(self.DUAL_IMU_FEATURE_DIM, dtype=np.float32)
        self.last_dual_imu_state = {}
        self.prev_front_sensor_pos = None
        self.prev_rear_sensor_pos = None
        self.prev_front_sensor_vel = np.zeros(3, dtype=np.float32)
        self.prev_rear_sensor_vel = np.zeros(3, dtype=np.float32)
        self.start_front_sensor_pos = None
        self.start_rear_sensor_pos = None

        min_step_length = 0.010 if self.height_field else 0.012
        max_step_length = 0.050 if self.height_field else 0.060
        min_step_velocity = 0.20 if self.height_field else 0.22
        max_step_velocity = 0.42 if self.height_field else 0.38
        leg_residual_limit = min(self.leg_step_residual_max, 0.012 if self.height_field else 0.010)
        self.action_space = spaces.Box(
            low=np.array(
                [
                    min_step_length,
                    min_step_velocity,
                    -leg_residual_limit,
                    -leg_residual_limit,
                    -leg_residual_limit,
                    -leg_residual_limit,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    max_step_length,
                    max_step_velocity,
                    leg_residual_limit,
                    leg_residual_limit,
                    leg_residual_limit,
                    leg_residual_limit,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        obs_dim = (
            16
            + self.ACTION_DIM
            + self.COMMAND_STATE_DIM
            + 8
            + 4
            + self.DUAL_IMU_FEATURE_DIM
            + self.imu_orientation_feature_dim
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        if self.requested_auto_yaw_gain is None:
            self.auto_yaw_gain = (
                self.DEFAULT_ROUGH_STABLE_AUTO_YAW_GAIN
                if self.height_field
                else self.DEFAULT_STABLE_AUTO_YAW_GAIN
            )
        if self.requested_gait_geometry_profile is None:
            self.gait_geometry_profile = (
                self.DEFAULT_ROUGH_STABLE_GEOMETRY_PROFILE
                if self.height_field
                else self.DEFAULT_STABLE_GEOMETRY_PROFILE
            )

    def _configure_gait_mode(self):
        super()._configure_gait_mode()
        if self.gait_mode != "four_phase":
            return

        if self.requested_auto_yaw_gain is None:
            self.auto_yaw_gain = (
                self.DEFAULT_ROUGH_STABLE_AUTO_YAW_GAIN
                if self.height_field
                else self.DEFAULT_STABLE_AUTO_YAW_GAIN
            )
        if self.requested_gait_geometry_profile is None:
            self.gait_geometry_profile = (
                self.DEFAULT_ROUGH_STABLE_GEOMETRY_PROFILE
                if self.height_field
                else self.DEFAULT_STABLE_GEOMETRY_PROFILE
            )
            geometry = self._get_four_phase_geometry_config(self.gait_geometry_profile)
            self.footprint_scale_x = geometry["footprint_scale_x"]
            self.footprint_scale_y = geometry["footprint_scale_y"]
            self.contact_offsets = geometry["contact_offsets"]
        if self.height_field:
            self.target_forward_speed = min(
                self.target_forward_speed,
                self.DEFAULT_ROUGH_STABLE_TARGET_FORWARD_SPEED,
            )
            self.clearance_height = max(self.clearance_height, 0.040)
            self.fall_penalty = max(self.fall_penalty, 40.0)
        else:
            self.target_forward_speed = min(
                self.target_forward_speed,
                self.DEFAULT_STABLE_TARGET_FORWARD_SPEED,
            )
            self.clearance_height = max(self.clearance_height, 0.028)
            self.fall_penalty = max(self.fall_penalty, 30.0)

    def _get_body_rotation(self):
        base_orn = self.base_env.spot.GetBaseOrientation()
        pybullet_client = self.base_env._pybullet_client
        return np.array(
            pybullet_client.getMatrixFromQuaternion(base_orn),
            dtype=np.float32,
        ).reshape(3, 3)

    def _get_virtual_sensor_positions(self):
        base_pos = np.array(self.base_env.spot.GetBasePosition(), dtype=np.float32)
        rotation = self._get_body_rotation()
        front_pos = base_pos + rotation @ self.FRONT_SENSOR_OFFSET
        rear_pos = base_pos + rotation @ self.REAR_SENSOR_OFFSET
        return front_pos.astype(np.float32), rear_pos.astype(np.float32)

    def _get_dual_imu_features(self):
        base_pos = np.array(self.base_env.spot.GetBasePosition(), dtype=np.float32)
        base_obs = np.array(self.base_env.return_state(), dtype=np.float32)
        base_orn = self.base_env.spot.GetBaseOrientation()
        roll, pitch, yaw = self.base_env._pybullet_client.getEulerFromQuaternion(base_orn)
        dt = self.base_env.control_time_step if self.base_env.control_time_step > 0 else self.base_env._time_step
        dt = max(dt, 1e-6)

        front_pos, rear_pos = self._get_virtual_sensor_positions()

        if self.prev_front_sensor_pos is None:
            front_vel = np.zeros(3, dtype=np.float32)
            rear_vel = np.zeros(3, dtype=np.float32)
        else:
            front_vel = (front_pos - self.prev_front_sensor_pos) / dt
            rear_vel = (rear_pos - self.prev_rear_sensor_pos) / dt

        front_acc = (front_vel - self.prev_front_sensor_vel) / dt
        rear_acc = (rear_vel - self.prev_rear_sensor_vel) / dt

        if self.start_front_sensor_pos is None:
            self.start_front_sensor_pos = front_pos.copy()
        if self.start_rear_sensor_pos is None:
            self.start_rear_sensor_pos = rear_pos.copy()

        front_rel = front_pos - self.start_front_sensor_pos
        rear_rel = rear_pos - self.start_rear_sensor_pos
        balance = np.array(
            [
                front_pos[1] - rear_pos[1],
                front_pos[2] - rear_pos[2],
                front_vel[2] - rear_vel[2],
                front_acc[2] - rear_acc[2],
            ],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [
                front_rel,
                rear_rel,
                front_vel,
                rear_vel,
                np.array([front_acc[2], rear_acc[2]], dtype=np.float32),
                balance,
            ]
        ).astype(np.float32)

        self.prev_front_sensor_pos = front_pos.copy()
        self.prev_rear_sensor_pos = rear_pos.copy()
        self.prev_front_sensor_vel = front_vel.copy()
        self.prev_rear_sensor_vel = rear_vel.copy()
        self.last_dual_imu_obs = obs
        self.last_dual_imu_state = {
            "front_pos": front_pos,
            "rear_pos": rear_pos,
            "front_vel": front_vel,
            "rear_vel": rear_vel,
            "front_acc": front_acc,
            "rear_acc": rear_acc,
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "gyro": base_obs[2:5].astype(np.float32),
            "base_pos": base_pos,
        }
        return obs

    def _split_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        base_step = float(action[0])
        step_velocity = float(action[1])
        leg_residuals = action[2: 2 + self.LEG_RESIDUAL_DIM].astype(np.float32)
        return base_step, step_velocity, leg_residuals

    def _smooth_stable_command(self, action):
        base_step, step_velocity, leg_residuals = self._split_action(action)
        self.cmd_step += self._stable_step_alpha * (base_step - self.cmd_step)
        self.cmd_vel += self._stable_velocity_alpha * (step_velocity - self.cmd_vel)
        self.leg_step_residuals += self._stable_leg_alpha * (
            leg_residuals - self.leg_step_residuals
        )
        return self.cmd_step, self.cmd_vel, self.leg_step_residuals.copy()

    def reset(self, seed=None, options=None):
        self.prev_action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        self.prev_leg_step_residual_action = np.zeros(self.LEG_RESIDUAL_DIM, dtype=np.float32)
        self.leg_step_residuals = np.zeros(self.LEG_RESIDUAL_DIM, dtype=np.float32)
        self.last_leg_step_lengths = np.zeros(self.LEG_RESIDUAL_DIM, dtype=np.float32)
        self.last_dual_imu_obs = np.zeros(self.DUAL_IMU_FEATURE_DIM, dtype=np.float32)
        self.last_dual_imu_state = {}
        self.prev_front_sensor_pos = None
        self.prev_rear_sensor_pos = None
        self.prev_front_sensor_vel = np.zeros(3, dtype=np.float32)
        self.prev_rear_sensor_vel = np.zeros(3, dtype=np.float32)
        self.start_front_sensor_pos = None
        self.start_rear_sensor_pos = None

        obs, info = super().reset(seed=seed, options=options)
        self.cmd_step = float(max(self.action_space.low[0], 0.018 if not self.height_field else 0.015))
        self.cmd_vel = float(max(self.action_space.low[1], 0.26 if not self.height_field else 0.22))
        self.prev_action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        self.prev_leg_step_residual_action = np.zeros(self.LEG_RESIDUAL_DIM, dtype=np.float32)
        self.last_leg_step_lengths = np.zeros(self.LEG_RESIDUAL_DIM, dtype=np.float32)
        self._get_dual_imu_features()
        obs = self._get_ml_observation()
        info.update(
            {
                "walk_variant": "dual_imu_stable_v1",
                "front_sensor_offset": self.FRONT_SENSOR_OFFSET.tolist(),
                "rear_sensor_offset": self.REAR_SENSOR_OFFSET.tolist(),
            }
        )
        return obs, info

    def _command_to_joint_angles(self, action):
        self.bz_step.ramp_up()

        step_length, step_velocity, leg_residuals = self._smooth_stable_command(action)
        base_state = self.base_env.return_state()
        stabilized_yaw_rate = float(
            np.clip(
                -self.base_env.return_yaw() * self.auto_yaw_gain,
                -0.08,
                0.08,
            )
        )
        self.last_stabilized_yaw_rate = stabilized_yaw_rate

        if self.height_field:
            self.bzg.Tswing = 0.20 if step_length > 0.020 else 0.18
        else:
            self.bzg.Tswing = 0.18 if step_length > 0.020 else 0.16

        leg_step_lengths = np.clip(
            step_length + leg_residuals,
            self.action_space.low[0],
            self.action_space.high[0],
        ).astype(np.float32)
        self.last_leg_step_lengths = leg_step_lengths

        self.bz_step.StepLength = float(np.mean(leg_step_lengths))
        self.bz_step.LateralFraction = 0.0
        self.bz_step.YawRate = stabilized_yaw_rate
        self.bz_step.StepVelocity = step_velocity
        self.cmd_lat = 0.0
        self.cmd_yaw = 0.0

        contacts = base_state[-4:]
        leg_length_map = {
            leg_name: leg_step_lengths[index]
            for index, leg_name in enumerate(self.LEG_NAMES)
        }
        self.T_bf = self.bzg.GenerateTrajectory(
            leg_length_map,
            0.0,
            stabilized_yaw_rate,
            step_velocity,
            self.T_bf0,
            self.T_bf,
            clearance_height=self.clearance_height,
            penetration_depth=self.penetration_depth,
            contacts=contacts,
        )

        pos, orn = self._compose_body_pose(base_state)
        joint_angles = self.spot_model.IK(orn, pos, self.T_bf)
        self.base_env.spot.GetExternalObservations(self.bzg, self.bz_step)
        return joint_angles.reshape(-1)

    def _get_ml_observation(self, base_obs=None, include_orientation_features=None):
        if base_obs is None:
            base_obs = self.base_env.return_state()
        if include_orientation_features is None:
            include_orientation_features = self.enable_imu_yaw

        base_obs = np.array(base_obs, dtype=np.float32)
        leg_phases = np.array(self.base_env.spot.LegPhases, dtype=np.float32)
        phase_sin = np.sin(2.0 * np.pi * leg_phases).astype(np.float32)
        phase_cos = np.cos(2.0 * np.pi * leg_phases).astype(np.float32)
        command_state = np.array(
            [
                self.cmd_step,
                self.cmd_vel,
                self.last_stabilized_yaw_rate,
                float(np.mean(self.last_leg_step_lengths)),
                float(np.std(self.last_leg_step_lengths)),
            ],
            dtype=np.float32,
        )
        dual_imu_obs = self._get_dual_imu_features()
        obs = np.concatenate(
            [
                base_obs,
                self.prev_action.astype(np.float32),
                command_state,
                phase_sin,
                phase_cos,
                self.last_leg_step_lengths.astype(np.float32),
                dual_imu_obs.astype(np.float32),
            ]
        ).astype(np.float32)
        if include_orientation_features:
            obs = np.concatenate([obs, self._get_orientation_features()]).astype(np.float32)
        return obs

    def _compute_reward(self, action, done=False):
        base_obs = np.array(self.base_env.return_state(), dtype=np.float32)
        roll = float(base_obs[0])
        pitch = float(base_obs[1])
        gx = float(base_obs[2])
        gy = float(base_obs[3])
        gz = float(base_obs[4])

        sensor_state = self.last_dual_imu_state or {}
        yaw = float(sensor_state.get("yaw", self.base_env.return_yaw()))
        base_pos = np.array(self.base_env.spot.GetBasePosition(), dtype=np.float32)
        dt = self.base_env.control_time_step if self.base_env.control_time_step > 0 else self.base_env._time_step
        forward_speed = (base_pos[0] - self.prev_base_pos[0]) / max(dt, 1e-6)
        self.last_forward_speed = float(forward_speed)
        forward_displacement = float(base_pos[0] - self.episode_start_pos[0])
        base_height_error = self._get_base_height_error()
        contacts = np.asarray(self.base_env.return_state()[-4:], dtype=np.float32)
        contact_count = int(np.sum(contacts > 0.5))

        front_pos = np.asarray(sensor_state.get("front_pos", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        rear_pos = np.asarray(sensor_state.get("rear_pos", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        front_vel = np.asarray(sensor_state.get("front_vel", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        rear_vel = np.asarray(sensor_state.get("rear_vel", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        front_acc = np.asarray(sensor_state.get("front_acc", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        rear_acc = np.asarray(sensor_state.get("rear_acc", np.zeros(3, dtype=np.float32)), dtype=np.float32)

        front_height_error = 0.0 if self.start_front_sensor_pos is None else float(front_pos[2] - self.start_front_sensor_pos[2])
        rear_height_error = 0.0 if self.start_rear_sensor_pos is None else float(rear_pos[2] - self.start_rear_sensor_pos[2])
        front_rear_height_diff = float(front_pos[2] - rear_pos[2])

        vel_reward = np.exp(-((forward_speed - self.target_forward_speed) ** 2) / 0.010)
        forward_progress_reward = 16.0 * max(0.0, forward_speed)
        distance_bonus = 0.80 * max(0.0, forward_displacement)
        stall_penalty = 8.0 * max(0.0, (0.70 * self.target_forward_speed) - forward_speed)
        upright_penalty = 2.8 * abs(roll) + 2.6 * abs(pitch) + 0.9 * abs(yaw)
        angular_penalty = 0.06 * (abs(gx) + abs(gy) + 0.5 * abs(gz))
        height_penalty = 1.2 * base_height_error
        sensor_height_penalty = (
            2.2 * (abs(front_height_error) + abs(rear_height_error))
            + 4.0 * abs(front_rear_height_diff)
        )
        sensor_vertical_motion_penalty = 0.16 * (
            abs(float(front_vel[2]))
            + abs(float(rear_vel[2]))
            + 0.06 * abs(float(front_acc[2]))
            + 0.06 * abs(float(rear_acc[2]))
        )
        lateral_penalty = 0.40 * (
            abs(float(front_pos[1]))
            + abs(float(rear_pos[1]))
            + abs(float(base_pos[1]))
        )
        support_penalty = 0.12 * max(0, 3 - contact_count)
        action_delta_penalty = 0.08 * np.linalg.norm(action - self.prev_action)
        leg_imbalance_penalty = 0.60 * float(np.std(self.last_leg_step_lengths))
        backward_penalty = 12.0 * max(0.0, -forward_speed)

        reward = (
            2.4 * vel_reward
            + forward_progress_reward
            + distance_bonus
            - stall_penalty
            - upright_penalty
            - angular_penalty
            - height_penalty
            - sensor_height_penalty
            - sensor_vertical_motion_penalty
            - lateral_penalty
            - support_penalty
            - action_delta_penalty
            - leg_imbalance_penalty
            - backward_penalty
        )

        if not self.base_env.is_fallen():
            reward += 0.18 if forward_speed > 0.03 else 0.02
        elif done:
            reward -= self.fall_penalty

        self.prev_base_pos = base_pos.copy()
        return float(reward)

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        joint_angles = self._command_to_joint_angles(action)
        self.base_env.pass_joint_angles(joint_angles)
        base_obs, _, done, info = self.base_env.step(np.zeros(4, dtype=np.float32))

        ml_obs = self._get_ml_observation(base_obs)
        reward = self._compute_reward(action, done=done)
        current_base_pos = np.array(self.base_env.spot.GetBasePosition(), dtype=np.float32)
        self.episode_step_count += 1
        self.prev_leg_step_residual_action = action[2:].copy()
        self.prev_action = action.copy()

        terminated = done
        truncated = self.episode_step_count >= self.max_episode_steps and not terminated

        info.update(
            {
                "walk_variant": "dual_imu_stable_v1",
                "body_pitch_bias_deg": float(self.body_pitch_bias_deg),
                "imu_yaw_enabled": bool(self.enable_imu_yaw),
                "imu_yaw_correction": float(self.imu_controller.yaw_corr),
                "forward_speed_target": self.target_forward_speed,
                "gait_mode": self.gait_mode,
                "terrain_profile": self.terrain_profile,
                "is_fallen": bool(self.base_env.is_fallen()),
                "base_position": current_base_pos,
                "forward_speed": float(self.last_forward_speed),
                "forward_displacement": float(current_base_pos[0] - self.episode_start_pos[0]),
                "base_height_error": float(self._get_base_height_error()),
                "episode_step_count": int(self.episode_step_count),
                "time_limit_reached": bool(truncated),
                "command": {
                    "step_length": float(self.cmd_step),
                    "step_velocity": float(self.cmd_vel),
                    "per_leg_step_lengths": self.last_leg_step_lengths.tolist(),
                    "per_leg_step_residuals": self.leg_step_residuals.tolist(),
                    "stabilized_yaw_rate": float(self.last_stabilized_yaw_rate),
                },
                "dual_imu": {
                    "front_pos": self.last_dual_imu_state.get("front_pos", np.zeros(3, dtype=np.float32)).tolist(),
                    "rear_pos": self.last_dual_imu_state.get("rear_pos", np.zeros(3, dtype=np.float32)).tolist(),
                    "front_vel": self.last_dual_imu_state.get("front_vel", np.zeros(3, dtype=np.float32)).tolist(),
                    "rear_vel": self.last_dual_imu_state.get("rear_vel", np.zeros(3, dtype=np.float32)).tolist(),
                    "front_acc": self.last_dual_imu_state.get("front_acc", np.zeros(3, dtype=np.float32)).tolist(),
                    "rear_acc": self.last_dual_imu_state.get("rear_acc", np.zeros(3, dtype=np.float32)).tolist(),
                    "roll": float(self.last_dual_imu_state.get("roll", 0.0)),
                    "pitch": float(self.last_dual_imu_state.get("pitch", 0.0)),
                    "yaw": float(self.last_dual_imu_state.get("yaw", 0.0)),
                },
            }
        )

        return ml_obs, reward, terminated, truncated, info


class SpotMLRoughDualIMUStableWalkEnv(SpotMLDualIMUStableWalkEnv):
    """Dual-IMU walker augmented with rough-terrain foothold features."""

    TERRAIN_FORWARD_PROBE_DISTANCE = 0.08
    TERRAIN_PROBE_START_Z = 0.30
    TERRAIN_PROBE_END_Z = -0.35
    CAMERA_FEATURE_DIM = 4
    TERRAIN_EXTRA_OBS_DIM = 12 + 12 + 4 + 4 + 4 + CAMERA_FEATURE_DIM
    DEFAULT_ROUGH_DUAL_IMU_CLEARANCE_HEIGHT = 0.034
    FRONT_LEG_STEP_SCALE = 0.82
    FRONT_LEG_SWING_HEIGHT_SCALE = 1.28
    FRONT_LEG_SWING_X_SCALE = 0.80
    FRONT_LEG_SWING_LIFT_BONUS = 0.008
    FRONT_LEG_STANCE_EXTENSION_SCALE = 0.34
    FRONT_LEG_STANCE_X_SCALE = 0.82
    FRONT_LEG_RESIDUAL_SCALE = 0.72
    FRONT_LEG_RESIDUAL_ALPHA = 0.22
    FRONT_LEG_MAX_STEP = 0.042
    FRONT_LEG_MIN_STEP = 0.030
    FRONT_LEG_REAR_MARGIN = 0.010
    FRONT_CONTACT_X_SHIFT = -0.014
    REAR_CONTACT_X_SHIFT = 0.006

    def __init__(
        self,
        *args,
        enable_camera_observation=True,
        camera_width=64,
        camera_height=48,
        camera_target_distance=1.0,
        camera_stabilize_roll_pitch=True,
        camera_pitch_offset_deg=-10.0,
        **kwargs,
    ):
        kwargs["terrain_profile"] = "rough"
        super().__init__(*args, **kwargs)

        self.enable_camera_observation = bool(enable_camera_observation)
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.camera_target_distance = float(camera_target_distance)
        self.camera_stabilize_roll_pitch = bool(camera_stabilize_roll_pitch)
        self.camera_pitch_offset_deg = float(camera_pitch_offset_deg)
        self.camera = None
        self.prev_foot_world_positions = None
        self.last_terrain_features = None

        if self.enable_camera_observation:
            self.camera = SpotCamera(
                self.base_env,
                width=self.camera_width,
                height=self.camera_height,
                target_distance=self.camera_target_distance,
                stabilize_roll_pitch=self.camera_stabilize_roll_pitch,
                pitch_offset_deg=self.camera_pitch_offset_deg,
            )

        base_obs_dim = int(self.observation_space.shape[0])
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_obs_dim + self.TERRAIN_EXTRA_OBS_DIM,),
            dtype=np.float32,
        )

        self.clearance_height = min(
            self.clearance_height,
            self.DEFAULT_ROUGH_DUAL_IMU_CLEARANCE_HEIGHT,
        )

    def _configure_gait_mode(self):
        super()._configure_gait_mode()
        if self.height_field:
            self.clearance_height = min(
                self.clearance_height,
                self.DEFAULT_ROUGH_DUAL_IMU_CLEARANCE_HEIGHT,
            )
            self.contact_offsets = {
                leg_name: offset.copy()
                for leg_name, offset in self.contact_offsets.items()
            }
            for leg_name in ("FL", "FR"):
                self.contact_offsets[leg_name][0] += self.FRONT_CONTACT_X_SHIFT
            for leg_name in ("BL", "BR"):
                self.contact_offsets[leg_name][0] += self.REAR_CONTACT_X_SHIFT

    def _shape_front_leg_trajectory(self, leg_step_lengths):
        shaped = np.asarray(leg_step_lengths, dtype=np.float32).copy()
        rear_reference = float(np.mean(shaped[2:])) if shaped.shape[0] >= 4 else float(np.mean(shaped))
        front_limit = min(
            float(self.action_space.high[0]),
            max(
                self.FRONT_LEG_MIN_STEP,
                min(
                    self.FRONT_LEG_MAX_STEP,
                    rear_reference + self.FRONT_LEG_REAR_MARGIN,
                ),
            ),
        )
        shaped[0] = np.clip(
            shaped[0] * self.FRONT_LEG_STEP_SCALE,
            float(self.action_space.low[0]),
            front_limit,
        )
        shaped[1] = np.clip(
            shaped[1] * self.FRONT_LEG_STEP_SCALE,
            float(self.action_space.low[0]),
            front_limit,
        )
        return shaped

    def _smooth_stable_command(self, action):
        base_step, step_velocity, leg_residuals = self._split_action(action)
        self.cmd_step += self._stable_step_alpha * (base_step - self.cmd_step)
        self.cmd_vel += self._stable_velocity_alpha * (step_velocity - self.cmd_vel)
        self.leg_step_residuals[:2] += self.FRONT_LEG_RESIDUAL_ALPHA * (
            leg_residuals[:2] - self.leg_step_residuals[:2]
        )
        self.leg_step_residuals[2:] += self._stable_leg_alpha * (
            leg_residuals[2:] - self.leg_step_residuals[2:]
        )
        return self.cmd_step, self.cmd_vel, self.leg_step_residuals.copy()

    def _reshape_front_leg_targets(self, contacts):
        contacts = np.asarray(contacts, dtype=np.float32)
        for leg_index, leg_name in enumerate(("FL", "FR")):
            baseline_x = float(self.T_bf0[leg_name][0, 3])
            generated_x = float(self.T_bf[leg_name][0, 3])
            baseline_z = float(self.T_bf0[leg_name][2, 3])
            generated_z = float(self.T_bf[leg_name][2, 3])
            x_delta = generated_x - baseline_x
            z_delta = generated_z - baseline_z

            if contacts[leg_index] > 0.5:
                self.T_bf[leg_name][0, 3] = baseline_x + (
                    x_delta * self.FRONT_LEG_STANCE_X_SCALE
                )
                if z_delta < 0.0:
                    self.T_bf[leg_name][2, 3] = baseline_z + (
                        z_delta * self.FRONT_LEG_STANCE_EXTENSION_SCALE
                    )
            else:
                self.T_bf[leg_name][0, 3] = baseline_x + (
                    x_delta * self.FRONT_LEG_SWING_X_SCALE
                )
                self.T_bf[leg_name][2, 3] = baseline_z + (
                    max(0.0, z_delta) * self.FRONT_LEG_SWING_HEIGHT_SCALE
                    + self.FRONT_LEG_SWING_LIFT_BONUS
                )

    def _get_foot_world_positions(self):
        pybullet_client = self.base_env._pybullet_client
        quadruped = self.base_env.spot.quadruped
        return np.array(
            [
                pybullet_client.getLinkState(quadruped, foot_id)[0]
                for foot_id in self.base_env.spot._foot_id_list
            ],
            dtype=np.float32,
        )

    def _world_to_body(self, points_world):
        base_pos = np.array(self.base_env.spot.GetBasePosition(), dtype=np.float32)
        rotation = self._get_body_rotation()
        rel_world = np.asarray(points_world, dtype=np.float32) - base_pos
        return (rotation.T @ rel_world.T).T

    def _get_target_foot_positions_body(self):
        return np.array(
            [self.T_bf[leg_name][:3, 3] for leg_name in self.LEG_NAMES],
            dtype=np.float32,
        )

    def _sample_terrain_heights(self, sample_points_world):
        pybullet_client = self.base_env._pybullet_client
        base_z = float(self.base_env.spot.GetBasePosition()[2])
        ray_starts = []
        ray_ends = []
        for point in sample_points_world:
            ray_starts.append(
                [float(point[0]), float(point[1]), base_z + self.TERRAIN_PROBE_START_Z]
            )
            ray_ends.append(
                [float(point[0]), float(point[1]), base_z + self.TERRAIN_PROBE_END_Z]
            )

        heights = []
        for point, hit in zip(sample_points_world, pybullet_client.rayTestBatch(ray_starts, ray_ends)):
            hit_body = hit[0]
            hit_pos = hit[3]
            if hit_body < 0:
                heights.append(float(point[2]))
            else:
                heights.append(float(hit_pos[2]))

        return np.array(heights, dtype=np.float32)

    def _get_camera_summary(self):
        if not self.enable_camera_observation or self.camera is None:
            return np.zeros(self.CAMERA_FEATURE_DIM, dtype=np.float32)

        try:
            _, depth = self.camera.get_rgbd_frame()
        except Exception:
            return np.zeros(self.CAMERA_FEATURE_DIM, dtype=np.float32)

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            return np.zeros(self.CAMERA_FEATURE_DIM, dtype=np.float32)

        h, w = depth.shape
        center_slice = depth[h // 3: (2 * h) // 3, w // 3: (2 * w) // 3]
        lower_center = depth[(2 * h) // 3:, w // 3: (2 * w) // 3]
        upper_center = depth[: h // 3, w // 3: (2 * w) // 3]
        left_lower = depth[h // 2:, : w // 2]
        right_lower = depth[h // 2:, w // 2:]
        return np.array(
            [
                float(np.min(lower_center)),
                float(np.mean(lower_center)),
                float(np.mean(center_slice) - np.mean(upper_center)),
                float(np.mean(left_lower) - np.mean(right_lower)),
            ],
            dtype=np.float32,
        )

    def _get_terrain_features(self):
        foot_world = self._get_foot_world_positions()
        foot_body = self._world_to_body(foot_world)
        target_body = self._get_target_foot_positions_body()

        rotation = self._get_body_rotation()
        forward_world = rotation[:, 0]
        forward_points_world = foot_world + (
            self.TERRAIN_FORWARD_PROBE_DISTANCE * forward_world.reshape(1, 3)
        )

        terrain_heights_current_world = self._sample_terrain_heights(foot_world)
        terrain_heights_forward_world = self._sample_terrain_heights(forward_points_world)
        base_z = float(self.base_env.spot.GetBasePosition()[2])
        terrain_heights_current = terrain_heights_current_world - base_z
        terrain_heights_forward = terrain_heights_forward_world - base_z
        foot_clearance = foot_world[:, 2] - terrain_heights_current_world
        camera_summary = self._get_camera_summary()

        return {
            "foot_world": foot_world,
            "foot_body": foot_body.astype(np.float32),
            "target_body": target_body.astype(np.float32),
            "terrain_heights_current": terrain_heights_current.astype(np.float32),
            "terrain_heights_forward": terrain_heights_forward.astype(np.float32),
            "foot_clearance": foot_clearance.astype(np.float32),
            "camera_summary": camera_summary.astype(np.float32),
        }

    def _command_to_joint_angles(self, action):
        self.bz_step.ramp_up()

        step_length, step_velocity, leg_residuals = self._smooth_stable_command(action)
        base_state = self.base_env.return_state()
        stabilized_yaw_rate = float(
            np.clip(
                -self.base_env.return_yaw() * self.auto_yaw_gain,
                -0.08,
                0.08,
            )
        )
        self.last_stabilized_yaw_rate = stabilized_yaw_rate

        self.bzg.Tswing = 0.20 if step_length > 0.020 else 0.18

        leg_residuals = np.asarray(leg_residuals, dtype=np.float32).copy()
        leg_residuals[0] *= self.FRONT_LEG_RESIDUAL_SCALE
        leg_residuals[1] *= self.FRONT_LEG_RESIDUAL_SCALE

        leg_step_lengths = np.clip(
            step_length + leg_residuals,
            self.action_space.low[0],
            self.action_space.high[0],
        ).astype(np.float32)
        leg_step_lengths = self._shape_front_leg_trajectory(leg_step_lengths)
        self.last_leg_step_lengths = leg_step_lengths

        self.bz_step.StepLength = float(np.mean(leg_step_lengths))
        self.bz_step.LateralFraction = 0.0
        self.bz_step.YawRate = stabilized_yaw_rate
        self.bz_step.StepVelocity = step_velocity
        self.cmd_lat = 0.0
        self.cmd_yaw = 0.0

        contacts = base_state[-4:]
        leg_length_map = {
            leg_name: leg_step_lengths[index]
            for index, leg_name in enumerate(self.LEG_NAMES)
        }
        self.T_bf = self.bzg.GenerateTrajectory(
            leg_length_map,
            0.0,
            stabilized_yaw_rate,
            step_velocity,
            self.T_bf0,
            self.T_bf,
            clearance_height=self.clearance_height,
            penetration_depth=self.penetration_depth,
            contacts=contacts,
        )

        self._reshape_front_leg_targets(contacts)

        pos, orn = self._compose_body_pose(base_state)
        joint_angles = self.spot_model.IK(orn, pos, self.T_bf)
        self.base_env.spot.GetExternalObservations(self.bzg, self.bz_step)
        return joint_angles.reshape(-1)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.prev_foot_world_positions = self._get_foot_world_positions()
        self.last_terrain_features = self._get_terrain_features()
        obs = self._get_ml_observation()
        info.update(
            {
                "walk_variant": "rough_dual_imu_stable_v1",
                "terrain_feature_mode": "rough_dual_imu",
                "camera_observation_enabled": bool(self.enable_camera_observation),
                "camera_roll_pitch_stabilized": bool(self.camera_stabilize_roll_pitch),
            }
        )
        return obs, info

    def _get_ml_observation(self, base_obs=None):
        base_obs = super()._get_ml_observation(base_obs)
        terrain = self._get_terrain_features()
        self.last_terrain_features = terrain
        extra_obs = np.concatenate(
            [
                terrain["foot_body"].reshape(-1),
                terrain["target_body"].reshape(-1),
                terrain["terrain_heights_current"],
                terrain["terrain_heights_forward"],
                terrain["foot_clearance"],
                terrain["camera_summary"],
            ]
        ).astype(np.float32)
        return np.concatenate([base_obs, extra_obs]).astype(np.float32)

    def _compute_reward(self, action, done=False):
        reward = super()._compute_reward(action, done=done)

        terrain = self.last_terrain_features or self._get_terrain_features()
        foot_world = terrain["foot_world"]
        foot_body = terrain["foot_body"]
        target_body = terrain["target_body"]
        foot_clearance = terrain["foot_clearance"]
        contacts = np.asarray(self.base_env.return_state()[-4:], dtype=np.float32)
        stance_mask = contacts > 0.5
        swing_mask = np.logical_not(stance_mask)

        slip_penalty = 0.0
        if self.prev_foot_world_positions is not None:
            stance_xy_delta = foot_world[:, :2] - self.prev_foot_world_positions[:, :2]
            slip_penalty = float(
                np.sum(np.linalg.norm(stance_xy_delta, axis=1) * stance_mask.astype(np.float32))
            )

        foothold_tracking_penalty = float(
            np.mean(np.linalg.norm(foot_body - target_body, axis=1))
        )

        clearance_tracking_reward = 0.0
        excess_clearance_penalty = 0.0
        if np.any(swing_mask):
            desired_clearance = np.maximum(
                0.016,
                terrain["terrain_heights_forward"][swing_mask] - terrain["terrain_heights_current"][swing_mask] + 0.012,
            )
            swing_leg_indices = np.flatnonzero(swing_mask)
            front_swing_mask = swing_leg_indices < 2
            if np.any(front_swing_mask):
                desired_clearance = desired_clearance.copy()
                desired_clearance[front_swing_mask] = np.minimum(
                    desired_clearance[front_swing_mask],
                    0.036,
                )
            swing_clearance = foot_clearance[swing_mask]
            clearance_tracking_reward = float(
                np.mean(
                    np.exp(
                        -((swing_clearance - desired_clearance) ** 2) / 0.00008
                    )
                )
            )
            excess_clearance_penalty = float(
                np.mean(
                    np.clip(
                        swing_clearance - (desired_clearance + 0.012),
                        0.0,
                        0.04,
                    )
                )
            )

        terrain_preview_bonus = float(
            np.mean(
                np.clip(
                    terrain["terrain_heights_forward"] - terrain["terrain_heights_current"],
                    0.0,
                    0.05,
                )
            )
        )

        reward += 0.18 * clearance_tracking_reward
        reward += 0.03 * terrain_preview_bonus
        reward -= 0.50 * excess_clearance_penalty
        reward -= 1.30 * slip_penalty
        reward -= 0.10 * foothold_tracking_penalty

        self.prev_foot_world_positions = foot_world.copy()
        return float(reward)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        terrain = self.last_terrain_features or self._get_terrain_features()
        info.update(
            {
                "walk_variant": "rough_dual_imu_stable_v1",
                "terrain_feature_mode": "rough_dual_imu",
                "camera_observation_enabled": bool(self.enable_camera_observation),
                "camera_roll_pitch_stabilized": bool(self.camera_stabilize_roll_pitch),
                "foot_world_positions": terrain["foot_world"].tolist(),
                "foot_body_positions": terrain["foot_body"].tolist(),
                "target_foot_positions_body": terrain["target_body"].tolist(),
                "terrain_heights_current": terrain["terrain_heights_current"].tolist(),
                "terrain_heights_forward": terrain["terrain_heights_forward"].tolist(),
                "foot_clearance": terrain["foot_clearance"].tolist(),
                "camera_summary": terrain["camera_summary"].tolist(),
                "camera_pitch_offset_deg": float(self.camera_pitch_offset_deg),
            }
        )
        return obs, reward, terminated, truncated, info


class SpotMLRoughTeacherJointResidualEnv(SpotMLRoughDualIMUStableWalkEnv):
    """Rough-terrain env driven by a longwalk teacher plus bounded joint residuals.

    The teacher policy supplies the baseline dual-IMU Bezier gait command that was
    already successful on flat ground. PPO only learns a bounded residual for each
    joint so it can adapt footholds and posture on rough terrain without having to
    rediscover basic forward walking from scratch.
    """

    JOINT_DIM = 12
    EXTRA_RESIDUAL_OBS_DIM = (JOINT_DIM * 5) + SpotMLDualIMUStableWalkEnv.ACTION_DIM
    DEFAULT_JOINT_RESIDUAL_LIMIT = 0.26
    DEFAULT_JOINT_RESIDUAL_ALPHA = 0.22
    DEFAULT_JOINT_TARGET_STEP_LIMIT = 0.24
    DEFAULT_SETTLE_STEPS = 4
    DEFAULT_TEACHER_CLEARANCE_HEIGHT = 0.040
    DEFAULT_TEACHER_STEP_ALPHA = 0.20
    DEFAULT_TEACHER_VELOCITY_ALPHA = 0.18
    DEFAULT_TEACHER_LEG_ALPHA = 0.18

    def __init__(
        self,
        *args,
        teacher_model_path,
        teacher_vecnormalize_path=None,
        joint_residual_limit=DEFAULT_JOINT_RESIDUAL_LIMIT,
        joint_residual_alpha=DEFAULT_JOINT_RESIDUAL_ALPHA,
        joint_target_step_limit=DEFAULT_JOINT_TARGET_STEP_LIMIT,
        settle_steps=DEFAULT_SETTLE_STEPS,
        **kwargs,
    ):
        if not teacher_model_path:
            raise ValueError("SpotMLRoughTeacherJointResidualEnv requires teacher_model_path.")

        kwargs["terrain_profile"] = "rough"
        super().__init__(*args, **kwargs)

        self.teacher_model_path = str(teacher_model_path)
        self.teacher_vecnormalize_path = (
            None if teacher_vecnormalize_path is None else str(teacher_vecnormalize_path)
        )
        self.joint_residual_limit = float(joint_residual_limit)
        self.joint_residual_alpha = float(joint_residual_alpha)
        self.joint_target_step_limit = float(joint_target_step_limit)
        self.settle_steps = int(settle_steps)
        self.clearance_height = max(
            self.clearance_height,
            self.DEFAULT_TEACHER_CLEARANCE_HEIGHT,
        )
        self._stable_step_alpha = max(self._stable_step_alpha, self.DEFAULT_TEACHER_STEP_ALPHA)
        self._stable_velocity_alpha = max(
            self._stable_velocity_alpha,
            self.DEFAULT_TEACHER_VELOCITY_ALPHA,
        )
        self._stable_leg_alpha = max(self._stable_leg_alpha, self.DEFAULT_TEACHER_LEG_ALPHA)

        self.teacher_action_low = self.action_space.low.astype(np.float32).copy()
        self.teacher_action_high = self.action_space.high.astype(np.float32).copy()

        self.action_space = spaces.Box(
            low=np.full(self.JOINT_DIM, -self.joint_residual_limit, dtype=np.float32),
            high=np.full(self.JOINT_DIM, self.joint_residual_limit, dtype=np.float32),
            dtype=np.float32,
        )

        base_obs_dim = int(self.observation_space.shape[0])
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_obs_dim + self.EXTRA_RESIDUAL_OBS_DIM,),
            dtype=np.float32,
        )

        self.teacher_model = None
        self.teacher_obs_mean = None
        self.teacher_obs_var = None
        self.teacher_clip_obs = 10.0
        self.teacher_epsilon = 1e-8
        self.teacher_uses_orientation_features = False

        self.prev_joint_residual_action = np.zeros(self.JOINT_DIM, dtype=np.float32)
        self.joint_residuals = np.zeros(self.JOINT_DIM, dtype=np.float32)
        self.last_teacher_action = np.zeros(
            SpotMLDualIMUStableWalkEnv.ACTION_DIM,
            dtype=np.float32,
        )
        self.last_teacher_joint_targets = np.zeros(self.JOINT_DIM, dtype=np.float32)
        self.last_commanded_joint_targets = np.zeros(self.JOINT_DIM, dtype=np.float32)
        self.last_motor_angles = np.zeros(self.JOINT_DIM, dtype=np.float32)
        self.last_motor_velocities = np.zeros(self.JOINT_DIM, dtype=np.float32)
        self.last_joint_tracking_error = np.zeros(self.JOINT_DIM, dtype=np.float32)

        self._load_teacher_controller()

    def _configure_gait_mode(self):
        # Start from the plain dual-IMU controller so the teacher's gait logic
        # stays intact on rough terrain without inheriting the extra front-leg
        # shaping experiments from SpotMLRoughDualIMUStableWalkEnv.
        SpotMLDualIMUStableWalkEnv._configure_gait_mode(self)

    def _load_teacher_controller(self):
        import pickle
        from stable_baselines3 import PPO

        self.teacher_model = PPO.load(self.teacher_model_path, device="auto")
        teacher_obs_dim = int(np.prod(self.teacher_model.observation_space.shape))
        base_teacher_obs_dim = SpotMLDualIMUStableWalkEnv._get_ml_observation(
            self,
            self.base_env.return_state(),
            include_orientation_features=False,
        ).shape[0]
        yaw_teacher_obs_dim = SpotMLDualIMUStableWalkEnv._get_ml_observation(
            self,
            self.base_env.return_state(),
            include_orientation_features=True,
        ).shape[0]
        self.teacher_uses_orientation_features = teacher_obs_dim == yaw_teacher_obs_dim

        if self.teacher_vecnormalize_path is None:
            return

        try:
            with open(self.teacher_vecnormalize_path, "rb") as file_handler:
                vecnormalize = pickle.load(file_handler)
        except Exception:
            return

        obs_rms = getattr(vecnormalize, "obs_rms", None)
        if obs_rms is None:
            return

        self.teacher_obs_mean = np.asarray(obs_rms.mean, dtype=np.float32)
        self.teacher_obs_var = np.asarray(obs_rms.var, dtype=np.float32)
        self.teacher_clip_obs = float(getattr(vecnormalize, "clip_obs", 10.0))
        self.teacher_epsilon = float(getattr(vecnormalize, "epsilon", 1e-8))
        self.teacher_uses_orientation_features = teacher_obs_dim == yaw_teacher_obs_dim
        if teacher_obs_dim not in (base_teacher_obs_dim, yaw_teacher_obs_dim):
            self.teacher_uses_orientation_features = False

    def _normalize_teacher_obs(self, teacher_obs):
        teacher_obs = np.asarray(teacher_obs, dtype=np.float32)
        if self.teacher_obs_mean is None or self.teacher_obs_var is None:
            return teacher_obs

        normalized = (teacher_obs - self.teacher_obs_mean) / np.sqrt(
            self.teacher_obs_var + self.teacher_epsilon
        )
        return np.clip(normalized, -self.teacher_clip_obs, self.teacher_clip_obs).astype(
            np.float32
        )

    def _predict_teacher_action(self, base_obs=None):
        teacher_obs = SpotMLDualIMUStableWalkEnv._get_ml_observation(
            self,
            base_obs,
            include_orientation_features=self.teacher_uses_orientation_features,
        )
        normalized_obs = self._normalize_teacher_obs(teacher_obs)
        teacher_action, _ = self.teacher_model.predict(normalized_obs, deterministic=True)
        teacher_action = np.asarray(teacher_action, dtype=np.float32).reshape(-1)
        teacher_action = np.clip(teacher_action, self.teacher_action_low, self.teacher_action_high)
        self.last_teacher_action = teacher_action.astype(np.float32)
        return self.last_teacher_action.copy()

    def _compute_teacher_joint_targets(self, teacher_action):
        teacher_action = np.asarray(teacher_action, dtype=np.float32)

        self.bz_step.ramp_up()

        base_step = float(
            np.clip(teacher_action[0], self.teacher_action_low[0], self.teacher_action_high[0])
        )
        step_velocity = float(
            np.clip(teacher_action[1], self.teacher_action_low[1], self.teacher_action_high[1])
        )
        leg_residuals = np.asarray(teacher_action[2: 2 + self.LEG_RESIDUAL_DIM], dtype=np.float32)

        self.cmd_step += self._stable_step_alpha * (base_step - self.cmd_step)
        self.cmd_vel += self._stable_velocity_alpha * (step_velocity - self.cmd_vel)
        self.leg_step_residuals += self._stable_leg_alpha * (
            leg_residuals - self.leg_step_residuals
        )

        base_state = self.base_env.return_state()
        stabilized_yaw_rate = float(
            np.clip(
                -self.base_env.return_yaw() * self.auto_yaw_gain,
                -0.08,
                0.08,
            )
        )
        self.last_stabilized_yaw_rate = stabilized_yaw_rate

        if self.height_field:
            self.bzg.Tswing = 0.20 if self.cmd_step > 0.020 else 0.18
        else:
            self.bzg.Tswing = 0.18 if self.cmd_step > 0.020 else 0.16

        leg_step_lengths = np.clip(
            self.cmd_step + self.leg_step_residuals,
            self.teacher_action_low[0],
            self.teacher_action_high[0],
        ).astype(np.float32)
        self.last_leg_step_lengths = leg_step_lengths

        self.bz_step.StepLength = float(np.mean(leg_step_lengths))
        self.bz_step.LateralFraction = 0.0
        self.bz_step.YawRate = stabilized_yaw_rate
        self.bz_step.StepVelocity = step_velocity
        self.cmd_lat = 0.0
        self.cmd_yaw = 0.0

        contacts = base_state[-4:]
        leg_length_map = {
            leg_name: leg_step_lengths[index]
            for index, leg_name in enumerate(self.LEG_NAMES)
        }
        self.T_bf = self.bzg.GenerateTrajectory(
            leg_length_map,
            0.0,
            stabilized_yaw_rate,
            step_velocity,
            self.T_bf0,
            self.T_bf,
            clearance_height=self.clearance_height,
            penetration_depth=self.penetration_depth,
            contacts=contacts,
        )

        pos, orn = self._compose_body_pose(base_state)
        joint_angles = self.spot_model.IK(orn, pos, self.T_bf).reshape(-1).astype(np.float32)
        self.base_env.spot.GetExternalObservations(self.bzg, self.bz_step)
        self.last_teacher_joint_targets = joint_angles
        return self.last_teacher_joint_targets.copy()

    def _command_to_joint_angles(self, action):
        residual_action = np.asarray(action, dtype=np.float32)
        residual_action = np.clip(residual_action, self.action_space.low, self.action_space.high)
        self.joint_residuals += self.joint_residual_alpha * (
            residual_action - self.joint_residuals
        )

        base_state = self.base_env.return_state()
        teacher_action = self._predict_teacher_action(base_state)
        teacher_joint_targets = self._compute_teacher_joint_targets(teacher_action)

        current_motor_angles = np.asarray(
            self.base_env.spot.GetMotorAngles(),
            dtype=np.float32,
        )
        current_motor_velocities = np.asarray(
            self.base_env.spot.GetMotorVelocities(),
            dtype=np.float32,
        )

        desired_joint_targets = teacher_joint_targets + self.joint_residuals
        limited_delta = np.clip(
            desired_joint_targets - current_motor_angles,
            -self.joint_target_step_limit,
            self.joint_target_step_limit,
        )
        final_joint_targets = current_motor_angles + limited_delta

        self.last_motor_angles = current_motor_angles
        self.last_motor_velocities = current_motor_velocities
        self.last_commanded_joint_targets = final_joint_targets.astype(np.float32)
        self.last_joint_tracking_error = current_motor_angles - self.last_commanded_joint_targets
        return final_joint_targets.astype(np.float32)

    def _get_ml_observation(self, base_obs=None):
        base_obs = SpotMLRoughDualIMUStableWalkEnv._get_ml_observation(self, base_obs)
        extra_obs = np.concatenate(
            [
                self.last_motor_angles,
                self.last_motor_velocities,
                self.last_teacher_action,
                self.last_teacher_joint_targets,
                self.prev_joint_residual_action,
                self.last_joint_tracking_error,
            ]
        ).astype(np.float32)
        return np.concatenate([base_obs, extra_obs]).astype(np.float32)

    def _compute_reward(self, action, done=False):
        reward = SpotMLRoughDualIMUStableWalkEnv._compute_reward(
            self,
            self.last_teacher_action,
            done=done,
        )

        residual_action = np.asarray(action, dtype=np.float32)
        residual_mag_penalty = 0.04 * float(np.mean(np.abs(self.joint_residuals)))
        residual_delta_penalty = 0.03 * float(
            np.mean(np.abs(residual_action - self.prev_joint_residual_action))
        )
        joint_tracking_penalty = 0.012 * float(np.mean(np.abs(self.last_joint_tracking_error)))
        motor_velocity_penalty = 0.001 * float(np.mean(np.abs(self.last_motor_velocities)))
        forward_commit_bonus = 1.25 * max(0.0, self.last_forward_speed)

        reward += forward_commit_bonus
        reward -= residual_mag_penalty
        reward -= residual_delta_penalty
        reward -= joint_tracking_penalty
        reward -= motor_velocity_penalty
        return float(reward)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        self.prev_joint_residual_action = np.zeros(self.JOINT_DIM, dtype=np.float32)
        self.joint_residuals = np.zeros(self.JOINT_DIM, dtype=np.float32)
        self.last_teacher_action = np.zeros(
            SpotMLDualIMUStableWalkEnv.ACTION_DIM,
            dtype=np.float32,
        )
        self.last_teacher_joint_targets = np.zeros(self.JOINT_DIM, dtype=np.float32)
        self.last_commanded_joint_targets = np.zeros(self.JOINT_DIM, dtype=np.float32)
        self.last_motor_angles = np.asarray(self.base_env.spot.GetMotorAngles(), dtype=np.float32)
        self.last_motor_velocities = np.asarray(
            self.base_env.spot.GetMotorVelocities(),
            dtype=np.float32,
        )
        self.last_joint_tracking_error = np.zeros(self.JOINT_DIM, dtype=np.float32)

        base_obs = self.base_env.return_state()
        teacher_action = self._predict_teacher_action(base_obs)
        teacher_joint_targets = self._compute_teacher_joint_targets(teacher_action)

        # Let the teacher settle the robot onto the terrain before PPO starts
        # adding residuals so each episode begins from a controllable stance.
        for _ in range(max(0, self.settle_steps)):
            self.base_env.pass_joint_angles(teacher_joint_targets)
            base_obs, _, done, _ = self.base_env.step(np.zeros(4, dtype=np.float32))
            if done:
                break
            teacher_action = self._predict_teacher_action(base_obs)
            teacher_joint_targets = self._compute_teacher_joint_targets(teacher_action)

        current_base_pos = np.array(self.base_env.spot.GetBasePosition(), dtype=np.float32)
        self.prev_base_pos = current_base_pos.copy()
        self.episode_start_pos = current_base_pos.copy()
        self.nominal_base_height = float(current_base_pos[2])
        self.episode_step_count = 0
        self.prev_foot_world_positions = self._get_foot_world_positions()
        self.last_terrain_features = self._get_terrain_features()
        self.last_motor_angles = np.asarray(self.base_env.spot.GetMotorAngles(), dtype=np.float32)
        self.last_motor_velocities = np.asarray(
            self.base_env.spot.GetMotorVelocities(),
            dtype=np.float32,
        )
        self.last_commanded_joint_targets = self.last_teacher_joint_targets.copy()
        self.last_joint_tracking_error = self.last_motor_angles - self.last_teacher_joint_targets

        obs = self._get_ml_observation(base_obs)
        info.update(
            {
                "walk_variant": "rough_teacher_joint_residual_v1",
                "terrain_feature_mode": "rough_teacher_joint_residual",
                "terrain_profile": self.terrain_profile,
                "gait_mode": self.gait_mode,
                "teacher_model_path": self.teacher_model_path,
                "teacher_vecnormalize_path": self.teacher_vecnormalize_path,
                "joint_residual_limit": float(self.joint_residual_limit),
                "forward_speed_target": self.target_forward_speed,
                "camera_observation_enabled": bool(self.enable_camera_observation),
                "camera_roll_pitch_stabilized": bool(self.camera_stabilize_roll_pitch),
                "camera_pitch_offset_deg": float(self.camera_pitch_offset_deg),
            }
        )
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        joint_angles = self._command_to_joint_angles(action)
        self.base_env.pass_joint_angles(joint_angles)
        base_obs, _, done, info = self.base_env.step(np.zeros(4, dtype=np.float32))

        ml_obs = self._get_ml_observation(base_obs)
        reward = self._compute_reward(action, done=done)
        current_base_pos = np.array(self.base_env.spot.GetBasePosition(), dtype=np.float32)
        self.episode_step_count += 1
        self.prev_joint_residual_action = action.copy()
        self.prev_action = self.last_teacher_action.copy()

        terminated = done
        truncated = self.episode_step_count >= self.max_episode_steps and not terminated

        info.update(
            {
                "walk_variant": "rough_teacher_joint_residual_v1",
                "terrain_feature_mode": "rough_teacher_joint_residual",
                "terrain_profile": self.terrain_profile,
                "gait_mode": self.gait_mode,
                "teacher_action": self.last_teacher_action.tolist(),
                "teacher_joint_targets": self.last_teacher_joint_targets.tolist(),
                "commanded_joint_targets": self.last_commanded_joint_targets.tolist(),
                "joint_residual_action": self.prev_joint_residual_action.tolist(),
                "joint_residual_smoothed": self.joint_residuals.tolist(),
                "joint_tracking_error": self.last_joint_tracking_error.tolist(),
                "forward_speed": float(self.last_forward_speed),
                "forward_speed_target": self.target_forward_speed,
                "forward_displacement": float(current_base_pos[0] - self.episode_start_pos[0]),
                "base_height_error": float(self._get_base_height_error()),
                "episode_step_count": int(self.episode_step_count),
                "time_limit_reached": bool(truncated),
                "camera_observation_enabled": bool(self.enable_camera_observation),
                "camera_roll_pitch_stabilized": bool(self.camera_stabilize_roll_pitch),
                "camera_pitch_offset_deg": float(self.camera_pitch_offset_deg),
            }
        )
        return ml_obs, reward, terminated, truncated, info
