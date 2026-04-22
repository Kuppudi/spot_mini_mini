import numpy as np
from gymnasium import spaces

from spot_ml import SpotMLWalkEnv
from spot_rough_terrain_ml import SpotMLRoughTerrainEnv


class SpotMLRoughTerrainHeightResidualEnv(SpotMLRoughTerrainEnv):
    """Rough-terrain env with per-leg swing-height residual control.

    This variant keeps the existing global crawl controller, but adds
    four leg-specific residual actions that only affect feet during
    swing. That gives the policy a way to adapt individual swing arcs
    over bumps without rewriting the whole gait scheduler.
    """

    BASE_ACTION_DIM = 4
    LEG_HEIGHT_RESIDUAL_DIM = 4
    EXTRA_RESIDUAL_OBS_DIM = 12

    DEFAULT_LEARNED_RESIDUAL_MAX = 0.035
    DEFAULT_AUTO_CLEARANCE_GAIN = 1.35
    DEFAULT_AUTO_CLEARANCE_BIAS = 0.006
    DEFAULT_AUTO_CLEARANCE_MAX = 0.040
    DEFAULT_SWING_CLEARANCE_TARGET = 0.020

    def __init__(
        self,
        *args,
        learned_leg_height_residual_max=DEFAULT_LEARNED_RESIDUAL_MAX,
        auto_clearance_gain=DEFAULT_AUTO_CLEARANCE_GAIN,
        auto_clearance_bias=DEFAULT_AUTO_CLEARANCE_BIAS,
        auto_clearance_max=DEFAULT_AUTO_CLEARANCE_MAX,
        swing_clearance_target=DEFAULT_SWING_CLEARANCE_TARGET,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.learned_leg_height_residual_max = float(learned_leg_height_residual_max)
        self.auto_clearance_gain = float(auto_clearance_gain)
        self.auto_clearance_bias = float(auto_clearance_bias)
        self.auto_clearance_max = float(auto_clearance_max)
        self.swing_clearance_target = float(swing_clearance_target)

        base_low = self.action_space.low.astype(np.float32)
        base_high = self.action_space.high.astype(np.float32)
        residual_low = np.zeros(self.LEG_HEIGHT_RESIDUAL_DIM, dtype=np.float32)
        residual_high = np.full(
            self.LEG_HEIGHT_RESIDUAL_DIM,
            self.learned_leg_height_residual_max,
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.concatenate([base_low, residual_low]),
            high=np.concatenate([base_high, residual_high]),
            dtype=np.float32,
        )

        base_obs_dim = int(self.observation_space.shape[0])
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_obs_dim + self.EXTRA_RESIDUAL_OBS_DIM,),
            dtype=np.float32,
        )

        self.prev_action = np.zeros(self.BASE_ACTION_DIM, dtype=np.float32)
        self.prev_leg_height_residual_action = np.zeros(
            self.LEG_HEIGHT_RESIDUAL_DIM,
            dtype=np.float32,
        )
        self.last_leg_height_residuals = np.zeros(
            self.LEG_HEIGHT_RESIDUAL_DIM,
            dtype=np.float32,
        )
        self.last_leg_auto_clearance = np.zeros(
            self.LEG_HEIGHT_RESIDUAL_DIM,
            dtype=np.float32,
        )
        self.last_leg_swing_mask = np.zeros(self.LEG_HEIGHT_RESIDUAL_DIM, dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.prev_action = np.zeros(self.BASE_ACTION_DIM, dtype=np.float32)
        self.prev_leg_height_residual_action = np.zeros(
            self.LEG_HEIGHT_RESIDUAL_DIM,
            dtype=np.float32,
        )
        self.last_leg_height_residuals = np.zeros(
            self.LEG_HEIGHT_RESIDUAL_DIM,
            dtype=np.float32,
        )
        self.last_leg_auto_clearance = np.zeros(
            self.LEG_HEIGHT_RESIDUAL_DIM,
            dtype=np.float32,
        )
        self.last_leg_swing_mask = np.zeros(self.LEG_HEIGHT_RESIDUAL_DIM, dtype=np.float32)
        obs = self._get_ml_observation()
        info.update(
            {
                "terrain_feature_mode": "foothold_height_residuals",
                "rough_env_variant": "foothold_height_residual_v3",
            }
        )
        return obs, info

    def _split_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        base_action = action[: self.BASE_ACTION_DIM]
        leg_height_action = action[
            self.BASE_ACTION_DIM: self.BASE_ACTION_DIM + self.LEG_HEIGHT_RESIDUAL_DIM
        ]
        leg_height_action = np.clip(
            leg_height_action,
            0.0,
            self.learned_leg_height_residual_max,
        ).astype(np.float32)
        return base_action, leg_height_action

    def _get_swing_mask(self):
        phases = np.asarray(getattr(self.bzg, "Phases", []), dtype=np.float32)
        if phases.shape != (self.LEG_HEIGHT_RESIDUAL_DIM,):
            phases = np.asarray(
                getattr(self.base_env.spot, "LegPhases", np.zeros(4, dtype=np.float32)),
                dtype=np.float32,
            )
        return (phases > 1.0).astype(np.float32)

    def _get_auto_clearance_targets(self, terrain=None):
        terrain = terrain or self.last_terrain_features or self._get_terrain_features()
        terrain_current = np.asarray(terrain["terrain_heights_current"], dtype=np.float32)
        terrain_forward = np.asarray(terrain["terrain_heights_forward"], dtype=np.float32)
        bump_delta = np.maximum(0.0, terrain_forward - terrain_current)
        auto_clearance = self.auto_clearance_bias + self.auto_clearance_gain * bump_delta
        return np.clip(auto_clearance, 0.0, self.auto_clearance_max).astype(np.float32)

    def _apply_leg_height_residuals(self, leg_height_action, terrain=None):
        swing_mask = self._get_swing_mask()
        auto_clearance = self._get_auto_clearance_targets(terrain)
        total_residual = np.clip(
            (auto_clearance + leg_height_action) * swing_mask,
            0.0,
            self.auto_clearance_max + self.learned_leg_height_residual_max,
        ).astype(np.float32)

        for leg_index, leg_name in enumerate(self.LEG_NAMES):
            self.T_bf[leg_name][2, 3] += float(total_residual[leg_index])

        self.last_leg_height_residuals = total_residual
        self.last_leg_auto_clearance = auto_clearance
        self.last_leg_swing_mask = swing_mask
        return total_residual

    def _command_to_joint_angles(self, action):
        self.bz_step.ramp_up()

        base_action, leg_height_action = self._split_action(action)
        step_length, lateral_fraction, yaw_rate, step_velocity = self._smooth_command(base_action)
        base_state = self.base_env.return_state()
        stabilized_yaw_rate = yaw_rate
        if abs(self.auto_yaw_gain) > 1e-8:
            stabilized_yaw_rate += -self.base_env.return_yaw() * self.auto_yaw_gain
        stabilized_yaw_rate = float(
            np.clip(
                stabilized_yaw_rate,
                self.action_space.low[2],
                self.action_space.high[2],
            )
        )
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

        self.bz_step.StepLength = step_length
        self.bz_step.LateralFraction = lateral_fraction
        self.bz_step.YawRate = stabilized_yaw_rate
        self.bz_step.StepVelocity = step_velocity

        contacts = base_state[-4:]
        self.T_bf = self.bzg.GenerateTrajectory(
            step_length,
            lateral_fraction,
            stabilized_yaw_rate,
            step_velocity,
            self.T_bf0,
            self.T_bf,
            clearance_height=self.clearance_height,
            penetration_depth=self.penetration_depth,
            contacts=contacts,
        )

        self._apply_leg_height_residuals(leg_height_action, self.last_terrain_features)

        pos, orn = self._compose_body_pose(base_state)
        joint_angles = self.spot_model.IK(orn, pos, self.T_bf)
        self.base_env.spot.GetExternalObservations(self.bzg, self.bz_step)
        return joint_angles.reshape(-1)

    def _get_ml_observation(self, base_obs=None):
        base_obs = super()._get_ml_observation(base_obs)
        residual_obs = np.concatenate(
            [
                self.last_leg_auto_clearance,
                self.prev_leg_height_residual_action,
                self.last_leg_swing_mask,
            ]
        ).astype(np.float32)
        return np.concatenate([base_obs, residual_obs]).astype(np.float32)

    def _compute_reward(self, done=False):
        reward = SpotMLWalkEnv._compute_reward(self, done=done)

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

        stance_tracking_penalty = 0.0
        if np.any(stance_mask):
            stance_tracking_penalty = float(
                np.mean(
                    np.linalg.norm(foot_body[stance_mask] - target_body[stance_mask], axis=1)
                )
            )

        adaptive_clearance_bonus = 0.0
        if np.any(swing_mask):
            desired_clearance = np.maximum(
                self.swing_clearance_target,
                self.last_leg_auto_clearance[swing_mask] + 0.010,
            )
            adaptive_clearance_bonus = float(
                np.mean(
                    np.clip(
                        foot_clearance[swing_mask] - desired_clearance,
                        0.0,
                        0.05,
                    )
                )
            )

        base_state = np.asarray(self.base_env.return_state(), dtype=np.float32)
        body_level_bonus = float(
            np.exp(-6.0 * (abs(float(base_state[0])) + abs(float(base_state[1]))))
        )

        reward += 0.60 * adaptive_clearance_bonus
        reward += 0.12 * body_level_bonus
        reward -= 1.10 * slip_penalty
        reward -= 0.05 * stance_tracking_penalty

        self.prev_foot_world_positions = foot_world.copy()
        return float(reward)

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        base_action, leg_height_action = self._split_action(action)

        joint_angles = self._command_to_joint_angles(action)
        self.base_env.pass_joint_angles(joint_angles)

        base_obs, _, done, info = self.base_env.step(np.zeros(4, dtype=np.float32))

        self.prev_action = base_action.copy()
        self.prev_leg_height_residual_action = leg_height_action.copy()
        ml_obs = self._get_ml_observation(base_obs)
        reward = self._compute_reward(done=done)
        base_height_error = self._get_base_height_error()
        current_base_pos = np.array(self.base_env.spot.GetBasePosition(), dtype=np.float32)
        self.episode_step_count += 1

        terminated = done
        truncated = self.episode_step_count >= self.max_episode_steps and not terminated

        terrain = self.last_terrain_features or self._get_terrain_features()
        info.update(
            {
                "forward_speed_target": self.target_forward_speed,
                "gait_mode": self.gait_mode,
                "terrain_profile": self.terrain_profile,
                "height_field": bool(self.height_field),
                "terrain_randomization": bool(self.terrain_randomization),
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
                },
                "terrain_feature_mode": "foothold_height_residuals",
                "rough_env_variant": "foothold_height_residual_v3",
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
                "leg_auto_clearance": self.last_leg_auto_clearance.tolist(),
                "leg_height_residual_action": self.prev_leg_height_residual_action.tolist(),
                "leg_total_height_residual": self.last_leg_height_residuals.tolist(),
                "leg_swing_mask": self.last_leg_swing_mask.tolist(),
            }
        )
        return ml_obs, reward, terminated, truncated, info
