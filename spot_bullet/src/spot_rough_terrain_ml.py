import numpy as np
from gymnasium import spaces

from camera_sensor import SpotCamera
from spot_ml import SpotMLWalkEnv


class SpotMLRoughTerrainEnv(SpotMLWalkEnv):
    """Rough-terrain specialization of the walk env.

    This keeps the global gait controller from SpotMLWalkEnv, but adds
    terrain-aware observations and reward hooks so we can iterate on
    foothold reasoning, terrain traversal, and later camera-driven path
    selection without overloading the flat-ground walk module.
    """

    LEG_NAMES = ("FL", "FR", "BL", "BR")
    TERRAIN_FORWARD_PROBE_DISTANCE = 0.08
    TERRAIN_PROBE_START_Z = 0.30
    TERRAIN_PROBE_END_Z = -0.35
    CAMERA_FEATURE_DIM = 4

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
        extra_dim = 12 + 12 + 4 + 4 + 4 + self.CAMERA_FEATURE_DIM
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_obs_dim + extra_dim,),
            dtype=np.float32,
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

    def _get_body_rotation(self):
        base_orn = self.base_env.spot.GetBaseOrientation()
        pybullet_client = self.base_env._pybullet_client
        return np.array(
            pybullet_client.getMatrixFromQuaternion(base_orn),
            dtype=np.float32,
        ).reshape(3, 3)

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

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.prev_foot_world_positions = self._get_foot_world_positions()
        self.last_terrain_features = self._get_terrain_features()
        obs = self._get_ml_observation()
        info.update(
            {
                "terrain_profile": "rough",
                "terrain_feature_mode": "foothold_aware",
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

    def _compute_reward(self, done=False):
        reward = super()._compute_reward(done=done)

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

        swing_clearance_bonus = 0.0
        if np.any(swing_mask):
            swing_clearance_bonus = float(
                np.mean(np.clip(foot_clearance[swing_mask] - 0.015, 0.0, 0.05))
            )

        reward += 0.30 * swing_clearance_bonus
        reward -= 1.75 * slip_penalty
        reward -= 0.12 * foothold_tracking_penalty

        self.prev_foot_world_positions = foot_world.copy()
        return float(reward)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        terrain = self.last_terrain_features or self._get_terrain_features()
        info.update(
            {
                "terrain_feature_mode": "foothold_aware",
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
