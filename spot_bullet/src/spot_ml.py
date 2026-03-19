import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from spotmicro.GymEnvs.spot_bezier_env import spotBezierEnv
from spotmicro.Kinematics.SpotKinematics import SpotModel
from spotmicro.GaitGenerator.Bezier import BezierGait
from spotmicro.OpenLoopSM.SpotOL import BezierStepper


class SpotMLWalkEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 render=False,
                 on_rack=False,
                 height_field=False,
                 draw_foot_path=False,
                 env_randomizer=None,
                 target_forward_speed=0.35):

        super().__init__()

        self.base_env = spotBezierEnv(
            render=render,
            on_rack=on_rack,
            height_field=height_field,
            draw_foot_path=draw_foot_path,
            env_randomizer=env_randomizer
        )

        self.target_forward_speed = target_forward_speed

        self.spot_model = SpotModel()
        self.T_bf0 = copy.deepcopy(self.spot_model.WorldToFoot)
        self.T_bf = copy.deepcopy(self.T_bf0)
        self.bzg = BezierGait(dt=self.base_env._time_step)
        self.bz_step = BezierStepper(dt=self.base_env._time_step, mode=0)

        self.cmd_step = 0.0
        self.cmd_lat = 0.0
        self.cmd_yaw = 0.0
        self.cmd_vel = 0.8

        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_base_pos = np.zeros(3, dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-0.08, -1.0, -1.2, 0.4], dtype=np.float32),
            high=np.array([ 0.08,  1.0,  1.2, 1.4], dtype=np.float32),
            dtype=np.float32
        )

        obs_dim = 16 + 4 + 4 + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def seed(self, seed=None):
        self.base_env.seed(seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        base_obs = self.base_env.reset()

        self.bzg.reset()
        self.bz_step = BezierStepper(dt=self.base_env._time_step, mode=0)

        self.T_bf0 = copy.deepcopy(self.spot_model.WorldToFoot)
        self.T_bf = copy.deepcopy(self.T_bf0)

        self.cmd_step = 0.0
        self.cmd_lat = 0.0
        self.cmd_yaw = 0.0
        self.cmd_vel = 0.8

        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_base_pos = np.array(self.base_env.spot.GetBasePosition(), dtype=np.float32)

        obs = self._get_ml_observation(base_obs)
        info = {}
        return obs, info

    def _smooth_command(self, action):
        alpha_step = 0.10
        alpha_lat = 0.12
        alpha_yaw = 0.10
        alpha_vel = 0.08

        target_step = float(action[0])
        target_lat = float(action[1])
        target_yaw = float(action[2])
        target_vel = float(action[3])

        if abs(target_lat) > 1e-4 and abs(target_step) < 1e-4:
            target_step = 0.06

        self.cmd_step += alpha_step * (target_step - self.cmd_step)
        self.cmd_lat += alpha_lat * (target_lat - self.cmd_lat)
        self.cmd_yaw += alpha_yaw * (target_yaw - self.cmd_yaw)
        self.cmd_vel += alpha_vel * (target_vel - self.cmd_vel)

        return self.cmd_step, self.cmd_lat, self.cmd_yaw, self.cmd_vel

    def _command_to_joint_angles(self, action):
        self.bz_step.ramp_up()

        step_length, lateral_fraction, yaw_rate, step_velocity = self._smooth_command(action)

        if abs(step_length) > 1e-4 and abs(yaw_rate) > 1e-4:
            self.bzg.Tswing = 0.42
        elif abs(yaw_rate) > 1e-4:
            self.bzg.Tswing = 0.40
        elif abs(lateral_fraction) > 1e-4:
            self.bzg.Tswing = 0.40
        else:
            self.bzg.Tswing = 0.28

        self.bz_step.StepLength = step_length
        self.bz_step.LateralFraction = lateral_fraction
        self.bz_step.YawRate = yaw_rate
        self.bz_step.StepVelocity = step_velocity

        contacts = self.base_env.return_state()[-4:]

        self.T_bf = self.bzg.GenerateTrajectory(
            step_length,
            lateral_fraction,
            yaw_rate,
            step_velocity,
            self.T_bf0,
            self.T_bf,
            clearance_height=0.05,
            penetration_depth=0.01,
            contacts=contacts
        )

        pos = np.array([0.0, 0.0, 0.0])
        orn = np.array([0.0, 0.0, 0.0])

        joint_angles = self.spot_model.IK(orn, pos, self.T_bf)
        self.base_env.spot.GetExternalObservations(self.bzg, self.bz_step)

        return joint_angles.reshape(-1)

    def _get_ml_observation(self, base_obs=None):
        if base_obs is None:
            base_obs = self.base_env.return_state()

        base_obs = np.array(base_obs, dtype=np.float32)
        base_height = np.array([self.base_env.spot.GetBasePosition()[2]], dtype=np.float32)

        cmd_state = np.array([
            self.cmd_step,
            self.cmd_lat,
            self.cmd_yaw,
            self.cmd_vel
        ], dtype=np.float32)

        obs = np.concatenate([
            base_obs,
            self.prev_action.astype(np.float32),
            cmd_state,
            base_height
        ]).astype(np.float32)

        return obs

    def _compute_reward(self):
        obs = np.array(self.base_env.return_state(), dtype=np.float32)

        roll = obs[0]
        pitch = obs[1]
        gx = obs[2]
        gy = obs[3]
        gz = obs[4]

        base_pos = np.array(self.base_env.spot.GetBasePosition(), dtype=np.float32)
        dt = self.base_env.control_time_step if self.base_env.control_time_step > 0 else self.base_env._time_step
        forward_speed = (base_pos[0] - self.prev_base_pos[0]) / max(dt, 1e-6)
        lateral_drift = abs(base_pos[1])

        vel_reward = np.exp(-((forward_speed - self.target_forward_speed) ** 2) / 0.02)
        upright_penalty = 0.8 * (abs(roll) + abs(pitch))
        ang_penalty = 0.05 * (abs(gx) + abs(gy) + 0.5 * abs(gz))

        action_delta_penalty = 0.20 * np.linalg.norm(
            np.array([self.cmd_step, self.cmd_lat, self.cmd_yaw, self.cmd_vel]) - self.prev_action
        )

        energy_penalty = 0.002 * np.abs(
            np.dot(self.base_env.spot.GetMotorTorques(), self.base_env.spot.GetMotorVelocities())
        )

        drift_penalty = 0.10 * lateral_drift

        reward = (
            2.0 * vel_reward
            - upright_penalty
            - ang_penalty
            - action_delta_penalty
            - energy_penalty
            - drift_penalty
        )

        if not self.base_env.is_fallen():
            reward += 0.05

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
        reward = self._compute_reward()

        terminated = done
        truncated = False

        info.update({
            "forward_speed_target": self.target_forward_speed,
            "base_position": self.base_env.spot.GetBasePosition(),
            "command": {
                "step_length": float(self.cmd_step),
                "lateral_fraction": float(self.cmd_lat),
                "yaw_rate": float(self.cmd_yaw),
                "step_velocity": float(self.cmd_vel),
            }
        })

        return ml_obs, reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        return self.base_env.close()