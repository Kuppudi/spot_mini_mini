import numpy as np


class SpotCamera:
    """
    Virtual camera mounted on the Spot robot body in PyBullet.
    """

    def __init__(self,
                 env,
                 width=320,
                 height=240,
                 fov=75.0,
                 near=0.02,
                 far=5.0,
                 camera_offset=(0.18, 0.0, 0.08),
                 target_distance=1.0):
        self.env = env
        self.p = env._pybullet_client

        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far

        # Camera position relative to robot base frame: (forward, left, up)
        self.camera_offset = np.array(camera_offset, dtype=float)

        # How far ahead the camera looks
        self.target_distance = target_distance

    def _get_camera_pose(self):
        """
        Compute camera world position and target using robot base pose.
        """
        base_pos = np.array(self.env.spot.GetBasePosition(), dtype=float)
        base_orn = self.env.spot.GetBaseOrientation()

        rot = np.array(self.p.getMatrixFromQuaternion(base_orn), dtype=float).reshape(3, 3)

        # Robot body axes in world frame
        forward = rot[:, 0]
        left = rot[:, 1]
        up = rot[:, 2]

        # Offset camera from body center
        cam_pos = (
            base_pos
            + self.camera_offset[0] * forward
            + self.camera_offset[1] * left
            + self.camera_offset[2] * up
        )

        # Look straight ahead
        target_pos = cam_pos + self.target_distance * forward

        return cam_pos, target_pos, up

    def get_rgb_frame(self):
        """
        Return RGB image from the robot-mounted camera.
        """
        cam_pos, target_pos, up = self._get_camera_pose()

        view_matrix = self.p.computeViewMatrix(
            cameraEyePosition=cam_pos.tolist(),
            cameraTargetPosition=target_pos.tolist(),
            cameraUpVector=up.tolist()
        )

        proj_matrix = self.p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=float(self.width) / float(self.height),
            nearVal=self.near,
            farVal=self.far
        )

        _, _, px, _, _ = self.p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self.p.ER_BULLET_HARDWARE_OPENGL
        )

        rgb = np.array(px, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3]
        return rgb

    def get_rgbd_frame(self):
        """
        Return RGB image and depth buffer.
        """
        cam_pos, target_pos, up = self._get_camera_pose()

        view_matrix = self.p.computeViewMatrix(
            cameraEyePosition=cam_pos.tolist(),
            cameraTargetPosition=target_pos.tolist(),
            cameraUpVector=up.tolist()
        )

        proj_matrix = self.p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=float(self.width) / float(self.height),
            nearVal=self.near,
            farVal=self.far
        )

        _, _, px, depth, _ = self.p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self.p.ER_BULLET_HARDWARE_OPENGL
        )

        rgb = np.array(px, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3]
        depth = np.array(depth).reshape(self.height, self.width)

        return rgb, depth