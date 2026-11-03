"""
Independent MuJoCo pose oracle for the SO-101.

The oracle sets the five arm joint coordinates, runs
MuJoCo forward kinematics, and returns the pose of the
static_finger_pad geometry relative to the baseframe site.
"""

from pathlib import Path

import mujoco
import numpy as np


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


class MuJoCoPoseOracle:

    def __init__(
        self,
        xml_path="model/scene.xml",
        base_site="baseframe",
        tool_geom="static_finger_pad",
    ):

        self.xml_path = Path(xml_path)

        self.model = mujoco.MjModel.from_xml_path(
            str(self.xml_path)
        )

        self.data = mujoco.MjData(self.model)

        #
        # Reference frame
        #

        self.base_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            base_site,
        )

        if self.base_site_id < 0:
            raise ValueError(
                f"MuJoCo site not found: {base_site}"
            )

        #
        # End-effector geometry
        #

        self.tool_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            tool_geom,
        )

        if self.tool_geom_id < 0:
            raise ValueError(
                f"MuJoCo geom not found: {tool_geom}"
            )

        #
        # Joint mapping
        #

        self.qpos_adr = {}
        self.joint_ranges = {}

        for name in JOINT_NAMES:

            jid = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )

            if jid < 0:
                raise ValueError(
                    f"MuJoCo joint not found: {name}"
                )

            self.qpos_adr[name] = int(
                self.model.jnt_qposadr[jid]
            )

            self.joint_ranges[name] = (
                self.model.jnt_range[jid]
                .astype(float)
                .copy()
            )

    def get_pose(self, q):
        """
        Return (p, R) of the static_finger_pad geometry
        relative to the baseframe site.

        Parameters
        ----------
        q : dict
            Dictionary of the five arm joint angles
            in radians.
        """

        #
        # Set joint coordinates
        #

        for name in JOINT_NAMES:
            self.data.qpos[
                self.qpos_adr[name]
            ] = float(q[name])

        #
        # Update all position-dependent quantities
        #

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        #
        # Base frame pose
        #

        p_w_b = self.data.site_xpos[
            self.base_site_id
        ].copy()

        R_w_b = (
            self.data.site_xmat[
                self.base_site_id
            ]
            .reshape(3, 3)
            .copy()
        )

        #
        # Tool pose
        #

        p_w_t = self.data.geom_xpos[
            self.tool_geom_id
        ].copy()

        R_w_t = (
            self.data.geom_xmat[
                self.tool_geom_id
            ]
            .reshape(3, 3)
            .copy()
        )

        #
        # Express tool pose relative to base frame
        #

        p_b_t = R_w_b.T @ (p_w_t - p_w_b)

        R_b_t = R_w_b.T @ R_w_t

        return p_b_t, R_b_t