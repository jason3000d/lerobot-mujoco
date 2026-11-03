"""MuJoCo simulator for a SO101 robot arm and a task cube.

This module builds a robot model from an XML scene, adds an optional cube in a
pseudo-random workspace position, and exposes helpers for interactive control and
predefined command sequences. The script can be run in either an interactive
"explore" mode or a scripted "command" mode.
"""

import mujoco
import mujoco.viewer
import time
import numpy as np


def cube_pos_from_id(cube_id, x_range=(0.15, 0.3), y_range=(-0.2, 0.2), z=0.0):
    """Return a deterministic cube position in the workspace for a given cube_id.

    The cube position is generated from a seeded random number generator so the
    same `cube_id` always produces the same location within the configured x/y
    bounds.
    """
    rng = np.random.default_rng(cube_id)
    x = rng.uniform(*x_range)
    y = rng.uniform(*y_range)
    return [float(x), float(y), z]


class Robot:
    """Thin wrapper around a MuJoCo model for a six-axis robot arm.

    The class wires up the robot's joint metadata, exposes actuator control via
    joint targets, and provides two usage modes:

    - `explore`: interactive viewer with live joint readings.
    - `command`: scripted waypoint execution in a single viewer session.
    """

    JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"]

    def __init__(self, scene_path="robot-assets/scene.xml", add_cube=True, cube_id=0):
        """Load the MuJoCo scene and initialize the robot state.

        Args:
            scene_path: Path to the XML model used to build the scene.
            add_cube: If true, inject a cube into the world at a deterministic
                location derived from `cube_id`.
            cube_id: Seed used for the cube's pseudo-random placement.
        """
        spec = mujoco.MjSpec.from_file(scene_path)
        self.cube_id = cube_id

        if add_cube:
            pos = cube_pos_from_id(cube_id)
            cube = spec.worldbody.add_body(name="cube", pos=pos)
            cube.add_freejoint()
            g = cube.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.012, 0.02, 0.012],
                mass=0.08, rgba=[1, 0, 0, 1],
                friction=[2, 0.5, 0.01],
                solref=[0.004, 1],
                solimp=[0.99, 0.999, 0.001, 0.5, 2],
            )
            g.condim = 6

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self.qpos_adr = {n: self.model.joint(n).qposadr[0] for n in self.JOINT_NAMES}
        self.act_id = {n: self.model.actuator(n).id for n in self.JOINT_NAMES}

        self.limits = {}
        for n in self.JOINT_NAMES:
            j = self.model.joint(n)
            self.limits[n] = (float(j.range[0]), float(j.range[1])) if j.limited else (-np.pi, np.pi)

    # ---- control ----
    def set_joint_targets(self, targets):
        """Set actuator targets for the robot.

        Args:
            targets: Either a mapping of joint names to target angles, or a
                sequence aligned with ``JOINT_NAMES``. Each target is clipped to
                the joint's configured min/max range before being sent to MuJoCo.
        """
        items = targets.items() if isinstance(targets, dict) else zip(self.JOINT_NAMES, targets)
        for name, q in items:
            lo, hi = self.limits[name]
            self.data.ctrl[self.act_id[name]] = float(np.clip(q, lo, hi))

    def get_joint_angles(self):
        """Return the current joint angles for all controlled robot joints.

        Returns:
            A dict keyed by the joint names in ``JOINT_NAMES`` with floating-point
            angle values in radians.
        """
        return {n: float(self.data.qpos[self.qpos_adr[n]]) for n in self.JOINT_NAMES}

    def _print_angles(self):
        """Print the current joint angles in a compact, human-readable format."""
        print(" | ".join(f"{n}: {q:+.3f}" for n, q in self.get_joint_angles().items()))

    # ---- MODE 1: explore ----
    def explore(self, minutes=10, print_hz=2):
        """Open the interactive viewer and keep the simulation running.

        In this mode the user can drag joints in the GUI while the live joint
        angles and cube position are printed at a configurable cadence.

        Args:
            minutes: Maximum runtime for the interactive session, in minutes.
            print_hz: Number of status prints per second while the viewer runs.
        """
        end = time.time() + minutes * 60
        last_print = 0.0
        interval = 1.0 / print_hz
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and time.time() < end:
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                now = time.time()
                if now - last_print >= interval:
                    self._print_angles()
                    print("cube:", self.data.body("cube").xpos)
                    last_print = now
                time.sleep(self.model.opt.timestep)
        print("Final pose:", {k: round(v, 3) for k, v in self.get_joint_angles().items()})
        print("cube:", self.data.body("cube").xpos)

    # ---- MODE 2: command ----
    def command(self, waypoints, duration=4.0):
        """Run a sequence of joint targets within a single viewer session.

        `waypoints` may be either a single pose specification or a list of poses.
        Each pose can be a mapping of joint names to target values, an array of
        values in `JOINT_NAMES` order, or a tuple of `(target, dur)` to override
        the default hold time for that step.

        Args:
            waypoints: One or more desired robot poses to execute in order.
            duration: Default number of seconds to hold each waypoint before
                moving to the next one.
        """
        if isinstance(waypoints, (dict, np.ndarray)) or (
            isinstance(waypoints, (list, tuple))
            and waypoints and isinstance(waypoints[0], (int, float))
        ):
            waypoints = [waypoints]

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            for wp in waypoints:
                target, dur = wp if isinstance(wp, tuple) else (wp, duration)

                self.set_joint_targets(target)
                end = time.time() + dur
                while viewer.is_running() and time.time() < end:
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    time.sleep(self.model.opt.timestep)

                if not viewer.is_running():
                    break
                self._print_angles()

        # Cube position after execution
        print("cube:", self.data.body("cube").xpos)
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "explore"
    cube_id = int(sys.argv[2]) if len(sys.argv) > 2 else -1

    if cube_id < 0 or cube_id > 9999:
        raise ValueError("cube_id must be last four digits of a student ID")

    robot = Robot(cube_id=cube_id)

    if mode == "command":
        robot.command([
            {"shoulder_pan": 0.768, "shoulder_lift": 0.542,   "elbow_flex": -0.253,
             "wrist_flex": 1.509,  "wrist_roll": 0.914, "gripper": 0.689},
            {"shoulder_pan": 0.768, "shoulder_lift": 0.542,   "elbow_flex": -0.253,
             "wrist_flex": 1.509,  "wrist_roll": 0.914, "gripper": 0.055},
            {"shoulder_pan": 0.038, "shoulder_lift": -0.384,   "elbow_flex": +0.423,
             "wrist_flex": 1.509,  "wrist_roll": 0.914, "gripper": -0.002},
            {"shoulder_pan": 0.038, "shoulder_lift": -0.384,   "elbow_flex": +0.423,
             "wrist_flex": 1.509,  "wrist_roll": 0.914, "gripper": 0.699},
        ], duration=1.0)
    else:
        robot.explore(minutes=10)