"""Compare one student FK result with the MuJoCo oracle.

Edit the five joint angles in JOINT_CONFIGURATION, then run:

    uv run python test_single_fk.py --xml .\robot-assets\\scene.xml

All joint angles are in radians.
"""

import argparse
import numpy as np

from mujoco_oracle import MuJoCoPoseOracle
from student_fk import forward_kinematics


# Edit these values to test one robot configuration.
JOINT_CONFIGURATION = {
    "shoulder_pan": np.pi/2,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": -np.pi/2,
    "wrist_roll": 0.0,
}


def validate_pose(position, rotation, source_name):
    """Convert a pose to arrays and verify its dimensions."""
    position = np.asarray(position, dtype=float)
    rotation = np.asarray(rotation, dtype=float)

    if position.shape != (3,):
        raise ValueError(
            f"{source_name} position must have shape (3,), "
            f"but has shape {position.shape}."
        )

    if rotation.shape != (3, 3):
        raise ValueError(
            f"{source_name} rotation must have shape (3, 3), "
            f"but has shape {rotation.shape}."
        )

    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{source_name} pose contains NaN or infinity.")

    return position, rotation


def main(xml_path):
    np.set_printoptions(precision=6, suppress=True)

    oracle = MuJoCoPoseOracle(xml_path)

    p_oracle, R_oracle = validate_pose(
        *oracle.get_pose(JOINT_CONFIGURATION),
        source_name="Oracle",
    )

    p_student, R_student = validate_pose(
        *forward_kinematics(JOINT_CONFIGURATION),
        source_name="Student FK",
    )

    position_error = np.linalg.norm(p_oracle - p_student)
    relative_rotation = R_oracle.T @ R_student
    rotation_error = 3.0 - np.trace(relative_rotation)

    # Remove a tiny negative result caused only by floating-point roundoff.
    if -1.0e-12 < rotation_error < 0.0:
        rotation_error = 0.0

    print("\nJOINT CONFIGURATION")
    for joint_name, angle in JOINT_CONFIGURATION.items():
        print(f"{joint_name:15s}: {angle: .6f} rad")

    print("\nMUJOCO ORACLE POSITION")
    print(p_oracle)

    print("\nSTUDENT FK POSITION")
    print(p_student)

    print("\nPOSITION ERROR")
    print(f"{position_error:.6f} m")

    print("\nMUJOCO ORACLE ROTATION")
    print(R_oracle)

    print("\nSTUDENT FK ROTATION")
    print(R_student)

    print("\nRELATIVE ROTATION: R_oracle.T @ R_student")
    print(relative_rotation)

    print("\nROTATION ERROR: 3 - trace(R_oracle.T @ R_student)")
    print(f"{rotation_error:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare one student FK result with the MuJoCo oracle."
    )
    parser.add_argument(
        "--xml",
        default="./robot-assets/scene.xml",
        help="Path to the MuJoCo scene XML file.",
    )
    args = parser.parse_args()

    main(args.xml)
