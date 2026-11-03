"""Instructor regression test for Homework 3 inverse kinematics."""

import numpy as np

from provided_fk import forward_kinematics
from student_ik import inverse_kinematics


POSITION_TOL = 1e-9
AXIS_TOL = 1e-9
NUM_TESTS = 100


rng = np.random.default_rng(4360)
failures = 0


for test_number in range(NUM_TESTS):

    # Generate a legal top-down configuration.
    q2 = rng.uniform(0.2, 0.8)
    q3 = rng.uniform(-0.2, 0.2)

    q = {
        "shoulder_pan": rng.uniform(-1.4, 1.4),
        "shoulder_lift": q2,
        "elbow_flex": q3,
        "wrist_flex": np.pi / 2 - q2 - q3,
        "wrist_roll": 0.0,
    }

    # Generate a target known to be reachable.
    target, unused_rotation = forward_kinematics(q)

    # Solve the inverse-kinematics problem.
    q_ik, reachable = inverse_kinematics(target)

    if not reachable:
        print("FAIL", test_number, "reported unreachable")
        failures += 1
        continue

    # Check the returned solution using forward kinematics.
    position, rotation = forward_kinematics(q_ik)

    position_error = np.linalg.norm(position - target)
    axis_error = np.linalg.norm(
        rotation[:, 2] - np.array([0.0, 0.0, 1.0])
    )

    if position_error > POSITION_TOL:
        print(
            "FAIL",
            test_number,
            "position error:",
            position_error,
        )
        failures += 1

    if axis_error > AXIS_TOL:
        print(
            "FAIL",
            test_number,
            "axis error:",
            axis_error,
        )
        failures += 1


# Check several clearly unreachable targets.
unreachable_targets = [
    np.array([1.0, 0.0, 0.2]),
    np.array([-0.5, 0.5, 0.8]),
    np.array([0.0388353, 0.0, 0.1166]),
]

for target in unreachable_targets:

    q_ik, reachable = inverse_kinematics(target)

    if reachable:
        print("FAIL accepted unreachable target:", target)
        failures += 1


print("Tests:", NUM_TESTS)
print("Failures:", failures)

if failures == 0:
    print("All tests passed.")