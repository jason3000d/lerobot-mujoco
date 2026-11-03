"""Run one IK target and check it with forward kinematics."""

import numpy as np

from provided_fk import forward_kinematics
from student_ik import inverse_kinematics


TARGET_POSITION = np.array([0.230144, -0.090251, 0.060482])


q, reachable = inverse_kinematics(TARGET_POSITION)

print("Target:", TARGET_POSITION)
print("Reachable:", reachable)

if reachable:

    print("Joint angles:")

    for name, value in q.items():
        print(name, value)

    position, rotation = forward_kinematics(q)

    position_error = np.linalg.norm(position - TARGET_POSITION)
    axis_error = np.linalg.norm(
        rotation[:, 2] - np.array([0.0, 0.0, 1.0])
    )

    print("FK position:", position)
    print("Position error:", position_error)
    print("Downward-axis error:", axis_error)

else:

    print("No solution.")