"""
Student submission file.

All joint angles are in radians.

Implement:

    dh_transform(theta, d, a, alpha)

and

    forward_kinematics(q)

The forward_kinematics() function must return:

    p : numpy.ndarray shape (3,)
        End-effector position

    R : numpy.ndarray shape (3,3)
        End-effector rotation matrix
"""

import numpy as np


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


def dh_transform(theta, d, a, alpha):
    """
    Standard Denavit-Hartenberg transform.

        T = Rz(theta) Tz(d) Tx(a) Rx(alpha)

    Parameters
    ----------
    theta : float
    d     : float
    a     : float
    alpha : float

    Returns
    -------
    T : numpy.ndarray shape (4,4)
    """

    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T_rz = np.array([
        [ct, -st, 0, 0],
        [st,  ct, 0, 0],
        [ 0,   0, 1, 0],
        [ 0,   0, 0, 1]
    ])
    T_z = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, d],
        [0, 0, 0, 1]
    ])
    T_x = np.array([
        [1, 0, 0, a],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    T_rx = np.array([
        [1, 0,  0,  0],
        [0, ca, -sa, 0],
        [0, sa,  ca, 0],
        [0,  0,  0,  1]
    ])

    return T_rz @ T_z @ T_x @ T_rx


def forward_kinematics(q):
    """
    Parameters
    ----------
    q : dict

        {
            "shoulder_pan"  : float,
            "shoulder_lift" : float,
            "elbow_flex"    : float,
            "wrist_flex"    : float,
            "wrist_roll"    : float
        }

    Returns
    -------
    p : numpy.ndarray shape (3,)
        End-effector position

    R : numpy.ndarray shape (3,3)
        End-effector orientation
    """

    # Convenient local variables
    q1 = q["shoulder_pan"]
    q2 = q["shoulder_lift"]
    q3 = q["elbow_flex"]
    q4 = q["wrist_flex"]
    q5 = q["wrist_roll"]

    link_2 = np.hypot(0.11257, 0.028)
    offset = np.arctan2(0.11257, 0.028)
    dh_table = [
        (0,                  0.0624,  0.0388353, -np.pi),
        (q1,                -0.0542,  0.0303992,  np.pi / 2),
        (np.pi - offset + q2, 0,      -link_2,     0),
        (offset + q3,         0,      -0.1349,     0),
        (np.pi / 2 + q4,      0,       0,          np.pi / 2),
        (q5,                 -0.1615,  0,          0),
    ]

    T = np.eye(4)
    for theta, d, a, alpha in dh_table:
        T = T @ dh_transform(theta, d, a, alpha)

    p = T[:3, 3]
    r = T[:3, :3]

    return p, r
