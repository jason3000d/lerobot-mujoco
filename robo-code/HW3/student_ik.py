"""Student submission file for Homework 3: Inverse Kinematics."""
import numpy as np

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# Robot dimensions in meters from the assignment specification.
l1 = 0.0388353
l2 = 0.0624
l3 = 0.0303992
l4 = 0.0542
l5 = 0.11257
l6 = 0.0280
l7 = 0.1349
l8 = 0.0611
l9 = 0.1004

L_A = np.hypot(l5, l6)
BETA = np.arctan2(l5, l6)
WRIST_OFFSET = l8 + l9

COS_TOL = 1e-9
JOINT_LIMITS = {
    "shoulder_pan": (-np.pi, np.pi),
    "shoulder_lift": (-np.pi, np.pi),
    "elbow_flex": (-np.pi, np.pi),
    "wrist_flex": (-np.pi, np.pi),
    "wrist_roll": (-np.pi, np.pi),
}


def inverse_kinematics(target):
    """Return (q, reachable) for one top-down, elbow-up solution.

    Parameters
    ----------
    target : array-like, shape (3,)
        Desired idealized tool-frame position [X,Y,Z] in meters.

    Returns
    -------
    q : dict
        Five joint angles in radians, keyed by JOINT_NAMES.
    reachable : bool
        True only when the requested target has a valid geometric solution.
    """
    target = np.asarray(target, dtype=float)
    if target.shape != (3,):
        raise ValueError("target must have shape (3,)")
    if not np.all(np.isfinite(target)):
        raise ValueError("target must be finite")

    nan_q = {name: float("nan") for name in JOINT_NAMES}
    X, Y, Z = target

    theta1 = np.arctan2(-Y, X - l1)
    r = np.hypot(X - l1, Y)
    if r < COS_TOL:
        return nan_q, False

    x_w = r - l3
    z_w = Z + WRIST_OFFSET - (l2 + l4)
    P = np.hypot(x_w, z_w)
    if P < COS_TOL or not np.isfinite(P):
        return nan_q, False

    cos_alpha = (L_A**2 + l7**2 - P**2) / (2.0 * L_A * l7)
    cos_gamma = (L_A**2 + P**2 - l7**2) / (2.0 * L_A * P)
    if abs(cos_alpha) > 1.0 + COS_TOL or abs(cos_gamma) > 1.0 + COS_TOL:
        return nan_q, False

    alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))
    gamma = np.arccos(np.clip(cos_gamma, -1.0, 1.0))
    delta = np.arctan2(z_w, x_w)

    theta3 = np.pi - BETA - alpha
    theta2 = BETA - gamma - delta
    theta4 = np.pi / 2.0 - theta2 - theta3

    q = {
        "shoulder_pan": theta1,
        "shoulder_lift": theta2,
        "elbow_flex": theta3,
        "wrist_flex": theta4,
        "wrist_roll": 0.0,
    }

    for name in JOINT_NAMES:
        lo, hi = JOINT_LIMITS[name]
        if not np.isfinite(q[name]) or not (lo <= q[name] <= hi):
            return nan_q, False

    return {k: float(v) for k, v in q.items()}, True
