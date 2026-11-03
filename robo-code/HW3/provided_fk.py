"""Instructor-provided forward kinematics for the SO-101.

This matches the Homework 2 reference convention. Angles are radians and
lengths are meters.
"""
import numpy as np

OFFSET = np.arctan2(0.11257, 0.028)
LINK_2 = np.hypot(0.11257, 0.028)
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

def dh_transform(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.0,     sa,     ca,    d],
        [0.0,    0.0,    0.0,  1.0],
    ], dtype=float)

def forward_kinematics(q):
    q1, q2, q3, q4, q5 = (float(q[name]) for name in JOINT_NAMES)
    table = [
        (0.0388353, -np.pi,       0.0624, 0.0),
        (0.0303992,  np.pi/2.0,  -0.0542, q1),
        (-LINK_2,    0.0,          0.0, np.pi-OFFSET+q2),
        (-0.1349,    0.0,          0.0, OFFSET+q3),
        (0.0,        np.pi/2.0,    0.0, np.pi/2.0+q4),
        (0.0,        0.0,        -0.1615, q5),
    ]
    T=np.eye(4)
    for a,alpha,d,theta in table:
        T = T @ dh_transform(theta,d,a,alpha)
    return T[:3,3].copy(), T[:3,:3].copy()
