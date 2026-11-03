"""
Modes (exactly one per run):
  --localize             hand-guide the tool point onto the cube
  --run                  execute the full sequence
"""

import argparse
import json
import time
from pathlib import Path
from lab1_utils import load_calibration, setup_motors, move_to_pose
import numpy as np


# CONFIGURATION VARIABLES
PORT_ID = "/dev/tty.usbmodem5B8E1147771" 
ROBOT_NAME = "hungeryBird" 

GRIPPER_OPEN = 60.0            
GRIPPER_CLOSED = 5.0
TARGET = [0.20, 0.00]          
SAFE_HEIGHT = 0.08            

 
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
CUBE_FILE = Path(__file__).with_name("cube.json")
 
# Robot reading (deg) -> model angle (rad):  q = SIGN * radians(reading - OFFSET)
OFFSET = {j: 0.0 for j in JOINTS}
SIGN = {j: 1 for j in JOINTS}
 
 
# LINK LENGTHS (m)
l1, l2, l3, l4 = 0.0388, 0.0624, 0.0304, 0.0542
l5, l6, l7 = 0.1126, 0.0280, 0.1349
l8, l9 = 0.0611, 0.1004
 
JOINT_LIMITS = { # radians
    "shoulder_pan":     (-1.92, 1.92),
    "shoulder_lift":    (-1.75, 1.75),
    "elbow_flex":       (-1.69, 1.69),
    "wrist_flex":       (-1.66, 1.66),
    "wrist_roll":       (-2.74, 2.74),
}

desired_position = {
    'shoulder_pan': 0.0,   # degrees
    'shoulder_lift': 0.0,
    'elbow_flex': 0.0,
    'wrist_flex': 0.0,
    'wrist_roll': 0.0,
    'gripper': GRIPPER_OPEN  # 0-100 range
}
move_time = 5.0  # seconds to reach desired position
hold_time = 2.0  # total time to hold at desired position
 
HOME = {j: desired_position[j] for j in JOINTS}
 
 
# FORWARD KINEMATICS 
def dh_transform(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])
 
 
def forward_kinematics(q):
    gamma = np.arctan2(l5, l6)
    dh_table = [   # (a, alpha, d, theta)
        (l1, -np.pi, l2, 0),
        (l3, np.pi/2, -l4, q["shoulder_pan"]),        
        (-np.sqrt(l5**2 + l6**2), 0, 0, q["shoulder_lift"] + (np.pi - gamma)),
        (-l7, 0, 0, q["elbow_flex"] + gamma),
        (0, np.pi/2, 0, q["wrist_flex"] + np.pi/2),
        (0, 0, -(l8 + l9), q["wrist_roll"]),
    ]
    T = np.eye(4)
    for a, alpha, d, theta in dh_table:
        T = T @ dh_transform(theta, d, a, alpha)
    return T[:3, 3], T[:3, :3]
 
 
# INVERSE KINEMATICS 
def inverse_kinematics(target):
    X, Y, Z = target
 
    theta1 = np.arctan2(-Y, X - l1)
    r = np.sqrt((X - l1)**2 + Y**2) - l3
    s = Z + l8 + l9 - (l2 + l4)
 
    la = np.sqrt(l5**2 + l6**2)
    beta = np.arctan2(l5, l6)
    P = np.sqrt(r**2 + s**2)
 
    cos_a = (la**2 + l7**2 - P**2) / (2 * la * l7)
    cos_g = (la**2 + P**2 - l7**2) / (2 * la * P)
    if P < 1e-9 or abs(cos_a) > 1 or abs(cos_g) > 1:
        return None                                  
 
    theta3 = np.pi - beta - np.arccos(cos_a)
    theta2 = beta - np.arccos(cos_g) - np.arctan2(s, r)
    theta4 = np.pi/2 - theta2 - theta3
    theta5 = 0.0
 
    q = {}
    for name, angle in zip(JOINTS, [theta1, theta2, theta3, theta4, theta5]):
        q[name] = (angle + np.pi) % (2*np.pi) - np.pi
    return q
 
 

def to_model(reading):
    """Robot reading (deg) -> model angles (rad)."""
    return {j: SIGN[j] * np.radians(reading[j] - OFFSET[j]) for j in JOINTS}
 
 
def to_robot(q):
    """Model angles (rad) -> robot command (deg)."""
    return {j: SIGN[j] * np.degrees(q[j]) + OFFSET[j] for j in JOINTS}
 
 
def solve(point):
    """IK for a point, checked for reachability, joint limits and FK agreement."""
    q = inverse_kinematics(point)
    if q is None:
        raise ValueError(f"{point} is unreachable")
    for j in JOINTS:
        if not JOINT_LIMITS[j][0] <= q[j] <= JOINT_LIMITS[j][1]:
            raise ValueError(f"{point}: {j} outside joint limits")
    p, _ = forward_kinematics(q)
    if np.linalg.norm(p - point) > 0.002:
        raise ValueError(f"{point}: FK check failed")
    return to_robot(q)
 
 
def move(bus, joints, gripper, label):
    print(f"{label}: {np.round(list(joints.values()), 1)}  gripper={gripper}")
    move_to_pose(bus, {**joints, "gripper": gripper}, move_time)
    time.sleep(0.5)
 
 
 
def mode_localized():
    calibration = load_calibration(ROBOT_NAME)
    bus = setup_motors(calibration, PORT_ID)
    try:
        bus.disable_torque()
        print("Torque is off. Move the arm by hand so the tool point is on the cube.")
        input("Hold the arm steady, then press Enter to record the cube position.")
        reading = bus.sync_read("Present_Position")
        cube, _ = forward_kinematics(to_model(reading))
        cube_data = {
            "p_cube": np.asarray(cube, dtype=float).tolist(),
            "q_cube_motor_deg": {joint: float(reading[joint]) for joint in JOINTS},
        }
        CUBE_FILE.write_text(json.dumps(cube_data, indent=2) + "\n")
        print("Cube position (m):", np.round(cube, 4).tolist())
        print(f"Saved cube position to {CUBE_FILE.name}.")
        return 0
    finally:
        bus.disconnect(disable_torque=True)


def mode_run():
    if not CUBE_FILE.exists():
        raise ValueError(f"{CUBE_FILE.name} not found; run --localize first")
    cube_data = json.loads(CUBE_FILE.read_text())
    cube = np.asarray(cube_data["p_cube"], dtype=float)
    if cube.shape != (3,) or not np.all(np.isfinite(cube)):
        raise ValueError(f"Invalid cube position in {CUBE_FILE.name}")

    x, y, z = cube
    tx, ty = TARGET
    up = z + SAFE_HEIGHT
    waypoints = [
        ("APPROACH", [x, y, up], GRIPPER_OPEN),
        ("GRASP", [x, y, z], GRIPPER_OPEN),
        ("CLOSE", [x, y, z], GRIPPER_CLOSED),
        ("LIFT", [x, y, up], GRIPPER_CLOSED),
        ("TRANSPORT", [tx, ty, up], GRIPPER_CLOSED),
        ("PLACE", [tx, ty, z], GRIPPER_CLOSED),
        ("OPEN", [tx, ty, z], GRIPPER_OPEN),
        ("RETREAT", [tx, ty, up], GRIPPER_OPEN),
    ]
    plan = [(label, solve(np.asarray(point)), gripper) for label, point, gripper in waypoints]
    print("All waypoints valid.")

    calibration = load_calibration(ROBOT_NAME)
    bus = setup_motors(calibration, PORT_ID)
    try:
        input("Support the arm and press Enter to start the pick-and-place sequence.")
        bus.sync_write("Goal_Position", bus.sync_read("Present_Position"))
        bus.enable_torque()
        move(bus, HOME, GRIPPER_OPEN, "HOME")
        for label, joints, gripper in plan:
            move(bus, joints, gripper, label)
        move(bus, HOME, GRIPPER_OPEN, "HOME")
        return 0
    finally:
        input("Support the arm, then press Enter to turn off the motors.")
        bus.disconnect(disable_torque=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--localize", action="store_true")
    g.add_argument("--run", action="store_true")
    args = ap.parse_args()
    try:
        if args.localize:
            return mode_localized()
        return mode_run()
    except (ValueError, KeyError) as exc:
        print(f"ABORT: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
