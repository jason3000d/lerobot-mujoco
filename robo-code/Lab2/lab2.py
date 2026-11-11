'''
AI-tool usage declaration: I use Claude for code assistance.
'''

import argparse
import heapq
import json
import time
from itertools import combinations
from pathlib import Path
from lab1_utils import load_calibration, setup_motors, move_to_pose
import numpy as np


# CONFIGURATION VARIABLES
PORT_ID = "/dev/tty.usbmodem5B8E1147771" 
ROBOT_NAME = "hungeryBird" 

GRIPPER_OPEN = 45.0            
GRIPPER_CLOSED = 5.0

 
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ENVIRONMENT_FILE = Path(__file__).with_name("environment.json")

# Robot reading (deg) -> model angle (rad): q = SIGN * radians(reading - OFFSET)
OFFSET = {joint: 0.0 for joint in JOINTS}
SIGN = {joint: 1 for joint in JOINTS}
 
 
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
move_time = 1.0  # seconds to reach desired position
# hold_time = 1.0  # total time to hold at desired position
 
HOME = {j: desired_position[j] for j in JOINTS}

"""
Each obstacle is stored as ((x_min, y_min), (x_max, y_max)).

obstacles = [
((0.20, 0.15), (0.30, 0.35)),
((0.40, 0.30), (0.55, 0.40)),
((0.62, 0.10), (0.72, 0.28)),
]

object_start = (0.12, 0.12)
goal = (0.78, 0.48)
object_radius = 0.025
transport_height = 0.10
"""

MAX_OBSTACLES = 25
OBSTACLE_CLEARANCE = 0.005
PATH_SAMPLE_STEP = 0.05
PATH_SEGMENT_TIME = 0.5

 
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
 
 
# basic conversion functions between robot readings and model angles
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
 
 
def move(bus, joints, gripper, label, duration=move_time):
    print(f"{label}: {np.round(list(joints.values()), 1)}  gripper={gripper}")
    move_to_pose(bus, {**joints, "gripper": gripper}, duration)

# Lab 2 configuration and motion functions
# Prompt for a finite X Y coordinate pair, in metres.
def prompt_xy(label):
    while True:
        raw = input(f"{label} (x, y in metres): ").strip().strip("()").replace(",", " ").split()
        if len(raw) != 2:
            print("Enter exactly two values: x y.")
            continue
        try:
            point = tuple(float(value) for value in raw)
        except ValueError:
            print("Coordinates must be numbers.")
            continue
        if not np.all(np.isfinite(point)):
            print("Coordinates must be finite numbers.")
            continue
        return point

# Collect obstacle rectangles and the object's start, goal, and radius.
def prompt_environment_configuration():
    while True:
        try:
            obstacle_count = int(input(f"Number of obstacles (0-{MAX_OBSTACLES}): "))
        except ValueError:
            print("Enter a whole number.")
            continue
        if 0 <= obstacle_count <= MAX_OBSTACLES:
            break
        print(f"Enter a number from 0 to {MAX_OBSTACLES}.")

    obstacles = []
    for index in range(obstacle_count):
        while True:
            print(f"Obstacle {index + 1}: enter the lower-left and upper-right corners.")
            minimum = prompt_xy("  minimum (x_min y_min)")
            maximum = prompt_xy("  maximum (x_max y_max)")
            if minimum[0] < maximum[0] and minimum[1] < maximum[1]:
                obstacles.append([list(minimum), list(maximum)])
                break
            print("Each minimum coordinate must be less than its corresponding maximum.")

    object_start = prompt_xy("Object start")
    goal = prompt_xy("Object goal")
    while True:
        try:
            object_radius = float(input("Object radius (metres): "))
        except ValueError:
            print("Radius must be a number.")
            continue
        if np.isfinite(object_radius) and object_radius > 0:
            break
        print("Radius must be finite and greater than zero.")

    while True:
        try:
            object_height = float(input("Object pickup/place height z (metres): "))
        except ValueError:
            print("Height must be a number.")
            continue
        if np.isfinite(object_height):
            break
        print("Height must be finite.")

    while True:
        try:
            transport_height = float(input("Transport height z (metres): "))
        except ValueError:
            print("Height must be a number.")
            continue
        if np.isfinite(transport_height) and transport_height > object_height:
            break
        print("Transport height must be finite and above the pickup/place height.")

    return {
        "obstacles": obstacles,
        "object_start": list(object_start),
        "goal": list(goal),
        "object_radius": object_radius,
        "object_height": object_height,
        "transport_height": transport_height,
    }


def mode_configure_obstacles():
    configuration = prompt_environment_configuration()
    ENVIRONMENT_FILE.write_text(json.dumps(configuration, indent=2) + "\n")
    print(f"Saved environment configuration to {ENVIRONMENT_FILE.name}.")
    print(json.dumps(configuration, indent=2))
    return 0

def point_inside_rectangle(point, rectangle):
    (x_min, y_min), (x_max, y_max) = rectangle
    return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max


def segment_intersects_rectangle(start, end, rectangle):
    """Treat touching an obstacle boundary as a collision."""
    (x_min, y_min), (x_max, y_max) = rectangle
    t_min, t_max = 0.0, 1.0
    for origin, delta, lower, upper in (
        (start[0], end[0] - start[0], x_min, x_max),
        (start[1], end[1] - start[1], y_min, y_max),
    ):
        if abs(delta) < 1e-12:
            if origin < lower or origin > upper:
                return False
            continue
        entry = min((lower - origin) / delta, (upper - origin) / delta)
        exit = max((lower - origin) / delta, (upper - origin) / delta)
        t_min = max(t_min, entry)
        t_max = min(t_max, exit)
        if t_min > t_max:
            return False
    return True


def plan_object_path(start, goal, obstacles, object_radius):
    """Find a shortest polyline with object-radius clearance around rectangles."""
    inflation = object_radius + OBSTACLE_CLEARANCE
    corner_offset = 0.003
    inflated_obstacles = []
    nodes = [tuple(start), tuple(goal)]

    for (x_min, y_min), (x_max, y_max) in obstacles:
        rectangle = (
            (x_min - inflation, y_min - inflation),
            (x_max + inflation, y_max + inflation),
        )
        inflated_obstacles.append(rectangle)
        left, bottom = rectangle[0]
        right, top = rectangle[1]
        nodes.extend((
            (left - corner_offset, bottom - corner_offset),
            (left - corner_offset, top + corner_offset),
            (right + corner_offset, bottom - corner_offset),
            (right + corner_offset, top + corner_offset),
        ))

    for point in (start, goal):
        if any(point_inside_rectangle(point, rect) for rect in inflated_obstacles):
            raise ValueError(f"Object position {point} is inside an obstacle clearance area")

    graph = [[] for _ in nodes]
    for first_index, second_index in combinations(range(len(nodes)), 2):
        first, second = nodes[first_index], nodes[second_index]
        blocked = any(
            segment_intersects_rectangle(first, second, rect)
            for rect in inflated_obstacles
        )
        if blocked:
            continue
        distance = float(np.linalg.norm(np.asarray(second) - np.asarray(first)))
        graph[first_index].append((second_index, distance))
        graph[second_index].append((first_index, distance))

    distances = [float("inf")] * len(nodes)
    previous = [None] * len(nodes)
    distances[0] = 0.0
    queue = [(0.0, 0)]
    while queue:
        distance, current = heapq.heappop(queue)
        if distance > distances[current]:
            continue
        if current == 1:
            break
        for neighbor, edge_length in graph[current]:
            candidate = distance + edge_length
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = current
                heapq.heappush(queue, (candidate, neighbor))

    if not np.isfinite(distances[1]):
        raise ValueError("No collision-free route exists between object start and goal")

    path = []
    current = 1
    while current is not None:
        path.append(nodes[current])
        current = previous[current]
    return list(reversed(path))


def sample_path(path):
    samples = [np.asarray(path[0], dtype=float)]
    for start, end in zip(path, path[1:]):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        distance = float(np.linalg.norm(end - start))
        count = max(1, int(np.ceil(distance / PATH_SAMPLE_STEP)))
        samples.extend(start + (end - start) * index / count for index in range(1, count + 1))
    return samples


def mode_run():
    if not ENVIRONMENT_FILE.exists():
        raise ValueError(f"{ENVIRONMENT_FILE.name} not found; run --configure_obstacles first")

    configuration = json.loads(ENVIRONMENT_FILE.read_text())
    object_start = np.asarray(configuration["object_start"], dtype=float) * 1.06
    goal = np.asarray(configuration["goal"], dtype=float)
    radius = float(configuration["object_radius"])
    pickup_z = float(configuration["object_height"])
    transport_height = float(configuration["transport_height"])
    if (object_start.shape != (2,) or goal.shape != (2,)
            or not np.all(np.isfinite(object_start)) or not np.all(np.isfinite(goal))):
        raise ValueError("Object start and goal must each be finite (x, y) points")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Object radius must be finite and greater than zero")
    if not np.isfinite(pickup_z):
        raise ValueError("Object pickup/place height must be finite")
    if not np.isfinite(transport_height) or transport_height <= pickup_z:
        raise ValueError("Transport height must be above object pickup/place height")

    raw_obstacles = configuration["obstacles"]
    if len(raw_obstacles) > MAX_OBSTACLES:
        raise ValueError(f"At most {MAX_OBSTACLES} obstacles are supported")
    obstacles = []
    for index, raw_rectangle in enumerate(raw_obstacles, start=1):
        rectangle = np.asarray(raw_rectangle, dtype=float)
        if rectangle.shape != (2, 2) or not np.all(np.isfinite(rectangle)):
            raise ValueError(f"Obstacle {index} must have finite minimum and maximum (x, y) corners")
        if np.any(rectangle[0] >= rectangle[1]):
            raise ValueError(f"Obstacle {index} minimum coordinates must be less than maximum coordinates")
        obstacles.append((tuple(rectangle[0]), tuple(rectangle[1])))

    path = plan_object_path(object_start, goal, obstacles, radius) 
    sampled_path = sample_path(path) # to create a denser set of points along the planned path
    print(path)
    # print(sampled_path)
    print("Planned object path (x, y):", [np.round(point, 4).tolist() for point in path])
    print(f"Clearance: object radius {radius:.4f} m + {OBSTACLE_CLEARANCE:.4f} m.")

    cartesian_targets = [
        ("APPROACH", [object_start[0], object_start[1], transport_height], GRIPPER_OPEN, move_time),
        ("PICK", [object_start[0], object_start[1], pickup_z], GRIPPER_OPEN, move_time),
        ("CLOSE", [object_start[0], object_start[1], pickup_z], GRIPPER_CLOSED, move_time),
        ("LIFT", [object_start[0], object_start[1], transport_height], GRIPPER_CLOSED, move_time),
    ]
    for index, point in enumerate(sampled_path[1:], start=1):
        cartesian_targets.append((
            f"TRANSPORT {index}/{len(sampled_path) - 1}",
            [float(point[0]), float(point[1]), transport_height],
            GRIPPER_CLOSED,
            PATH_SEGMENT_TIME,
        ))
    cartesian_targets.extend((
        ("PLACE", [goal[0], goal[1], pickup_z], GRIPPER_CLOSED, move_time),
        ("OPEN", [goal[0], goal[1], pickup_z], GRIPPER_OPEN, move_time),
        ("RETREAT", [goal[0], goal[1], transport_height], GRIPPER_OPEN, move_time),
    ))

    plan = [
        (label, solve(np.asarray(point, dtype=float)), gripper, duration)
        for label, point, gripper, duration in cartesian_targets
    ]
    print(f"All {len(plan)} motion targets passed IK and joint-limit checks.")

    calibration = load_calibration(ROBOT_NAME)
    bus = setup_motors(calibration, PORT_ID)
    try:
        input("Clear the workspace, support the arm, then press Enter to begin.")
        bus.sync_write("Goal_Position", bus.sync_read("Present_Position"))
        bus.enable_torque()
        move(bus, HOME, GRIPPER_OPEN, "HOME")
        for label, joints, gripper, duration in plan:
            move(bus, joints, gripper, label, duration)
        move(bus, HOME, GRIPPER_OPEN, "HOME")
        return 0
    finally:
        try:
            input("Support the arm, then press Enter to turn off the motors.")
        finally:
            bus.disconnect(disable_torque=True)

"""
Modes:
  --configure-obstacles     enter the 2D task environment
  --run                plan and execute the obstacle-avoiding pick and place
"""

# main function
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--configure-obstacles", "--configure_obstacles", dest="configure_obstacles", action="store_true")
    g.add_argument("--run", action="store_true")
    args = ap.parse_args()
    try:
        if args.configure_obstacles:
            return mode_configure_obstacles()
        if args.run:
            return mode_run()
    except (ValueError, KeyError) as exc:
        print(f"ABORT: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
