"""SO-101 Lab 1: pick-and-place using HW2 FK + HW3 IK.

Modes (exactly one per run):
  --check-target X Y Z   offline: IK + FK round-trip for one target (no robot)
  --read-state           connect, read joints, print FK position (torque off)
  --gripper-test         step the gripper through candidate values to find open/closed
  --fk-validate          hand-guide the arm to 3-5 poses, enter ruler measurements, get errors
  --localize             hand-guide the tool point onto the cube; saves cube.json
  --plan                 offline: build + validate every waypoint from cube.json (no robot)
  --run                  execute the full sequence (use --step and --dry-run first)

Safety: nothing in --check-target / --plan / --read-state / --fk-validate / --localize
sends a position goal. Only --run and --gripper-test move motors.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

LAB_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = LAB_DIRECTORY.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LAB_DIRECTORY))

from HW2.student_fk import JOINT_NAMES, forward_kinematics
from HW3.student_ik import JOINT_LIMITS, inverse_kinematics

# ----------------------------------------------------------------------------
# CONFIGURATION RECORD (handout section 3.4) -- fill these in
# ----------------------------------------------------------------------------
PORT_ID = "/dev/tty.usbmodem5B8E1147771"   # from `uv run lerobot-find-port`
ROBOT_NAME = "hungeryBird"                 # must match the ID used at calibration
CALIBRATION_FILE = "calibration"           # lab1_utils adds calibration_files/ and .json

# ----------------------------------------------------------------------------
# JOINT MAPPING: motor degrees <-> FK radians.  If FK validation shows a flipped
# sign or a constant offset on a joint, change it HERE (one line), nowhere else.
#   q_fk_deg = SIGN * motor_deg + OFFSET_DEG
# ----------------------------------------------------------------------------
SIGN = {n: 1.0 for n in JOINT_NAMES}
OFFSET_DEG = {n: 0.0 for n in JOINT_NAMES}

# ----------------------------------------------------------------------------
# GRIPPER (0-100 range). VERIFY with --gripper-test before the real run.
# ----------------------------------------------------------------------------
GRIPPER_OPEN = 60.0
GRIPPER_CLOSED = 10.0     # tune so it squeezes the cube without stalling hard

# ----------------------------------------------------------------------------
# MOTION / PLANNING PARAMETERS
# ----------------------------------------------------------------------------
HOME_TARGET = (0.18, 0.00, 0.06)  # tool position (m) that defines the HOME pose (via IK)
PLACE_XY = (0.20, 0.00)           # required drop-off point in the base frame
APPROACH_OFFSET = 0.08         # m above cube for APPROACH
LIFT_OFFSET = 0.04                # m above cube for LIFT
TRANSPORT_OFFSET = 0.04           # m above place height for TRANSPORT
CART_STEP = 0.01                  # m between IK samples on straight-line segments
MAX_JOINT_JUMP_DEG = 12.0         # between consecutive samples; larger = branch flip -> abort
MAX_JOINT_SPEED_DPS = 20.0        # deg/s for the largest-moving joint (before --speed scaling)
TRACKING_ABORT_DEG = 15.0         # abort if commanded-vs-actual error exceeds this
TRACKING_WARN_DEG = 4.0           # warn at end of a move if final error exceeds this
IK_FK_TOLERANCE_M = 1e-6
LOOP_DT = 0.02                    # 50 Hz command loop

CUBE_FILE = LAB_DIRECTORY / "cube.json"
FK_CSV = LAB_DIRECTORY / "fk_validation.csv"


SHORT = {"shoulder_pan": "pan", "shoulder_lift": "lift", "elbow_flex": "elbow",
         "wrist_flex": "wflex", "wrist_roll": "roll"}


class Abort(Exception):
    pass


# ----------------------------------------------------------------------------
# Unit / convention conversion
# ----------------------------------------------------------------------------
def motor_to_fk(motor_deg):
    """Motor readings (degrees) -> FK joint dict (radians)."""
    out = {}
    for n in JOINT_NAMES:
        if n not in motor_deg:
            raise ValueError(f"missing joint reading: {n}")
        v = float(motor_deg[n])
        if not math.isfinite(v):
            raise ValueError(f"non-finite reading for {n}")
        out[n] = math.radians(SIGN[n] * v + OFFSET_DEG[n])
    return out


def fk_to_motor(q_rad):
    """FK/IK joint dict (radians) -> motor command dict (degrees)."""
    out = {}
    for n in JOINT_NAMES:
        v = float(q_rad[n])
        if not math.isfinite(v):
            raise ValueError(f"non-finite angle for {n}")
        out[n] = SIGN[n] * (math.degrees(v) - OFFSET_DEG[n])
    return out


def fk_position_from_motor(motor_deg):
    p, _ = forward_kinematics(motor_to_fk(motor_deg))
    return np.asarray(p, dtype=float)


# ----------------------------------------------------------------------------
# IK validation (handout section 7)
# ----------------------------------------------------------------------------
def solve_validated(target, label=""):
    """IK -> reject NaN / out-of-limit -> FK round-trip. Returns joint dict (rad)."""
    q, ok = inverse_kinematics(target)
    if not ok:
        raise Abort(f"IK rejected {label} target {np.round(target, 4).tolist()} "
                    "(unreachable or joint limit)")
    for n in JOINT_NAMES:
        lo, hi = JOINT_LIMITS[n]
        if not (lo <= q[n] <= hi):
            raise Abort(f"{label}: {n} outside limits")
    p, _ = forward_kinematics(q)
    err = float(np.linalg.norm(np.asarray(p) - np.asarray(target)))
    if not math.isfinite(err) or err > IK_FK_TOLERANCE_M:
        raise Abort(f"{label}: FK/IK error {err:.3g} m exceeds {IK_FK_TOLERANCE_M:g} m")
    return q


def straight_line_path(p0, p1, label):
    """IK samples along a straight Cartesian line (so LIFT/APPROACH are truly vertical)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(1, int(math.ceil(np.linalg.norm(p1 - p0) / CART_STEP)))
    path = []
    for i in range(n + 1):
        p = p0 + (p1 - p0) * (i / n)
        path.append(fk_to_motor(solve_validated(p, f"{label}[{i}/{n}]")))
    check_no_branch_flip(path, label)
    return path


def check_no_branch_flip(path, label):
    for a, b in zip(path, path[1:]):
        jump = max(abs(a[n] - b[n]) for n in JOINT_NAMES)
        if jump > MAX_JOINT_JUMP_DEG:
            raise Abort(f"{label}: {jump:.1f} deg jump between samples (IK branch change)")


def build_waypoints(
   p_cube,
   approach=APPROACH_OFFSET,
   lift=LIFT_OFFSET,
   transport=TRANSPORT_OFFSET,
):
   p_cube = np.asarray(p_cube, dtype=float)

   if p_cube.shape != (3,) or not np.all(np.isfinite(p_cube)):
       raise ValueError(f"Invalid cube position: {p_cube}")

   # The assignment's fixed placement XY coordinates.
   place_h = p_cube[2]
   p_target = np.array([PLACE_XY[0], PLACE_XY[1], place_h])
   p_cube_multiplied = p_cube * [1.1, 1.1, 1.0]  # Slightly offset for approach)

   waypoints = {
       "HOME": np.array(HOME_TARGET, dtype=float),
       "APPROACH": p_cube_multiplied + np.array([0.0, 0.0, approach]),
       "GRASP": p_cube.copy() + np.array([0.02, 0.02, 0.0]),
       "LIFT": p_cube + np.array([0.0, 0.0, lift]),
       "TRANSPORT": p_target + np.array([0.0, 0.0, transport]),
       "PLACE": p_target,
    #    "RETREAT": p_target + np.array([0.0, 0.0, transport]),
        "RETREAT": np.array(HOME_TARGET, dtype=float),
   }

   return waypoints

def build_plan(p_cube, **offsets):
    """Validate EVERYTHING offline. Returns ordered list of steps."""
    w = build_waypoints(p_cube, **offsets)
    home_q = fk_to_motor(solve_validated(w["HOME"], "HOME"))
    # (name, kind, data)  kind: 'path' = list of motor-degree dicts; 'grip' = gripper target
    steps = [
        ("HOME", "path", [home_q]),
        ("APPROACH", "path", straight_line_path(w["HOME"], w["APPROACH"], "HOME->APPROACH")),
        ("GRASP", "path", straight_line_path(w["APPROACH"], w["GRASP"], "APPROACH->GRASP")),
        ("CLOSE", "grip", GRIPPER_CLOSED),
        ("LIFT", "path", straight_line_path(w["GRASP"], w["LIFT"], "GRASP->LIFT")),
        ("TRANSPORT", "path", straight_line_path(w["LIFT"], w["TRANSPORT"], "LIFT->TRANSPORT")),
        ("PLACE", "path", straight_line_path(w["TRANSPORT"], w["PLACE"], "TRANSPORT->PLACE")),
        ("OPEN", "grip", GRIPPER_OPEN),
        ("RETREAT", "path", straight_line_path(w["PLACE"], w["RETREAT"], "PLACE->RETREAT")),
        ("HOME (return)", "path", [home_q]),
    ]
    return w, steps


def print_plan(w, steps):
    print("\nWaypoints (tool position in base frame, metres):")
    for k, v in w.items():
        print(f"  {k:<10} {np.round(v, 4).tolist()}")
    print("\nValidated sequence (end-of-step motor angles, degrees):")
    for name, kind, data in steps:
        if kind == "path":
            q = data[-1]
            print(f"  {name:<14} {len(data):>3} samples  " +
                  "  ".join(f"{SHORT[n]}={q[n]:7.1f}" for n in JOINT_NAMES))
        else:
            print(f"  {name:<14} gripper -> {data}")
    print("All IK solutions: finite, inside joint limits, FK error < "
          f"{IK_FK_TOLERANCE_M:g} m, no branch flips.\n")


# ----------------------------------------------------------------------------
# Robot connection + motion
# ----------------------------------------------------------------------------
def connect():
    from lab1_utils import load_calibration, setup_motors  # lazy: needs lerobot installed
    calibration = load_calibration(CALIBRATION_FILE)
    bus = setup_motors(calibration, PORT_ID)
    # setup_motors re-enables torque when it finishes; start from a known-limp state.
    bus.disable_torque()
    return bus


def read_motor_degrees(bus, samples=1):
    acc = None
    for _ in range(samples):
        r = bus.sync_read("Present_Position")
        if acc is None:
            acc = {k: 0.0 for k in r}
        for k, v in r.items():
            acc[k] += float(v)
        if samples > 1:
            time.sleep(0.02)
    return {k: v / samples for k, v in acc.items()}


def hold_here_and_enable(bus):
    """Make goal == present BEFORE torque, so enabling torque cannot jerk the arm."""
    now = bus.sync_read("Present_Position")
    bus.sync_write("Goal_Position", {k: float(v) for k, v in now.items()}, normalize=True)
    bus.enable_torque()


def _command(bus, arm_deg, gripper):
    cmd = {n: float(arm_deg[n]) for n in JOINT_NAMES}
    cmd["gripper"] = float(gripper)
    bus.sync_write("Goal_Position", cmd, normalize=True)


def follow_path(bus, path, gripper, speed=1.0, label=""):
    """Stream a smooth joint trajectory through `path` (list of motor-degree dicts)."""
    now = read_motor_degrees(bus)
    start = {n: now[n] for n in JOINT_NAMES}
    pts = [start] + list(path)
    seg = [max(abs(a[n] - b[n]) for n in JOINT_NAMES) for a, b in zip(pts, pts[1:])]
    total = sum(seg)
    if total < 0.2:                          # already there
        return
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    # smoothstep: peak speed = 1.5x average, so divide by 1.5 to respect the speed cap
    duration = 1.5 * total / (MAX_JOINT_SPEED_DPS * speed)
    duration = max(duration, 0.5)
    t0 = time.time()
    k = 0
    while True:
        u = min((time.time() - t0) / duration, 1.0)
        s = 3 * u * u - 2 * u * u * u * 1.0
        d = s * total
        i = min(int(np.searchsorted(cum, d, side="right")) - 1, len(seg) - 1)
        i = max(i, 0)
        a = (d - cum[i]) / seg[i] if seg[i] > 1e-9 else 1.0
        a = min(max(a, 0.0), 1.0)
        cmd = {n: (1 - a) * pts[i][n] + a * pts[i + 1][n] for n in JOINT_NAMES}
        _command(bus, cmd, gripper)
        k += 1
        if k % 5 == 0:
            cur = read_motor_degrees(bus)
            err = max(abs(cur[n] - cmd[n]) for n in JOINT_NAMES)
            if err > TRACKING_ABORT_DEG:
                raise Abort(f"{label}: tracking error {err:.1f} deg (blocked or stalled?)")
        if u >= 1.0:
            break
        time.sleep(LOOP_DT)
    time.sleep(0.15)
    cur = read_motor_degrees(bus)
    err = max(abs(cur[n] - path[-1][n]) for n in JOINT_NAMES)
    if err > TRACKING_WARN_DEG:
        print(f"  WARNING: {label} ended {err:.1f} deg from target")


def move_gripper(bus, arm_deg, target, seconds=1.5):
    cur = read_motor_degrees(bus)["gripper"]
    t0 = time.time()
    while True:
        u = min((time.time() - t0) / seconds, 1.0)
        _command(bus, arm_deg, cur + (target - cur) * (3 * u * u - 2 * u ** 3))
        if u >= 1.0:
            break
        time.sleep(LOOP_DT)
    time.sleep(0.3)


def freeze(bus):
    try:
        now = bus.sync_read("Present_Position")
        bus.sync_write("Goal_Position", {k: float(v) for k, v in now.items()}, normalize=True)
    except Exception as exc:  # pragma: no cover
        print(f"(freeze failed: {exc})")


def shutdown(bus):
    print("\nSupport the arm. Torque will be disabled on Enter.")
    try:
        input("Press Enter: ")
    except EOFError:
        pass
    bus.disconnect(disable_torque=True)
    print("Disconnected; motor torque disabled.")


# ----------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------
def mode_check_target(target):
    q = solve_validated(target, "target")
    p, _ = forward_kinematics(q)
    print(f"Target (m):            {[float(v) for v in target]}")
    print(f"IK angles (rad):       { {k: round(v, 5) for k, v in q.items()} }")
    print(f"Motor angles (deg):    { {k: round(v, 2) for k, v in fk_to_motor(q).items()} }")
    print(f"FK of solution (m):    {np.round(p, 9).tolist()}")
    print(f"FK error:              {np.linalg.norm(np.asarray(p) - target):.3e} m  (limit {IK_FK_TOLERANCE_M:g})")
    print("Joint-limit check:     PASS")
    return 0


def mode_read_state():
    bus = connect()
    try:
        deg = read_motor_degrees(bus)
        print(f"Robot ID: {ROBOT_NAME}   Port: {PORT_ID}")
        print("Motor positions (arm degrees; gripper 0-100):")
        for k, v in deg.items():
            print(f"  {k:<14}{v:9.3f}")
        rad = motor_to_fk(deg)
        print("FK joint values (rad): " + ", ".join(f"{n}={rad[n]:.4f}" for n in JOINT_NAMES))
        print(f"FK tool position (m):  {np.round(fk_position_from_motor(deg), 4).tolist()}")
    finally:
        bus.disconnect(disable_torque=True)
        print("Disconnected; motor torque disabled.")
    return 0


def mode_gripper_test():
    bus = connect()
    try:
        print("Gripper test. Torque on for the gripper; arm joints held where they are.")
        input("Support the arm, clear the gripper, press Enter: ")
        hold_here_and_enable(bus)
        arm = read_motor_degrees(bus)
        while True:
            s = input("Gripper value 0-100 (blank to finish): ").strip()
            if not s:
                break
            val = float(s)
            if not 0.0 <= val <= 100.0:
                print("out of range")
                continue
            move_gripper(bus, arm, val, 1.0)
            print(f"  now reads {read_motor_degrees(bus)['gripper']:.1f}")
        print("Record your OPEN and CLOSED values into GRIPPER_OPEN / GRIPPER_CLOSED.")
    finally:
        freeze(bus)
        shutdown(bus)
    return 0


def read_measured_position_mm():
    """Prompt until a finite x y z measurement in millimetres is provided."""
    while True:
        raw = input("  measured x y z in mm (space separated): ").strip()
        values = raw.split()
        if len(values) != 3:
            print("  Enter exactly three numbers: x y z in millimetres.")
            continue
        try:
            measured = np.asarray([float(value) for value in values], dtype=float)
        except ValueError:
            print("  Coordinates must be numbers; try again.")
            continue
        if not np.all(np.isfinite(measured)):
            print("  Coordinates must be finite numbers; try again.")
            continue
        return measured / 1e3


def mode_fk_validate():
    bus = connect()
    rows = []
    try:
        print("FK validation. Torque is OFF; move the arm by hand to 3-5 distinct poses.")
        print("Measure the tool point from the marked base-frame origin with a ruler.")
        i = 1
        while True:
            input(f"\nPose {i}: hold the arm steady, press Enter to record (Ctrl-C to stop)... ")
            deg = read_motor_degrees(bus, samples=20)
            pred = fk_position_from_motor(deg)
            print("  joints (deg): " + ", ".join(f"{n}={deg[n]:.1f}" for n in JOINT_NAMES))
            print(f"  FK predicts (mm): x={pred[0]*1e3:.1f}  y={pred[1]*1e3:.1f}  z={pred[2]*1e3:.1f}")
            meas = read_measured_position_mm()
            err = meas - pred
            rows.append((deg, pred, meas, err))
            print(f"  error vector (mm): {np.round(err*1e3, 1).tolist()}   |e| = {np.linalg.norm(err)*1e3:.1f} mm")
            i += 1
            if i > 5:
                break
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        bus.disconnect(disable_torque=True)
    if rows:
        errs = np.array([np.linalg.norm(r[3]) for r in rows])
        bias = np.mean([r[3] for r in rows], axis=0)
        print(f"\nPoses: {len(rows)}   mean error = {errs.mean()*1e3:.1f} mm   max = {errs.max()*1e3:.1f} mm")
        print(f"Mean error vector (bias, mm): {np.round(bias*1e3, 1).tolist()}")
        print("A large consistent bias in one axis means check SIGN/OFFSET_DEG, units, or link lengths.")
        with open(FK_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pose", *[f"{n}_deg" for n in JOINT_NAMES], "xFK_m", "yFK_m", "zFK_m",
                        "xmeas_m", "ymeas_m", "zmeas_m", "error_m"])
            for j, (d, p, m_, e) in enumerate(rows, 1):
                w.writerow([j, *[f"{d[n]:.3f}" for n in JOINT_NAMES], *p.round(5), *m_.round(5),
                            f"{np.linalg.norm(e):.5f}"])
        print(f"Saved {FK_CSV.name}")
    return 0


def mode_localize():
    bus = connect()
    try:
        print("Cube localization. Torque is OFF.")
        print("Guide the tool reference point (centre between the fingertips) onto the cube pickup point.")
        input("Hold the arm steady and press Enter to record... ")
        deg = read_motor_degrees(bus, samples=30)
        p_cube = fk_position_from_motor(deg)
        print(f"p_cube (m) = {np.round(p_cube, 4).tolist()}")
        CUBE_FILE.write_text(json.dumps({"p_cube": p_cube.tolist(),
                                         "q_cube_motor_deg": {n: deg[n] for n in JOINT_NAMES}}, indent=2))
        print(f"Saved {CUBE_FILE.name}. Now run --plan.")
    finally:
        bus.disconnect(disable_torque=True)
    return 0


def load_cube():
    if not CUBE_FILE.exists():
        raise Abort("cube.json not found; run --localize first")
    return np.array(json.loads(CUBE_FILE.read_text())["p_cube"], float)


def mode_plan(args):
    w, steps = build_plan(load_cube(), approach=args.approach, lift=args.lift,
                          transport=args.transport)
    print_plan(w, steps)
    return 0


def mode_run(args):
    w, steps = build_plan(load_cube(), approach=args.approach, lift=args.lift,
                          transport=args.transport)
    print_plan(w, steps)
    print(f"Mode: {'DRY RUN (no cube)' if args.dry_run else 'LIVE'}   speed x{args.speed}   "
          f"step-by-step: {args.step}")
    print("Clamp the base, clear the workspace and cables, keep hands clear, one person on the power switch.")
    if input("Type GO to start: ").strip() != "GO":
        print("cancelled")
        return 1

    bus = connect()
    try:
        arm = read_motor_degrees(bus)
        print("Starting joints (deg): " + ", ".join(f"{n}={arm[n]:.1f}" for n in JOINT_NAMES))
        hold_here_and_enable(bus)
        gripper = GRIPPER_OPEN
        for name, kind, data in steps:
            if args.step:
                input(f"\n[{name}] press Enter to execute (Ctrl-C to abort)... ")
            else:
                print(f"[{name}]")
            if kind == "path":
                follow_path(bus, data, gripper, speed=args.speed, label=name)
            else:
                arm_now = {n: read_motor_degrees(bus)[n] for n in JOINT_NAMES}
                move_gripper(bus, arm_now, data)
                gripper = data
            cur = read_motor_degrees(bus)
            print("   tool now (m): " + str(np.round(fk_position_from_motor(cur), 4).tolist()))
        print("\nSequence complete.")
        return 0
    except (Abort, KeyboardInterrupt) as exc:
        print(f"\n*** ABORT: {exc if str(exc) else 'operator interrupt'} -- freezing at current pose ***")
        freeze(bus)
        return 2
    finally:
        shutdown(bus)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check-target", nargs=3, type=float, metavar=("X", "Y", "Z"))
    g.add_argument("--read-state", action="store_true")
    g.add_argument("--gripper-test", action="store_true")
    g.add_argument("--fk-validate", action="store_true")
    g.add_argument("--localize", action="store_true")
    g.add_argument("--plan", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="label run as no-cube dry run")
    ap.add_argument("--step", action="store_true", help="pause for Enter before every state")
    ap.add_argument("--speed", type=float, default=1.0, help="speed multiplier (use 0.5 first)")
    ap.add_argument("--approach", type=float, default=APPROACH_OFFSET)
    ap.add_argument("--lift", type=float, default=LIFT_OFFSET)
    ap.add_argument("--transport", type=float, default=TRANSPORT_OFFSET)
    args = ap.parse_args()
    try:
        if args.check_target:
            return mode_check_target(np.array(args.check_target))
        if args.read_state:
            return mode_read_state()
        if args.gripper_test:
            return mode_gripper_test()
        if args.fk_validate:
            return mode_fk_validate()
        if args.localize:
            return mode_localize()
        if args.plan:
            return mode_plan(args)
        return mode_run(args)
    except Abort as exc:
        print(f"ABORT: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
