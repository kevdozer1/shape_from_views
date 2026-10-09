"""Camera poses: quaternions, look-at cameras, and poses from state estimates.

Conventions (the same as COLMAP and OpenCV):
  * A pose is world-to-camera:  X_cam = R @ X_world + t.  The camera centre is C = -R.T @ t.
  * Camera axes: +x points RIGHT in the image, +y points DOWN, +z points forward (viewing direction).
  * Generated poses: world +Z is up and the target (the object) sits at the origin.

State estimates describe where the observer is around the target:
  distance  - observer to target
  yaw       - azimuth around the target's vertical axis; yaw 0 puts the observer on the -Y axis
  pitch     - elevation above the target's horizontal plane
  roll      - rotation of the camera about its own viewing direction
The state files store yaw/pitch/roll as differences from frame 0, so the absolute value is
baseline + difference (the baseline is frame 0's absolute attitude).
"""
import math

import numpy as np


def qvec_to_rotmat(qvec):
    """Rotation matrix of a (qw, qx, qy, qz) quaternion."""
    n = math.sqrt(qvec[0] * qvec[0] + qvec[1] * qvec[1] + qvec[2] * qvec[2] + qvec[3] * qvec[3])
    w, x, y, z = qvec[0] / n, qvec[1] / n, qvec[2] / n, qvec[3] / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rotmat_to_qvec(R):
    """(qw, qx, qy, qz) of a rotation matrix, normalised and with qw >= 0."""
    R = [[float(v) for v in row] for row in R]
    trace = R[0][0] + R[1][1] + R[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1) * 2
        q = (s / 4, (R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s)
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1 + R[0][0] - R[1][1] - R[2][2]) * 2
        q = ((R[2][1] - R[1][2]) / s, s / 4, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s)
    elif R[1][1] > R[2][2]:
        s = math.sqrt(1 + R[1][1] - R[0][0] - R[2][2]) * 2
        q = ((R[0][2] - R[2][0]) / s, (R[0][1] + R[1][0]) / s, s / 4, (R[1][2] + R[2][1]) / s)
    else:
        s = math.sqrt(1 + R[2][2] - R[0][0] - R[1][1]) * 2
        q = ((R[1][0] - R[0][1]) / s, (R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, s / 4)
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    sign = -1.0 if q[0] < 0 else 1.0
    return tuple(sign * v / n for v in q)


def camera_center(R, t):
    """C = -R^T t, written out so every implementation rounds the same way."""
    return np.array([-(R[0][j] * t[0] + R[1][j] * t[1] + R[2][j] * t[2]) for j in range(3)])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _normalize(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / n, v[1] / n, v[2] / n)


def look_at(C, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    """World-to-camera rotation of a camera at C looking at target, with OpenCV axes.

    The rows of R are the camera's right, down and forward directions in world coordinates.
    (The original C++ used right = up x forward and kept "up" as the camera's y axis, which
    gives x = LEFT and y = UP: every image ends up rotated 180 degrees.)
    """
    forward = _normalize((target[0] - C[0], target[1] - C[1], target[2] - C[2]))
    right = _cross(forward, up)
    if math.sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]) < 1e-9:
        right = _cross(forward, (0.0, 1.0, 0.0))   # looking straight up/down: pick any "up"
    right = _normalize(right)
    down = _cross(forward, right)
    return np.array([right, down, forward])


def pose_from_state(distance, yaw_deg, pitch_deg, roll_deg=0.0):
    """(R, t) of an observer at the given distance/azimuth/elevation looking at the origin."""
    deg = math.pi / 180.0
    yaw, pitch, roll = yaw_deg * deg, pitch_deg * deg, roll_deg * deg
    C = (distance * math.cos(pitch) * math.sin(yaw),
         -distance * math.cos(pitch) * math.cos(yaw),
         distance * math.sin(pitch))
    R0 = look_at(C)
    c, s = math.cos(roll), math.sin(roll)
    Rroll = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]   # spin about the viewing axis
    R = np.array([[Rroll[i][0] * R0[0][j] + Rroll[i][1] * R0[1][j] + Rroll[i][2] * R0[2][j]
                   for j in range(3)] for i in range(3)])
    t = np.array([-(R[i][0] * C[0] + R[i][1] * C[1] + R[i][2] * C[2]) for i in range(3)])
    return R, t


def state_from_pose(R, distance):
    """The inverse of pose_from_state: (distance, yaw, pitch, roll) of the camera with world-to-camera
    rotation R that sits `distance` from the target and looks straight at it (pitch must not be +-90)."""
    C = [-distance * R[2][j] for j in range(3)]          # it looks along R[2], at the origin
    yaw = math.degrees(math.atan2(C[0], -C[1]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, C[2] / distance))))
    L = look_at(C)                                       # the same camera before its roll
    # R = Rroll @ L, so Rroll = R @ L^T = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    c = R[0][0] * L[0][0] + R[0][1] * L[0][1] + R[0][2] * L[0][2]
    s = R[1][0] * L[0][0] + R[1][1] * L[0][1] + R[1][2] * L[0][2]
    return distance, yaw, pitch, math.degrees(math.atan2(s, c))


def orbit_states(n_images, rings):
    """Absolute (distance, yaw, pitch, roll) per frame for cameras circling the origin.

    rings = [(radius, height), ...]; the images are split evenly over the rings in order and
    each ring is a full 360-degree circle starting at yaw 0.
    """
    if n_images % len(rings) != 0:
        raise ValueError(f"{n_images} images cannot be split evenly over {len(rings)} rings")
    per_ring = n_images // len(rings)
    states = []
    for i in range(n_images):
        radius, height = rings[i // per_ring]
        yaw = (360.0 / per_ring) * (i % per_ring)
        distance = math.sqrt(radius * radius + height * height)
        pitch = math.atan2(height, radius) * (180.0 / math.pi)
        states.append((distance, yaw, pitch, 0.0))
    return states


def to_differential(states):
    """Split absolute states into (frame-0 baseline, per-frame differences); yaw wrapped to (-180, 180]."""
    _, yaw0, pitch0, roll0 = states[0]
    differences = []
    for distance, yaw, pitch, roll in states:
        d_yaw = yaw - yaw0
        while d_yaw > 180:
            d_yaw -= 360
        while d_yaw <= -180:
            d_yaw += 360
        differences.append((distance, d_yaw, pitch - pitch0, roll - roll0))
    return (yaw0, pitch0, roll0), differences
