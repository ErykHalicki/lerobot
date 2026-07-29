"""Gravity torque model for the reBot B601-DM arm, in plain numpy.

Gravity compensation only needs the generalized gravity vector g(q), which for a
serial chain is the gradient of potential energy:

    U(q)   = -sum_j m_j * (g_vec . c_j(q))
    g(q)_i = dU/dq_i = sum_{j >= i} m_j * (-g_vec) . (z_i x (c_j - o_i))

where c_j is link j's centre of mass in world frame, o_i / z_i are joint i's
origin and axis in world frame. That is exactly what pinocchio's
computeGeneralizedGravity returns, in about forty lines and with no dependency
beyond numpy -- worth it on the arm's Raspberry Pi, where pinocchio drags in
eigenpy and boost.

Link masses, centres of mass and joint origins are transcribed from the
manufacturer's reBot-DevArm_fixend URDF. Joint angles are the raw Damiao motor
angles in radians, so they line up with the URDF zero pose as long as the motor
zeros were set there (both lerobot-calibrate and the vendor's zeroing script set
zeros at the pose the arm is physically held in).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.81])

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
)


@dataclass(frozen=True)
class Link:
    """One rigid body, its mass, and how its frame attaches to its parent.

    `xyz`/`rpy` place this frame in the parent link's frame. `axis` is the
    revolute axis in that frame, or None for a rigidly attached body. `com` is
    the centre of mass in this link's own frame.
    """

    name: str
    parent: int  # index into LINKS, -1 for the base
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float] | None
    mass: float
    com: tuple[float, float, float]
    effort: float = 0.0  # motor torque limit, N.m; 0 for fixed bodies


# Order matters: revolute links appear in joint order, so LINKS[i].axis is not
# None exactly for the i-th entry of JOINT_NAMES, and every link is a descendant
# of the revolute links before it.
LINKS: tuple[Link, ...] = (
    Link(
        name="link1",
        parent=-1,
        xyz=(-8.416e-05, 0.0, 0.08465),
        rpy=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        mass=0.1613,
        com=(0.000113614552951627, -0.000616319527051323, 0.0236476372671394),
        effort=27.0,
    ),
    Link(
        name="link2",
        parent=0,
        xyz=(0.020084, 0.031625, 0.05555),
        rpy=(-1.5708, 0.0, 0.0),
        axis=(0.0, 0.0, -1.0),
        mass=1.3266,
        com=(-0.13225622308888, -0.0030617036386309, -0.0308306967030205),
        effort=27.0,
    ),
    Link(
        name="link3",
        parent=1,
        xyz=(-0.264, 0.0, 0.0),
        rpy=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        mass=0.8353,
        com=(0.121040035791843, -0.0536211076627949, -0.0310137854608077),
        effort=27.0,
    ),
    Link(
        name="link4",
        parent=2,
        xyz=(0.2426, -0.054, -0.001625),
        rpy=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        mass=0.52,
        com=(0.0608200956293136, -0.0511711906613122, -0.030299458623927),
        effort=7.0,
    ),
    Link(
        name="link5",
        parent=3,
        xyz=(0.078308, -0.0375, -0.03),
        rpy=(-1.5708, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        mass=0.383,
        com=(-0.00502802058982517, 1.73866206692364e-06, 0.0386233236326755),
        effort=7.0,
    ),
    Link(
        name="link6",
        parent=4,
        xyz=(0.028008, 0.0, 0.04),
        rpy=(0.0, 1.5708, 0.0),
        axis=(0.0, 0.0, 1.0),
        mass=0.3663,
        com=(3.76418727127126e-06, -0.000100908819946677, 0.0253308606425965),
        effort=7.0,
    ),
    # Gripper assembly, modelled as one rigid body bolted to link6. Add a
    # payload by raising this mass and shifting its com, or via payload= below.
    Link(
        name="end_link",
        parent=5,
        xyz=(0.0, 0.0, 0.15539),
        rpy=(0.0, -1.5708, 3.1415),
        axis=None,
        mass=0.45,
        com=(-0.0737654295815033, -9.5080995865868e-06, 7.04327840286845e-06),
    ),
)

NUM_JOINTS = sum(1 for link in LINKS if link.axis is not None)
EFFORT_LIMITS = np.array([link.effort for link in LINKS if link.axis is not None])

_IDENTITY = np.eye(3)
_ORIGIN = np.zeros(3)


def _rpy_to_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw to a rotation matrix: Rz(y) Ry(p) Rx(r)."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    """Rodrigues rotation of `angle` radians about a unit `axis`."""
    x, y, z = axis
    c, s, t = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


# Every per-link quantity that doesn't depend on q, built once at import: this
# is the arm's fixed geometry, so rebuilding it per call is pure overhead.
_RPY = tuple(_rpy_to_matrix(link.rpy) for link in LINKS)
_XYZ = tuple(np.asarray(link.xyz, dtype=float) for link in LINKS)
_COM = tuple(np.asarray(link.com, dtype=float) for link in LINKS)
_AXIS = tuple(
    np.asarray(link.axis, dtype=float) if link.axis is not None else None for link in LINKS
)
_MASS = tuple(link.mass for link in LINKS)


def _triple_product(u: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """u . (a x b), written out because np.cross costs far more than the six
    multiplies it does, and this runs per joint per send tick."""
    return float(
        u[0] * (a[1] * b[2] - a[2] * b[1])
        + u[1] * (a[2] * b[0] - a[0] * b[2])
        + u[2] * (a[0] * b[1] - a[1] * b[0])
    )


def link_frames(q: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """World (rotation, origin) of every link frame at joint configuration `q`."""
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.shape[0] != NUM_JOINTS:
        raise ValueError(f"q must have {NUM_JOINTS} entries, got {q.shape[0]}")

    frames: list[tuple[np.ndarray, np.ndarray]] = []
    joint_index = 0
    for i, link in enumerate(LINKS):
        if link.parent < 0:
            rot_parent, pos_parent = _IDENTITY, _ORIGIN
        else:
            rot_parent, pos_parent = frames[link.parent]
        # Joint frame: fixed offset from the parent, before the joint rotates.
        rot_joint = rot_parent @ _RPY[i]
        pos = pos_parent + rot_parent @ _XYZ[i]
        if link.axis is not None:
            rot = rot_joint @ _axis_rotation(link.axis, q[joint_index])
            joint_index += 1
        else:
            rot = rot_joint
        frames.append((rot, pos))
    return frames


def gravity_torque(
    q: np.ndarray,
    gravity: np.ndarray = GRAVITY,
    payload: float = 0.0,
    payload_com: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Joint torques that hold the arm static against gravity, in N.m.

    `payload` adds a point mass (kg) at `payload_com`, given in the end
    effector's frame, on top of the modelled gripper.
    """
    frames = link_frames(q)
    up = -np.asarray(gravity, dtype=float)

    masses = list(_MASS)
    coms = [pos + rot @ _COM[i] for i, (rot, pos) in enumerate(frames)]
    if payload:
        rot, pos = frames[-1]
        masses.append(payload)
        coms.append(pos + rot @ np.asarray(payload_com, dtype=float))

    # Everything from body i outward, lumped: the moment sum is linear in the
    # bodies' positions, so the whole outboard chain reduces to one mass at one
    # point, and each joint needs a single moment arm rather than a term per body.
    suffix_mass = [0.0] * len(masses)
    suffix_moment = [_ORIGIN] * len(masses)
    running_mass = 0.0
    running_moment = _ORIGIN
    for j in range(len(masses) - 1, -1, -1):
        running_mass += masses[j]
        running_moment = running_moment + masses[j] * coms[j]
        suffix_mass[j] = running_mass
        suffix_moment[j] = running_moment

    tau = np.zeros(NUM_JOINTS)
    joint_index = 0
    for i, link in enumerate(LINKS):
        if link.axis is None:
            continue
        rot, origin = frames[i]
        axis_world = rot @ _AXIS[i]
        arm = suffix_moment[i] - suffix_mass[i] * origin
        tau[joint_index] = _triple_product(up, axis_world, arm)
        joint_index += 1
    return tau


def potential_energy(
    q: np.ndarray,
    gravity: np.ndarray = GRAVITY,
    payload: float = 0.0,
    payload_com: Sequence[float] = (0.0, 0.0, 0.0),
) -> float:
    """Gravitational potential energy in joules; gravity_torque is its gradient."""
    frames = link_frames(q)
    energy = 0.0
    for link, (rot, pos) in zip(LINKS, frames, strict=True):
        energy -= link.mass * np.asarray(gravity) @ (pos + rot @ np.asarray(link.com))
    if payload:
        rot, pos = frames[-1]
        energy -= (
            payload
            * np.asarray(gravity)
            @ (pos + rot @ np.asarray(payload_com, dtype=float))
        )
    return float(energy)
