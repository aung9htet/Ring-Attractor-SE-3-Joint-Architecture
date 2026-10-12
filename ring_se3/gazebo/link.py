"""Gazebo's own account of where the hand is.

The loop takes only joint angles from Gazebo (:class:`ring_so3.gazebo.link.RosLink`, copied from
the SO-3 repo).  A pose goal is also a position, and every "true position" of the results is
worked out from joint angles with the chain's kinematics.  So those kinematics are checked
against the simulator: :class:`PoseRosLink` adds the pose of a link taken from Gazebo's link
states, and :func:`kinematics_check` compares the two at held poses.

The chains are given in the frame of ``torso_lift_link``.  Link ``arm_k_link`` is the frame of
joint k.  Gazebo joins links that hang on a fixed joint to their parent, so ``arm_tool_link`` does
not exist there: the end point of ``tiago7`` is taken as the tool's fixed offset in
``arm_7_link``.  A shorter chain ends at the origin of the next joint, which Gazebo has as a link
of its own; that is the stricter check, since it does not use the chain's own end offset.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ring_so3.chains import TIAGO_BASE_LINK
from ring_so3.gazebo.link import ARM_JOINTS, MODEL_NAME, RosLink
from ring_so3.kinematics import Chain, rotation_angle

from .. import kinematics


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """The rotation of a unit quaternion (x, y, z, w)."""

    n = float(np.sqrt(x * x + y * y + z * z + w * w))
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def arm_link(k: int) -> str:
    """The link that joint k (1 .. 7) moves."""

    return ARM_JOINTS[int(k) - 1].replace("_joint", "_link")


class PoseRosLink(RosLink):
    """:class:`~ring_so3.gazebo.link.RosLink` with the pose of a link from Gazebo's link states."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from gazebo_msgs.srv import GetLinkState

        self._rospy.wait_for_service("/gazebo/get_link_state", timeout=30.0)
        self._get_link = self._rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)

    def _world(self, link: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            state = self._get_link(link_name="%s::%s" % (MODEL_NAME, link), reference_frame="world")
        except Exception:  # noqa: BLE001 - the service is gone: unknown
            return None
        if not state.success:
            return None
        p, o = state.link_state.pose.position, state.link_state.pose.orientation
        return np.array([p.x, p.y, p.z], dtype=float), quaternion_matrix(o.x, o.y, o.z, o.w)

    def link_pose(self, link: str, reference: str = TIAGO_BASE_LINK) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Origin (3,) and orientation (3, 3) of ``link`` in the frame of ``reference``; None when Gazebo has no such link.

        Both poses are asked for in the world frame and put together here.  Asked for a relative pose, Gazebo's
        service gives the right origin but multiplies the two orientations in the wrong order, which shows as
        soon as the robot's base is not exactly upright (measured: up to 0.9 degrees)."""

        body, frame = self._world(link), self._world(reference)
        if body is None or frame is None:
            return None
        return frame[1].T @ (body[0] - frame[0]), frame[1].T @ body[1]


def end_point(link: Any, chain: Chain) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Position and orientation of the chain's end point as Gazebo has them, in the chain's base frame.
    ``link``: anything with :meth:`PoseRosLink.link_pose`."""

    last = link.link_pose(arm_link(chain.n))
    if last is None:
        return None
    origin, rotation = last
    position = origin + rotation @ np.array(chain.tool_offset, dtype=float)
    if chain.n < len(ARM_JOINTS):                       # the next joint's origin is a link of its own in Gazebo
        following = link.link_pose(arm_link(chain.n + 1))
        if following is not None:
            position = following[0]
    return position, rotation @ np.array(chain.tool_fixed, dtype=float)


def kinematics_check(link: Any, robot: Any, chain: Chain, poses: Sequence[Sequence[float]], log=print) -> Dict[str, Any]:
    """Move the Gazebo arm to each of ``poses`` and compare the end point Gazebo has with the chain's kinematics at
    the joint angles Gazebo reports.  ``robot``: a :class:`ring_so3.robots.GazeboRobot` on ``link``."""

    rows: List[Dict[str, Any]] = []
    for index, pose in enumerate(poses):
        robot.reset(pose)
        q = robot.read()
        seen = end_point(link, chain)
        if seen is None:
            rows.append({"index": index, "q": q.tolist(), "position_error_m": None, "orientation_error_deg": None})
            continue
        position, rotation = seen
        row = {"index": index, "q": q.tolist(), "gazebo_position": position.tolist(), "chain_position": kinematics.position(chain, q).tolist(),
               "position_error_m": float(np.linalg.norm(position - kinematics.position(chain, q))),
               "orientation_error_deg": float(np.degrees(rotation_angle(rotation, chain.rotation(q)))),
               "asked_off_rad": float(np.max(np.abs(np.asarray(pose, dtype=float) - q)))}
        rows.append(row)
        if log is not None:
            log("pose %d: Gazebo and the chain's kinematics differ by %.2f mm and %.3f deg (the arm is %.4f rad from the pose asked for)" % (
                index, 1000.0 * row["position_error_m"], row["orientation_error_deg"], row["asked_off_rad"]))
    known = [r for r in rows if r["position_error_m"] is not None]
    return {"experiment": "gazebo_kinematics_check", "chain": chain.name, "base_link": TIAGO_BASE_LINK, "end_link": arm_link(chain.n), "rows": rows,
            "largest_position_error_m": max((r["position_error_m"] for r in known), default=None),
            "largest_orientation_error_deg": max((r["orientation_error_deg"] for r in known), default=None)}


__all__ = ["PoseRosLink", "arm_link", "end_point", "kinematics_check", "quaternion_matrix"]
