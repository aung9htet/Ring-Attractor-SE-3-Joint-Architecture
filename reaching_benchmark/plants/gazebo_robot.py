"""The TIAGo arm in Gazebo as the robot of the closed loop, advanced in lock step with the network.

Physics stays paused.  Every ``follow`` sends the targets to the arm's trajectory controller and then lets
exactly one tick of physics pass (:meth:`RosLink.step`: unpause, wait until ``/clock`` has advanced by the
tick, pause).  Targets follow an absolute schedule, so the small overshoot of one step is not added to the
next.  Nothing here calls ``rospy.sleep``: with ``/use_sim_time`` it would wait on a paused clock for ever.
The mobile base is not anchored; :meth:`RosLink.put_base_back` returns it to the origin before each trial.

Needs ROS Noetic and a running Gazebo (``docker/run.bash gazebo up NAME``); imported only by the Gazebo
benchmark and by the simulator's ``--gazebo`` mode.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Empty
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from reaching_benchmark.geometry.chains import Chain

ARM_JOINTS: Tuple[str, ...] = tuple("arm_%d_joint" % k for k in range(1, 8))
ARM_COMMAND_TOPIC = "/arm_controller/command"
JOINT_STATES_TOPIC = "/joint_states"
CAMERA_TOPIC = "/ring_ik_camera/image_raw"
MODEL_NAME = "tiago"
#: the arm is still when no joint has moved more than this (rad) over this many ticks (the wrist's reported
#: velocities are not usable: they read above 1 rad/s while the angle does not change)
STILL_RAD = 0.003
STILL_TICKS = 6


class StepTimeout(RuntimeError):
    pass


class RosLink:
    """Joint states in, arm targets out, physics in lock step, and the world camera's latest frame."""

    def __init__(self, node_name: str = "reaching_benchmark", physics_step_s: float = 0.001, step_timeout_s: float = 20.0,
                 camera_topic: Optional[str] = CAMERA_TOPIC) -> None:
        self.physics_step_s = float(physics_step_s)
        self.step_timeout_s = float(step_timeout_s)
        self._lock = threading.Lock()
        self._clock = threading.Condition()
        self._sim_time_s = 0.0
        self._state: Optional[Tuple[List[float], List[float], float]] = None
        self._frame = None
        self._frame_count = 0
        self._schedule_s: Optional[float] = None
        self.overshoot_s = 0.0
        rospy.init_node(node_name, anonymous=True, disable_signals=True)
        self._command = rospy.Publisher(ARM_COMMAND_TOPIC, JointTrajectory, queue_size=1)
        rospy.Subscriber(JOINT_STATES_TOPIC, JointState, self._on_joints, queue_size=1)
        rospy.Subscriber("/clock", Clock, self._on_clock, queue_size=1)
        if camera_topic:
            rospy.Subscriber(camera_topic, Image, self._on_image, queue_size=1, buff_size=2 ** 24)
        for name in ("/gazebo/pause_physics", "/gazebo/unpause_physics", "/gazebo/get_model_state", "/gazebo/set_model_state"):
            rospy.wait_for_service(name, timeout=30.0)
        self._pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self._unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self._get_model = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self._set_model = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self._unpause()                        # joint states come only while physics runs
        deadline = time.monotonic() + 30.0
        while self._state is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._state is None:
            raise RuntimeError("no joint states from Gazebo within 30 s")
        self._pause()

    def _on_joints(self, message: JointState) -> None:
        positions = dict(zip(message.name, message.position))
        velocities = dict(zip(message.name, message.velocity)) if message.velocity else {}
        if any(name not in positions for name in ARM_JOINTS):
            return
        with self._lock:
            self._state = ([float(positions[name]) for name in ARM_JOINTS], [float(velocities.get(name, 0.0)) for name in ARM_JOINTS],
                           message.header.stamp.to_sec())

    def _on_clock(self, message: Clock) -> None:
        with self._clock:
            self._sim_time_s = message.clock.to_sec()
            self._clock.notify_all()

    def _on_image(self, message: Image) -> None:
        with self._lock:
            self._frame = message
            self._frame_count += 1

    def frames(self) -> int:
        """Camera frames received so far (0: the camera gives no picture)."""

        return self._frame_count

    def joint_state(self) -> Tuple[List[float], List[float], float]:
        with self._lock:
            return self._state

    def sim_time_s(self) -> float:
        with self._clock:
            return self._sim_time_s

    def base_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            state = self._get_model(model_name=MODEL_NAME, relative_entity_name="world")
        except rospy.ServiceException:
            return None
        if not state.success:
            return None
        o = state.pose.orientation
        yaw = math.atan2(2.0 * (o.w * o.z + o.x * o.y), 1.0 - 2.0 * (o.y * o.y + o.z * o.z))
        return float(state.pose.position.x), float(state.pose.position.y), float(yaw)

    def put_base_back(self) -> bool:
        try:
            current = self._get_model(model_name=MODEL_NAME, relative_entity_name="world")
            if not current.success:
                return False
            state = ModelState()
            state.model_name, state.reference_frame = MODEL_NAME, "world"
            state.pose.position.z = current.pose.position.z
            state.pose.orientation.w = 1.0
            return bool(self._set_model(model_state=state).success)
        except rospy.ServiceException:
            return False

    def jpeg(self, quality: int = 70) -> Optional[bytes]:
        """The latest camera frame as JPEG (None without a picture: the camera needs a GPU container)."""

        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        channels = 3 if frame.encoding in ("rgb8", "bgr8") else 1
        image = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, channels)
        if frame.encoding == "rgb8":
            image = image[:, :, ::-1]
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return encoded.tobytes() if ok else None

    def send(self, positions: Sequence[float], time_s: float) -> None:
        """Targets for the seven arm joints, to be reached ``time_s`` from now (sim time)."""

        message = JointTrajectory()
        message.header.stamp = rospy.Time.from_sec(self.sim_time_s())
        message.joint_names = list(ARM_JOINTS)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in positions]
        point.time_from_start = rospy.Duration.from_sec(float(time_s))
        message.points.append(point)
        self._command.publish(message)

    def step(self, iterations: int) -> float:
        """Let ``iterations`` physics steps pass; returns the sim time afterwards."""

        before = self.sim_time_s()
        span = int(iterations) * self.physics_step_s
        target = before + span
        if self._schedule_s is not None and self._schedule_s + span - before >= 0.5 * span:
            target = self._schedule_s + span
        self._unpause()
        deadline = time.monotonic() + self.step_timeout_s
        try:
            with self._clock:
                while self._sim_time_s < target - 1e-9:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise StepTimeout("Gazebo clock at %.3f s, target %.3f s" % (self._sim_time_s, target))
                    self._clock.wait(min(remaining, 0.05))
        finally:
            self._pause()
        self._schedule_s = target
        after = self.sim_time_s()
        self.overshoot_s = after - target
        time.sleep(0.004)                      # the joint states of the last physics step are still on their way
        return after


class GazeboRobot:
    """The robot interface of plants/light_robot.py over a :class:`RosLink`; the chain's joints are the arm's first ones."""

    name = "gazebo"
    tolerance = 0.01
    place_speed = 1.0

    def __init__(self, chain: Chain, link: RosLink) -> None:
        if not chain.name.startswith("tiago"):
            raise ValueError("Gazebo has the TIAGo arm only; %s is not one of its chains" % chain.name)
        self.chain = chain
        self.link = link
        self.limits = chain.limits()
        self.q = np.zeros(chain.n)
        self.others = np.zeros(0)
        self.lag = np.zeros(chain.n)                 # target of the last tick minus where the joint is now
        self.velocity = np.zeros(chain.n)            # the velocity commanded in the last tick (the write-ahead uses it)
        #: jammed joints and the angle each is held at (see :meth:`lock`)
        self.locked: Dict[int, float] = {}
        self.read()
        self.target = self.q.copy()

    def read(self) -> np.ndarray:
        positions, velocities, _ = self.link.joint_state()
        self.q = np.array(positions[: self.chain.n], dtype=float)
        #: the joints' velocities as Gazebo reports them (the learned baselines' state; the RAK model does not read it)
        self.measured_velocity = np.array(velocities[: self.chain.n], dtype=float) if len(velocities) >= self.chain.n else np.zeros(self.chain.n)
        self.others = np.array(positions[self.chain.n:], dtype=float)
        return self.q

    def difference(self, target: Sequence[float]) -> np.ndarray:
        return np.asarray(target, dtype=float) - self.q

    def follow(self, targets: Sequence[float], time_s: float, dt: float, exact: bool = False) -> np.ndarray:
        wanted = np.array(targets, dtype=float)
        if self.locked and not exact:                # a jammed joint is held where it jammed
            for joint, angle in self.locked.items():
                wanted[joint] = angle
        self.target = np.clip(np.where(np.isfinite(wanted), wanted, self.q), self.limits[:, 0], self.limits[:, 1])
        self.velocity = (self.target - self.q) / max(float(time_s), float(dt))
        self.link.send(list(self.target) + list(self.others), max(float(time_s), float(dt)))
        self.link.step(int(round(float(dt) / self.link.physics_step_s)))
        self.read()
        self.lag = self.target - self.q
        return self.q

    def reset(self, q: Sequence[float], speed: float = 1.0, dt: float = 0.05, settle_s: float = 1.0) -> None:
        """Move to ``q`` at up to ``speed`` rad/s and wait until the arm is there and has stopped moving."""

        goal = np.clip(np.asarray(q, dtype=float), self.limits[:, 0], self.limits[:, 1])
        self.link.put_base_back()
        recent: List[np.ndarray] = []
        for _ in range(2000):
            difference = goal - self.q
            recent = (recent + [self.q.copy()])[-STILL_TICKS:]
            still = len(recent) == STILL_TICKS and float(np.max(np.abs(recent[-1] - recent[0]))) < STILL_RAD
            if np.max(np.abs(difference)) < self.tolerance and still:
                break
            self.follow(self.q + np.clip(difference, -speed * 4 * dt, speed * 4 * dt), 4 * dt, dt)
        for _ in range(int(round(settle_s / dt))):
            self.follow(goal, 4 * dt, dt)

    def displace(self, joint: int, angle: float) -> None:
        raise NotImplementedError("a joint cannot be pushed from outside in Gazebo")

    def lock(self, joint: int) -> None:
        """Jam a joint, as a brake would. From now on its trajectory target is the angle it had when it jammed, whatever
        the network asks for, so the controller holds it there against gravity and the other joints' motion. Any
        drift the controller allows is real and is recorded. Moves by hand (``exact``) still move it."""

        self.locked[int(joint)] = float(self.q[int(joint)])

    def unlock(self) -> None:
        self.locked.clear()

    def describe(self) -> dict:
        return {"robot": self.name, "physics_step_s": self.link.physics_step_s, "base": self.link.base_pose(), "locked": sorted(self.locked)}
