"""How results leave this module."""
from __future__ import annotations

import json
import sys
from typing import List

from .types import IdentityResult


class StdoutPublisher:
    """One JSON object per line, one line per frame."""

    def publish(self, results: List[IdentityResult]) -> None:
        sys.stdout.write(json.dumps([r.to_dict() for r in results]) + "\n")
        sys.stdout.flush()

    def close(self) -> None:
        pass


class Ros2Publisher:
    """Same JSON, as std_msgs/String on a ROS 2 topic. UNTESTED: needs a ROS 2 install."""

    def __init__(self, topic: str = "/person_id/result") -> None:
        import rclpy
        from std_msgs.msg import String

        self._rclpy, self._String = rclpy, String
        rclpy.init()
        self._node = rclpy.create_node("person_id")
        self._pub = self._node.create_publisher(String, topic, 10)

    def publish(self, results: List[IdentityResult]) -> None:
        msg = self._String()
        msg.data = json.dumps([r.to_dict() for r in results])
        self._pub.publish(msg)

    def close(self) -> None:
        self._node.destroy_node()
        self._rclpy.shutdown()


def build_publisher(kind: str, ros2_topic: str = "/person_id/result"):
    if kind == "stdout":
        return StdoutPublisher()
    if kind == "ros2":
        return Ros2Publisher(ros2_topic)
    raise ValueError(f"Unknown publisher '{kind}' (use 'stdout' or 'ros2')")
