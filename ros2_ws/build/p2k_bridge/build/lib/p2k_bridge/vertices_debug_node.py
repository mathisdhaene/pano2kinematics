#!/usr/bin/env python3
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class VerticesDebugNode(Node):
    def __init__(self) -> None:
        super().__init__("vertices_debug")
        self.declare_parameter("topic", "/p2k/vertices")
        self.declare_parameter("max_points", 5)
        self.declare_parameter("every_n", 1)

        topic = self.get_parameter("topic").get_parameter_value().string_value
        self._max_points = int(self.get_parameter("max_points").get_parameter_value().integer_value)
        self._every_n = max(1, int(self.get_parameter("every_n").get_parameter_value().integer_value))
        self._frame_idx = 0

        self.create_subscription(PointCloud2, topic, self._cb, 10)
        self.get_logger().info(
            f"Listening {topic} (printing first {self._max_points} points every {self._every_n} frame(s))"
        )

    def _cb(self, msg: PointCloud2) -> None:
        self._frame_idx += 1
        if self._frame_idx % self._every_n != 0:
            return

        pts: List[Tuple[float, float, float]] = list(
            point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False)
        )
        shown = pts[: self._max_points]
        self.get_logger().info(
            f"frame={self._frame_idx} n={len(pts)} first={[(round(x, 4), round(y, 4), round(z, 4)) for x, y, z in shown]}"
        )


def main() -> None:
    rclpy.init()
    node = VerticesDebugNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
