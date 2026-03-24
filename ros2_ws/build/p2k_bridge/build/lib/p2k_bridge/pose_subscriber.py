#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class PoseOperationNode(Node):
    def __init__(self) -> None:
        super().__init__("pose_operation")

        self.declare_parameter("topic", "/p2k/vertices")
        self.declare_parameter("log_every_n", 30)
        topic = self.get_parameter("topic").get_parameter_value().string_value
        self._log_every_n = max(
            1,
            int(self.get_parameter("log_every_n").get_parameter_value().integer_value),
        )
        self._frame_idx = 0

        self._sub = self.create_subscription(
            PointCloud2,
            topic,
            self._callback,
            10,
        )

        self.get_logger().info(
            f"Listening to {topic} and reporting every {self._log_every_n} frame(s)"
        )

    def _callback(self, msg: PointCloud2) -> None:
        self._frame_idx += 1

        points = point_cloud2.read_points_numpy(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=False,
        )

        if points.size == 0:
            self.get_logger().warn("Received empty point cloud")
            return

        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        finite_mask = np.isfinite(points).all(axis=1)
        valid_points = points[finite_mask]

        if valid_points.size == 0:
            self.get_logger().warn("Received point cloud but all vertices are invalid")
            return

        if self._frame_idx % self._log_every_n != 0:
            return

        centroid = valid_points.mean(axis=0)
        mins = valid_points.min(axis=0)
        maxs = valid_points.max(axis=0)
        self.get_logger().info(
            "frame=%d vertices=%d valid=%d centroid=(%.3f, %.3f, %.3f) "
            "bounds_min=(%.3f, %.3f, %.3f) bounds_max=(%.3f, %.3f, %.3f)"
            % (
                self._frame_idx,
                points.shape[0],
                valid_points.shape[0],
                centroid[0],
                centroid[1],
                centroid[2],
                mins[0],
                mins[1],
                mins[2],
                maxs[0],
                maxs[1],
                maxs[2],
            )
        )


def main() -> None:
    rclpy.init()
    node = PoseOperationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
