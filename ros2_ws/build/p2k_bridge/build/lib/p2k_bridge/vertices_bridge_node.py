#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from .socket_utils import TcpServerFrameReader


class VerticesBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("vertices_bridge")
        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 5555)
        self.declare_parameter("num_vertices", 20)
        self.declare_parameter("topic", "/p2k/vertices")
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("poll_hz", 200.0)

        host = self.get_parameter("host").get_parameter_value().string_value
        port = self.get_parameter("port").get_parameter_value().integer_value
        self._num_vertices = self.get_parameter("num_vertices").get_parameter_value().integer_value
        topic = self.get_parameter("topic").get_parameter_value().string_value
        self._frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        poll_hz = self.get_parameter("poll_hz").get_parameter_value().double_value

        if self._num_vertices <= 0:
            raise ValueError("num_vertices must be > 0")

        self._bytes_per_frame = int(self._num_vertices * 3 * 4)
        self._reader = TcpServerFrameReader(host=host, port=int(port), timeout_s=0.2)
        self._pub = self.create_publisher(PointCloud2, topic, 10)
        self._listening = self._reader.start()
        self._connected = False

        period = 1.0 / max(poll_hz, 1.0)
        self.create_timer(period, self._tick)
        self.get_logger().info(
            f"vertices_bridge listening on {host}:{int(port)} "
            f"num_vertices={self._num_vertices} topic={topic}"
        )

    def _tick(self) -> None:
        if not self._listening:
            self._listening = self._reader.start()
            return

        if not self._connected:
            self._connected = self._reader.accept_if_needed()
            if self._connected:
                self.get_logger().info("Publisher connected on vertices socket")
            return

        payload = self._reader.recv_exact(self._bytes_per_frame)
        if payload is None:
            if self._connected:
                self.get_logger().warn("Vertices publisher disconnected, waiting for reconnect...")
            self._connected = False
            return

        arr = np.frombuffer(payload, dtype=np.float32).reshape(self._num_vertices, 3)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self._frame_id
        msg = point_cloud2.create_cloud_xyz32(header, arr.tolist())
        self._pub.publish(msg)

    def destroy_node(self) -> bool:
        self._reader.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = VerticesBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
