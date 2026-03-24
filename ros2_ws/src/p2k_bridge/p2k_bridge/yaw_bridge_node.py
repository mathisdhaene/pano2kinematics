#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from .socket_utils import TcpServerFrameReader


def yaw_deg_to_quaternion(yaw_deg: float) -> tuple[float, float, float, float]:
    yaw_rad = math.radians(float(yaw_deg))
    half = 0.5 * yaw_rad
    return (0.0, 0.0, math.sin(half), math.cos(half))


class YawBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("yaw_bridge")
        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 5556)
        self.declare_parameter("topic", "/p2k/yaw_pose")
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("poll_hz", 200.0)

        host = self.get_parameter("host").get_parameter_value().string_value
        port = self.get_parameter("port").get_parameter_value().integer_value
        topic = self.get_parameter("topic").get_parameter_value().string_value
        self._frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        poll_hz = self.get_parameter("poll_hz").get_parameter_value().double_value

        self._reader = TcpServerFrameReader(host=host, port=int(port), timeout_s=0.2)
        self._pub = self.create_publisher(PoseStamped, topic, 10)
        self._listening = self._reader.start()
        self._connected = False

        period = 1.0 / max(poll_hz, 1.0)
        self.create_timer(period, self._tick)
        self.get_logger().info(f"yaw_bridge listening on {host}:{int(port)} topic={topic}")

    def _tick(self) -> None:
        if not self._listening:
            self._listening = self._reader.start()
            return

        if not self._connected:
            self._connected = self._reader.accept_if_needed()
            if self._connected:
                self.get_logger().info("Publisher connected on yaw socket")
            return

        payload = self._reader.recv_exact(4)
        if payload is None:
            if self._connected:
                self.get_logger().warn("Yaw publisher disconnected, waiting for reconnect...")
            self._connected = False
            return

        yaw_deg = float(np.frombuffer(payload, dtype=np.float32)[0])
        qx, qy, qz, qw = yaw_deg_to_quaternion(yaw_deg)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.pose.position.x = 0.0
        msg.pose.position.y = 0.0
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self._pub.publish(msg)

    def destroy_node(self) -> bool:
        self._reader.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = YawBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
