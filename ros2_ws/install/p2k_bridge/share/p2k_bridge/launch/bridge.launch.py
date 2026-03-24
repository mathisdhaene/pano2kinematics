from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("p2k_bridge")
    params_file = pkg_share + "/config/bridge.yaml"
    return LaunchDescription(
        [
            Node(
                package="p2k_bridge",
                executable="vertices_bridge",
                name="vertices_bridge",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="p2k_bridge",
                executable="yaw_bridge",
                name="yaw_bridge",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="p2k_bridge",
                executable="pose_operation",
                name="pose_operation",
                output="screen",
                parameters=[params_file],
            ),
        ]
    )
