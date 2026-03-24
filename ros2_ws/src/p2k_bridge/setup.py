from setuptools import find_packages, setup

package_name = "p2k_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/bridge.launch.py"]),
        ("share/" + package_name + "/config", ["config/bridge.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mathis",
    maintainer_email="mathis@example.com",
    description="Socket to ROS2 bridge for pano2kinematics streams.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vertices_bridge = p2k_bridge.vertices_bridge_node:main",
            "yaw_bridge = p2k_bridge.yaw_bridge_node:main",
            "vertices_debug = p2k_bridge.vertices_debug_node:main",
            "pose_operation = p2k_bridge.pose_subscriber:main",
        ],
    },
)
