# ROS2 Workspace

This workspace contains bridge nodes to convert TCP socket streams into ROS2 topics.

## Layout

- `src/p2k_bridge`: socket-to-ROS bridge nodes
  - `vertices_bridge`: reads vertices socket, publishes `sensor_msgs/PointCloud2`
  - `yaw_bridge`: reads yaw socket, publishes `geometry_msgs/PoseStamped`
  - `pose_operation`: subscribes to `/p2k/vertices` and computes basic diagnostics on the received vertices

## Build

```bash
cd ros2_ws
colcon build --packages-select p2k_bridge
source install/setup.bash
```

## Run

```bash
ros2 launch p2k_bridge bridge.launch.py
```

Override parameters at launch time as needed (ports, topic names, frame id, vertex count).
