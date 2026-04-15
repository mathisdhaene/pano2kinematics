import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/mathis/PANORAMICS/deps/pano2kinematics/ros2_ws/install/p2k_bridge'
