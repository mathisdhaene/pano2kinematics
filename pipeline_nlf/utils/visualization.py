import matplotlib.pyplot as plt


def plot_keypoints_3d(keypoints):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x_vals = [kp[0] for kp in keypoints]
    y_vals = [kp[1] for kp in keypoints]
    z_vals = [kp[2] for kp in keypoints]

    ax.scatter(x_vals, y_vals, z_vals, c="r", marker="o")
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    ax.set_title("3D Keypoints Debug")
    plt.show()
