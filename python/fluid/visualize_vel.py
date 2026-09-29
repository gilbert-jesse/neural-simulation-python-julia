import numpy as np
import matplotlib.pyplot as plt

def velocity_to_cell_centers(u, v):
    u_center = 0.5 * (
        u[:, :-1] + u[:, 1:]
    )

    v_center = 0.5 * (
        v[:-1, :] + v[1:, :]
    )

    return u_center, v_center


def plot_velocity(u, v, h, title="Velocity_field"):
    ny = u.shape[0]
    nx = v.shape[1]

    u_center, v_center = velocity_to_cell_centers(u, v)

    x = (np.arange(nx) + 0.5) * h
    y = (np.arange(ny) + 0.5) * h

    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(7, 7))
    plt.quiver(
        X,
        Y,
        u_center,
        v_center,
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)

    plt.axis("equal")
    plt.show()



