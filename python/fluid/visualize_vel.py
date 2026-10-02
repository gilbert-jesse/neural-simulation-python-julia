import numpy as np
import matplotlib.pyplot as plt
from divergence import divergence

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

def initialize_velocity(nx, ny, h):

    Lx = nx * h
    Ly = ny * h

    u = np.zeros((ny, nx+1))
    v = np.zeros((ny+1, nx))

    x_u = (np.arange(nx)) * h
    y_u = (np.arange(ny) + 0.5) * h

    X_u, Y_u = np.meshgrid(
        x_u, y_u
    )

    u[:, :-1] = -np.sin(2.0 * np.pi * Y_u / Ly)
    u[:, -1] = u[:, 0]

    x_v = (np.arange(nx) + 0.5) * h
    y_v = (np.arange(ny)) * h

    X_v, Y_v = np.meshgrid(
        x_v, y_v
    )

    v[:-1, :] = np.sin(2.0 * np.pi * X_v / Lx)

    v[-1, :] = v[0, :]

    return u, v

if __name__ == "__main__":
    u, v = initialize_velocity(
        nx = 32,
        ny = 32,
        h = 1.0,
    )

    print(f"initial divergence: {np.linalg.norm(divergence(u, v, 1.0))}")

    plot_velocity(
        u, 
        v, 
        h = 1.0,
        title="Initial_Velocity"
    )


