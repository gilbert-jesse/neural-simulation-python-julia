import numpy as np
from divergence import divergence
from pressure_gradient import pressure_gradient_staggered
from advect_velocity import advect_velocity
from project_velocity import project_velocity

def initialize_velocity(nx, ny, h):
    Lx = nx * h
    Ly = ny * h

    u = np.zeros((ny, nx+1))
    v = np.zeros((ny+1, nx))

    # u_locations
    x_u = (np.arange(nx) * h)
    y_u = (np.arange(ny) + 0.5 * h)

    X_u, Y_u = np.meshgrid(x_u, y_u)

    u[:, :-1] = -np.sin(
        2.0 * np.pi * Y_u / Ly
    )

    u[:, -1] = u[:, 0]

    # v_locations

    x_v = (np.arange(nx) + 0.5 * h)
    y_v = np.arange(ny) * h

    X_v, Y_v = np.meshgrid(x_v, y_v)

    v[:-1, :] = np.sin(
        2.0 * np.pi * X_v / Lx
    )

    v[-1, :] = v[0, :]

    return u, v

if  __name__ == "__main__":
    nx = 32
    ny = 32

    h = 1.0
    dt = 0.1

    u, v = initialize_velocity(
        nx,
        ny,
        h,
    )

    div_initial = divergence(
    u,
    v,
    h,
    )

    print(
        "Initial divergence:",
        np.linalg.norm(div_initial)
    )


    u_adv, v_adv = advect_velocity(
        u,
        v,
        dt,
        h,
    )


    div_advected = divergence(
        u_adv,
        v_adv,
        h,
    )

    print(
        "After advection:",
        np.linalg.norm(div_advected)
    )

    u_new, v_new, p = project_velocity(
    u_adv,
    v_adv,
    dt=dt,
    rho=1.0,
    h=h,
    iter=1000,
    )

    print(
        "After projection:",
        np.linalg.norm(
            divergence(u_new, v_new, h)
        )
    )





