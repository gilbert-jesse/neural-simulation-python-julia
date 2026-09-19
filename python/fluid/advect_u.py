import numpy as np
from python.fluid.bilinear_interpolation import bilinear_interpolation
from python.fluid.sample_velocity import sample_velocity
def advect_u(u, v, dt, h):
    ny = u.shape[0]
    nx = u.shape[1] - 1

    j = np.arange(nx)
    i = np.arange(ny)

    x = j * h
    y = (i + 0.5) * h

    X, Y = np.meshgrid(x, y)

    vel_x, vel_y = sample_velocity(
        u,
        v,
        X,
        Y,
        h,
    )

    X_back = X - dt * vel_x
    Y_back = Y - dt * vel_y

    u_unique = u[:, :-1]
    u_advected = bilinear_interpolation(
        u_unique,
        X_back,
        Y_back,
        h,
        x_offset=0.0,
        y_offset=0.5,
    )
    u_advected = np.column_stack([u_advected, u_advected[:, 0]])

    return u_advected




    