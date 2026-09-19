import numpy as np
from python.fluid.bilinear_interpolation import bilinear_interpolation
from python.fluid.sample_velocity import sample_velocity

def advect_v(u, v, dt, h):
    nx = v.shape[0] - 1
    ny = v.shape[1]

    j = np.arange(nx)
    i = np.arange(ny)

    x = (j + 0.5) * h
    y = i * h

    X, Y = np.meshgrid(x, y)

    vel_x, vel_y = sample_velocity(
        u, v, X, Y, h,
    )

    X_back = X - dt * vel_x
    Y_back = Y - dt * vel_y

    v_unique = v[:-1, :]

    v_advected = bilinear_interpolation(
        v_unique,
        X_back,
        Y_back,
        h,
        x_offset = 0.5,
        y_offset = 0.0,
    )

    v_advected = np.vstack([v_advected, v_advected[0, :]])
    return v_advected

