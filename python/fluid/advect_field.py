import numpy as np
from sample_velocity import sample_velocity
from bilinear_interpolation import bilinear_interpolation

def advect_field(
        field,
        u,
        v,
        dt,
        h,
        x_offset,
        y_offset,
        return_departure=False,
):
    ny, nx = field.shape

    x = (np.arange(nx) + x_offset) * h
    y = (np.arange(ny) + y_offset) * h

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

    field_advected = bilinear_interpolation(
        field,
        X_back,
        Y_back,
        h,
        x_offset=x_offset,
        y_offset=y_offset,
    )

    if return_departure:
        return X_back, Y_back, field_advected

    return field_advected