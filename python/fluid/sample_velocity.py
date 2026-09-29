import numpy as np
from bilinear_interpolation import bilinear_interpolation
def sample_velocity(
        u, v, x, y, h
):
    u_unique = u[:, :-1]
    v_unique = v[:-1, :]

    u_sample = bilinear_interpolation(
        u_unique,
        x,
        y,
        h,
        x_offset=0.0,
        y_offset=0.5
    )

    v_sample = bilinear_interpolation(
        v_unique,
        x,
        y,
        h,
        x_offset=0.5,
        y_offset=0.0
    )
    return u_sample, v_sample