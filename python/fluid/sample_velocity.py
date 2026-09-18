import numpy as np
from python.fluid.bilinear_interpolation import sample_periodic
def sample_velocity(
        u, v, x, y, h
):
    u_unique = u[:, :-1]
    v_unique = v[:-1, :]

    u_sample = sample_periodic(
        u_unique,
        x,
        y,
        h,
        x_offset=0.0,
        y_offset=0.5
    )

    v_sample = sample_periodic(
        v_unique,
        x,
        y,
        h,
        x_offset=0.5,
        y_offset=0.0
    )
    return u_sample, v_sample