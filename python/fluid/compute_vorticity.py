import numpy as np
from visualize_vel import velocity_to_cell_centers
def compute_vorticity(u, v, h):

    u_center, v_center = velocity_to_cell_centers(
        u,
        v,
    )

    dv_dx = (
        np.roll(v_center, -1, axis=1)
        - np.roll(v_center, 1, axis=1)
    ) / (2.0 * h)

    du_dy = (
        np.roll(u_center, -1, axis=0)
        - np.roll(u_center, 1, axis=0)
    ) / (2.0 * h)

    omega = dv_dx - du_dy

    return omega