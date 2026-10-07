from bilinear_interpolation import bilinear_interpolation
import numpy as np
from advect_field import advect_field
from interpolation_bounds import interpolation_bounds

def maccormark_advect_velocity(
        u,
        v,
        dt,
        h,
):
    u_old = u[:, :-1]
    v_old = v[:-1, :]

    #Forward

    X_u_departed, Y_u_departed, u_hat = advect_field(
        u_old,
        u,
        v,
        dt,
        h,
        x_offset = 0.0,
        y_offset = 0.5,
        return_departure = True,
    )

    X_v_departed, Y_v_departed, v_hat = advect_field(
        v_old,
        u,
        v,
        dt,
        h,
        x_offset = 0.5,
        y_offset = 0.0,
        return_departure = True,
    )

    u_hat_full = np.column_stack([u_hat, u_hat[:, 0]])
    v_hat_full = np.vstack([v_hat, v_hat[0, :]])

    #Backward

    u_back = advect_field(
        u_hat,
        u_hat_full,
        v_hat_full,
        -dt,
        h,
        x_offset=0.0,
        y_offset=0.5,
        return_departure=False,
    )

    v_back = advect_field(
        v_hat,
        u_hat_full,
        v_hat_full,
        -dt,
        h,
        x_offset=0.5,
        y_offset=0.0,
        return_departure=False,

    )

    u_corrected = (
        u_hat
        + 0.5 * (u_old - u_back)
    )

    v_corrected = (
        v_hat 
        + 0.5 * (v_old - v_back)
    )

    u_min, u_max = interpolation_bounds(
        u_old,
        X_u_departed,
        Y_u_departed,
        h,
        x_offset=0.0,
        y_offset=0.5,
    )

    v_min, v_max = interpolation_bounds(
        v_old,
        X_v_departed,
        Y_v_departed,
        h,
        x_offset=0.5,
        y_offset=0.0,
    )

    u_limited = np.clip(
        u_corrected,
        u_min,
        u_max,
    )

    v_limited = np.clip(
        v_corrected,
        v_min,
        v_max,
    )

    u_new = np.column_stack(
        [u_limited, u_limited[:, 0]]
    )

    v_new = np.vstack(
        [v_limited, v_limited[0, :]]
    )

    return u_new, v_new





    

