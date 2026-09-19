from python.fluid.advect_u import advect_u
from python.fluid.advect_v import advect_v

def advect_velocity(u, v, dt, h):
    u_old = u.copy()
    v_old = v.copy()

    u_advected = advect_u(
        u_old,
        v_old,
        dt,
        h,
    )

    v_advected = advect_v(
        u_old,
        v_old,
        dt,
        h,
    )

    return u_advected, v_advected
