from advect_velocity import advect_velocity
from project_velocity import project_velocity
def fluid_step(u, v, dt, rho, h, n_iter,):
    u_star, v_star = advect_velocity(
        u,
        v,
        dt,
        h
    )

    u_new, v_new, p = project_velocity(
        u_star,
        v_star,
        dt,
        rho=rho,
        h=h,
        n_iter = n_iter,
    )

    return u_new, v_new, p
