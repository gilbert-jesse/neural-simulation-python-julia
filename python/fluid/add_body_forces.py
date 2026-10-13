def add_body_forces(
    u,
    v,
    dt,
    force_x=0.0,
    force_y=0.0,
):
    u_new = u + dt * force_x
    v_new = v + dt * force_y

    u_new[:, -1] = u_new[:, 0]
    v_new[-1, :] = v_new[0, :]

    return u_new, v_new
