def advect_u(u, v, dt, h):
    ny = u.shape[0]
    nx = u.shape[1] - 1

    j = np.arange(nx)
    i = np.arange(ny)
    

    