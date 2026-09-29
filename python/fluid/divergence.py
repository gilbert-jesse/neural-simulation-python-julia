import numpy as np
def divergence(u, v, h):
    """
    We first compute the divergence machinery.
    U has shape (Ny, Nx+1)
    V has shape (Ny+1, Nx)

    This will return (Ny, Nx) shape, divergence at pressure centers
    """
    du_dx = (u[:, 1:] - u[:, :-1]) / h
    dv_dy = (v[1:, :] - v[:-1, :]) / h

    return du_dx + dv_dy

if __name__ == "__main__":
    nx = 3
    ny = 3

    u = np.zeros((nx, ny+1))
    v = np.zeros((nx+1, ny))

    u[1, 2] = 1.0

    div_field = divergence(u, v, 1.0)
    print(div_field[0])
