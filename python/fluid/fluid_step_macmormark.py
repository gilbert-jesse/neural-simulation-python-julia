from macmormark_advect_velocity import maccormark_advect_velocity
from project_velocity import project_velocity
from energy import kinetic_energy
from intialize_velocity import initialize_velocity
import numpy as np
from divergence import divergence
def fluid_step(u, v, dt, rho, h, n_iter,):
        
    u_star, v_star = maccormark_advect_velocity(
        u,
        v,
        dt,
        h,
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

if __name__ == "__main__":

    nx = 32
    ny = 32
    h = 1.0
    dt = 0.1
    rho = 1.0
    n_steps = 50
    iter = 500

    u, v = initialize_velocity(
        nx,
        ny,
        h,
    )

    E_before = kinetic_energy(u, v)

    u_star, v_star = maccormark_advect_velocity(
        u,
        v,
        dt,
        h,
    )

    E_after_adv = kinetic_energy(
        u_star,
        v_star,
    )

    print("Energy before:", E_before)
    print("Energy after MacCormack:", E_after_adv)
    print(
        "MacCormack energy loss:",
        E_before - E_after_adv
    )

    print(
        "Divergence after MacCormack:",
        np.linalg.norm(
            divergence(u_star, v_star, h)
        )
    )
