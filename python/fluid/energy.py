from intialize_velocity import initialize_velocity
import numpy as np
import matplotlib.pyplot as plt
from advect_velocity import advect_velocity
from divergence import divergence
from project_velocity import project_velocity
from visualize_vel import velocity_to_cell_centers


def kinetic_energy(u, v):
    u_center, v_center = velocity_to_cell_centers(
        u, v
    )

    return 0.5 * np.mean(u_center**2 + v_center**2)


nx = 32
ny = 32
h = 1.0
dt = 0.1
rho = 1.0
n_steps = 50
iter = 500

u, v = initialize_velocity(nx,
                           ny,
                           h,)
divergence_before_projection = []
divergence_after_projection = []
energy_before = []
energy_after_advection = []
energy_after_projection = []


u_initial = u.copy()
v_initial = v.copy()


for step in range(n_steps):

    energy_before.append(kinetic_energy(u, v))

    #advection

    u_star, v_star = advect_velocity(u, v, dt, h,)

    div_star = np.linalg.norm(divergence(u_star, v_star, h,))

    energy_after_advection.append(kinetic_energy(u_star, v_star))

    #project

    u, v, p = project_velocity(
        u_star,
        v_star,
        dt=dt,
        rho=rho,
        h=h,
        iter=iter,
    )

    energy_after_projection.append(kinetic_energy(u, v))

    div_new = np.linalg.norm(divergence(u, v, h,))

    divergence_before_projection.append(
        div_star
    )

    divergence_after_projection.append(
        div_new
    )


E_before = np.array(energy_before)
E_adv = np.array(energy_after_advection)
E_proj = np.array(energy_after_projection)

loss_advection = E_before - E_adv

loss_projection = E_adv - E_proj

loss_total = E_before - E_proj

print(
    f"Mean energy before: {np.mean(E_before)}\n"
    f"Mean loss from advection: {np.mean(loss_advection)}\n"
    f"Mean loss from projection: {np.mean(loss_projection)}\n"
    f"Mean total loss: {np.mean(loss_total)}"
)


plt.figure(figsize=(8, 5))

plt.plot(
    np.arange(1, n_steps + 1),
    energy_before,
    label = "energy_before"
)

plt.plot(
    np.arange(1, n_steps + 1),
    energy_after_advection,
    label = "energy_after_advection"
)

plt.plot(
    np.arange(1, n_steps + 1),
    energy_after_projection,
    label = "energy_after_projection"
)

plt.xlabel("Timestep")
plt.ylabel("Mean kinetic energy")
plt.title("Kinetic energy")

plt.legend()

plt.show()


