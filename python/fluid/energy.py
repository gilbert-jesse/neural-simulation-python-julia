from intialize_velocity import initialize_velocity
import numpy as np
import matplotlib.pyplot as plt
from advect_velocity import advect_velocity
from divergence import divergence
from project_velocity import project_velocity
from visualize_vel import velocity_to_cell_centers
from macmormark_advect_velocity import maccormark_advect_velocity


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
divergence_before_projection_MC = []

divergence_after_projection = []
divergence_after_projection_MC = []

energy_before_SL = []
energy_after_advection_SL = []
energy_after_advection_MC = []
energy_before_MC = []
energy_after_projection_MC = []
energy_after_projection_SL = []




u_initial = u.copy()
v_initial = v.copy()


for step in range(n_steps):

    energy_before_SL.append(kinetic_energy(u, v))
    energy_before_MC.append(kinetic_energy(u, v))


    #advection

    u_star, v_star = advect_velocity(u, v, dt, h,)
    u_star_mc, v_star_mc = maccormark_advect_velocity(u, v, dt, h)

    div_star = np.linalg.norm(divergence(u_star, v_star, h,))
    div_star_MC = np.linalg.norm(divergence(u_star_mc, v_star_mc, h,))


    energy_after_advection_SL.append(kinetic_energy(u_star, v_star))
    energy_after_advection_MC.append(kinetic_energy(u_star_mc, v_star_mc))


    #project

    u, v, p = project_velocity(
        u_star,
        v_star,
        dt=dt,
        rho=rho,
        h=h,
        iter=iter,
    )

    u_MC, v_MC, p_MC = project_velocity(
            u_star_mc,
            v_star_mc,
            dt=dt,
            rho=rho,
            h=h,
            iter=iter,
        )

    energy_after_projection_SL.append(kinetic_energy(u, v))
    energy_after_projection_MC.append(kinetic_energy(u_MC, v_MC))


    div_new = np.linalg.norm(divergence(u, v, h,))
    div_new_MC = np.linalg.norm(divergence(u_MC, v_MC, h,))


    divergence_before_projection.append(
        div_star
    )
    divergence_before_projection_MC.append(
        div_star_MC
    )

    divergence_after_projection.append(
        div_new
    )

    divergence_after_projection_MC.append(
        div_new_MC
    )


E_before = np.array(energy_before_SL)
E_adv = np.array(energy_after_advection_SL)
E_proj = np.array(energy_after_projection_SL)

E_before_MC = np.array(energy_before_MC)
E_adv_MC = np.array(energy_after_advection_MC)
E_proj_MC = np.array(energy_after_projection_MC)

loss_advection = E_before - E_adv

loss_projection = E_adv - E_proj

loss_total = E_before - E_proj

loss_advection_MC = E_before_MC - E_adv_MC

loss_projection_MC = E_adv_MC - E_proj_MC

loss_total_MC = E_before_MC - E_proj_MC


print(
    f"Mean energy before_SL: {np.mean(E_before)}\n"
    f"Mean loss from advection_SL: {np.mean(loss_advection)}\n"
    f"Mean loss from projection_SL: {np.mean(loss_projection)}\n"
    f"Mean total loss_SL: {np.mean(loss_total)}\n"
    f"Mean energy before_MC: {np.mean(E_before_MC)}\n"
    f"Mean loss from advection_MC: {np.mean(loss_advection_MC)}\n"
    f"Mean loss from projection_MC: {np.mean(loss_projection_MC)}\n"
    f"Mean total loss_MC: {np.mean(loss_total_MC)}"
)


# plt.figure(figsize=(8, 5))
fig, ax = plt.subplots(1, 2, figsize=(8, 5))

ax[0].plot(
    np.arange(1, n_steps + 1),
    energy_before_SL,
    label = "energy_before"
)

ax[0].plot(
    np.arange(1, n_steps + 1),
    energy_after_advection_SL,
    label = "energy_after_advection"
)

ax[0].plot(
    np.arange(1, n_steps + 1),
    energy_after_projection_SL,
    label = "energy_after_projection"
)

ax[0].set_xlabel("Timestep")
ax[0].set_ylabel("Mean kinetic energy_SL")
ax[0].set_title("Kinetic energy_SL")

######

ax[1].plot(
    np.arange(1, n_steps + 1),
    energy_before_MC,
    label = "energy_before_MC"
)

ax[1].plot(
    np.arange(1, n_steps + 1),
    energy_after_advection_MC,
    label = "energy_after_advection_MC"
)

ax[1].plot(
    np.arange(1, n_steps + 1),
    energy_after_projection_MC,
    label = "energy_after_projection_MC"
)

ax[1].set_xlabel("Timestep")
ax[1].set_ylabel("Mean kinetic energy_MC")
ax[1].set_title("Kinetic energy_MC")

plt.legend()

plt.show()


