from intialize_velocity import initialize_velocity
import numpy as np
import matplotlib.pyplot as plt
from advect_velocity import advect_velocity
from divergence import divergence
from project_velocity import project_velocity
from visualize_vel import plot_velocity

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

u_initial = u.copy()
v_initial = v.copy()

for step in range(n_steps):

    #advection

    u_star, v_star = advect_velocity(u, v, dt, h,)

    div_star = np.linalg.norm(divergence(u_star, v_star, h,))

    #project

    u, v, p = project_velocity(
        u_star,
        v_star,
        dt=dt,
        rho=rho,
        h=h,
        iter=iter,
    )

    div_new = np.linalg.norm(divergence(u, v, h,))

    divergence_before_projection.append(
        div_star
    )

    divergence_after_projection.append(
        div_new
    )



    print(
        f"step {step + 1:3d} | "
        f"before = {div_star:.6e} | "
        f"after = {div_new:.6e}"
    )


steps = np.arange(
    1,
    n_steps + 1,
)

plt.figure(figsize=(8, 5))

plt.plot(
    steps,
    divergence_before_projection,
    label="Before projection",
)

plt.plot(
    steps,
    divergence_after_projection,
    label="After projection",
)

plt.yscale("log")

plt.xlabel("Timestep")
plt.ylabel("L2 divergence")
plt.title("Divergence through simulation")

plt.legend()

plt.show()
