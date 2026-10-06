import numpy as np
import matplotlib.pyplot as plt
# from python.fluid.divergence import divergence
# from python.fluid.jacobi import jacobi
# from python.fluid.laplacian import laplacian
# from python.fluid.pressure_gradient import pressure_gradient_staggered
# from python.fluid.project_velocity import project_velocity

# u = np.zeros((3, 4))
# v = np.zeros((4, 3))

# u[1, 2] = 1.0

# u, v, p = project_velocity(u, v)

# corr_div = divergence(u, v, h=1.0)
# print(corr_div)

# u_div = divergence(u, v, h=1.0)

# jaco = jacobi(u_div, h=1.0, iter=100)
# print(jaco)

# p_div_x, p_div_y = pressure_gradient_staggered(jaco, h=1.0)

# u_corr_x = u - p_div_x
# v_corr_y = v - p_div_y

# corr_div = divergence(u_corr_x, v_corr_y, h=1.0)
# print(corr_div)

# print(np.mod(12, 11))

# print(np.floor(1.9).    astype(int))
# print(np.arange(3) * "h")


x = np.arange(10)
y = np.arange(10)

X, Y = np.meshgrid(x, y)

x_cont = np.zeros_like(x, dtype=float)
y_cont = np.zeros_like(y)

x_cont = -np.sin(
    2.0 * np.pi * Y / 50
)

y_cont = np.sin(
    2.0 * np.pi * X / 50
)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(x_cont, cmap="viridis", origin="lower")
ax[1].imshow(y_cont, cmap="viridis", origin="lower")
plt.legend()

plt.show()




