import numpy as np
from divergence import divergence
from jacobi import jacobi
from pressure_gradient import pressure_gradient_staggered

def project_velocity(u_star, v_star, dt=1.0, rho=1.0, h=1.0, iter=100):
    velocity_star_div = divergence(u_star, v_star, h)
    b = (rho / dt) * velocity_star_div
    jacobi_pressure = jacobi(b, h, iter=iter)
    pressure_div_x, pressure_div_y = pressure_gradient_staggered(jacobi_pressure, h=h)
    u = u_star - (dt / rho) * pressure_div_x
    v = v_star - (dt / rho) * pressure_div_y

    return u, v, jacobi_pressure
