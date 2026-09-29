import numpy as np
def bilinear_interpolation(
        field,
        x, 
        y,
        h,
        x_offset,
        y_offset,
):
    ny, nx = field.shape

    xi = x / h - x_offset
    yi = y / h - y_offset

    i0_raw = np.floor(xi).astype(int)
    j0_raw = np.floor(yi).astype(int)

    fx = xi - np.floor(xi)
    fy = yi - np.floor(yi)

    i0 = np.mod(i0_raw, nx)
    j0 = np.mod(j0_raw, ny)

    i1 = np.mod(i0_raw + 1, nx)
    j1 = np.mod(j0_raw + 1, ny)

    f00 = field[j0, i0]
    f10 = field[j0, i1]
    f01 = field[j1, i0]
    f11 = field[j1, i1]

    return(
        (1 - fx) * (1 - fy) * f00
        + fx * (1 - fy) * f10 
        + (1 - fx) * fy * f01
        + fx * fy * f11
    )






