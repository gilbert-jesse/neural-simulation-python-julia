import numpy as np
def interpolation_bounds(
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

    col0_raw = np.floor(xi).astype(int)
    row0_raw = np.floor(yi).astype(int)

    col0 = np.mod(col0_raw, nx)
    col1 = np.mod(col0_raw + 1, nx)
    row0 = np.mod(row0_raw, ny)
    row1 = np.mod(row0_raw + 1, ny)

    f00 = field[row0, col0]
    f10 = field[row0, col1]
    f01 = field[row1, col0]
    f11 = field[row1, col1]

    q_max = np.max(f00, f10, f01, f11)
    q_min = np.min(f00, f10, f01, f11)

    return q_max, q_min
    

