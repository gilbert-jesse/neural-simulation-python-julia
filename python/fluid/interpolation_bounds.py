import numpy as np


def interpolation_bounds(
    field,
    x,
    y,
    h,
    x_offset,
    y_offset,
):
    """
    Return the minimum and maximum of the four
    native field values surrounding each query point.
    """

    ny, nx = field.shape

    # Physical coordinates -> native array coordinates
    x_grid = x / h - x_offset
    y_grid = y / h - y_offset

    col0_raw = np.floor(x_grid).astype(int)
    row0_raw = np.floor(y_grid).astype(int)

    col0 = np.mod(col0_raw, nx)
    col1 = np.mod(col0_raw + 1, nx)

    row0 = np.mod(row0_raw, ny)
    row1 = np.mod(row0_raw + 1, ny)

    f00 = field[row0, col0]
    f10 = field[row0, col1]
    f01 = field[row1, col0]
    f11 = field[row1, col1]

    field_min = np.minimum.reduce(
        [f00, f10, f01, f11]
    )

    field_max = np.maximum.reduce(
        [f00, f10, f01, f11]
    )

    return field_min, field_max