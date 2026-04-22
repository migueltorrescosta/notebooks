import numpy as np


def generate_spin_matrices(dim: int) -> tuple[np.ndarray, np.ndarray]:
    spin = (dim - 1) / 2
    # Vectorized construction of Jz (diagonal matrix)
    j = np.arange(dim)
    magnetic_numbers = spin - j
    jz = np.diag(magnetic_numbers)

    # Vectorized construction of Jx (off-diagonal matrix)
    off_diags = 0.5 * np.sqrt(
        (spin - magnetic_numbers[:-1] + 1) * (spin + magnetic_numbers[:-1])
    )
    jx = np.zeros((dim, dim))
    jx[np.arange(dim - 1), np.arange(1, dim)] = off_diags
    jx[np.arange(1, dim), np.arange(dim - 1)] = off_diags

    return np.array(jx, dtype=float), np.array(jz, dtype=float)
