import numpy as np

# Returns the J_x / J_z spin matrices of dimension dim
def generate_spin_matrices(dim: int) -> [np.array, np.array]:
    spin = (dim - 1) / 2
    jx = np.zeros((dim, dim))
    jz = np.zeros((dim, dim))
    for j in range(dim):
        magnetic_number = spin - j
        jz[j, j] = magnetic_number
        if j < dim - 1:
            jx[j, j + 1] = jx[j + 1, j] = 0.5 * np.sqrt((spin - magnetic_number + 1) * (spin + magnetic_number))
    return np.array(jx, dtype=float), np.array(jz, dtype=float) # j_x & j_z
