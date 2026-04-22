import functools

import numpy as np
import pandas as pd
import streamlit as st

from src.plotting import plot_array

st.set_page_config(page_title="Heisenberg Model", page_icon="⛓️", layout="wide")

st.header("Heisenberg Model", divider="blue")

with st.expander("📖 Methodology", expanded=False):
    st.markdown("""
    **Transverse Field Heisenberg Model** describes a 1D spin chain with nearest-neighbor interactions.
    
    **The Hamiltonian:**
    $$H = H_J + H_U = J \\sum_{i=0}^{N} \\sigma_i^x \\sigma_{i+1}^x + \\frac{U}{2} \\sum_{i=1}^{N} \\sigma_i^z$$
    
    Where:
    - $J$: Coupling strength between neighboring spins (ferromagnetic if $J > 0$)
    - $U$: Transverse field strength along the z-direction
    - $\\sigma_i^x, \\sigma_i^z$: Pauli matrices acting on site $i$
    
    **Methodology:**
    1. **Hilbert Space**: For $N$ spin-1/2 particles, the Hilbert space dimension is $2^N$
    2. **Operator Construction**: Build $H$ as a $2^N \\times 2^N$ matrix using Kronecker products
       - $H_J$: Sum of $\\sigma^x_i \\sigma^x_{i+1}$ for all bonds
       - $H_U$: Sum of $\\sigma^z_i$ for all sites (weighted by $U/2$)
    3. **Diagonalization**: Compute all eigenvalues and eigenvectors using `np.linalg.eig`
    4. **Expectation Values**: Calculate $\\langle n|\\sigma_i^z|n\\rangle$ for each eigenstate $n$ using Einstein summation
    
    **Physical Context:** This model exhibits quantum phase transitions between ferromagnetic and paramagnetic phases,
    and serves as a testbed for many-body physics and quantum information concepts.
    
    **Note:** Limited to $N < 26$ sites due to exponential growth of Hilbert space ($2^N$ states).
    """)

with st.sidebar:
    st.header("Parameters", divider="blue")
    c1, c2, c3 = st.columns(3)
    with c1:
        j = st.number_input("$J$", value=1)
    with c2:
        u = st.number_input("$U$", value=1)
    with c3:
        n_sites = st.number_input("$N$", value=3)
        assert n_sites < 26, "We can't handle such a large number of sites"

    debug_mode = st.toggle("Debug mode", value=True)

st.header("Setup", divider="blue")
st.markdown("Transverse Heisenberg Model")
st.markdown(
    f"$H = H_J + H_U = {j} \\sum_{{i=0}}^{{ {n_sites + 1} }} \\sigma^x_i \\sigma^x_{{i+1}} + \\frac{{{u}}}{{2}} \\sum_{{i=1}}^{{{n_sites}}} \\sigma^z_i$"
)

sigma_x = np.array([[0, 1], [1, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
eye_2 = np.array([[1, 0], [0, 1]])

# Optimized: precompute identities once, then modify in-place for each coupling term
identities = [eye_2] * n_sites
hamiltonian_coupling = np.zeros_like(functools.reduce(np.kron, identities))
for i in range(n_sites - 1):
    op = identities.copy()
    op[i] = sigma_x
    op[i + 1] = sigma_x
    hamiltonian_coupling += functools.reduce(np.kron, op)

hamiltonian_local = functools.reduce(np.kron, [sigma_z for _ in range(1, n_sites + 1)])
hamiltonian = j * hamiltonian_local + (0.5 * u) * hamiltonian_coupling

info_0, info_1, info_2 = st.tabs(["$H$", "$H_J$", "$H_U$"])
with info_0:
    plot_array(hamiltonian, key="h")
with info_1:
    plot_array(hamiltonian_coupling, key="hj")
with info_2:
    plot_array(hamiltonian_local, key="hu")

vals, vectors = np.linalg.eig(hamiltonian)

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_sites]

energy_levels = pd.DataFrame(vals, columns=["Energy"])
level_expectations = np.array(
    [
        np.einsum(alphabet + f" -> {alphabet[j]}", np.reshape(vector, [2] * n_sites))
        for j in range(n_sites)
        for vector in vectors
    ]
)
level_expectations = level_expectations.reshape(n_sites, len(vectors), 2)

if debug_mode:
    info = {
        "alphabet": alphabet,
        "sites": list(range(n_sites)),
        "n_vectors": len(vectors),
        "resulting vector": np.einsum(
            alphabet + f" -> {alphabet[0]}", np.reshape(vectors[0], [2] * n_sites)
        ),
        "resulting shape": level_expectations.shape,
        "generator": list((i, j) for i in range(3) for j in range(4)),
    }

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.dataframe(vectors)
    with col2:
        st.dataframe(energy_levels)
    with col3:
        st.dataframe(level_expectations[:, :, 0].T)

st.header("Energy levels", divider="orange")

left, center, right = st.columns(3)

with left:
    st.bar_chart(sorted(vals))
with center:
    plot_array(level_expectations[:, :, 0])
with right:
    plot_array(level_expectations[:, :, 1])
