"""Heisenberg Model UI page - imports physics from src.heisenberg_model."""

import pandas as pd
import streamlit as st

from src.physics.heisenberg_model import (
    heisenberg_hamiltonian,
    diagonalize_hamiltonian,
    compute_expectation_values,
)
from src.visualization.plotting import plot_array

st.set_page_config(page_title="Heisenberg Model", page_icon="⛓️", layout="wide")

st.header("Heisenberg Model", divider="blue")

with st.expander("📖 Methodology", expanded=False):
    st.markdown(r"""
    **Transverse Field Heisenberg Model** describes a 1D spin chain with nearest-neighbor interactions.
    
    **The Hamiltonian:**
    $$H = H_J + H_U = J \sum_{i=0}^{N} \sigma_i^x\sigma_{i+1}^x + \frac{U}{2} \sum_{i=1}^{N} \sigma_i^z$$
    
    Where:
    - $J$: Coupling strength between neighboring spins (ferromagnetic if $J > 0$)
    - $U$: Transverse field strength along the z-direction
    - $\sigma_i^x, \sigma_i^z$: Pauli matrices acting on site $i$
    
    **Methodology:**
    1. **Hilbert Space**: For $N$ spin-1/2 particles, the Hilbert space dimension is $2^N$
    2. **Operator Construction**: Build $H$ as a $2^N \times 2^N$ matrix using Kronecker products
    3. **Diagonalization**: Compute all eigenvalues and eigenvectors using `np.linalg.eig`
    4. **Expectation Values**: Calculate $\langle n|\sigma_i^z|n\rangle$ for each eigenstate $n$
    
    **Physical Context:** This model exhibits quantum phase transitions between ferromagnetic and paramagnetic phases.
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
    f"$H = H_J + H_U = {j} sum_{{i=0}}^{{ {n_sites + 1} }} \sigma^x_i\sigma^x_{{i+1}} + \frac{{{u}}}{{2}} sum_{{i=1}}^{{{n_sites}}} \sigma^z_i$"
)

# Build Hamiltonian using physics module
hamiltonian = heisenberg_hamiltonian(n_sites, j, u)

# Diagonalize
eigenvalues, eigenvectors = diagonalize_hamiltonian(n_sites, j, u)

info_0, info_1, info_2 = st.tabs([r"$H$", r"$H_J$", r"$H_U$"])
with info_0:
    plot_array(hamiltonian, key="h")
with info_1:
    plot_array(hamiltonian @ hamiltonian - hamiltonian, key="hj")  # placeholder
with info_2:
    plot_array(hamiltonian - hamiltonian, key="hu")  # placeholder

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_sites]

energy_levels = pd.DataFrame(eigenvalues, columns=["Energy"])

# Compute expectation values
level_expectations = compute_expectation_values(n_sites, eigenvectors)
level_expectations = level_expectations.reshape(n_sites, len(eigenvectors), 2)

if debug_mode:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.dataframe(eigenvectors)
    with col2:
        st.dataframe(energy_levels)
    with col3:
        st.dataframe(level_expectations[:, :, 0].T)

st.header("Energy levels", divider="orange")

left, center, right = st.columns(3)
with left:
    st.bar_chart(sorted(eigenvalues))
with center:
    plot_array(level_expectations[:, :, 0])
with right:
    plot_array(level_expectations[:, :, 1])
