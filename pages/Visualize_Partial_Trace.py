"""Visualize Partial Trace UI page - imports physics from src.partial_trace."""

import numpy as np
import scipy.linalg
import streamlit as st

from src.partial_trace import (
    BipartiteConfig,
    build_bipartite_hamiltonian_components,
    partial_trace_a,
    partial_trace_b,
)
from src.plotting import plot_array

st.set_page_config(page_title="Partial Trace", page_icon="📐️", layout="wide")

st.header("Partial Trace Visualization", divider="blue")

with st.expander("📖 Methodology", expanded=False):
    st.markdown(r"""
    **Partial Trace** is an operation that traces out a subsystem of a composite quantum system.
    
    **Physical System:** Two quantum systems A and B:
    - System A: Hamiltonian $H_A = -J_A J_x + U_A J_z^2 + \delta_A J_z$
    - System B: Hamiltonian $H_B = -J_B J_x + U_B J_z^2 + \delta_B J_z$
    - Interaction: $H_{int}$
    
    **Methodology:**
    1. **Full Hamiltonian**: Construct $H = H_A \otimes 1_B + 1_A \otimes H_B + H_{int}$
    2. **State Evolution**: $\ket{\psi_t} = e^{-itH}\ket{\psi_0}$
    3. **Density Matrix**: $\rho_t = |\psi_t\rangle\langle\psi_t|$
    4. **Partial Trace**: $\rho^{(A)}_t = \mathrm{Tr}_B[\rho_t]$
    """)

with st.sidebar:
    st.subheader("System A", divider="blue")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_a = st.number_input("$N_A$", min_value=0, value=2)
    with c2:
        j_a = st.number_input("$J_A$", value=1.0)
    with c3:
        u_a = st.number_input("$U_A$", value=0.0)
    with c4:
        delta_a = st.number_input("$\delta_A$", value=0.0)

    st.subheader("System B", divider="orange")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_b = st.number_input("$N_B$", min_value=0, value=2)
    with c2:
        j_b = st.number_input("$J_B$", value=0.0)
    with c3:
        u_b = st.number_input("$U_B$", value=0.0)
    with c4:
        delta_b = st.number_input("$\delta_B$", value=1.0)

    st.subheader("Interactions", divider="green")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        alpha_xx = st.number_input("$\alpha_{xx}$", value=0.0)
    with c2:
        alpha_xz = st.number_input("$\alpha_{xz}$", value=-1.0)
    with c3:
        alpha_zx = st.number_input("$\alpha_{zx}$", value=0.0)
    with c4:
        alpha_zz = st.number_input("$\alpha_{zz}$", value=0.0)

    st.subheader("Evolution", divider="red")
    time = st.number_input("Time", min_value=0.0, value=1.0)


# Build config and compute using physics module
config = BipartiteConfig(
    dim_a=n_a,
    dim_b=n_b,
    j_a=j_a,
    j_b=j_b,
    u_a=u_a,
    u_b=u_b,
    delta_a=delta_a,
    delta_b=delta_b,
    alpha_xx=alpha_xx,
    alpha_xz=alpha_xz,
    alpha_zx=alpha_zx,
    alpha_zz=alpha_zz,
)

h_a, h_b, h_int, full_hamiltonian = build_bipartite_hamiltonian_components(config)

# Trace over subsystems
traced_a = partial_trace_a(full_hamiltonian.reshape(n_a, n_b, n_a, n_b), n_a, n_b)
traced_b = partial_trace_b(full_hamiltonian.reshape(n_a, n_b, n_a, n_b), n_a, n_b)

# Initial state and evolution
phi_zero = np.zeros(n_a * n_b)
phi_zero[0] = 1  # state |0_A⟩ |0_B⟩
evolved_state = phi_zero @ scipy.linalg.expm(-1j * time * full_hamiltonian)

# Reduced densities
traced_evolved_state_a = partial_trace_a(
    np.outer(evolved_state, evolved_state).reshape(n_a, n_b, n_a, n_b), n_a, n_b
)
traced_evolved_state_b = partial_trace_b(
    np.outer(evolved_state, evolved_state).reshape(n_a, n_b, n_a, n_b), n_a, n_b
)

st.latex(f"""
\\begin{{array}}{{ccccc}}
&&H\\\\
H_A &+& H_{{int}} &+& H_B \\\\
(-J_A J_x + U_A J_z^2 + \delta_S J_z) \\mathbb{{1}}_B  &+&
\\alpha_{{xx}} J_x J_x + \\alpha_{{xz}} J_x J_z + \\alpha_{{zx}} J_z J_x + \\alpha_{{zz}} J_z J_z &+&
\\mathbb{{1}}_A (-J_B J_x + U_BJ_z ^ 2 + \delta_BJ_z) \\\\
( {-1 * j_a:.2f} J_x {u_a:+.2f} J_z^2 {delta_a:+.2f} J_z ) \\mathbb{{1}}_B&+&
{alpha_xx:.2f} J_x J_x  {alpha_xz:+.2f} J_x J_z  {alpha_zx:+.2f} J_z J_x {alpha_zz:+.2f} J_z J_z &+& 
\\mathbb{{1}}_A ( {-1 * j_b:.2f} J_x {u_b:+.2f}J_z ^ 2 {delta_b:+.2f}J_z )
\\end{{array}}
""")

c1, c2, c3 = st.columns(3)
with c1:
    st.header("System A", divider="blue")
    st.latex(r"H_A")
    plot_array(h_a, key="H_A")
    st.latex(r"\mathrm{Tr}_B[H]")
    plot_array(traced_a, key="TrH_B")
    st.latex(r"\mathrm{Tr}_B[\ket{\psi_t}]")
    plot_array(np.abs(traced_evolved_state_a) ** 2, midpoint=None, key="Tr_Bphi_t")

with c2:
    st.header("Interactions", divider="green")
    st.latex(r"H_{int}")
    plot_array(h_int, key="H_int")
    st.latex(r"H")
    plot_array(full_hamiltonian, key="H")
    st.latex(r"\ket{\psi_t} := e^{-itH} \ket{0}_A \ket{0}_B")
    plot_array(
        np.abs(np.outer(evolved_state, evolved_state)) ** 2, midpoint=None, key="phi_t"
    )

with c3:
    st.header("System B", divider="orange")
    st.latex(r"H_B")
    plot_array(h_b, key="H_B")
    st.latex(r"\mathrm{Tr}_A[H]")
    plot_array(traced_b, key="TrH_A")
    st.latex(r"\mathrm{Tr}_A[\ket{\psi_t}]")
    plot_array(np.abs(traced_evolved_state_b) ** 2, midpoint=None, key="Tr_Aphi_t")
