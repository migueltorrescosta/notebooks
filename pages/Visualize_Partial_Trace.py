import functools
import numpy as np
import scipy.linalg
import streamlit as st
from src.angular_momentum import generate_spin_matrices
from src.plotting import plot_array


st.set_page_config(page_title="Partial Trace", page_icon="📐️", layout="wide")


with st.sidebar:
    st.subheader("System A", divider="blue")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_a = st.number_input("$N_A$", min_value=0, value=2)
    with c2:
        j_a = st.number_input("$J_A$", value=1.)
    with c3:
        u_a = st.number_input("$U_A$", value=0.)
    with c4:
        delta_a = st.number_input("$\\delta_A$", value=0.)

    st.subheader("System B", divider="orange")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_b = st.number_input("$N_B$", min_value=0, value=2)
    with c2:
        j_b = st.number_input("$J_B$", value=0.)
    with c3:
        u_b = st.number_input("$U_B$", value=0.)
    with c4:
        delta_b = st.number_input("$\\delta_B$", value=1.)

    st.subheader("Interactions", divider="green")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        alpha_xx = st.number_input("$\\alpha_{xx}$", value=0.)
    with c2:
        alpha_xz = st.number_input("$\\alpha_{xz}$", value=-1.)
    with c3:
        alpha_zx = st.number_input("$\\alpha_{zx}$", value=0.)
    with c4:
        alpha_zz = st.number_input("$\\alpha_{zz}$", value=0.)

    st.subheader("Evolution", divider="red")
    time = st.number_input("Time", min_value=0., value=1.)

jxa, jza = generate_spin_matrices(n_a)
jxb, jzb = generate_spin_matrices(n_b)

hamiltonian_a = -1 * j_a * jxa + u_a * jza @ jza + delta_a * jza
hamiltonian_b = -1 * j_b * jxb + u_b * jzb @ jzb + delta_b * jzb

interaction_hamiltonian = functools.reduce(lambda x, y: x + y, [
    alpha_xx * np.kron(jxa, jxb),
    alpha_xz * np.kron(jxa, jzb),
    alpha_zx * np.kron(jza, jxb),
    alpha_zz * np.kron(jza, jzb),
])

full_hamiltonian = functools.reduce(lambda x, y: x + y, [
    np.kron(hamiltonian_a, np.divide(np.eye(n_b), n_b)),
    np.kron(np.divide(np.eye(n_a), n_a), hamiltonian_b),
    interaction_hamiltonian,
])

traced_a = np.trace(np.array(full_hamiltonian).reshape(n_a, n_b, n_a, n_b), axis1=1, axis2=3)
traced_b = np.trace(np.array(full_hamiltonian).reshape(n_a, n_b, n_a, n_b), axis1=0, axis2=2)

phi_zero = np.zeros(n_a * n_b)
phi_zero[0] = 1 # state 0_A 0_B

evolved_state = phi_zero @ scipy.linalg.expm(-1j *time*full_hamiltonian)
traced_evolved_state_a = np.trace(np.array(np.outer(evolved_state, evolved_state)).reshape(n_a, n_b, n_a, n_b), axis1=1, axis2=3)
traced_evolved_state_b = np.trace(np.array(np.outer(evolved_state, evolved_state)).reshape(n_a, n_b, n_a, n_b), axis1=0, axis2=2)

st.latex(f"""
    \\begin{{array}}{{ccccc}}
    &&H&& \\\\
    H_A &+& H_{{int}} &+& H_B \\\\
    (-J_A J_x + U_A J_z^2 + \\delta_S J_z) \\mathbb{1}_B  &+&
    \\alpha_{{xx}} J_x J_x + \\alpha_{{xz}} J_x J_z + \\alpha_{{zx}} J_z J_x + \\alpha_{{zz}} J_z J_z &+&
    \\mathbb{1}_A (-J_B J_x + U_BJ_z ^ 2 + \\delta_BJ_z) \\\\
    ( {-1 * j_a:.2f} J_x {j_b:+.2f} J_z^2 {delta_a:+.2f} J_z ) \\mathbb{1}_B&+&
    {alpha_xx:.2f} J_x J_x  {alpha_xz:+.2f} J_x J_z  {alpha_zx:+.2f} J_z J_x {alpha_zz:+.2f} J_z J_z &+& 
    \\mathbb{1}_A ( {-1 * j_b:.2f} J_x {u_b:+.2f}J_z ^ 2 {delta_b:+.2f}J_z )
    
    \\end{{array}}
    """)

c1, c2, c3 = st.columns(3)
with c1:
    st.header("System A", divider="blue")
    st.latex("H_A")
    plot_array(hamiltonian_a, key="H_A")
    st.latex("\\mathrm{Tr}_B[H]")
    plot_array(traced_a, key="TrH_B")
    st.latex("\\mathrm{Tr}_B[\\ket{\\psi_t}]")
    plot_array(np.abs(traced_evolved_state_a)**2, midpoint=None, key="Tr_Bphi_t")

with c2:
    st.header("Interactions", divider="green")
    st.latex("H_{int}")
    plot_array(interaction_hamiltonian, key="H_int")
    st.latex("H")
    plot_array(full_hamiltonian, key="H")
    st.latex("\\ket{\\psi_t} := e^{-itH} \\ket{0}_A \\ket{0}_B")
    plot_array(np.abs(np.outer(evolved_state, evolved_state))**2, midpoint=None, key="phi_t")

with c3:
    st.header("System B", divider="orange")
    st.latex("H_B")
    plot_array(hamiltonian_b, key="H_B")
    st.latex("\\mathrm{Tr}_A[H]")
    plot_array(traced_b, key="TrH_A")
    st.latex("\\mathrm{Tr}_A[\\ket{\\psi_t}]")
    plot_array(np.abs(traced_evolved_state_b)**2, midpoint=None, key="Tr_Aphi_t")
