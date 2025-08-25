import functools

import numpy as np
import pandas as pd
import streamlit as st

from src.plotting import plot_array

st.set_page_config(page_title="Heisenberg Model", page_icon="⛓️", layout="wide")

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
st.markdown(f"$H = H_J + H_U = {j} \\sum_{{i=0}}^{{ {n_sites + 1} }} \\sigma^x_i \\sigma^x_{{i+1}} + \\frac{{{u}}}{{2}} \\sum_{{i=1}}^{{{ n_sites}}} \\sigma^z_i$")

sigma_x = np.array([[0,1],[1,0]])
sigma_z = np.array([[1,0],[0,-1]])
eye_2 = np.array([[1,0],[0,1]])

hamiltonian_coupling = functools.reduce(
    lambda a,b: a+b,
    [
        functools.reduce(np.kron, [sigma_x if j==i or j==i+1 else eye_2 for j in range(1, n_sites+1)])
        for i in range(1,n_sites)
    ]
)

hamiltonian_local = functools.reduce(np.kron, [sigma_z for _ in range(1, n_sites+1)])
hamiltonian = j * hamiltonian_local + (.5 * u ) * hamiltonian_coupling

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
level_expectations = np.array([
    np.einsum(
        alphabet + f" -> {alphabet[j]}",
        np.reshape(vector, [2]*n_sites)
    )
    for j in range(n_sites)
    for vector in vectors
])
level_expectations = level_expectations.reshape(n_sites, len(vectors), 2)

if debug_mode:
    info = {
        "alphabet": alphabet,
        "sites": list(range(n_sites)),
        "n_vectors": len(vectors),
        "resulting vector": np.einsum(
            alphabet + f" -> {alphabet[0]}",
            np.reshape(vectors[0], [2]*n_sites)
        ),
        "resulting shape": level_expectations.shape,
        "generator": list((i,j) for i in range(3) for j in range(4))

    }

    i,j,k,l = st.columns(4)
    with i:
        st.dataframe(vectors)
    with j:
        st.dataframe(energy_levels)
    with k:
        st.dataframe(level_expectations[:,:,0].T)

st.header("Energy levels", divider="orange")

left, center, right = st.columns(3)

with left:
    st.bar_chart(sorted(vals))
with center:
    plot_array(level_expectations[:,:,0])
with right:
    plot_array(level_expectations[:,:,1])

