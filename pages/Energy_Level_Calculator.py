from dataclasses import dataclass
from plotly import express as px
from typing import Callable, List
import numpy as np
import pandas as pd
import scipy
import streamlit as st

from src.enums import WavePacket, PotentialFunction, BoundaryCondition
from src.plotting import plot_array

st.set_page_config(page_title="Energy level calculator", page_icon="🛢", layout="wide")

st.cache_data.clear()
st.cache_resource.clear()

# START
with st.sidebar:
    st.header("Settings", divider="orange")
    c1, c2 = st.columns(2)
    with c1:
        x_min = st.number_input("$x_{min}$", value=-3.)
    with c2:
        x_max = st.number_input("$x_{max}$", value=3.)
    assert x_min < x_max, f"We need x_min < x_max"

    c1, c2 = st.columns(2)
    with c1:
        resolution = st.number_input("$N_x$", value=200)
    with c2:
        number_of_energy_levels = st.number_input("$E_{levels}$", min_value=2, value=20)

    boundary_condition: BoundaryCondition = st.selectbox("Boundary Condition", [f.value for f in BoundaryCondition])

    valid_x = np.linspace(x_min, x_max, resolution)

    st.header("Potential $V(x)$", divider="blue")
    potential_function: PotentialFunction = st.selectbox("$V(x)$", [f.value for f in PotentialFunction])
    match potential_function:

        case PotentialFunction.Quadratic.value:
            st.latex("a(x-c)^2")
            potential_increase = st.number_input("$a$", value=10)
            potential_center = st.number_input("$c$", value=0.0)
            potential_x = lambda x: potential_increase * (potential_center - x) ** 2

        case PotentialFunction.Quartic.value:
            st.latex("a(x-c)^4")
            potential_increase = st.number_input("$a$", value=.05)
            potential_center = st.number_input("$c$", value=0.0)
            potential_x = lambda x: potential_increase * (potential_center - x) ** 4

        case PotentialFunction.Trigonometric.value:
            width = x_max - x_min
            st.latex("a\\cos(\\phi + 2 \\pi k x)")
            amplitude = st.number_input("$a$", min_value=.0, value=1.)
            phase = st.number_input("$\\phi$", value=0.0)
            width = x_max - x_min
            k = st.number_input("$k$ ( number of wells )", min_value=.0, value=4.)
            potential_x = lambda x: amplitude * np.cos(phase + np.divide(k * 2 * np.pi * x, width))

        case PotentialFunction.Uniform.value:
            st.latex("ax")
            a = st.number_input("$a$", value=1.)
            potential_x = lambda x: a * x

        case PotentialFunction.DoubleWell.value:
            st.latex("a ( x^4 - 2 x^2 ) + be^{-cx^2}")
            a = st.number_input("$a$", min_value=.0, value=1.)
            b = st.number_input("$b$", min_value=.0, value=30.)
            c = st.number_input("$c$", min_value=.0, value=3.)
            potential_x = lambda x: a * (x ** 4 + 2 * x ** 2) + b * np.exp(-1 * c * x ** 2)


# Based on https://medium.com/@natsunoyuki/quantum-mechanics-with-python-de2a7f8edd1f
def build_1d_hamiltonian(
        inner_n: int,
        inner_dx: float,
        inner_potential_function: Callable[[float], float],
        inner_boundary_condition: BoundaryCondition,
) -> np.array:
    # https://docs.scipy.org/doc/scipy/reference/sparse.html#sparse-array-classes
    inner_hamiltonian = scipy.sparse.eye(inner_n, inner_n, format='lil') * 2
    # P^2 term
    for i in range(inner_n - 1):
        inner_hamiltonian[i, i + 1] = -1
        inner_hamiltonian[i + 1, i] = -1

    # Making 0 and inner_n-1 "neighbours"
    if inner_boundary_condition == BoundaryCondition.Cyclic:
        inner_hamiltonian[0, inner_n - 1] = -1
        inner_hamiltonian[inner_n - 1, 0] = -1

    inner_hamiltonian = np.divide(inner_hamiltonian, inner_dx ** 2)
    # V(X) term
    for i in range(inner_n):
        inner_hamiltonian[i, i] = inner_hamiltonian[i, i] + inner_potential_function(valid_x[i])

    return inner_hamiltonian.tocsc()


st.header("Potential $V(x)$", divider="blue")

st.line_chart(
    pd.DataFrame({
        "Potential": map(potential_x, valid_x),
        "x": valid_x,
    }),
    x="x"
)

st.header("Energy levels $\\ket{n}$", divider="green")

hamiltonian = build_1d_hamiltonian(
    inner_n=resolution,
    inner_dx=valid_x[1] - valid_x[0],  # noqa
    inner_potential_function=potential_x,
    inner_boundary_condition=BoundaryCondition[boundary_condition]
)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html
eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(hamiltonian, k=number_of_energy_levels, which="SM")


@dataclass
class EnergyLevel:
    level: int
    energy: float
    wave_function: np.array


energy_levels = [
    EnergyLevel(level=l, energy=e, wave_function=wf)
    for l, e, wf
    in zip(range(number_of_energy_levels), np.real(eigenvalues), eigenvectors.T)
]

tabs = st.tabs([str(e) for e in range(number_of_energy_levels)])
for tab, energy_level in zip(tabs, energy_levels):
    with tab:
        st.line_chart(
            pd.DataFrame({
                "Re": np.real(energy_level.wave_function),
                "Im": np.imag(energy_level.wave_function),
                # "L_2": np.abs(energy_level.wave_function),
                "x": valid_x
            }),
            x="x"
        )
        st.markdown("Copy pastable values:")
        st.code(energy_level.wave_function, language="python")
