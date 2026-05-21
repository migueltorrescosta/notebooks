"""Energy level calculator for quantum systems.

Computes and visualizes eigenvalues, eigenstates, and energy spectra
for Hamiltonians of interest in quantum metrology (Heisenberg model,
spin systems, etc.).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy
import streamlit as st

from src.evolution.quantum_time_evolution import build_1d_hamiltonian
from src.utils.enums import BoundaryCondition, PotentialFunction
from src.visualization.plotting import plot_array

st.set_page_config(
    page_title="QM | Energy Level Calculator",
    page_icon="🛢",
    layout="wide",
)

st.cache_data.clear()
st.cache_resource.clear()

st.header("QM | Energy Level Calculator", divider="blue")

with st.expander("📖 Methodology", expanded=False):
    st.markdown("""
    **Energy Level Calculator** solves the 1D time-independent Schrödinger equation to find eigenenergies and eigenstates.

    **The Problem:** Find solutions to $H\\ket{\\psi_n} = E_n \\ket{\\psi_n}$ where
    $$H = \\frac{\\hat{p}^2}{2m} + V(x)$$

    **Methodology:**
    1. **Spatial Discretization**: Discretize the domain $[x_{min}, x_{max}]$ into $N_x$ grid points
    2. **Kinetic Operator**: Approximate $\\hat{p}^2$ using the finite difference matrix:
       $$(\\hat{p}^2)_{i,j} \\approx \\frac{\\psi_{i+1} - 2\\psi_i + \\psi_{i-1}}{(\\Delta x)^2}$$
       with optional cyclic boundary conditions for periodic potentials
    3. **Hamiltonian Assembly**: Construct the $N_x \\times N_x$ sparse matrix $H = T + V$
    4. **Eigensolver**: Use `scipy.sparse.linalg.eigs` to find the $N$ lowest eigenvalues/eigenvectors
    5. **Normalization**: Verify orthonormality of eigenstates via inner product matrix

    **Physical Context:** The potential $V(x)$ can be quadratic (harmonic oscillator), quartic, trigonometric (multi-well),
    uniform (linear potential), or double-well (tunneling systems).

    **Validation:** Orthonormality error should be $< 10^{-8}$ for reliable results.
    """)

# START
with st.sidebar:
    st.header("Settings", divider="gray")
    c1, c2 = st.columns(2)
    with c1:
        x_min = st.number_input("$x_{min}$", value=-3.0)
    with c2:
        x_max = st.number_input("$x_{max}$", value=3.0)
    assert x_min < x_max, "We need x_min < x_max"

    c1, c2 = st.columns(2)
    with c1:
        resolution: int = int(st.number_input("$N_x$", value=200))
    with c2:
        number_of_energy_levels = int(
            st.number_input("$E_{levels}$", min_value=2, value=20),
        )

    boundary_condition_str = st.selectbox(
        "Boundary Condition",
        [f.value for f in BoundaryCondition],
    )
    boundary_condition: BoundaryCondition = BoundaryCondition(boundary_condition_str)
    valid_x = np.linspace(x_min, x_max, resolution)

    st.header("Potential $V(x)$", divider="green")
    potential_function_str = st.selectbox(
        "$V(x)$",
        [f.value for f in PotentialFunction],
    )
    potential_function: PotentialFunction = PotentialFunction(potential_function_str)

    # Create widgets FIRST, then define pure functions that use those values
    # This avoids the duplicate key issue when the function is called multiple times

    # Default potential (overridden by match branches below)
    def potential_x(x: float) -> float:
        return 0.0

    match potential_function:
        case PotentialFunction.Quadratic.value:
            st.latex(r"a(x-c)^2")
            a = st.number_input("$a$", value=10.0)
            c = st.number_input("$c$", value=0.0)

            def potential_x(x: float) -> float:
                return a * (c - x) ** 2

        case PotentialFunction.Quartic.value:
            st.latex(r"a(x-c)^4")
            a = st.number_input("$a$", value=0.05)
            c = st.number_input("$c$", value=0.0)

            def potential_x(x: float) -> float:
                return a * (c - x) ** 4

        case PotentialFunction.Trigonometric.value:
            st.latex(r"a\\cos(\\phi + 2 \pi k x)")
            width = x_max - x_min
            a = st.number_input("$a$", min_value=0.0, value=1.0)
            phi = st.number_input("$\\phi$", value=0.0)
            k = st.number_input("$k$ ( number of wells )", min_value=0.0, value=4.0)

            def potential_x(x: float) -> float:
                return a * np.cos(phi + np.divide(k * 2 * np.pi * x, width))

        case PotentialFunction.Uniform.value:
            st.latex(r"ax")
            a = st.number_input("$a$", value=1.0)

            def potential_x(x: float) -> float:
                return a * x

        case PotentialFunction.DoubleWell.value:
            st.latex(r"a ( x^4 - 2 x^2 ) + be^{-cx^2}")
            a = st.number_input("$a$", min_value=0.0, value=1.0)
            b = st.number_input("$b$", min_value=0.0, value=30.0)
            c = st.number_input("$c$", min_value=0.0, value=3.0)

            def potential_x(x: float) -> float:
                return a * (x**4 + 2 * x**2) + b * np.exp(-1 * c * x**2)


st.subheader("Potential $V(x)$")

st.line_chart(
    pd.DataFrame(
        {
            "Potential": map(potential_x, valid_x),
            "x": valid_x,
        },
    ),
    x="x",
    height=200,
)

hamiltonian = build_1d_hamiltonian(
    spatial_points=resolution,
    dx=valid_x[1] - valid_x[0],
    potential_function=potential_x,
    boundary_condition=boundary_condition,
    mass=0.5,  # Match existing page behaviour (T = tridiag(-1,2,-1) / dx²)
    x_grid=valid_x,  # Evaluate V(x) at the actual spatial positions
)
eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(
    hamiltonian,
    k=number_of_energy_levels,
    which="SM",
)


@dataclass
class EnergyLevel:
    level: int
    energy: float
    wave_function: np.ndarray


energy_levels = [
    EnergyLevel(level=level, energy=energy, wave_function=wf)
    for level, energy, wf in zip(
        range(number_of_energy_levels),
        np.real(eigenvalues),
        eigenvectors.T,
        strict=False,
    )
]

with st.sidebar:
    st.header("Orthonormality check", divider="orange")
    error_matrix = np.abs(eigenvectors.T @ eigenvectors) - np.eye(
        number_of_energy_levels,
    )
    st.caption(f"Biggest error: {np.max(error_matrix):.2g}")
    plot_array(error_matrix)

st.subheader("Energy levels $\\ket{n}$")
st.caption(
    f"Showing first {min(5, number_of_energy_levels)} of {number_of_energy_levels} levels",
)

# Use tabs for first few levels
tabs = st.tabs([str(e) for e in range(min(5, number_of_energy_levels))])
for tab, energy_level in zip(tabs, energy_levels[:5], strict=False):
    with tab:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.line_chart(
                pd.DataFrame(
                    {
                        "Re": np.real(energy_level.wave_function),
                        "Im": np.imag(energy_level.wave_function),
                        "x": valid_x,
                    },
                ),
                x="x",
                height=200,
            )
        with c2:
            st.metric("Energy", f"{energy_level.energy:.4f}")
            with st.expander("Show values"):
                st.code(energy_level.wave_function, language="python")
