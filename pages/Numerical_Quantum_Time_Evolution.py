"""Numerical Quantum Time Evolution UI page - imports physics from src.quantum_time_evolution."""

from dataclasses import dataclass
from functools import partial
from plotly import express as px
from typing import List, cast
import numpy as np
import pandas as pd
import scipy
import streamlit as st

from src.enums import WavePacket, PotentialFunction, BoundaryCondition
from src.quantum_time_evolution import (
    gaussian_wave_packet,
    step_wave_packet,
    potential_quadratic,
    potential_quartic,
    potential_trigonometric,
    potential_uniform,
    potential_double_well,
    build_1d_hamiltonian,
    compute_energy_levels,
    normalize_energy_levels,
    TimeEvolver,
)
from src.plotting import plot_array


st.set_page_config(page_title="Numerical quantum Time Evolution", page_icon="🦖️", layout="wide")

st.cache_data.clear()
st.cache_resource.clear()

st.header("Numerical Quantum Time Evolution", divider="blue")

with st.expander("📖 Methodology", expanded=False):
    st.markdown(r"""
    **Quantum Time Evolution** evolves an initial quantum state under a time-independent Hamiltonian using spectral decomposition.
    
    **The Problem:** Solve the time-dependent Schrödinger equation:
    $$i\hbar \frac{d}{dt}|\psi(t)\rangle = H|\psi(t)\rangle$$
    
    **Methodology:**
    1. **Hamiltonian Construction**: Build the discretized Hamiltonian matrix $H = \frac{\hat{p}^2}{2m} + V(x)$
       using finite differences on a spatial grid of $N_x$ points
    2. **Eigendecomposition**: Find eigenvalues $E_i$ and eigenvectors $\ket{E_i}$ of $H$
    3. **Initial State Decomposition**: Expand $\ket{\psi_0}$ in the energy eigenbasis:
       $$\ket{\psi_0} = \sum_i \lambda_i \ket{E_i}, \quad \lambda_i = \langle E_i|\psi_0\rangle$$
    4. **Time Evolution**: Apply the unitary evolution operator:
       $$\ket{\psi(t)} = \sum_i \lambda_i e^{-iE_it/\hbar}\ket{E_i}$$
    5. **Visualization**: Generate a heatmap of $|\psi(x,t)|^2$ over time
    
    **Initial State Options:**
    - **Gaussian wave packet**: $\psi(x) = e^{-d(x-x_0)^2 + ipx}$ with position/momentum
    - **Step function**: $\psi(x) = e^{ipx} \cdot \mathbb{1}_{[r,s]}(x)$ for localized states
    
    **Validation:** Verify orthonormality of eigenstates and conservation of total probability.
    
    **Note:** The algorithm uses $\hbar = 1$ units and $\Delta t = 0.1$ in the phase evolution.
    """)

# START
with st.sidebar:
    st.header("Setup", divider="blue")
    c1, c2, c3 = st.columns(3)
    with c1:
        x_min = st.number_input("$x_{min}$", value=-3.0)
    with c2:
        x_max = st.number_input("$x_{max}$", value=3.0)
    with c3:
        number_of_spatial_points = st.number_input("$N_x$", value=1000)
    c1, c2, c3 = st.columns(3)
    with c1:
        number_of_energy_levels = st.number_input("$E_{levels}$", min_value=2, value=20)
    with c2:
        time = st.number_input("$t$", min_value=0.00001, value=10.0)
    with c3:
        trotterization_steps = st.number_input("$N_t$", min_value=5, value=500)

    assert x_min < x_max, "We need x_min < x_max"

    valid_x = np.linspace(x_min, x_max, number_of_spatial_points)

    c1, c2 = st.columns(2)
    with c1:
        st.header(r"Initial state $\ket{\psi_0}$", divider="orange")
        initial_wave_packet = st.selectbox(
            r"$\psi_0(x)$", [psi.value for psi in WavePacket]
        )
        match initial_wave_packet:
            case WavePacket.Gaussian.value:
                st.latex(r"e^{{-d(x-x_0)^2 + ipx }}")
                d = st.number_input("$d$", min_value=0.0, value=1.0)
                initial_momentum = st.number_input("$p$", value=2.0)
                initial_center_of_mass = st.number_input("$x_0$", value=-1.0)
                phi_zero = gaussian_wave_packet(valid_x, d=d, x0=initial_center_of_mass, p=initial_momentum)
            case WavePacket.Step.value:
                st.latex(r"\mathbb{1}_{{[r,s]}}e^{{ipx}}")
                r = st.number_input("$r$", value=-1.0)
                s = st.number_input("$s$", min_value=r, value=1.0)
                momentum = st.number_input("$p$", value=5.0)
                phi_zero = step_wave_packet(valid_x, r=r, s=s, p=momentum)

    phi_zero = np.array(phi_zero)
    phi_zero /= np.sqrt(np.sum(np.abs(phi_zero) ** 2))

    with c2:
        st.header(r"Potential $V(x)$", divider="green")
        potential_function = st.selectbox(
            r"$V(x)$", [f.value for f in PotentialFunction]
        )
        match potential_function:
            case PotentialFunction.Quadratic.value:
                st.latex(r"a(x-c)^2")
                potential_increase = st.number_input("$a$", value=0.2)
                potential_center = st.number_input("$c$", value=0.0)
                potential_x = partial(potential_quadratic, a=potential_increase, c=potential_center)
            case PotentialFunction.Quartic.value:
                st.latex(r"a(x-c)^4")
                potential_increase = st.number_input("$a$", value=0.05)
                potential_center = st.number_input("$c$", value=0.0)
                potential_x = partial(potential_quartic, a=potential_increase, c=potential_center)
            case PotentialFunction.Trigonometric.value:
                width = x_max - x_min
                st.latex(r"a\cos(\phi + 2\pi kx)")
                amplitude = st.number_input("$a$", min_value=0.0, value=1.0)
                phase = st.number_input(r"$\phi$", value=0.0)
                width = x_max - x_min
                k = st.number_input("$k$", min_value=0.0, value=4.0)
                potential_x = partial(potential_trigonometric, amplitude=amplitude, phase=phase, k=k, width=width)
            case PotentialFunction.Uniform.value:
                st.latex(r"ax")
                a = st.number_input("$a$", value=1.0)
                potential_x = partial(potential_uniform, a=a)
            case PotentialFunction.DoubleWell.value:
                st.latex(r"a ( x^4 - 2 x^2 ) + be^{-cx^2}")
                a = st.number_input("$a$", min_value=0.0, value=1.0)
                b = st.number_input("$b$", min_value=0.0, value=30.0)
                c = st.number_input("$c$", min_value=0.0, value=3.0)
                potential_x = partial(potential_double_well, a=a, b=b, c=c)

        boundary_condition = st.selectbox("Boundary Condition", [f.value for f in BoundaryCondition])

st.header("Setup", divider="blue")

methodology_info = {
    "Methodology": "These tabs describe the methodology used",
    "Wave": f"We choose a 1-dimensional {initial_wave_packet} as the original state $\ket{{\psi_0}}$.",
    "Potential": f"We choose a {potential_function} potential with {boundary_condition} boundary.",
    "Hamiltonian": f"""We calculate the Hamiltonian using the tri-diagonal representation of $H=\frac{{\hat{{P}}^2}}{{2m}} + V(x)$,
    with ${number_of_spatial_points}$ spatial points.""",
    "Energy levels": f"We calculate the {number_of_energy_levels} lowest eigenvalues/eigenvectors.",
    "Decomposition": f"""We decompose $\ket{{\psi_0}}$ by projecting it into $\ket{{E_i}}$,
    for $i \in \{{ 1, 2, \dots, {number_of_energy_levels} }}$.""",
    "Evolution": f"We evolve $\ket{{\psi_0}}$ and produce a heatmap up to time $t={time:g}$.",
}

tabs = st.tabs(list(methodology_info.keys()))
for tab, content in zip(tabs, list(methodology_info.keys())):
    with tab:
        st.markdown(methodology_info[content])

c1, c2, c3 = st.columns([5, 1, 5])
with c1:
    st.header(r"Initial state $\ket{\psi_0}$", divider="orange")
    st.line_chart(pd.DataFrame({
        "Re": np.real(phi_zero),
        "Im": np.imag(phi_zero),
        "Norm": np.sqrt(np.real(phi_zero) ** 2 + np.imag(phi_zero) ** 2),
        "x": valid_x,
    }), x="x")


# Build Hamiltonian using physics module
def build_hamiltonian_wrapper(
    inner_n: int,
    inner_dx: float,
    inner_potential_function,  # type: ignore
    inner_boundary_condition: BoundaryCondition,
):
    return build_1d_hamiltonian(inner_n, inner_dx, inner_potential_function, inner_boundary_condition)


hamiltonian = build_hamiltonian_wrapper(
    inner_n=number_of_spatial_points,
    inner_dx=valid_x[1] - valid_x[0],
    inner_potential_function=potential_x,
    inner_boundary_condition=BoundaryCondition[boundary_condition],
)

# Compute energy levels
levels = compute_energy_levels(hamiltonian, phi_zero, number_of_energy_levels)

# Normalize
wf_matrix, components, energies = normalize_energy_levels(levels)

with c3:
    st.header(r"Potential $V(x)$", divider="green")
    st.line_chart(pd.DataFrame({
        "Potential": map(potential_x, valid_x),
        "x": valid_x,
    }), x="x")


@dataclass
class EnergyLevel:
    level: int
    energy: float
    wave_function: np.ndarray
    component: float


energy_levels = [
    EnergyLevel(level=level, energy=energy, wave_function=wf, component=phi_zero.T @ wf)
    for level, energy, wf in zip(range(number_of_energy_levels), np.real(energies), wf_matrix.T)
]

# Rotate wave functions so that all components are real valued
for el in energy_levels:
    assert 0 < np.abs(el.component) < 1, f"{el.component} not in [0,1]"
    el.wave_function = np.divide(el.wave_function, el.component) * np.abs(el.component)
    el.component = np.abs(el.component)

with c3:
    st.subheader(r"Lowest energy levels $\ket{E_i}$")
    visible_energy_levels: List[int] = st.multiselect(
        "Visible energy levels",
        options=[el.level for el in energy_levels],
        default=range(min(4, number_of_energy_levels)),
    )
    st.line_chart(pd.DataFrame({
        "x": valid_x,
        **{str(el.level): np.abs(el.wave_function) ** 2 for el in energy_levels if el.level in visible_energy_levels},
    }), x="x")

# Orthonormality check
ortho_error = np.sum(np.abs(np.real(wf_matrix.conj().T @ wf_matrix) - np.eye(number_of_energy_levels)))
with c3:
    st.caption(f"Orthonormality error = {ortho_error:g}")
assert orthonormality_error < 1e-8, "Your energy levels are not orthonormal"

explained_psi_zero = np.sum([np.abs(el.component) ** 2 for el in energy_levels])

with c1:
    st.subheader(r"Decomposition $\ket{\psi_0} = \sum_i \lambda_i \ket{E_i}$")
    st.scatter_chart(pd.DataFrame({
        "probability": [np.abs(el.component) ** 2 for el in energy_levels],
        "level": [el.level for el in energy_levels],
        "energy": [el.energy for el in energy_levels],
    }), x="level", y="energy", size="probability")
    st.caption(f"We know {100 * np.sum([np.abs(el.component) ** 2 for el in energy_levels]):.20g}% of $\ket{{\psi_0}}$")

with c2:
    st.caption("\\begin{array}{c}100\\ket{\psi_0}\\\\ \\\parallel\\\\" + "\\\\".join(
        [f"{100 * el.component:+.2f}\\ket{{ {el.level} }}" for el in energy_levels]
    ) + "\\end{array}")

# Create evolver and compute time evolution
evolver = TimeEvolver(wf_matrix, components, energies)

st.header("Evolution", divider="red")
st.markdown("Green represents an higher wave function amplitude, purple a lower amplitude")


def evolve(t: float) -> np.ndarray:
    phases = components * np.exp(-0.1j * t * energies)
    wf = np.einsum("k,kx->x", phases, wf_matrix)
    return wf / np.linalg.norm(wf)


time_evolution_data = pd.DataFrame(
    data=np.array([np.abs(evolve(temp_t)) ** 2 for temp_t in np.linspace(0, time, trotterization_steps + 1)]).T,
    columns=np.linspace(0, time, trotterization_steps + 1),
    index=valid_x,
)

plot_array(np.array(time_evolution_data.T), midpoint=None, text_auto=False)

st.header(f"Final state $\ket{{\psi_{{{time:g}}} }}$")
phi_time = evolve(t=time)
phi_time_df = pd.DataFrame({
    "Re": np.real(phi_time),
    "Im": np.imag(phi_time),
    "Norm": np.abs(phi_time),
    "x": valid_x,
})

st.line_chart(phi_time_df, x="x")

fig = px.line_3d(phi_time_df, x="x", y="Im", z="Re")
st.plotly_chart(fig, use_container_width=True)