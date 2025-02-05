from dataclasses import dataclass
from plotly import express as px
from typing import Callable, List
import numpy as np
import pandas as pd
import scipy
import streamlit as st

from src.enums import WavePacket, PotentialFunction, BoundaryCondition
from src.plotting import plot_array

st.set_page_config(page_title="Numerical quantum Time Evolution", page_icon="🦖️", layout="wide")

st.cache_data.clear()
st.cache_resource.clear()

# START
with st.sidebar:

    st.header("Setup", divider="blue")
    c1, c2, c3 = st.columns(3)
    with c1:
        x_min = st.number_input("$x_{min}$", value=-3.)
    with c2:
        x_max = st.number_input("$x_{max}$", value=3.)
    with c3:
        number_of_spatial_points = st.number_input("$N_x$", value=1000)
    c1, c2, c3 = st.columns(3)
    with c1:
        number_of_energy_levels = st.number_input("$E_{levels}$", min_value=2, value=20)
    with c2:
        time = st.number_input("$t$", min_value=.00001, value=10.)
    with c3:
        trotterization_steps = st.number_input("$N_t$", min_value=5, value=500)

    assert x_min < x_max, f"We need x_min < x_max"

    valid_x = np.linspace(x_min, x_max, number_of_spatial_points)

    c1, c2 = st.columns(2)
    with c1:
        st.header("Initial state $\\ket{\\psi_0}$", divider="orange")
        initial_wave_packet: WavePacket = st.selectbox("$\\psi_0(x)$", [psi.value for psi in WavePacket])
        match initial_wave_packet:
            case WavePacket.Gaussian.value:
                st.latex("e^{{-d(x-x_0)^2 + ipx }}")
                d = st.number_input("$d$", min_value=.0, value=1.)
                initial_momentum = st.number_input("$p$", value=2.)
                initial_center_of_mass = st.number_input("$x_0$", value=-1.)
                phi_zero_x = lambda x: np.exp(-1 * d * (x - initial_center_of_mass) ** 2 - initial_momentum * 1.j * x)
            case WavePacket.Step.value:
                st.latex("\\mathbb{1}_{{[r,s]}}e^{{ipx}}")
                r = st.number_input("$r$", value=-1.)
                s = st.number_input("$s$", min_value=r, value=1.)
                momentum = st.number_input("$p$", value=5.)
                phi_zero_x = lambda x: np.exp(1j * momentum * x) * (x >= r) * (x <= s)

        phi_zero = np.array([phi_zero_x(x) for x in valid_x])
        phi_zero /= np.sqrt(np.sum(np.abs(phi_zero) ** 2))

    with c2:
        st.header("Potential $V(x)$", divider="green")
        potential_function: PotentialFunction = st.selectbox("$V(x)$", [f.value for f in PotentialFunction])
        match potential_function:

            case PotentialFunction.Quadratic.value:
                st.latex("a(x-c)^2")
                potential_increase = st.number_input("$a$", value=.2)
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
                width = x_max-x_min
                k = st.number_input("$k$", min_value=.0, value=4.)
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
                potential_x = lambda x: a * (x**4 + 2*x**2) + b*np.exp(-1*c*x**2)

        boundary_condition: BoundaryCondition = st.selectbox("Boundary Condition", [f.value for f in BoundaryCondition])

st.header("Setup", divider="blue")

methodology_info = {
    "Methodology": "These tabs describe the methodology used",
    "Wave": f"""
        We choose a 1-dimensional {initial_wave_packet} as the original state $\\ket{{\\psi_0}}$.
    """,
    "Potential": f"""
        We choose a {potential_function} potential with {boundary_condition} boundary.
    """,
    "Hamiltonian": f"""
        We calculate the Hamiltonian using the tri-diagonal representation of $H=\\frac{{\\hat{{P}}^2}}{{2m}} + V(x)$,
        with ${number_of_spatial_points}$ spatial points,
        i.e. using a ${number_of_spatial_points} \\times {number_of_spatial_points}$ array.
    """,
    "Energy levels": f"""
        We calculate the {number_of_energy_levels} lowest eigenvalues/eigenvectors for the given Hamiltonian.
    """,
    "Decomposition": f"""
        We decompose $\\ket{{\\psi_0}}$ by projecting it into $\\ket{{E_i}}$, for $i \\in \\{{ 1, 2, \\dots, {number_of_energy_levels} \\}}$.
    """,
    "Evolution" : f"""
        We evolve $\\ket{{\\psi_0}}$ by evolving the individual eigenstates.
        We produce a heatmap of the time evolution up to time $t={time:g}$.
    """
}

tabs = st.tabs(methodology_info.keys())
for tab, content in zip(tabs, methodology_info.keys()):
    with tab:
        st.markdown(methodology_info[content])

c1, c2, c3 = st.columns([5, 1, 5])
with c1:
    st.header("Initial state $\\ket{\\psi_0}$", divider="orange")
    st.line_chart(
        pd.DataFrame({
            "Re": np.real(phi_zero),
            "Im": np.imag(phi_zero),
            "Norm": np.sqrt(np.real(phi_zero) ** 2 + np.imag(phi_zero) ** 2),
            "x": valid_x
        }),
        x="x"
    )


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


hamiltonian = build_1d_hamiltonian(
    inner_n=number_of_spatial_points,
    inner_dx=valid_x[1] - valid_x[0],  # noqa
    inner_potential_function=potential_x,
    inner_boundary_condition=BoundaryCondition[boundary_condition]
)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html
eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(hamiltonian, k=number_of_energy_levels, which="SM")

with c3:
    st.header("Potential $V(x)$", divider="green")
    st.line_chart(
        pd.DataFrame({
            "Potential": map(potential_x, valid_x),
            "x": valid_x,
        }),
        x="x"
    )


@dataclass
class EnergyLevel:
    level: int
    energy: float
    wave_function: np.array
    component: float  # phi_zero.T @ wave_function


energy_levels = [
    EnergyLevel(level=l, energy=e, wave_function=wf, component=phi_zero.T @ wf)
    for l, e, wf
    in zip(
        range(number_of_energy_levels),
        np.real(eigenvalues),
        eigenvectors.T
    )
]

# Rotate wave functions so that all components are real valued
for el in energy_levels:
    assert 0 < np.abs(el.component) < 1, f"{el.component} $\\nin [0,1]$  "
    el.wave_function = np.divide(el.wave_function, el.component) * np.abs(el.component)  # Turn coefficients real
    el.component = np.abs(el.component)

with c3:
    st.subheader("Lowest energy levels $\\ket{E_i}$")
    visible_energy_levels: List[int] = st.multiselect(
        "Visible energy levels",
        options=[el.level for el in energy_levels],
        default=range(min(4, number_of_energy_levels))
    )
    st.line_chart(
        pd.DataFrame({
            "x": valid_x,
            **{str(el.level): np.abs(el.wave_function) ** 2 for el in energy_levels if
               el.level in visible_energy_levels}
        }),
        x="x"
    )

# st.text("Orthogonality check via inner product table")
orthonormality_error = np.sum(np.abs(
    np.real(np.conjugate(eigenvectors.T) @ eigenvectors) \
    - np.eye(number_of_energy_levels, number_of_energy_levels)
))

with c3:
    st.caption("Orthonormality check")
    st.caption(
        f"\n $\\left \\| (\\sum_i \\ket{{\\lambda_i}})(\\sum_i \\bra{{\\lambda_i}}) - \\mathbb{{1}}) \\right \\|_{{sup}} = {orthonormality_error:g}$")

# plot_array(np.real(np.conjugate(eigenvectors.T) @ eigenvectors))
assert orthonormality_error < 1e-8, f"Your energy levels are not orthonormal"

explained_psi_zero = np.sum([np.abs(el.component) ** 2 for el in energy_levels])
explanation = pd.DataFrame({"component": el.component, "level": str(el.level)} for el in energy_levels)

with c1:
    st.subheader("Decomposition $\\ket{\\psi_0} = \\sum_i \\lambda_i \\ket{E_i}$")
    st.scatter_chart(
        pd.DataFrame({
            "probability": [np.abs(el.component) ** 2 for el in energy_levels],
            "level": [el.level for el in energy_levels],
            "energy": [el.energy for el in energy_levels]
        }),
        x='level',
        y="energy",
        size="probability"
    )
    st.caption(
        f"""We known ${100 * np.sum([np.abs(el.component) ** 2 for el in energy_levels]):.20g} \\%$ of $\\ket{{\\psi_0}}$""")

with c2:
    st.caption(f"""
        $\\begin{{array}}{{c}} 100\\ket{{\\psi_0}} \\\\ \\parallel \\\\
        """ + "\\\\".join([
        f"{100 * el.component:+.2f}\\ket{{ {el.level} }}"
        for el
        in energy_levels
    ]) + "\\end{array}$"
               )

st.header("Evolution", divider="red")
st.markdown("Green represents an higher wave function amplitude, purple a lower amplitude")

def evolve(t: float) -> np.array:
    wf = np.sum([
        np.multiply(
            el.component * np.exp(-1 * .1j * t * el.energy),
            el.wave_function
        )
        for el
        in energy_levels],
        axis=0
    )
    return np.divide(wf, np.sqrt(np.sum(np.abs(wf) ** 2)))


time_evolution_data = pd.DataFrame(
    data=np.array([
        np.abs(evolve(t=temp_t))
        for temp_t
        in np.linspace(0, time, trotterization_steps + 1)
    ]).T,
    columns=np.linspace(0, time, trotterization_steps + 1),
    index=valid_x,
)

plot_array(time_evolution_data.T, midpoint=None, text_auto=False)

st.header(f"Final state $\\ket{{\\psi_{{ {time:g} }} }}$")
phi_time = evolve(t=time)
phi_time_df = pd.DataFrame({
    "Re": np.real(phi_time),
    "Im": np.imag(phi_time),
    "Norm": np.abs(phi_time),
    "x": valid_x
})

st.line_chart(
    phi_time_df,
    x="x"
)

fig = px.line_3d(phi_time_df, x="x", y="Im", z="Re")
st.plotly_chart(fig, use_container_width=True)
