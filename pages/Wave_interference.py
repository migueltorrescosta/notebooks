from functools import partial

from matplotlib import pyplot as plt
from src.visualization.plotting import plot_array
from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Interference Patterns", page_icon="〰", layout="wide")

st.header("Wave Interference", divider="blue")

with st.expander("📖 Methodology", expanded=False):
    st.markdown("""
    **Wave Interference** visualizes the superposition of two coherent waves and the resulting interference patterns.
    
    **Physical Model:**
    Two plane waves propagating in the 2D plane:
    $$\\psi_a(x,y) = a \\cdot e^{2i\\pi\\lambda_a^{-1}(\\sin\\theta_a \\cdot x + \\cos\\theta_a \\cdot y)}$$
    $$\\psi_b(x,y) = b \\cdot e^{2i\\pi\\lambda_b^{-1}(\\sin\\theta_b \\cdot x + \\cos\\theta_b \\cdot y)}$$
    
    Where:
    - $\\lambda_a, \\lambda_b$: Wavelengths of each wave
    - $\\theta_a, \\theta_b$: Propagation angles
    - $a, b$: Amplitudes (normalized to 1)
    
    **Methodology:**
    1. **Grid Generation**: Create a uniform grid of $(x,y)$ points
    2. **Wave Evaluation**: Compute complex values of both waves at each grid point
    3. **Superposition**: $\\psi_{total} = \\psi_a + \\psi_b$
    4. **Visualization**:
       - Real part: Shows wave-like oscillations
       - Imaginary part: Phase-shifted view
       - Amplitude $|\\psi|^2$: Shows interference fringes (constructive/destructive)
    5. **Vector Field**: Phase portrait showing direction of local wave vector
    
    **Key Phenomena:**
    - **Constructive interference**: When waves are in phase ($\\Delta\\phi = 2\\pi n$)
    - **Destructive interference**: When waves are out of phase ($\\Delta\\phi = (2n+1)\\pi$)
    - **Moiré patterns**: Arise when two periodic patterns with slightly different angles or wavelengths overlap
    
    **Applications:** Understanding optical interferometry, quantum measurement theory, and wave physics.
    """)


def wave_a(x: float, y: float, wavelength: float, angle: float) -> complex:
    return np.exp(2j * wavelength * np.pi * (np.sin(angle) * x + np.cos(angle) * y))


def wave_b(x: float, y: float, wavelength: float, angle: float) -> complex:
    return np.exp(2j * wavelength * np.pi * (np.sin(angle) * x + np.cos(angle) * y))


# START
with st.sidebar:
    st.header("Setup", divider="gray")
    c1, c2, c3 = st.columns(3)
    with c1:
        wavelength_a = st.number_input("$\\lambda_a$", value=5.0)
    with c2:
        angle_a = st.slider("$\\theta_a$", min_value=0.0, max_value=1.0, value=0.24)
    with c3:
        st.latex(f"\\bold{{a}} = {wavelength_a:g}e^{{2 i \\pi {angle_a:g}}}")

    c1, c2, c3 = st.columns(3)
    with c1:
        wavelength_b = st.number_input("$\\lambda_b$", value=5.0)
    with c2:
        angle_b = st.slider("$\\theta_b$", min_value=0.0, max_value=1.0, value=0.26)
    with c3:
        st.latex(f"\\bold{{b}} = {wavelength_b:g}e^{{2 i \\pi {angle_b:g}i}}")

    resolution = st.number_input("Resolution", value=20)

    wave_a = partial(wave_a, wavelength_a, angle_a)
    wave_b = partial(wave_b, wavelength_b, angle_b)

st.header("Setup", divider="gray")  # SETUP/CONFIG
st.markdown("""
    This page aims to interactively show the interference pattern caused by two waves.
    In particular this can be used to visualize the [Moiré effect](https://en.wikipedia.org/wiki/Moir%C3%A9_pattern).
    """)

grid = list(
    itertools.product(np.linspace(0, 1, resolution), np.linspace(0, 1, resolution))
)
df = pd.DataFrame(
    [
        {
            "x": x,
            "y": y,
            "wave_a": wave_a(x, y),
            "wave_b": wave_b(x, y),
            "f": wave_a(x, y) + wave_b(x, y),
        }
        for x, y in tqdm(grid, total=resolution**2)
    ]
)

st.markdown(grid)
# st.dataframe(df.T)
# st.dataframe(df.pivot(columns="x", index="y", values="wave_a"))

st.header("Real part", divider="orange")
c1, c2, c3 = st.columns(3)
with c1:
    plot_array(
        np.real(df.pivot(columns="x", index="y", values="wave_a").values),
        text_auto=False,
    )
with c2:
    plot_array(
        np.real(df.pivot(columns="x", index="y", values="wave_b").values),
        text_auto=False,
    )
with c3:
    plot_array(
        np.real(df.pivot(columns="x", index="y", values="f").values), text_auto=False
    )

st.header("Imaginary part", divider="green")
c1, c2, c3 = st.columns(3)
with c1:
    plot_array(
        np.imag(df.pivot(columns="x", index="y", values="wave_a").values),
        text_auto=False,
    )
with c2:
    plot_array(
        np.imag(df.pivot(columns="x", index="y", values="wave_b").values),
        text_auto=False,
    )
with c3:
    plot_array(
        np.imag(df.pivot(columns="x", index="y", values="f").values), text_auto=False
    )

st.header(r"$\| \cdot \|_2$ Norm", divider="green")
plot_array(np.abs(df.pivot(columns="x", index="y", values="f").values), text_auto=False)
# st.header("Raw data", divider="green")
# st.dataframe(df.pivot(columns="y", index="x", values="f"))

# QUIVER view
# https://matplotlib.org/2.0.2/examples/pylab_examples/quiver_demo.html

width = 6
Y, X = np.mgrid[
    -width : width : np.divide(width, 10), -width : width : np.divide(width, 10)
]
wave_a_grid = np.array([wave_a(x, y) for (x, y) in zip(X, Y)])
wave_b_grid = np.array([wave_b(x, y) for (x, y) in zip(X, Y)])

fig, ax = plt.subplots(figsize=(6, 6))
ax.quiver(X, Y, np.real(wave_a_grid), np.imag(wave_a_grid), color="orange")
ax.quiver(X, Y, np.real(wave_b_grid), np.imag(wave_b_grid), color="green")
ax.quiver(
    X,
    Y,
    np.real(wave_a_grid + wave_b_grid),
    np.imag(wave_a_grid + wave_b_grid),
    color="red",
)

# strm = plt.quiver(X, Y, U, V, color="red", linewidth=1)
# fig.colorbar(strm.lines)
# ax.set_title("Phase Plane")
st.pyplot(fig)
