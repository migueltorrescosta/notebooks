from matplotlib import pyplot as plt
from src.plotting import plot_array
from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Interference Patterns", page_icon="〰", layout="wide")

# START
with st.sidebar:
    st.header("Setup", divider="blue")
    c1, c2, c3 = st.columns(3)
    with c1:
        wavelength_a = st.number_input("$\\lambda_a$", value=5.)
    with c2:
        angle_a = st.slider("$\\theta_a$", min_value=0., max_value=1., value=.24)
    with c3:
        st.latex(f"\\bold{{a}} = {wavelength_a:g}e^{{2 i \\pi {angle_a:g}}}")
        wave_a = lambda x, y: np.exp(2j *wavelength_a * np.pi * (np.sin(angle_a) * x + np.cos(angle_a) * y))

    c1, c2, c3 = st.columns(3)
    with c1:
        wavelength_b = st.number_input("$\\lambda_b$", value=5.)
    with c2:
        angle_b = st.slider("$\\theta_b$", min_value=0., max_value=1., value=.26)
    with c3:
        st.latex(f"\\bold{{b}} = {wavelength_b:g}e^{{2 i \\pi {angle_b:g}i}}")
        wave_b = lambda x, y: np.exp(2j * wavelength_b * np.pi * ( np.sin(angle_b) * x + np.cos(angle_b) * y))

    resolution = st.number_input("Resolution", value=20)

st.header("Setup", divider="blue")
st.markdown("""
    This page aims to interactively show the interference pattern caused by two waves.
    In particular this can be used to visualize the [Moiré effect](https://en.wikipedia.org/wiki/Moir%C3%A9_pattern).
    """)

grid = itertools.product(
    np.linspace(0, 1, resolution),
    np.linspace(0, 1, resolution)
)
df = pd.DataFrame([
    {
        "x": x,
        "y": y,
        "wave_a": wave_a(x, y),
        "wave_b": wave_b(x, y),
        "f": wave_a(x, y) + wave_b(x, y),
    }
    for x, y
    in tqdm(grid, total=resolution ** 2)
])

st.markdown(grid)
# st.dataframe(df.T)
# st.dataframe(df.pivot(columns="x", index="y", values="wave_a"))

st.header("Real part", divider="orange")
c1, c2, c3 = st.columns(3)
with c1:
    plot_array(np.real(df.pivot(columns="x", index="y", values="wave_a")), text_auto=False)
with c2:
    plot_array(np.real(df.pivot(columns="x", index="y", values="wave_b")), text_auto=False)
with c3:
    plot_array(np.real(df.pivot(columns="x", index="y", values="f")), text_auto=False)

st.header("Imaginary part", divider="green")
c1, c2, c3 = st.columns(3)
with c1:
    plot_array(np.imag(df.pivot(columns="x", index="y", values="wave_a")), text_auto=False)
with c2:
    plot_array(np.imag(df.pivot(columns="x", index="y", values="wave_b")), text_auto=False)
with c3:
    plot_array(np.imag(df.pivot(columns="x", index="y", values="f")), text_auto=False)

st.header("$\\| \\cdot \\|_2$ Norm", divider="green")
plot_array(np.abs(df.pivot(columns="x", index="y", values="f")), text_auto=False)
# st.header("Raw data", divider="green")
# st.dataframe(df.pivot(columns="y", index="x", values="f"))

# QUIVER view
# https://matplotlib.org/2.0.2/examples/pylab_examples/quiver_demo.html

width = 6
Y, X = np.mgrid[-width:width:np.divide(width, 10), -width:width:np.divide(width, 10)]
wave_a_grid = np.array([wave_a(x, y) for (x, y) in zip(X, Y)])
wave_b_grid = np.array([wave_b(x, y) for (x, y) in zip(X, Y)])

fig, ax = plt.subplots(figsize=(6, 6))
ax.quiver(X, Y, np.real(wave_a_grid), np.imag(wave_a_grid), color="orange")
ax.quiver(X, Y, np.real(wave_b_grid), np.imag(wave_b_grid), color="green")
ax.quiver(X, Y, np.real(wave_a_grid + wave_b_grid), np.imag(wave_a_grid + wave_b_grid), color="red")

# strm = plt.quiver(X, Y, U, V, color="red", linewidth=1)
# fig.colorbar(strm.lines)
# ax.set_title("Phase Plane")
st.pyplot(fig)
