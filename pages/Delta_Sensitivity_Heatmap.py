import itertools
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

cpus = multiprocessing.cpu_count()

st.set_page_config(page_title="Delta Sensitivity Heatmap", page_icon="📈️", layout="wide")

# SENSITIVITY CALCULATION
def sensitivity(
        n: int,  # Ancillary dimension
        k: int,  # Level
        j_s: float,  # system_tunneling_strength
        delta_s: float,  # system_energy_shift
        alpha_x: float,  # sigma_x_coupling_coefficient
        alpha_z: float,  # sigma_z_coupling_coefficient
        t: float  # time
):
    # Returns the sensitivity wrt the tunneling strength and wrt the energy shift of the system for a given level |k>
    assert k <= n, "The level k must be smaller than the ancilliary dimension n"
    assert k >= 0, "The level k must be greater than or equal to 0"

    x_coefficient = alpha_x * np.divide(n - 2 * k, 2) - j_s
    z_coefficient = alpha_z * np.divide(n - 2 * k, 2) + delta_s
    omega_k = np.sqrt(x_coefficient ** 2 + z_coefficient ** 2)

    sensitivity_to_j = np.multiply(
        np.sin(omega_k * t) ** 2,
        np.divide(alpha_x * x_coefficient, omega_k ** 2)
    )
    sensitivity_to_delta = np.multiply(
        np.sin(omega_k * t) ** 2,
        np.divide(alpha_z * z_coefficient, omega_k ** 2)
    )
    return {
        "n": n,
        "k": k,
        "j_s": j_s,
        "delta_s": delta_s,
        "alpha_x": alpha_x,
        "alpha_z": alpha_z,
        "t": t,
        "omega_k": omega_k,
        "sensitivity_to_j": sensitivity_to_j,
        "sensitivity_to_delta": sensitivity_to_delta,
    }


# LAYOUT
st.latex(r"""H =
    ( -J_S \sigma_x + \delta_S \sigma_z) +
    ( U_A J_z^2 + \delta_AJ_z ) +
    (\alpha_x \sigma_x J_z + \alpha_z \sigma_z J_z )
""")
st.latex(r"""\mathrm{Tr}[\rho_t^{(S)}\sigma_z] =
    \sum_{k=0}^{N}
    \braket{k | \rho^{(A)} | k}
    \left ( \cos^2 (\omega_k t) + \sin^2 (\omega_k t) \left ( \frac{\alpha_z \frac{N-2k}{2} + \delta_S}{\omega_k} \right )^2 - \sin^2(\omega_k t) \left ( \frac{\alpha_x \frac{N-2k}{2} - J_S}{\omega_k } \right ) \right )^2
""")

sensitivity_column_1, sensitivity_column_2 = st.columns(2)

variables_of_interest = ["j_s", "delta_s", "alpha_x", "alpha_z", "t"]

# CONTROLS
with st.sidebar:
    st.subheader("System", divider="blue")
    c1, c2=st.columns(2)
    with c1:
        j_s = st.number_input("$J_S$:", -10., 10., 0.)
    with c2:
        delta_s = st.number_input("$\\delta_S$", -10., 10., 0.)

    st.subheader("Ancillary", divider="orange")
    c1, c2 = st.columns(2)
    with c1:
        n = st.number_input("$N$", 0, 20, 4)
    with c2:
        k = st.number_input("$k$ ( initial state )", 0, n, 1)

    st.subheader("Interactions", divider="green")
    c1, c2, c3 = st.columns(3)
    with c1:
        alpha_x = st.number_input("$\\alpha_x$", 0., 10., 1.)
    with c2:
        alpha_z = st.number_input("$\\alpha_z$", 0., 10., 1.)
    with c3:
        t = st.number_input("$t$", 0., 20., 3.)

# DATAFRAME CREATION
resolution = [round(v, 3) for v in np.linspace(-5, 5, 500 + 1)]
generator = itertools.product(resolution, repeat=2)
star_generator = [
    (n, k, j_s, delta_s, alpha_x, alpha_z, t)
    for (alpha_x, alpha_z)
    in itertools.product(resolution, repeat=2)
]
pool = multiprocessing.Pool(processes=cpus)
pool.starmap(sensitivity, star_generator)  # Parallelizes the data generation
sensitivity_df = pd.DataFrame(
    data=pool.starmap(sensitivity, star_generator),
    columns=["n", "k", "j_s", "delta_s", "alpha_x", "alpha_z", "t", "sensitivity_to_j", "sensitivity_to_delta"]
)


def plot_sensitivity(df, title: str, values: str):
    fig, ax = plt.subplots()
    ax.set_title(title)
    sns.heatmap(
        df.pivot(index="alpha_x", columns="alpha_z", values=values),
        ax=ax,
        vmin=-1,
        vmax=1,
        cmap="viridis"
    )
    st.write(fig)


# PLOTS
with sensitivity_column_1:
    plot_sensitivity(sensitivity_df, title="Sensitivity to J_S", values="sensitivity_to_j")

with sensitivity_column_2:
    plot_sensitivity(sensitivity_df, title="Sensitivity to delta_S", values="sensitivity_to_delta")

st.latex(
    r"""\omega_k :=  \sqrt{\left ( \alpha_z \frac{N-2k}{2} + \delta_S \right )^2 +  \left ( \alpha_x \frac{N-2k}{2} - J_S \right )^2}"""
)

# st.text("Full data")
# sensitivity_df.T
