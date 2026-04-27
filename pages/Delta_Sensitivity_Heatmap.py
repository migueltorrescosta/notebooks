"""Delta Sensitivity Heatmap UI page - imports physics from src.sensitivity_analysis."""

import itertools
import multiprocessing
import numpy as np
import pandas as pd
import streamlit as st

from src.analysis.sensitivity_analysis import sensitivity

cpus = multiprocessing.cpu_count()

st.set_page_config(
    page_title="Delta Sensitivity Heatmap", page_icon="📈️", layout="wide"
)

st.header("Delta Sensitivity Heatmap", divider="blue")

with st.expander("📖 Methodology", expanded=False):
    st.markdown(r"""
    **Sensitivity Analysis** quantifies how the system observable depends on parameter variations.
    
    **Physical System:** A reduced model where the ancillary system is prepared in a single Fock state $\ket{k}$:
    $$H = (-J_S \sigma_x + \delta_S \sigma_z) + \alpha_x \sigma_x J_z + \alpha_z \sigma_z J_z$$
    
    The system observable is:
    $$\mathrm{Tr}[\rho_t^{(S)}\sigma_z] = \sum_{k=0}^{N} \braket{k|\rho^{(A)}|k} (\cos^2(\omega_k t) + \sin^2(\omega_k t) (\frac{\alpha_z \frac{N-2k}{2} + \delta_S}{\omega_k})^2 - \sin^2(\omega_k t) (\frac{\alpha_x \frac{N-2k}{2} - J_S}{\omega_k})^2)$$
    
    **Methodology:**
    1. **Rabi Frequency**: Compute $\omega_k$
    2. **Sensitivity to $J_S$**: $\frac{\partial \langle\sigma_z\rangle}{\partial J_S}$
    3. **Sensitivity to $\delta_S$**: $\frac{\partial \langle\sigma_z\rangle}{\partial \delta_S}$
    """)


# SENSITIVITY CALCULATION is now in physics module
st.latex(r"""H = ( -J_S \sigma_x + \delta_S \sigma_z) +
    ( U_A J_z^2 + \delta_AJ_z ) +
    (\alpha_x \sigma_x J_z + \alpha_z \sigma_z J_z )
""")
st.latex(r"""\mathrm{Tr}[\rho_t^{(S)}\sigma_z] = \sum_{k=0}^{N}\braket{k | \rho^{(A)} | k}
    \left ( \cos^2 (\omega_k t) + \sin^2 (\omega_k t) \left ( \frac{\alpha_z \frac{N-2k}{2} + \delta_S}{\omega_k} \right )^2 - \sin^2(\omega_k t) \left ( \frac{\alpha_x \frac{N-2k}{2} - J_S }{\omega_k } \right ) \right )^2""")

sensitivity_column_1, sensitivity_column_2 = st.columns(2)

variables_of_interest = ["j_s", "delta_s", "alpha_x", "alpha_z", "t"]

# CONTROLS
with st.sidebar:
    st.subheader("System", divider="blue")
    c1, c2 = st.columns(2)
    with c1:
        j_s = st.number_input("$J_S$:", -10.0, 10.0, 0.0)
    with c2:
        delta_s = st.number_input(r"$\delta_S$", -10.0, 10.0, 0.0)

    st.subheader("Ancillary", divider="orange")
    c1, c2 = st.columns(2)
    with c1:
        n = st.number_input("$N$", 0, 20, 4)
    with c2:
        k = st.number_input("$k$ ( initial state )", 0, n, 1)

    st.subheader("Interactions", divider="green")
    c1, c2, c3 = st.columns(3)
    with c1:
        alpha_x = st.number_input(r"$\alpha_x$", 0.0, 10.0, 1.0)
    with c2:
        alpha_z = st.number_input(r"$\alpha_z$", 0.0, 10.0, 1.0)
    with c3:
        t = st.number_input("$t$", 0.0, 20.0, 3.0)


# DATAFRAME CREATION using physics module
@st.cache_data
def compute_sensitivity_df(
    n: int, k: int, j_s: float, delta_s: float, alpha_x: float, alpha_z: float, t: float
) -> pd.DataFrame:
    resolution = [round(v, 3) for v in np.linspace(-5, 5, 51)]
    star_generator = [
        (n, k, j_s, delta_s, ax, az, t)
        for (ax, az) in itertools.product(resolution, repeat=2)
    ]
    pool = multiprocessing.Pool(processes=cpus)
    results = pool.starmap(sensitivity, star_generator)
    return pd.DataFrame(
        data=results,
        columns=[
            "n",
            "k",
            "j_s",
            "delta_s",
            "alpha_x",
            "alpha_z",
            "t",
            "omega_k",
            "sensitivity_to_j",
            "sensitivity_to_delta",
        ],
    )


sensitivity_df = compute_sensitivity_df(n, k, j_s, delta_s, alpha_x, alpha_z, t)


# PLOTS
def plot_sensitivity(df: pd.DataFrame, title: str, values: str):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots()
    ax.set_title(title)
    sns.heatmap(
        df.pivot(index="alpha_x", columns="alpha_z", values=values),
        ax=ax,
        vmin=-1,
        vmax=1,
        cmap="viridis",
    )
    st.pyplot(fig)


with sensitivity_column_1:
    plot_sensitivity(sensitivity_df, "Sensitivity to J_S", "sensitivity_to_j")

with sensitivity_column_2:
    plot_sensitivity(sensitivity_df, "Sensitivity to delta_S", "sensitivity_to_delta")

st.latex(
    r"""\omega_k :=  \sqrt{\left ( \alpha_z \frac{N-2k}{2} + \delta_S \right )^2 +  \left ( \alpha_x \frac{N-2k}{2} - J_S \right )^2}"""
)
