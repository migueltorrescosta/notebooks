"""Delta Estimation UI page - imports physics from src.delta_estimation."""

from numpy.polynomial import Polynomial
from src.angular_momentum import generate_spin_matrices
from src.delta_estimation import DeltaEstimationConfig, generate_hamiltonian, full_calculation
from src.plotting import plot_array
from tqdm import tqdm
from typing import Any, cast
import multiprocessing
import numpy as np
import pandas as pd
import scipy
import streamlit as st

tqdm.pandas()

# LAYOUT
st.set_page_config(page_title="Delta Optimization", page_icon="📈️", layout="wide")

st.header("Delta Optimization", divider="blue")

with st.expander("📖 Methodology", expanded=False):
    st.markdown("""
    **Delta Optimization** estimates unknown parameters in a bipartite quantum system using the Cramer-Rao bound.
    
    **Physical System:** A two-dimensional system qubit (S) coupled to an N-dimensional ancillary system (A):
    $$H = H_S \\otimes \\mathbb{1}_A + \\mathbb{1}_S \\otimes H_A + H_{int}$$
    
    Where:
    - $H_S = -J_S \\sigma_x + \\delta_S \\sigma_z$ (system Hamiltonian)
    - $H_A = -J_A J_x + U_A J_z^2 + \\delta_A J_z$ (ancillary Hamiltonian with spin operators $J_x, J_z$)
    - $H_{int} = \\alpha_{xx}\\sigma_x J_x + \\alpha_{xz}\\sigma_x J_z + \\alpha_{zx}\\sigma_z J_x + \\alpha_{zz}\\sigma_z J_z$
    
    **Methodology:**
    1. **State Evolution**: Evolve the initial state $\\ket{\\psi_0} = \ket{0}_S \otimes \ket{k}_A$ under the Hamiltonian
    2. **Partial Trace**: Trace out the ancillary system to obtain the reduced system density matrix $\rho_t^{(S)}$
    3. **Observable Measurement**: Compute $\langle \\sigma_z \rangle = \mathrm{Tr}[\rho_t^{(S)} \\sigma_z]$
    4. **Parameter Estimation**: Use polynomial fitting to estimate $\delta_S$ from the observable
    5. **Likelihood Analysis**: Construct likelihood function from binomial statistics of repeated measurements
    
    **Key Result:** The posterior distribution over $\delta_S$ is obtained by combining the physical model with experimental observations,
    providing both a point estimate and uncertainty bounds.
    """)

# INPUTS
with st.sidebar:
    st.header("System evolution", divider="blue")

    st.subheader("System controls")
    c1, c2 = st.columns(2)
    with c1:
        j_s = st.number_input("$J_S$", value=-5.2515, step=0.0001)
    with c2:
        delta_s = st.number_input("$\\delta_S$", value=3.0, step=0.0001)

    st.subheader("Ancillary setup")
    c1, c2 = st.columns(2)
    with c1:
        dim_a = st.number_input(
            "$N$ ( Ancillary dim )", min_value=0, value=5, max_value=100
        )
    with c2:
        k = st.number_input("$\ket{k}$", min_value=0, value=0, max_value=dim_a - 1)

    st.subheader("Ancillary controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        j_a = st.number_input("$J_A$", value=0.27688, step=0.0001)
    with c2:
        u_a = st.number_input("$U_A$", value=3.9666, step=0.0001)
    with c3:
        delta_a = st.number_input("$\delta_A$", value=-3.8515472, step=0.0001)

    st.subheader("Interaction controls")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        alpha_xx = st.number_input(r"$\alpha_{xx}$", value=0.501046930, step=0.0001)
    with c2:
        alpha_xz = st.number_input(r"$\alpha_{xz}$", value=-0.843229248, step=0.0001)
    with c3:
        alpha_zx = st.number_input(r"$\alpha_{zx}$", value=-1.66364957, step=0.0001)
    with c4:
        alpha_zz = st.number_input(r"$\alpha_{zz}$", value=-3.09656175, step=0.0001)


# Recorded variables for DataFrame
RECORDED_VARS = [
    "time",
    "<0|rho_system_t|0>",
    "<1|rho_system_t|1>",
    "expected_sigma_z",
    "variance_sigma_z",
    "delta_s",
]


# RELEVANT OPERATORS - using physics module
info_0, info_1, info_2, info_3, info_4 = st.tabs(
    [r"$\sigma_x$", r"$\sigma_z$", r"$J_x$", r"$J_z$", r"$H$"]
)

temp_sigma_x, temp_sigma_z = generate_spin_matrices(dim=2)
with info_0:
    plot_array(temp_sigma_x, key="system_jx")
with info_1:
    plot_array(temp_sigma_z, key="system_jz")

temp_jx, temp_jz = generate_spin_matrices(dim=dim_a)
with info_2:
    plot_array(temp_jx)
with info_3:
    plot_array(temp_jz)

with info_4:
    config = DeltaEstimationConfig(
        ancillary_dimension=dim_a,
        ancillary_initial_state=k,
        j_s=j_s,
        delta_s=delta_s,
        j_a=j_a,
        u_a=u_a,
        delta_a=delta_a,
        alpha_xx=alpha_xx,
        alpha_xz=alpha_xz,
        alpha_zx=alpha_zx,
        alpha_zz=alpha_zz,
    )
    static_hamiltonian = generate_hamiltonian(config)
    plot_array(static_hamiltonian)

st.header("System evolution", divider="blue")

st.latex(f"""
    \\begin{{array}}{{rrccccc}}
    H &=& H_S &+& H_A &+& H_{{int}} \\\\
    &=&  -J_S \\sigma_x + \\delta_S \\sigma_z  &+&
     -J_A J_x + U_AJ_z ^ 2 + \\delta_AJ_z &+&
     \\alpha_{{xx}} \\sigma_x J_x + \\alpha_{{xz}} \\sigma_x J_z + \\alpha_{{zx}} \\sigma_z J_x + \\alpha_{{zz}} \\sigma_z J_z \\\\
    &=&  {-1 * j_s:.2f} \\sigma_x {delta_s:+.2f} \\sigma_z &+&
     {-1 * j_a:.2f} J_x {u_a:+.2f}J_z ^ 2 {delta_a:+.2f}J_z &+& 
     {alpha_xx:.2f} \\sigma_x J_x  {alpha_xz:+.2f} \\sigma_x J_z  {alpha_zx:+.2f} \\sigma_z J_x {alpha_zz:+.2f} \\sigma_z J_z
    \\end{{array}}
    """)


# DATAFRAME CREATION using physics module
@st.cache_data
def compute_evolution_df(dim_a: int, k: int, granularity: int = 100) -> pd.DataFrame:
    base_config = DeltaEstimationConfig(ancillary_dimension=dim_a, ancillary_initial_state=k)
    j_s = base_config.j_s
    delta_s = base_config.delta_s
    j_a = base_config.j_a
    u_a = base_config.u_a
    delta_a = base_config.delta_a
    alpha_xx = base_config.alpha_xx
    alpha_xz = base_config.alpha_xz
    alpha_zx = base_config.alpha_zx
    alpha_zz = base_config.alpha_zz
    
    iterable_1 = []
    for t_val in np.round(np.linspace(0, 10, granularity + 1), 3):
        c = DeltaEstimationConfig(
            ancillary_dimension=dim_a,
            ancillary_initial_state=k,
            j_s=j_s,
            delta_s=delta_s,
            j_a=j_a,
            u_a=u_a,
            delta_a=delta_a,
            alpha_xx=alpha_xx,
            alpha_xz=alpha_xz,
            alpha_zx=alpha_zx,
            alpha_zz=alpha_zz,
            t=float(t_val),
        )
        iterable_1.append(c)
    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpus)
    data = pool.map(full_calculation, iterable_1)
    return pd.DataFrame(data=data, columns=RECORDED_VARS)


df = compute_evolution_df(dim_a, k)

# PLOTS
st.line_chart(
    data=df,
    x="time",
    y=[v for v in RECORDED_VARS if v not in ["time", "delta_s"]],
)

# ESTIMATION CONTROLS
st.header(r"""Estimating $\delta_S$""", divider="orange")

with st.sidebar:
    st.header(r"""Estimating $\delta_S$""", divider="orange")
    c1, c2, c3 = st.columns(3)
    with c1:
        time = st.number_input("$t$", min_value=0.0, value=9.580, step=0.0001)
    with c2:
        guessed_delta_s = st.number_input(
            r"$\hat{\delta}_s$", value=delta_s, step=0.0001
        )
    with c3:
        delta_s_var = st.number_input(
            r"$\Delta \delta_s$", min_value=0.01, value=1.0, step=0.0001
        )

# Find closest time value in cached dataframe
closest_idx = (df["time"] - time).abs().idxmin()
true_probability = float(cast(Any, df.loc[closest_idx, "<1|rho_system_t|1>"]))


@st.cache_data
def compute_estimation_df(
    guessed_delta_s: float, delta_s_var: float, time: float, dim_a: int, k: int
) -> pd.DataFrame:
    base_config = DeltaEstimationConfig(ancillary_dimension=dim_a, ancillary_initial_state=k)
    config_vars = {}
    for field in ['j_s', 'delta_s', 'j_a', 'u_a', 'delta_a', 'alpha_xx', 'alpha_xz', 'alpha_zx', 'alpha_zz']:
        config_vars[field] = getattr(base_config, field)
    
    iterable_2 = []
    for d_val in np.round(np.linspace(guessed_delta_s - delta_s_var, guessed_delta_s + delta_s_var, 51), 3):
        c = DeltaEstimationConfig(**config_vars, delta_s=float(d_val), t=float(time))
        iterable_2.append(c)
    
    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpus)
    data = pool.map(full_calculation, iterable_2)
    df = pd.DataFrame(data=data, columns=RECORDED_VARS)
    df.drop("time", axis=1, inplace=True)
    return df


estimation_df = compute_estimation_df(guessed_delta_s, delta_s_var, time, dim_a, k)

polynomial_fit = cast(Any, Polynomial).fit(
    np.array(estimation_df["expected_sigma_z"]),
    np.array(estimation_df["delta_s"]) - guessed_delta_s,
    deg=1,
)
a0, a1 = polynomial_fit.coef


st.latex(f"""
\mathrm{{Tr}}[\sigma_z \rho^{{(S)}}_t] \approx
{a0:.3f}
{a1:+.3f}(\delta_S - \hat{{\delta}}_S)
+ O((\delta_S - \hat{{\delta}}_S)^2) \\
\Downarrow \\
<1|\rho_{{ {time:.2f} }}^{{(S)}}|1> = {true_probability * 100:.2f}%
""")


# PLOTS
st.line_chart(
    data=estimation_df,
    x="delta_s",
    y=["<0|rho_system_t|0>", "<1|rho_system_t|1>", "expected_sigma_z", "variance_sigma_z"],
)

# Calculating expected likelihood based on observations
with st.sidebar:
    st.subheader("Likelihood", divider="green")
    c1, c2 = st.columns(2)
    with c1:
        n_trials = st.number_input("$N_{trials}$", value=50)
    with c2:
        confidence_interval = st.number_input(
            "Confidence", value=0.9, min_value=0.0001, max_value=0.9999, step=0.0001
        )
        confidence_interval_multiplier = float(scipy.stats.norm.interval(confidence_interval)[1])
    if n_trials > 500:
        st.error(f"Since $N_{{trials}} = {n_trials} \geq 500$, this will be slooow ⚠️")
    show_log_likelihood = st.toggle("Show log likelihood", value=False)


def calculate_probability_density_function(n: int, p: float) -> np.ndarray:
    dist = scipy.stats.binom(n=n, p=p)
    cumulative_probabilities = np.array([dist.cdf(v) for v in range(n_trials + 1)])
    return cumulative_probabilities - np.array([0, *cumulative_probabilities[:-1]])


def calculate_likelihood(
    prob: float,
    true_pdf: np.ndarray = calculate_probability_density_function(n=n_trials, p=true_probability),
) -> float:
    inner_pdf = calculate_probability_density_function(n=n_trials, p=prob)
    return np.dot(inner_pdf, true_pdf)


estimation_df["likelihood"] = estimation_df["<1|rho_system_t|1>"].progress_apply(calculate_likelihood)
estimation_df["likelihood"] = np.divide(estimation_df["likelihood"], estimation_df["likelihood"].mean())


st.subheader("Likelihood", divider="green")
likelihood_arr = np.array(estimation_df["likelihood"])
likelihood_sum = float(np.sum(likelihood_arr))
delta_s_arr = np.array(estimation_df["delta_s"])
estimated_delta_mean = float(np.dot(delta_s_arr, likelihood_arr)) / likelihood_sum
estimated_delta_var = float(np.dot(((delta_s_arr - estimated_delta_mean) ** 2), likelihood_arr)) / likelihood_sum
st.latex(f"\delta_s \approx {estimated_delta_mean:.6f} \pm {confidence_interval_multiplier * np.sqrt(estimated_delta_var):.6f}")

estimation_df["loglikelihood"] = np.log(estimation_df["likelihood"])
estimation_df["loglikelihood"] -= min(estimation_df["loglikelihood"])
y_variable = "loglikelihood" if show_log_likelihood else "likelihood"
st.area_chart(
    data=estimation_df[["delta_s", y_variable]],
    x="delta_s",
    y=[y_variable],
)

# HISTORY
st.header("History", divider="red")
history_data: dict[str, float] = {
    "j_s": j_s,
    "delta_s": delta_s,
    "dim_a": float(dim_a),
    "k": float(k),
    "j_a": j_a,
    "u_a": u_a,
    "delta_a": delta_a,
    "alpha_xx": alpha_xx,
    "alpha_xz": alpha_xz,
    "alpha_zx": alpha_zx,
    "alpha_zz": alpha_zz,
    "time": time,
    "guessed_delta_s": guessed_delta_s,
    "delta_s_var": delta_s_var,
    "log_2_n_trials": float(np.log2(n_trials)),
    "confidence_interval": confidence_interval,
    "confidence_interval_multiplier": confidence_interval_multiplier,
    "estimated_delta_mean": float(estimated_delta_mean),
    "estimated_delta_var": float(estimated_delta_var),
    "log_2_estimated_delta_var": float(np.log2(estimated_delta_var)),
}

with st.sidebar:
    st.header("History", divider="red")
    if st.button("Delete session data ( irreversible )", type="primary"):
        st.session_state.pop("experiment_history_df")

    history_y_axis = st.selectbox("y-axis", sorted(list(history_data.keys())))
    history_x_axis = st.selectbox("x-axis", sorted(list(history_data.keys())))

if "experiment_history_df" not in st.session_state:
    st.session_state.experiment_history_df = pd.DataFrame([history_data])
else:
    st.session_state.experiment_history_df.reset_index(drop=True, inplace=True)
    st.session_state.experiment_history_df.loc[len(st.session_state.experiment_history_df)] = history_data
    st.session_state.experiment_history_df.drop_duplicates(inplace=True)

c1, c2 = st.columns(2)
with c1:
    st.scatter_chart(st.session_state.experiment_history_df, x=history_x_axis, y=history_y_axis)
with c2:
    plot_array(st.session_state.experiment_history_df.T, midpoint=None)