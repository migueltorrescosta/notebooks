from dataclasses import dataclass
from src.angular_momentum import generate_spin_matrices
from src.plotting import plot_array
from tqdm import tqdm
from typing import Dict
import functools
import multiprocessing
import numpy as np
import pandas as pd
import scipy
import streamlit as st

tqdm.pandas()

# LAYOUT
st.set_page_config(page_title="Delta Optimization", page_icon="📈️", layout="wide")

# INPUTS
with st.sidebar:
    st.header("System evolution", divider="blue")

    st.subheader("System controls")
    c1, c2 = st.columns(2)
    with c1:
        j_s = st.number_input("$J_S$", value=-5.2515, step=.0001)
    with c2:
        delta_s = st.number_input("$\\delta_S$", value=3., step=.0001)

    st.subheader("Ancillary setup")
    c1, c2 = st.columns(2)
    with c1:
        dim_a = st.number_input("$N$ ( Ancillary dim )", min_value=0, value=5, max_value=100)
    with c2:
        k = st.number_input("$\\ket{k}$", min_value=0, value=0, max_value=dim_a - 1)

    st.subheader("Ancillary controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        j_a = st.number_input("$J_A$", value=0.27688, step=.0001)
    with c2:
        u_a = st.number_input("$U_A$", value=3.9666, step=.0001)
    with c3:
        delta_a = st.number_input("$\\delta_A$", value=-3.8515472, step=.0001)

    st.subheader("Interaction controls")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        alpha_xx = st.number_input(r"$\alpha_{xx}$", value=0.501046930, step=.0001)
    with c2:
        alpha_xz = st.number_input(r"$\alpha_{xz}$", value=-0.843229248, step=.0001)
    with c3:
        alpha_zx = st.number_input(r"$\alpha_{zx}$", value=-1.66364957, step=.0001)
    with c4:
        alpha_zz = st.number_input(r"$\alpha_{zz}$", value=-3.09656175, step=.0001)

# Assumptions
# 1. dim_S = 2

# AUXILIARY FUNCTIONS

@st.cache_data
def generate_initial_state(ancillary_dimension: int, initial_state: int) -> np.array:
    # |rho_0><rho_0| \otimes |k><k|
    assert initial_state <= ancillary_dimension
    rho0 = np.array([1, 0])
    rho_aux_0 = np.zeros(ancillary_dimension)
    rho_aux_0[initial_state] = 1
    return np.kron(
        np.outer(rho0, rho0),
        np.outer(rho_aux_0, rho_aux_0),
    )


@dataclass
class RunOptions:
    ancillary_dimension: int = dim_a
    ancillary_initial_state: int = k
    j_s: float = j_s
    delta_s: float = delta_s
    j_a: float = j_a
    u_a: float = u_a
    delta_a: float = delta_a
    alpha_xx: float = alpha_xx
    alpha_xz: float = alpha_xz
    alpha_zx: float = alpha_zx
    alpha_zz: float = alpha_zz
    t: float = 0


@dataclass
class Settings:
    recorded_variables = [
        "time",
        "<0|rho_system_t|0>",
        "<1|rho_system_t|1>",
        "expected_sigma_z",
        "variance_sigma_z",
        "delta_s"
    ]
    sigma_x, sigma_z = generate_spin_matrices(dim=2)
    jx, jz = generate_spin_matrices(dim=dim_a)
    initial_state = generate_initial_state(ancillary_dimension=dim_a, initial_state=k)


def generate_hamiltonian(run_options: RunOptions) -> np.array:
    system_hamiltonian = np.kron(
        -1 * run_options.j_s * Settings.sigma_x + run_options.delta_s * Settings.sigma_z,
        np.divide(np.eye(run_options.ancillary_dimension), run_options.ancillary_dimension)
    )
    ancillary_hamiltonian = np.kron(
        np.divide(np.eye(2), 2),
        -1 * run_options.j_a * Settings.jx + run_options.u_a * Settings.jz @ Settings.jz + run_options.delta_a * Settings.jz
    )
    interaction_hamiltonian = functools.reduce(
        lambda x, y: x + y,
        [
            run_options.alpha_xx * np.kron(Settings.sigma_x, Settings.jx),
            run_options.alpha_xz * np.kron(Settings.sigma_x, Settings.jz),
            run_options.alpha_zx * np.kron(Settings.sigma_z, Settings.jx),
            run_options.alpha_zz * np.kron(Settings.sigma_z, Settings.jz),
        ]
    )
    return system_hamiltonian + ancillary_hamiltonian + interaction_hamiltonian


def generate_evolved_system_state(
        hamiltonian: np.array,
        initial_state: np.array,
        time: float
) -> np.array:
    return np.array(
        scipy.linalg.expm(-1j * time * hamiltonian) @ initial_state @ scipy.linalg.expm(1j * time * hamiltonian),
        dtype=complex
    )


def trace_out_ancillary(full_system: np.ndarray) -> np.ndarray:
    derived_ancillary_dimension: int = full_system.shape[0] // 2  # only works if dim_S = 2
    return np.trace(
        np.array(full_system).reshape(2, derived_ancillary_dimension, 2, derived_ancillary_dimension),
        axis1=1,
        axis2=3
    )


def full_calculation(run_options: RunOptions) -> Dict[str, float]:
    hamiltonian = generate_hamiltonian(
        run_options=run_options
    )
    initial_state = generate_initial_state(
        ancillary_dimension=run_options.ancillary_dimension,
        initial_state=run_options.ancillary_initial_state
    )
    rho_t = generate_evolved_system_state(
        hamiltonian=hamiltonian,
        initial_state=initial_state,
        time=run_options.t
    )
    rho_system_t = trace_out_ancillary(
        full_system=rho_t
    )
    observable = np.array([[1, 0], [0, -1]])  # sigma_z

    return {
        "time": run_options.t,
        "<0|rho_system_t|0>": rho_system_t[0][0].real,
        "<1|rho_system_t|1>": rho_system_t[1][1].real,
        "expected_sigma_z": np.trace(rho_system_t @ observable).real,
        "variance_sigma_z": 1 - np.trace(rho_system_t @ observable).real ** 2,
        "delta_s": run_options.delta_s
    }


# RELEVANT OPERATORS
info_0, info_1, info_2, info_3, info_4 = st.tabs(["$\\sigma_x$", "$\\sigma_z$", "$J_x$", "$J_z$", "$H$"])

temp_sigma_x, temp_sigma_z = generate_spin_matrices(dim=2)
with info_0:
    plot_array(temp_sigma_x, key="system_jx")

with info_1:
    plot_array(temp_sigma_z, key="system_jz")

temp_sigma_x, temp_sigma_z = generate_spin_matrices(dim=dim_a)
with info_2:
    plot_array(temp_sigma_x)

with info_3:
    plot_array(temp_sigma_z)

with info_4:
    static_hamiltonian = generate_hamiltonian(run_options=RunOptions())
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

# DATAFRAME CREATION
granularity = 500
iterable = (RunOptions(t=time)  # For some weird reason, starmap needs a tuple, even if it has a single element
    for time
    in np.round(np.linspace(0, 10, granularity + 1), 3)
)

data = map(full_calculation, iterable)
df = pd.DataFrame(
    data=data,
    columns=Settings.recorded_variables
)

# PLOTS
# st.dataframe(data=df)
st.line_chart(
    data=df,
    x="time",
    y=[v for v in Settings.recorded_variables if v not in ["time", "delta_s"]],
)

# ESTIMATION CONTROLS
st.header(r"""Estimating $\delta_S$""", divider="orange")

with st.sidebar:
    st.header(r"""Estimating $\delta_S$""", divider="orange")
    c1, c2, c3 = st.columns(3)
    with c1:
        time = st.number_input("$t$", min_value=0., value=9.580, step=.0001)
    with c2:
        guessed_delta_s = st.number_input("$\\hat{\\delta}_s$", value=delta_s, step=.0001)
    with c3:
        delta_s_var = st.number_input("$\\Delta \\delta_s$", min_value=0.01, value=1., step=.0001)

true_probability = float(np.array(df[df["time"] == time]["<1|rho_system_t|1>"])[0])

# DATAFRAME CREATION
iterable = (
    (RunOptions(delta_s=inner_delta_s, t=time),) # starmap needs a tuple, even if it has a single element
    for inner_delta_s
    in np.round(np.linspace(guessed_delta_s - delta_s_var, guessed_delta_s + delta_s_var, 200 + 1), 3)
)


cpus = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cpus)
df = pd.DataFrame(
    data=pool.starmap(full_calculation, iterable),
    columns=Settings.recorded_variables
)

df.drop("time", axis=1, inplace=True)

polynomial_fit = np.polynomial.Polynomial.fit(
    np.array(df["expected_sigma_z"]),
    np.array(df["delta_s"]) - guessed_delta_s,
    deg=1
)
a0, a1 = polynomial_fit.coef


st.latex(f"""
\\mathrm{{Tr}}[\\sigma_z \\rho^{{(S)}}_t] \\approx
{a0:.3f}
{a1:+.3f}(\\delta_S - \\hat{{\\delta}}_S)
+ O((\\delta_S - \\hat{{\\delta}}_S)^2) \\\\
\\Downarrow \\\\
<1|\\rho_{{ {time:.2f} }}^{{(S)}}|1> = {true_probability * 100:.2f}\\%
""")


# PLOTS
st.line_chart(
    data=df,
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
        confidence_interval = st.number_input("Confidence", value=.9, min_value=0.0001, max_value=.9999, step=.0001)
        confidence_interval_multiplier = scipy.stats.norm.interval(confidence_interval)[1]
    if n_trials>500:
        st.error(f"Since $N_{{trials}} = {n_trials} \\geq 500$, this will be slooow ⚠️")
    show_log_likelihood = st.toggle("Show log likelihood", value=False)


def calculate_probability_density_function(n: int, p: float) -> np.array:
    # Returns full binomial pdf
    # Example: (n=3, p=.6 ) -> [0.064, 0.288, 0.432, 0.216]
    dist = scipy.stats.binom(n=n, p=p)
    cumulative_probabilities = np.array([dist.cdf(v) for v in range(n_trials + 1)])
    return cumulative_probabilities - np.array([0, *cumulative_probabilities[:-1]])


def calculate_likelihood(
        prob: float,
        true_pdf: np.array = calculate_probability_density_function(n=n_trials, p=true_probability)
) -> float:
    inner_pdf = calculate_probability_density_function(n=n_trials, p=prob)
    return np.dot(inner_pdf, true_pdf)


df["likelihood"] = df["<1|rho_system_t|1>"].progress_apply(calculate_likelihood)
df["likelihood"] = np.divide(df["likelihood"], df["likelihood"].mean())


st.subheader("Likelihood", divider="green")
estimated_delta_mean = np.divide(df["delta_s"] @ df["likelihood"], df["likelihood"].sum())
estimated_delta_var = np.divide((df["delta_s"]-estimated_delta_mean)**2 @ df["likelihood"], df["likelihood"].sum())
st.latex(f"\\delta_s \\approx {estimated_delta_mean:.6f} \\pm {confidence_interval_multiplier * np.sqrt(estimated_delta_var):.6f}")

df["loglikelihood"] = np.log(df["likelihood"])
df["loglikelihood"] -= min(df["loglikelihood"])
y_variable = "loglikelihood" if show_log_likelihood else "likelihood"
st.area_chart(
    data=df[["delta_s", y_variable]],
    x="delta_s",
    y=[y_variable],
)

# HISTORY
st.header("History", divider="red")
data = {
    "j_s": j_s,
    "delta_s": delta_s,
    "dim_a": dim_a,
    "k": k,
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
    "log_2_n_trials": np.log2(n_trials),
    "confidence_interval": confidence_interval,
    "confidence_interval_multiplier": confidence_interval_multiplier,
    "estimated_delta_mean": estimated_delta_mean,
    "estimated_delta_var": estimated_delta_var,
    "log_2_estimated_delta_var": np.log2(estimated_delta_var),
}

with st.sidebar:
    st.header("History", divider="red")

    with st.sidebar:
        if st.button("Delete session data ( irreversible )", type="primary"):
            st.session_state.pop("experiment_history_df")

    history_y_axis = st.selectbox("y-axis", sorted(data.keys()))
    history_x_axis = st.selectbox("x-axis", sorted(data.keys()))

if 'experiment_history_df' not in st.session_state:
    st.session_state.experiment_history_df = pd.DataFrame([data])
else:
    st.session_state.experiment_history_df.reset_index(drop=True, inplace=True)
    st.session_state.experiment_history_df.loc[len(st.session_state.experiment_history_df)] = data
    st.session_state.experiment_history_df.drop_duplicates(inplace=True)

c1, c2 = st.columns(2)
with c1:
    st.scatter_chart(
        st.session_state.experiment_history_df,
        x=history_x_axis,
        y=history_y_axis,
    )
with c2:
    plot_array(st.session_state.experiment_history_df.T, midpoint=None)

# 2D likelihood array, useful for debugging
# fig = plot_array(
#     np.array([calculate_probability_density_function(n=n_trials, p=p) for p in np.array(df["<1|rho_system_t|1>"])]),
#     midpoint=max(pdf) / 2
# )
# st.plotly_chart(fig)

# NEXT: Plot the variance of our likelihood as a function of the Number of trials n_trials

