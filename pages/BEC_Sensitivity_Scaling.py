"""
BEC Sensitivity Scaling Analysis Page.

This page analyzes how phase estimation sensitivity Δφ scales with atom number N
for different quantum states in a BEC interferometer:
- CSS (Coherent Spin State): Standard quantum limit, Δφ ∝ 1/√N
- SSS (Spin Squeezed State): Heisenberg scaling, Δφ ∝ N^(-2/3)
- Twin-Fock: Near-Heisenberg scaling, Δφ ∝ 1/N
- NOON: Heisenberg limit, Δφ ∝ 1/N

Features:
- Selectable states (CSS, SSS, Twin-Fock, NOON or ALL)
- N range sweep from 10 to 1000
- Noise channels (one-body loss, two-body loss, phase diffusion)
- Method toggle: Lindblad vs TWA
- Log-log plot of Δφ vs N
- Scaling exponent comparison

Physical Model:
- Dicke basis |J, m⟩ with J = N/2, dimension N+1
- OAT Hamiltonian: H = χ J_z² for squeezing
- Phase accumulation: φ = χ T * N (linear in N)
- Phase sensitivity: Δφ = √(Var(J_z)) / |∂⟨J_z⟩/∂φ|
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from plotly import graph_objects as go

from src.analysis.bec_sensitivity import (
    compute_sensitivity_vs_n as _compute_sensitivity_vn,
)
from src.analysis.scaling_fit import compute_scaling_exponent
from src.physics.noise_channels import NoiseConfig

# Page configuration
st.set_page_config(
    page_title="Metro | BEC Sensitivity Scaling",
    page_icon="📊",
    layout="wide",
)


# =============================================================================
# Cached wrapper for compute_sensitivity_vs_n
# =============================================================================


@st.cache_data
def compute_sensitivity_vs_n(
    state_type: str,
    N_range: tuple[int, int],
    N_points: int,
    chi: float,
    noise_config: NoiseConfig,
    method: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Cached wrapper around :func:`src.analysis.bec_sensitivity.compute_sensitivity_vs_n`.

    See :func:`src.analysis.bec_sensitivity.compute_sensitivity_vs_n` for
    full documentation.

    """
    return _compute_sensitivity_vn(
        state_type=state_type,
        N_range=N_range,
        N_points=N_points,
        chi=chi,
        noise_config=noise_config,
        method=method,
        seed=seed,
    )


# =============================================================================
# UI Layout
# =============================================================================


st.header("Metro | BEC Sensitivity Scaling", divider="blue")


with st.expander("📖 Methodology", expanded=False):
    st.markdown(r"""
    **Physical Model:**

    Phase estimation sensitivity Δφ measures the smallest detectable phase shift.
    For different quantum states, the scaling with atom number N varies:

    | State | Scaling | Exponent α |
    |-------|--------|-----------|
    | CSS (Coherent Spin) | Δφ ∝ 1/√N | -0.5 |
    | SSS (Spin Squeezed) | Δφ ∝ N^(-2/3) | -2/3 |
    | Twin-Fock | Δφ ∝ 1/N | -1.0 |
    | NOON | Δφ ∝ 1/N | -1.0 |

    **Methods:**
    - **Lindblad**: Full quantum master equation (limited to N ≤ 20)
    - **TWA**: Truncated Wigner approximation (scales to N ~ 1000)

    **Noise Channels:**
    - One-body loss (γ₁): Atomic loss
    - Two-body loss (γ₂): Pairwise loss
    - Phase diffusion (γ_φ): Dephasing
    """)


# =============================================================================
# Sidebar Controls
# =============================================================================


with st.sidebar:
    st.header("Configuration", divider="gray")

    # State selection
    st.subheader("State Selection", divider="gray")
    state_options = ["All", "CSS", "SSS", "Twin-Fock", "NOON"]
    selected_state = st.radio(
        "Quantum State",
        state_options,
        index=0,
        help="Select quantum state for sensitivity analysis",
    )

    # N range
    st.subheader("N Sweep Range", divider="gray")
    c1, c2 = st.columns(2)
    with c1:
        N_min = st.number_input("N min", min_value=10, value=20, step=10)
    with c2:
        N_max = st.number_input("N max", min_value=N_min + 10, value=200, step=50)
    N_points = st.slider("N points", min_value=5, value=10, max_value=20)

    # Method toggle
    st.subheader("Method", divider="gray")
    method = st.segmented_control(
        "Simulation Method",
        ["Lindblad", "TWA"],
        default="TWA",
        help="Lindblad for small N, TWA for large N",
    )
    if method is None:
        method = "TWA"  # Default fallback

    # Noise parameters
    st.subheader("Noise Channels", divider="orange")
    include_one_body = st.checkbox("One-body loss (γ₁)", value=False)
    gamma_1 = st.slider("γ₁", 0.0, 0.1, 0.0) if include_one_body else 0.0

    include_two_body = st.checkbox("Two-body loss (γ₂)", value=False)
    gamma_2 = st.slider("γ₂", 0.0, 0.05, 0.0) if include_two_body else 0.0

    include_phase_diff = st.checkbox("Phase diffusion (γ_φ)", value=False)
    gamma_phi = st.slider("γ_φ", 0.0, 0.1, 0.0) if include_phase_diff else 0.0

    # OAT strength
    st.subheader("OAT Parameters", divider="green")
    chi = st.number_input("χ (OAT strength)", value=1.0, min_value=0.0)

    # Random seed
    st.subheader("Reproducibility", divider="gray")
    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=42,
        help="For reproducible random sampling in TWA/Lindblad simulations.",
    )

    # Export
    st.subheader("Export", divider="red")
    export_csv = st.button("Export CSV", type="secondary")


# Build noise config
noise_config = NoiseConfig(
    gamma_1=gamma_1,
    gamma_2=gamma_2,
    gamma_phi=gamma_phi,
)

# Determine states to analyze
if selected_state == "All":
    states_to_analyze = ["CSS", "SSS", "Twin-Fock", "NOON"]
else:
    states_to_analyze = [selected_state]


# =============================================================================
# Main Computation
# =============================================================================


st.subheader("Phase Sensitivity vs N")

# Create placeholder for progress
progress_bar = st.progress(0)
status_text = st.empty()

# Store all results
all_results = {}
scaling_exponents = {}

# States color mapping
state_colors = {
    "CSS": "#1f77b4",
    "SSS": "#ff7f0e",
    "Twin-Fock": "#2ca02c",
    "NOON": "#d62728",
}

for idx, state_type in enumerate(states_to_analyze):
    progress_bar.progress((idx + 1) / len(states_to_analyze))
    status_text.text(f"Computing {state_type}...")

    df = compute_sensitivity_vs_n(
        state_type=state_type,
        N_range=(N_min, N_max),
        N_points=N_points,
        chi=chi,
        noise_config=noise_config,
        method=method,
        seed=int(seed),
    )

    if len(df) > 1:
        # Compute scaling exponent
        α = compute_scaling_exponent(
            np.array(df["N"].values),
            np.array(df["delta_phi"].values),
        )
        scaling_exponents[state_type] = α
        all_results[state_type] = df

progress_bar.empty()
status_text.text("")

# =============================================================================
# Log-Log Plot
# =============================================================================


if all_results:
    fig = go.Figure()

    # Reference lines for SQL and HL
    N_ref = np.array([N_min, N_max])
    delta_sql = 1.0 / np.sqrt(N_ref)
    delta_hl = 1.0 / N_ref

    # Add reference lines
    fig.add_trace(
        go.Scatter(
            x=N_ref,
            y=delta_sql,
            mode="lines",
            name="SQL (1/√N)",
            line={"dash": "dash", "color": "gray"},
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=N_ref,
            y=delta_hl,
            mode="lines",
            name="HL (1/N)",
            line={"dash": "dot", "color": "gray"},
        ),
    )

    # Add data traces
    for state_type, df in all_results.items():
        fig.add_trace(
            go.Scatter(
                x=df["N"],
                y=df["delta_phi"],
                mode="lines+markers",
                name=state_type,
                line={"color": state_colors.get(state_type, "blue")},
                marker={"size": 6},
            ),
        )

    fig.update_layout(
        xaxis_type="log",
        yaxis_type="log",
        xaxis_title="N (atom number)",
        yaxis_title="Δφ (phase uncertainty)",
        template="plotly_white",
        height=350,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    )

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Scaling Exponent Comparison
# =============================================================================


if scaling_exponents:
    st.subheader("Scaling Exponent Comparison")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Create comparison bar chart
        fig_bar = go.Figure()

        states = list(scaling_exponents.keys())
        exponents = list(scaling_exponents.values())
        colors = [state_colors.get(s, "blue") for s in states]

        # Reference values
        ref_exponents = {"CSS": -0.5, "SSS": -2 / 3, "Twin-Fock": -1.0, "NOON": -1.0}

        fig_bar.add_trace(
            go.Bar(
                name="Computed",
                x=states,
                y=exponents,
                marker_color=colors,
            ),
        )

        # Add reference lines
        for state_type, ref in ref_exponents.items():
            if state_type in states:
                fig_bar.add_hline(
                    y=ref,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=f"Ref: {ref:.2f}",
                )

        fig_bar.update_layout(
            yaxis_title="Scaling exponent α",
            yaxis_range=[min(exponents) - 0.2, 0.0],
            template="plotly_white",
            height=250,
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        # Display table
        results_df = pd.DataFrame(
            [
                {
                    "State": state_type,
                    "α": f"{α:.3f}",
                    "Ref": f"{ref_exponents.get(state_type, 'N/A'):.3f}",
                }
                for state_type, α in scaling_exponents.items()
            ],
        )

        st.table(results_df)


# =============================================================================
# Export
# =============================================================================


if export_csv and all_results:
    # Combine all results
    export_dfs = []
    for state_type, df in all_results.items():
        df_export = df.copy()
        df_export["state_type"] = state_type
        export_dfs.append(df_export)

    combined_df = pd.concat(export_dfs, ignore_index=True)

    # Save to CSV
    csv_path = Path("bec_sensitivity_scaling.csv")
    combined_df.to_csv(csv_path, index=False)

    st.download_button(
        label="Download CSV",
        data=csv_path.read_bytes(),
        file_name="bec_sensitivity_scaling.csv",
        mime="text/csv",
    )


# =============================================================================
# Summary
# =============================================================================


st.subheader("Summary")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("States Analyzed", str(len(states_to_analyze)))
with col2:
    st.metric("N Range", f"{N_min} - {N_max}")
with col3:
    st.metric("Method", method)
with col4:
    has_noise = (
        noise_config.gamma_1 > 0
        or noise_config.gamma_2 > 0
        or noise_config.gamma_phi > 0
    )
    st.metric("Noise", "On" if has_noise else "Off")

st.caption(
    f"BEC Scaling | {len(states_to_analyze)} states | "
    f"N ∈ [{N_min}, {N_max}] | Method: {method}",
)
