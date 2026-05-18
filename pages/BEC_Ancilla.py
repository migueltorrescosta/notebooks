"""
BEC Ancilla-Enhanced Metrology Page.

This page demonstrates ancilla-enhanced phase estimation in BEC
interferometers using the Tensor Tree Network (TTN) method.

Features:
- N slider: 1-20 (TTN tractable range)
- State types: coherent, NOON, hybrid
- Coupling slider: λ [0, 5] Hz
- TTN bond dimension growth visualization
- Phase sensitivity comparison (with/without ancilla)

Physical Model:
- Main system: N atoms in Dicke basis
- Ancilla: N auxiliary atoms
- Interaction: λ * J_z ⊗ J_z (entanglement)
- Phase shift: φ on main system
- Measurement: σ_z on main only

Key Result:
With ancilla: Δφ ∝ 1/N² (probe entanglement)
Without ancilla: Δφ ∝ 1/N (standard)
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from plotly import graph_objects as go

from src.algorithms.spin_squeezing import coherent_spin_state
from src.physics.bec_ancilla_system import (
    compute_phase_sensitivity,
    compute_ttn_bond_growth,
    generate_system_state,
)
from src.physics.noise_channels import NoiseConfig

# Page configuration
st.set_page_config(
    page_title="Metro | BEC Ancilla",
    page_icon="🔗",
    layout="wide",
)


# =============================================================================
# UI Layout
# =============================================================================


st.header("Metro | BEC Ancilla-Enhanced Metrology", divider="blue")


with st.expander("📖 Methodology", expanded=False):
    st.markdown(r"""
    **Physical Model:**

    An ancilla-enhanced BEC interferometer uses additional quantum
    resources (auxiliary atoms) to enhance phase estimation.

    **Circuit:**
    ```
    System: |ψ⟩ ──┬──[Phase φ]──[Measure σ_z]
    Ancilla: |0⟩ ──┘
                ↑
           Coupling λ
    ```

    **Key Parameters:**
    - $N$: Number of atoms in main system
    - $\chi$: OAT squeezing strength
    - $\lambda$: System-ancilla coupling
    - $T$: Evolution time

    **Sensitivity Scaling:**
    | Configuration | Δφ Scaling |
    |--------------|------------|
    | No ancilla | 1/√N (SQL) |
    | With ancilla | 1/N (HL) |
    | Entangled probe | 1/N² |

    **TTN:** Tensor Tree Network provides efficient
    representation via bond dimension truncation.
    """)


# =============================================================================
# Sidebar Controls
# =============================================================================


with st.sidebar:
    st.header("Parameters", divider="gray")

    # N slider
    N = st.slider(
        "N (atom number)",
        min_value=1,
        max_value=20,
        value=10,
        help="TTN tractable range (N ≤ 20)",
    )

    # State type
    state_type = st.selectbox(
        "State type",
        ["coherent", "noon", "hybrid"],
        format_func=lambda x: {
            "coherent": "Coherent (CSS)",
            "noon": "NOON State",
            "hybrid": "Hybrid (squeezed + coherent)",
        }[x],
    )

    # OAT strength
    chi = st.number_input("χ (OAT strength)", value=1.0, min_value=0.0)

    # Coupling strength
    lambda_coupling = st.slider(
        "λ (coupling strength)",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )

    # Evolution time
    T = st.number_input("T (evolution time)", value=1.0, min_value=0.0)

    # TTN options
    show_ttn = st.toggle("Show TTN bond dimension", value=True)


# =============================================================================
# Main Computation
# =============================================================================


# Generate states
system_state = generate_system_state(N, state_type, chi, T)

# Compute sensitivity without ancilla (reference)
noise_config = NoiseConfig()
results_no_ancilla = compute_phase_sensitivity(
    N=N,
    state=system_state,
    chi=chi,
    T=T,
    lambda_coupling=0.0,
    has_ancilla=False,
    noise_config=noise_config,
)

# Compute sensitivity with ancilla
results_with_ancilla = compute_phase_sensitivity(
    N=N,
    state=system_state,
    chi=chi,
    T=T,
    lambda_coupling=lambda_coupling,
    has_ancilla=True,
    noise_config=noise_config,
)


# =============================================================================
# Results Display
# =============================================================================


st.subheader("Phase Sensitivity Comparison")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Δφ (no ancilla)",
        f"{results_no_ancilla['delta_phi']:.4f}",
        f"SQL: {results_no_ancilla['delta_phi_sql']:.4f}",
    )
with col2:
    st.metric(
        "Δφ (with ancilla)",
        f"{results_with_ancilla['delta_phi_enhanced']:.4f}",
        f"HL: {results_no_ancilla['delta_phi_hl']:.4f}",
    )
with col3:
    st.metric(
        "Enhancement",
        f"{results_with_ancilla['enhancement']:.2f}x",
    )
with col4:
    improvement = (
        results_no_ancilla["delta_phi"] / results_with_ancilla["delta_phi_enhanced"]
    )
    st.metric("Improvement", f"{improvement:.2f}x")


# =============================================================================
# Comparison Plot
# =============================================================================


st.subheader("Sensitivity vs N")

# Compute across N range
N_range = np.arange(1, N + 1)
delta_no_ancilla = []
delta_with_ancilla = []

for n_atoms in N_range:
    state_n = coherent_spin_state(n_atoms)

    # No ancilla
    res_no = compute_phase_sensitivity(
        N=n_atoms,
        state=state_n,
        chi=chi,
        T=T,
        lambda_coupling=0.0,
        has_ancilla=False,
        noise_config=noise_config,
    )
    delta_no_ancilla.append(res_no["delta_phi"])

    # With ancilla
    res_with = compute_phase_sensitivity(
        N=n_atoms,
        state=state_n,
        chi=chi,
        T=T,
        lambda_coupling=lambda_coupling,
        has_ancilla=True,
        noise_config=noise_config,
    )
    delta_with_ancilla.append(res_with["delta_phi_enhanced"])

# Plot
fig = go.Figure()

# Add reference lines
fig.add_trace(
    go.Scatter(
        x=N_range,
        y=1.0 / np.sqrt(N_range),
        mode="lines",
        name="SQL (1/√N)",
        line={"dash": "dash", "color": "gray"},
    ),
)
fig.add_trace(
    go.Scatter(
        x=N_range,
        y=1.0 / N_range,
        mode="lines",
        name="HL (1/N)",
        line={"dash": "dot", "color": "gray"},
    ),
)

# Add computed data
fig.add_trace(
    go.Scatter(
        x=N_range,
        y=delta_no_ancilla,
        mode="lines+markers",
        name="Without ancilla",
        line={"color": "blue"},
        marker={"size": 8},
    ),
)
fig.add_trace(
    go.Scatter(
        x=N_range,
        y=delta_with_ancilla,
        mode="lines+markers",
        name="With ancilla",
        line={"color": "red"},
        marker={"size": 8},
    ),
)

fig.update_layout(
    xaxis_title="N (atom number)",
    yaxis_title="Δφ (phase uncertainty)",
    yaxis_type="log",
    template="plotly_white",
    height=400,
    legend={"xanchor": "right", "y": 0.99, "x": 0.99},
)

st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# TTN Bond Dimension Growth
# =============================================================================


if show_ttn:
    st.subheader("TTN Bond Dimension")

    # Compute bond dimension growth
    ttn_data = compute_ttn_bond_growth(N, system_state)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Max Bond Dim", str(ttn_data["max_bond_dim"]))
        st.metric("Compression", f"{(2**N) / ttn_data['max_bond_dim']:.1f}x")
    with col2:
        # Plot bond dimension vs epsilon
        fig_ttn = go.Figure()
        fig_ttn.add_trace(
            go.Bar(
                x=[f"ε = {e:.0e}" for e in ttn_data["epsilons"]],
                y=ttn_data["bond_dims"],
                marker_color="green",
            ),
        )
        fig_ttn.update_layout(
            xaxis_title="SVD Threshold ε",
            yaxis_title="Bond Dimension",
            template="plotly_white",
            height=250,
        )
        st.plotly_chart(fig_ttn, use_container_width=True)


# =============================================================================
# Summary Statistics
# =============================================================================


st.subheader("Summary")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("N", str(N))
with col2:
    st.metric("State", state_type)
with col3:
    st.metric("λ", f"{lambda_coupling:.2f}")
with col4:
    st.metric("TTN Bond Dim", str(ttn_data.get("max_bond_dim", "N/A")))

st.caption(rf"""
Δφ (no ancilla) = {results_no_ancilla["delta_phi"]:.4f} |
Δφ (with ancilla) = {results_with_ancilla["delta_phi_enhanced"]:.4f} |
Enhancement = {results_with_ancilla["enhancement"]:.2f}x |
ξ² = {improvement:.2f}
""")

st.caption(
    f"BEC Ancilla | N={N} | λ={lambda_coupling:.2f} | "
    f"TTN: {ttn_data.get('max_bond_dim', 'N/A')}",
)
