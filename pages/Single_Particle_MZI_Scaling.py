"""Single-Particle MZI: Sensitivity Scaling with Holding Time.

This page simulates a single-particle Mach-Zehnder interferometer and
verifies the Δθ ∝ 1/T_H scaling (exponent α = -1) via log-log analysis.

Imports simulation functions from src/ modules directly.
"""

import numpy as np
import streamlit as st
from plotly import graph_objects as go

from src.analysis.scaling_fit import fit_scaling_exponent
from src.physics.single_particle_mzi import (
    compute_sensitivity_sweep,
    run_validation,
)

st.set_page_config(
    page_title="MZI | Single-Particle Scaling",
    page_icon="📐",
    layout="wide",
)

st.header(
    "MZI | Single-Particle: Sensitivity Scaling with Holding Time",
    divider="blue",
)

# =============================================================================
# Methodology
# =============================================================================

with st.expander("📖 Methodology", expanded=False):
    st.markdown(r"""
    **Physical Model:**

    A single particle ($N = 1$, spin-$1/2$ equivalent) in a Mach-Zehnder
    interferometer with Hamiltonian $H = \theta J_z$ applied during a
    holding time $T_H$.

    **Circuit:**
    ```
    |1,0⟩ → U_BS → exp(-i θ T_H J_z) → U_BS → ⟨J_z⟩
    ```

    **Analytical Result:**
    - $\langle J_z \rangle = -\frac{1}{2} \cos(\theta T_H)$
    - $\mathrm{Var}(J_z) = \frac{1}{4} \sin^2(\theta T_H)$
    - $\partial\langle J_z\rangle/\partial\theta = \frac{T_H}{2} \sin(\theta T_H)$
    - $\boxed{\Delta\theta = \sqrt{\mathrm{Var}(J_z)} / |\partial\langle J_z\rangle/\partial\theta| = 1/T_H}$
    - Scaling exponent: $\alpha = -1$ (standard quantum limit)

    **Key Assumptions:** Pure states, no decoherence, instantaneous beam
    splitters, ideal $J_z$ measurement with infinite statistics.
    """)

# =============================================================================
# Sidebar Controls
# =============================================================================

with st.sidebar:
    st.header("Parameters", divider="gray")

    theta_true = st.number_input(
        r"True $\theta$ (rad/unit time)",
        value=1.0,
        min_value=0.01,
        max_value=10.0,
        step=0.01,
        format="%.2f",
    )

    st.subheader("T_H Sweep")
    t_h_min = st.number_input(
        "Min T_H",
        value=0.1,
        min_value=0.01,
        max_value=10.0,
        step=0.1,
        format="%.2f",
    )
    t_h_max = st.number_input(
        "Max T_H",
        value=100.0,
        min_value=1.0,
        max_value=1000.0,
        step=10.0,
        format="%.1f",
    )
    n_points = st.slider(
        "Number of points",
        min_value=10,
        max_value=2000,
        value=50,
    )

    st.subheader("Numerics")
    fd_delta = st.number_input(
        "Finite-difference step δ",
        value=1e-6,
        min_value=1e-10,
        max_value=1e-3,
        format="%.0e",
        step=1e-7,
    )

    run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

# =============================================================================
# Validation Section
# =============================================================================

st.subheader("Validation Checks")

validation_cols = st.columns(4)

# Run lightweight validations
with st.spinner("Running validation checks..."):
    val = run_validation(theta=theta_true, t_h=1.0)

with validation_cols[0]:
    st.metric(
        "State Normalized",
        "✅" if val["state_normalized"] else "❌",
        help=f"||ψ|| = {val['norm']:.10f}",
    )

with validation_cols[1]:
    st.metric(
        "BS Unitary",
        "✅" if val["bs_unitary"] else "❌",
        help="U_BS U_BS† = I",
    )

with validation_cols[2]:
    st.metric(
        "Δθ = 1/T_H",
        "✅" if val["delta_theta_matches_theory"] else "❌",
        help=f"Δθ = {val['delta_theta_analytical']:.6e}, "
        f"1/T_H = {val['delta_theta_theory']:.6e}",
    )

with validation_cols[3]:
    st.metric(
        "∂⟨J_z⟩/∂θ matches",
        "✅" if val["derivative_match"] else "❌",
        help=f"Relative diff: {val['derivative_relative_diff']:.2e}",
    )

# =============================================================================
# Simulation
# =============================================================================

if run_btn:
    with st.spinner("Running sensitivity sweep..."):
        # Full sensitivity sweep
        df = compute_sensitivity_sweep(
            theta=theta_true,
            t_h_min=t_h_min,
            t_h_max=t_h_max,
            n_points=n_points,
            delta_fd=fd_delta,
        )

        # Fit scaling exponent via log-log regression
        result_analytical = fit_scaling_exponent(
            df["T_H"].to_numpy(),
            df["delta_theta_analytical"].to_numpy(),
        )
        result_numerical = fit_scaling_exponent(
            df["T_H"].to_numpy(),
            df["delta_theta_numerical"].to_numpy(),
        )
        alpha_analytical = result_analytical.alpha
        r_sq_analytical = result_analytical.R_squared
        alpha_numerical = result_numerical.alpha
        r_sq_numerical = result_numerical.R_squared

    # -------------------------------------------------------------------------
    # Summary Metrics
    # -------------------------------------------------------------------------

    st.subheader("Scaling Results")
    metric_cols = st.columns(4)

    with metric_cols[0]:
        st.metric(
            r"α (analytical ∂⟨J_z⟩/∂θ)",
            f"{alpha_analytical:.4f}" if np.isfinite(alpha_analytical) else "N/A",
            delta=f"{alpha_analytical - (-1.0):+.4f}"
            if np.isfinite(alpha_analytical)
            else None,
            delta_color="off",
        )

    with metric_cols[1]:
        st.metric(
            r"α (numerical ∂⟨J_z⟩/∂θ)",
            f"{alpha_numerical:.4f}" if np.isfinite(alpha_numerical) else "N/A",
            delta=f"{alpha_numerical - (-1.0):+.4f}"
            if np.isfinite(alpha_numerical)
            else None,
            delta_color="off",
        )

    with metric_cols[2]:
        st.metric(
            "R² (analytical)",
            f"{r_sq_analytical:.6f}" if np.isfinite(r_sq_analytical) else "N/A",
        )

    with metric_cols[3]:
        st.metric(
            "R² (numerical)",
            f"{r_sq_numerical:.6f}" if np.isfinite(r_sq_numerical) else "N/A",
        )

    st.caption(
        "Expected exponent α = −1.000 (standard quantum limit for a single "
        "probe with increasing interrogation time).",
    )

    # -------------------------------------------------------------------------
    # Plot 1: log-log Δθ vs T_H
    # -------------------------------------------------------------------------

    st.subheader("Δθ vs T_H (Log-Log)")

    fig1 = go.Figure()

    # Theory reference: 1/T_H
    t_ref = np.array([t_h_min, t_h_max])
    dt_ref = 1.0 / t_ref
    fig1.add_trace(
        go.Scatter(
            x=t_ref,
            y=dt_ref,
            mode="lines",
            name=r"$1/T_H$ (theory, α=-1)",
            line={"dash": "dash", "color": "gray", "width": 2},
        ),
    )

    # Analytical data (non-fringe points)
    fringe_points = df[df["is_fringe_extremum"]]
    clean_points = df[~df["is_fringe_extremum"]]

    fig1.add_trace(
        go.Scatter(
            x=clean_points["T_H"],
            y=clean_points["delta_theta_analytical"],
            mode="markers",
            name=r"Δθ (analytical ∂⟨J_z⟩/∂θ)",
            marker={"color": "#1f77b4", "size": 6, "symbol": "circle"},
        ),
    )

    fig1.add_trace(
        go.Scatter(
            x=clean_points["T_H"],
            y=clean_points["delta_theta_numerical"],
            mode="markers",
            name=r"Δθ (numerical ∂⟨J_z⟩/∂θ)",
            marker={"color": "#ff7f0e", "size": 6, "symbol": "x"},
        ),
    )

    if not fringe_points.empty:
        fig1.add_trace(
            go.Scatter(
                x=fringe_points["T_H"],
                y=fringe_points["delta_theta_analytical"],
                mode="markers",
                name="Fringe extremum (excluded)",
                marker={
                    "color": "red",
                    "size": 8,
                    "symbol": "star",
                    "opacity": 0.6,
                },
            ),
        )

    fig1.update_layout(
        xaxis_type="log",
        yaxis_type="log",
        xaxis_title="Holding Time T_H",
        yaxis_title="Δθ (sensitivity)",
        template="plotly_white",
        height=400,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        hovermode="x unified",
    )

    st.plotly_chart(fig1, use_container_width=True)

    # -------------------------------------------------------------------------
    # Plot 2: ⟨J_z⟩ vs T_H (oscillatory signal)
    # -------------------------------------------------------------------------

    st.subheader("⟨J_z⟩ vs T_H")

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=df["T_H"],
            y=df["jz_mean"],
            mode="lines+markers",
            name=r"⟨J_z⟩",
            line={"color": "#2ca02c", "width": 2},
            marker={"size": 5},
        ),
    )

    # Analytical envelope
    t_dense = np.logspace(np.log10(t_h_min), np.log10(t_h_max), 500)
    fig2.add_trace(
        go.Scatter(
            x=t_dense,
            y=-0.5 * np.cos(theta_true * t_dense),
            mode="lines",
            name=r"$-\frac{1}{2}\cos(\theta T_H)$",
            line={"dash": "dot", "color": "gray", "width": 1},
        ),
    )

    fig2.update_layout(
        xaxis_type="log",
        xaxis_title="Holding Time T_H",
        yaxis_title=r"⟨J_z⟩",
        template="plotly_white",
        height=300,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    )

    st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------------------------------------------------
    # Plot 3: ∂⟨J_z⟩/∂θ comparison
    # -------------------------------------------------------------------------

    st.subheader(r"∂⟨J_z⟩/∂θ: Analytical vs Numerical")

    fig3 = go.Figure()

    fig3.add_trace(
        go.Scatter(
            x=df["T_H"],
            y=df["d_jz_analytical"],
            mode="lines+markers",
            name=r"Analytical: $\frac{T_H}{2}\sin(\theta T_H)$",
            line={"color": "#1f77b4", "width": 2},
            marker={"size": 5},
        ),
    )

    fig3.add_trace(
        go.Scatter(
            x=df["T_H"],
            y=df["d_jz_numerical"],
            mode="markers",
            name=f"Numerical (δ = {fd_delta:.0e})",
            marker={"color": "#ff7f0e", "size": 4, "symbol": "x"},
        ),
    )

    # Relative difference
    rel_diff = np.abs(
        (df["d_jz_analytical"] - df["d_jz_numerical"])
        / np.maximum(np.abs(df["d_jz_analytical"]), 1e-15),
    )
    fig3.add_trace(
        go.Scatter(
            x=df["T_H"],
            y=rel_diff,
            mode="lines+markers",
            name="Relative difference",
            yaxis="y2",
            line={"color": "red", "width": 1, "dash": "dot"},
            marker={"size": 4, "symbol": "diamond"},
        ),
    )

    fig3.update_layout(
        xaxis_type="log",
        yaxis_type="log" if len(df) > 0 and np.min(rel_diff) > 0 else "linear",
        xaxis_title="Holding Time T_H",
        yaxis_title="∂⟨J_z⟩/∂θ",
        template="plotly_white",
        height=350,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        yaxis2={
            "title": "Relative difference",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
    )

    st.plotly_chart(fig3, use_container_width=True)

    # -------------------------------------------------------------------------
    # Raw data table
    # -------------------------------------------------------------------------

    with st.expander("📊 View Raw Data", expanded=False):
        display_df = df.copy()
        display_df["T_H"] = display_df["T_H"].map("{:.4f}".format)
        display_df["jz_mean"] = display_df["jz_mean"].map("{:.6f}".format)
        display_df["jz_var"] = display_df["jz_var"].map("{:.6e}".format)
        display_df["d_jz_analytical"] = display_df["d_jz_analytical"].map(
            "{:.6e}".format,
        )
        display_df["d_jz_numerical"] = display_df["d_jz_numerical"].map("{:.6e}".format)
        display_df["delta_theta_analytical"] = display_df["delta_theta_analytical"].map(
            "{:.6e}".format,
        )
        display_df["delta_theta_numerical"] = display_df["delta_theta_numerical"].map(
            "{:.6e}".format,
        )
        display_df["delta_theta_theory"] = display_df["delta_theta_theory"].map(
            "{:.6e}".format,
        )
        display_df["abs_sin"] = display_df["abs_sin"].map("{:.2e}".format)

        st.dataframe(
            display_df[
                [
                    "T_H",
                    "jz_mean",
                    "jz_var",
                    "d_jz_analytical",
                    "d_jz_numerical",
                    "delta_theta_analytical",
                    "delta_theta_numerical",
                    "delta_theta_theory",
                    "is_fringe_extremum",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

else:
    st.info("👈 Configure parameters in the sidebar and press **Run Simulation**.")
