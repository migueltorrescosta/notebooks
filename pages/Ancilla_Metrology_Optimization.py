"""
Streamlit page: Ancilla-Assisted Metrology Optimisation via Nelder–Mead.

Implements the full-parameter optimisation of a two-qubit (system + ancilla)
Mach–Zehnder interferometer for estimating an unknown phase rate θ.

References:
- `articles/2026-05-12-Ancilla-Assisted-Metrology-Optimization.md`
"""

from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.analysis.ancilla_optimization import (
    build_two_qubit_operators,
    compute_convergence_metric,
    run_theta_scan,
    sensitivity_objective,
    validate_bs_unitarity,
    validate_hold_unitarity,
    validate_operators,
    validate_sensitivity_reasonable,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ancilla Metrology Optimisation",
    page_icon="🎛️",
    layout="wide",
)

st.title("🎛️ Ancilla-Assisted Metrology Optimisation")
st.markdown(
    r"""
    **Nelder–Mead optimisation of sensitivity $\Delta\theta$ for a two-qubit
    (system + ancilla) Mach–Zehnder interferometer.** 11 free parameters:
    4 for the initial product state $|\psi_S\rangle \otimes |\psi_A\rangle$,
    2 beam-splitter durations, holding time $T_H$, and 4 interaction coefficients
    $\alpha_{xx}, \alpha_{xz}, \alpha_{zx}, \alpha_{zz}$.
    """
)

# ── Validate core infrastructure ─────────────────────────────────────────────
with st.expander("🔧 Validation Checks", expanded=False):
    vcol1, vcol2, vcol3, vcol4 = st.columns(4)
    ops = build_two_qubit_operators()
    ops_ok = validate_operators(ops)
    bs_ok = validate_bs_unitarity()
    hold_ok = validate_hold_unitarity()
    sens_ok = validate_sensitivity_reasonable()

    vcol1.success("✅ Operators Hermitian & SU(2)") if ops_ok else vcol1.error("❌")
    vcol2.success("✅ BS unitaries unitary") if bs_ok else vcol2.error("❌")
    vcol3.success("✅ Hold unitaries unitary") if hold_ok else vcol3.error("❌")
    vcol4.success("✅ SQL sensitivity check") if sens_ok else vcol4.error("❌")

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Optimisation Parameters")

    theta_min = st.number_input(
        "θ min", value=0.1, min_value=0.01, step=0.1, format="%.2f"
    )
    theta_max = st.number_input(
        "θ max", value=2.0, min_value=0.1, step=0.1, format="%.2f"
    )
    n_theta = st.slider(
        "Number of θ values", min_value=2, max_value=20, value=6, step=1
    )

    n_restarts = st.slider(
        "Restarts per θ",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        help="Number of random-start Nelder–Mead runs per θ value. More restarts = better coverage.",
    )

    maxiter = st.slider(
        "Nelder–Mead max iterations",
        min_value=50,
        max_value=5000,
        value=500,
        step=50,
        help="Maximum iterations per Nelder–Mead run.",
    )

    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=42,
        help="For reproducible random starts.",
    )

    run_button = st.button(
        "▶ Run θ Scan",
        type="primary",
        use_container_width=True,
    )

# ── Main content ─────────────────────────────────────────────────────────────

# Show theoretical SQL reference
st.subheader("Reference: Standard Quantum Limit (SQL)")
st.markdown(
    r"""
    For a **decoupled system** ($\alpha = 0$) with an optimal initial state
    and 50/50 beam splitters, the sensitivity is:
    $$
    \Delta\theta_{\text{SQL}} = \frac{1}{T_H}
    $$
    The Nelder–Mead optimiser explores the full 11-parameter space seeking
    $\Delta\theta < \Delta\theta_{\text{SQL}}$ through ancilla interaction.
    """
)

# θ values to scan
theta_values = np.linspace(theta_min, theta_max, n_theta)
st.caption(f"θ scan: {n_theta} values from {theta_min:.2f} to {theta_max:.2f}")

if run_button:
    with st.spinner(
        f"Running Nelder–Mead optimisation over {n_theta} θ values "
        f"× {n_restarts} restarts = {n_theta * n_restarts} runs..."
    ):
        start = time.time()
        scan_result = run_theta_scan(
            theta_values=theta_values,
            n_restarts=n_restarts,
            seed=int(seed),
            maxiter=maxiter,
        )
        elapsed = time.time() - start

    st.success(
        f"Scan completed in {elapsed:.1f}s "
        f"({elapsed / (n_theta * n_restarts):.2f}s per run)"
    )

    # ── Summary metrics ──────────────────────────────────────────────────
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    best_overall = float(np.min(scan_result.best_per_theta))
    worst_overall = float(np.max(scan_result.best_per_theta))
    mean_overall = float(np.mean(scan_result.best_per_theta))

    with mcol1:
        st.metric("Best Δθ (overall)", f"{best_overall:.4f}")
    with mcol2:
        st.metric("Worst Δθ", f"{worst_overall:.4f}")
    with mcol3:
        st.metric("Mean Δθ", f"{mean_overall:.4f}")
    with mcol4:
        # Best improvement over the SQL for the optimal T_H*
        best_result = min(
            (r for results in scan_result.all_results.values() for r in results),
            key=lambda r: r.delta_theta_opt,
        )
        t_h_star = best_result.params_opt[6]
        sql_ref = 1.0 / t_h_star if t_h_star > 0 else float("inf")
        improvement = (
            (1.0 - best_result.delta_theta_opt / sql_ref) * 100
            if np.isfinite(sql_ref)
            else 0.0
        )
        st.metric("Best improvement vs SQL", f"{improvement:.1f}%")

    # ── Results table ────────────────────────────────────────────────────
    st.subheader("Results per θ Value")

    # Build a table
    best_results = []
    convergence_metrics = {}
    for theta in theta_values:
        theta_results = scan_result.all_results.get(theta, [])
        best_for_theta = min(theta_results, key=lambda r: r.delta_theta_opt)
        best_results.append(best_for_theta)
        convergence_metrics[theta] = compute_convergence_metric(theta_results)

    table_data = {
        "θ": [],
        "Best Δθ": [],
        "Δθ_SQL": [],
        "vs SQL": [],
        "Spread": [],
        "T_H*": [],
        "Success": [],
        "NFEV": [],
        "α_xx*": [],
        "α_xz*": [],
        "α_zx*": [],
        "α_zz*": [],
    }
    for r in best_results:
        t_h_star = r.params_opt[6]
        sql_dtheta = 1.0 / t_h_star if t_h_star > 0 else float("inf")
        imp_str = (
            f"{((1.0 - r.delta_theta_opt / sql_dtheta) * 100):.1f}%"
            if np.isfinite(sql_dtheta) and sql_dtheta > 0
            else "—"
        )
        spread = convergence_metrics.get(r.theta_true, 0.0)
        spread_str = f"{spread:.3f}" + (" ✅" if spread < 0.10 else " ⚠️")

        table_data["θ"].append(f"{r.theta_true:.2f}")
        table_data["Best Δθ"].append(f"{r.delta_theta_opt:.4f}")
        table_data["Δθ_SQL"].append(
            f"{sql_dtheta:.4f}" if np.isfinite(sql_dtheta) else "—"
        )
        table_data["vs SQL"].append(imp_str)
        table_data["Spread"].append(spread_str)
        table_data["T_H*"].append(f"{t_h_star:.4f}")
        table_data["Success"].append("✅" if r.success else "⚠️")
        table_data["NFEV"].append(str(r.nfev))
        table_data["α_xx*"].append(f"{r.params_opt[7]:.4f}")
        table_data["α_xz*"].append(f"{r.params_opt[8]:.4f}")
        table_data["α_zx*"].append(f"{r.params_opt[9]:.4f}")
        table_data["α_zz*"].append(f"{r.params_opt[10]:.4f}")

    st.table(table_data)

    # ── Sensitivity vs θ plot ────────────────────────────────────────────
    st.subheader("Sensitivity Δθ vs θ")

    fig = go.Figure()

    # Best per θ
    fig.add_trace(
        go.Scatter(
            x=theta_values,
            y=scan_result.best_per_theta,
            mode="lines+markers",
            name="Best Δθ (Nelder–Mead)",
            line=dict(color="firebrick", width=2),
            marker=dict(size=8),
        )
    )

    # SQL reference: dynamic per-θ based on optimal T_H*
    sql_y = [
        1.0 / best_results[i].params_opt[6]
        if best_results[i].params_opt[6] > 0
        else float("inf")
        for i in range(len(theta_values))
    ]
    fig.add_trace(
        go.Scatter(
            x=theta_values,
            y=sql_y,
            mode="lines",
            name="SQL (1/T_H*)",
            line=dict(color="gray", width=2, dash="dash"),
        )
    )

    # All individual results
    all_thetas = []
    all_deltas = []
    for theta in theta_values:
        for r in scan_result.all_results.get(theta, []):
            all_thetas.append(theta)
            all_deltas.append(r.delta_theta_opt)

    fig.add_trace(
        go.Scatter(
            x=all_thetas,
            y=all_deltas,
            mode="markers",
            name="Individual runs",
            marker=dict(color="rgba(100, 100, 100, 0.3)", size=4),
            showlegend=True,
        )
    )

    fig.update_layout(
        title="Δθ vs True θ",
        xaxis_title="True θ",
        yaxis_title="Δθ",
        yaxis_type="log",
        height=450,
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Optimal parameters visualisation ─────────────────────────────────
    with st.expander("Optimal Parameters Details", expanded=False):
        st.markdown("#### Optimal parameters for each θ value")
        param_names = [
            "θ_S",
            "φ_S",
            "θ_A",
            "φ_A",
            "T_BS1",
            "T_BS2",
            "T_H",
            "α_xx",
            "α_xz",
            "α_zx",
            "α_zz",
        ]

        # Build a heatmap of parameters
        param_matrix = np.array([r.params_opt for r in best_results])

        fig_params = go.Figure()
        for i, name in enumerate(param_names):
            fig_params.add_trace(
                go.Scatter(
                    x=theta_values,
                    y=param_matrix[:, i],
                    mode="lines+markers",
                    name=name,
                )
            )

        fig_params.update_layout(
            title="Optimal Parameters vs θ",
            xaxis_title="True θ",
            yaxis_title="Parameter value",
            height=400,
            legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.95),
        )

        st.plotly_chart(fig_params, use_container_width=True)

        # Show parameter correlation
        st.markdown("#### Parameter correlation matrix")
        corr = np.corrcoef(param_matrix.T)
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr,
                x=param_names,
                y=param_names,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                text=np.round(corr, 2),
                texttemplate="%{text}",
            )
        )
        fig_corr.update_layout(
            title="Parameter Correlation (across θ values)",
            height=500,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Interactive single-run test ──────────────────────────────────────
    with st.expander("🧪 Test Single Configuration", expanded=False):
        st.markdown(
            """
            Manually set the 11 parameters to probe the sensitivity landscape.
            """
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            theta_S_test = st.slider("θ_S", 0.0, np.pi, 0.0, 0.01)
            phi_S_test = st.slider("φ_S", 0.0, 2.0 * np.pi, 0.0, 0.01)
            theta_A_test = st.slider("θ_A", 0.0, np.pi, 0.0, 0.01)
            phi_A_test = st.slider("φ_A", 0.0, 2.0 * np.pi, 0.0, 0.01)
        with col2:
            T_BS1_test = st.slider("T_BS1", 0.0, np.pi, np.pi / 2, 0.01)
            T_BS2_test = st.slider("T_BS2", 0.0, np.pi, np.pi / 2, 0.01)
            T_H_test = st.slider("T_H", 0.0, 5.0, 1.0, 0.1)
            theta_test = st.slider("True θ", 0.1, 5.0, 1.0, 0.1)
        with col3:
            a_xx_test = st.slider("α_xx", -2.0, 2.0, 0.0, 0.01)
            a_xz_test = st.slider("α_xz", -2.0, 2.0, 0.0, 0.01)
            a_zx_test = st.slider("α_zx", -2.0, 2.0, 0.0, 0.01)
            a_zz_test = st.slider("α_zz", -2.0, 2.0, 0.0, 0.01)

        test_params = np.array(
            [
                theta_S_test,
                phi_S_test,
                theta_A_test,
                phi_A_test,
                T_BS1_test,
                T_BS2_test,
                T_H_test,
                a_xx_test,
                a_xz_test,
                a_zx_test,
                a_zz_test,
            ]
        )

        if st.button("Compute Sensitivity", key="test_btn"):
            with st.spinner("Computing..."):
                ops = build_two_qubit_operators()
                val = sensitivity_objective(test_params, theta_test, ops)
                sql_ref = 1.0 / T_H_test

            st.metric("Δθ", f"{val:.6f}", delta=f"SQL: {sql_ref:.6f}")
            if val < sql_ref:
                st.success("🎯 Below SQL — potential ancilla enhancement!")
            elif np.isinf(val):
                st.warning("⚠️ At fringe extremum (derivative ≈ 0)")
            else:
                st.info("At or above SQL.")

    # ── Conclusion ────────────────────────────────────────────────────────
    st.subheader("Summary")
    # Compute improvement relative to the SQL at the best T_H*
    best_theta = min(
        (r for results in scan_result.all_results.values() for r in results),
        key=lambda r: r.delta_theta_opt,
    )
    t_h_best = best_theta.params_opt[6]
    sql_best = 1.0 / t_h_best if t_h_best > 0 else float("inf")
    improvement_best = (
        (1.0 - best_theta.delta_theta_opt / sql_best) * 100
        if np.isfinite(sql_best)
        else 0.0
    )
    if improvement_best > 5.0:
        st.success(
            rf"""
            The Nelder–Mead optimiser found configurations with
            **{improvement_best:.1f}% improvement** over the SQL baseline.
            The optimal parameters deviate significantly from the decoupled
            case, suggesting that ancilla interaction can enhance sensitivity
            when the full parameter space is explored.
            """
        )
    elif improvement_best > 0.0:
        st.info(
            rf"""
            Modest improvement ({improvement_best:.1f}%) over SQL observed.
            The optimal configurations are near the decoupled limit,
            suggesting limited ancilla benefit for this parameter regime.
            """
        )
    else:
        st.warning(
            """
            No improvement over SQL observed in this scan. The Nelder–Mead
            optimisation may benefit from more restarts or wider search bounds.
            """
        )

else:
    st.info("👈 Configure parameters in the sidebar and click **▶ Run θ Scan**")

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    r"""
    **Model**: $\mathcal{H} = \mathbb{C}^2 \otimes \mathbb{C}^2$ |
    **Method**: Nelder–Mead (11 parameters) |
    **Objective**: $\Delta\theta = \sqrt{\text{Var}(J_z^S)} / |\partial\langle J_z^S\rangle/\partial\theta|$ |
    🔗 See `articles/2026-05-12-Ancilla-Assisted-Metrology-Optimization.md` for the full plan.
    """
)
