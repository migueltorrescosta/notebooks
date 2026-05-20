"""
Streamlit page: Ancilla-Assisted Metrology Optimisation via Nelder–Mead.

Implements the full-parameter optimisation of a two-qubit (system + ancilla)
Mach–Zehnder interferometer for estimating an unknown phase rate θ.
Supports both S-only (J_z^S) and joint (J_z^S + J_z^A) measurement operators.

References:
- `reports/2026-05-12/2026-05-12-Ancilla-Assisted-Metrology-Optimization.md`
- `reports/2026-05-15/2026-05-15-Ancilla-Assisted-Metrology-Joint-Measurement.md`

"""

from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.analysis.ancilla_optimization import (
    build_joint_operator,
    build_two_qubit_operators,
    compute_convergence_metric,
    compute_covariance,
    compute_expectation_and_variance,
    compute_reduced_purity,
    compute_sensitivity,
    evolve_full,
    get_default_bounds,
    random_search_alpha,
    run_theta_scan,
    scan_alpha_single_parameter,
    scan_alpha_with_reoptimisation,
    two_qubit_state,
    validate_bs_unitarity,
    validate_derivative_stability,
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
    Choose between **S-only** ($J_z^S$) and **joint** ($M = J_z^S + J_z^A$)
    measurement operators from the sidebar.
    """,
)

# ── Validate core infrastructure ─────────────────────────────────────────────
with st.expander("🔧 Validation Checks", expanded=False):
    vcol1, vcol2, vcol3, vcol4, vcol5 = st.columns(5)
    ops = build_two_qubit_operators()
    ops_ok = validate_operators(ops)
    bs_ok = validate_bs_unitarity()
    hold_ok = validate_hold_unitarity()
    sens_ok = validate_sensitivity_reasonable()

    # Derivative stability check (uses SQL-optimal configuration)
    psi0_check = two_qubit_state(0.0, 0.0, 0.0, 0.0)
    alpha_check: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    try:
        deriv_ok = validate_derivative_stability(
            psi0=psi0_check,
            T_BS1=np.pi / 2,
            T_BS2=np.pi / 2,
            T_H=1.0,
            theta_true=1.0,
            alpha=alpha_check,
            ops=ops,
        )
    except AssertionError:
        deriv_ok = False

    vcol1.success("✅ Operators Hermitian & SU(2)") if ops_ok else vcol1.error("❌")
    vcol2.success("✅ BS unitaries unitary") if bs_ok else vcol2.error("❌")
    vcol3.success("✅ Hold unitaries unitary") if hold_ok else vcol3.error("❌")
    vcol4.success("✅ SQL sensitivity check") if sens_ok else vcol4.error("❌")
    vcol5.success("✅ FD derivative stable") if deriv_ok else vcol5.error("❌")

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Optimisation Parameters")

    theta_min = st.number_input(
        "θ min",
        value=0.1,
        min_value=0.01,
        step=0.1,
        format="%.2f",
    )
    theta_max = st.number_input(
        "θ max",
        value=5.0,
        min_value=0.1,
        step=0.1,
        format="%.2f",
    )
    n_theta = st.slider(
        "Number of θ values",
        min_value=2,
        max_value=20,
        value=6,
        step=1,
    )

    n_restarts = st.slider(
        "Restarts per θ",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="Number of random-start Nelder–Mead runs per θ value. More restarts = better coverage.",
    )

    meas_choice = st.radio(
        "Measurement operator",
        ["S-only (J_z^S)", "Joint M = J_z^S + J_z^A"],
        index=0,
        help=(
            "**S-only**: objective uses Var(J_z^S) — subsystem measurement on the system qubit. "
            "**Joint**: objective uses Var(J_z^S + J_z^A) — readout of both qubits, "
            "capturing S–A correlations via the covariance term."
        ),
    )

    st.divider()

    with st.expander("📐 Advanced: Bounds", expanded=False):
        st.markdown(
            "Adjust parameter search bounds. Use `T_H max = 20.0` to replicate the expanded-range investigation in the report.",
        )
        t_h_min = st.number_input(
            "T_H min",
            value=0.0,
            min_value=0.0,
            max_value=50.0,
            step=0.5,
            format="%.1f",
        )
        t_h_max = st.number_input(
            "T_H max",
            value=5.0,
            min_value=0.1,
            max_value=50.0,
            step=0.5,
            format="%.1f",
            help="Default = 5.0. Article's expanded-bound investigation used 20.0.",
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

tab1, tab2, tab3 = st.tabs(["θ-Scan", "α-Scan", "Re-optimisation"])

with tab1:
    # Show theoretical SQL reference
    st.subheader("Reference: Standard Quantum Limit (SQL)")
    meas_label = (
        "\\text{Var}(J_z^S)"
        if meas_choice == "S-only (J_z^S)"
        else "\\text{Var}(J_z^S + J_z^A)"
    )
    st.markdown(
        rf"""
        For a **decoupled system** ($\alpha = 0$) with an optimal initial state
        and 50/50 beam splitters, both S-only and joint measurements saturate
        $$
        \Delta\theta_{{\text{{SQL}}}} = \frac{{1}}{{T_H}}
        $$
        The Nelder–Mead optimiser explores the full 11-parameter space, seeking
        $\Delta\theta < \Delta\theta_{{\text{{SQL}}}}$ via ancilla interaction.
        The objective currently uses ${meas_label}$.
        """,
    )

    # θ values to scan
    theta_values = np.linspace(theta_min, theta_max, n_theta)
    st.caption(f"θ scan: {n_theta} values from {theta_min:.2f} to {theta_max:.2f}")

    # Build bounds dict (override T_H only if user changed it)
    bounds = get_default_bounds()
    if "t_h_min" in locals() and "t_h_max" in locals():
        bounds["T_H"] = (float(t_h_min), float(t_h_max))

    # Select measurement operator based on sidebar choice
    ops = build_two_qubit_operators()
    meas_op: np.ndarray | None = (
        None if meas_choice == "S-only (J_z^S)" else build_joint_operator(ops)
    )
    meas_label_short = "S-only" if meas_choice == "S-only (J_z^S)" else "Joint"

    if run_button:
        with st.spinner(
            f"Running Nelder–Mead optimisation over {n_theta} θ values "
            f"× {n_restarts} restarts = {n_theta * n_restarts} runs... "
            f"(T_H ∈ [{bounds['T_H'][0]:.1f}, {bounds['T_H'][1]:.1f}], "
            f"measurement: {meas_label_short})",
        ):
            start = time.time()
            scan_result = run_theta_scan(
                theta_values=theta_values,
                n_restarts=n_restarts,
                seed=int(seed),
                maxiter=maxiter,
                bounds=bounds,
                meas_op=meas_op,
            )
            elapsed = time.time() - start

            # Also run a comparison scan with the opposite measurement for cross-reference
            # (only if requested meas_op differs from default — i.e. joint mode)
            if meas_op is not None:
                st.info("⏳ Running companion S-only scan for comparison...")
                sonly_scan_result = run_theta_scan(
                    theta_values=theta_values,
                    n_restarts=n_restarts,
                    seed=int(seed),
                    maxiter=maxiter,
                    bounds=bounds,
                    meas_op=None,
                )

                # Compute comparison stats for each θ
                comparison_table_data: dict[str, list[str]] = {
                    "θ": [],
                    f"Δθ_{meas_label_short} (best)": [],
                    "Δθ S-only (best)": [],
                    "Joint < S-only?": [],
                }
                for theta in theta_values:
                    joint_results = scan_result.all_results.get(theta, [])
                    sonly_results = sonly_scan_result.all_results.get(theta, [])
                    best_joint = (
                        min(
                            joint_results, key=lambda r: r.delta_theta_opt
                        ).delta_theta_opt
                        if joint_results
                        else float("inf")
                    )
                    best_sonly = (
                        min(
                            sonly_results, key=lambda r: r.delta_theta_opt
                        ).delta_theta_opt
                        if sonly_results
                        else float("inf")
                    )
                    comparison_table_data["θ"].append(f"{theta:.2f}")
                    comparison_table_data[f"Δθ_{meas_label_short} (best)"].append(
                        f"{best_joint:.4f}" if np.isfinite(best_joint) else "∞"
                    )
                    comparison_table_data["Δθ S-only (best)"].append(
                        f"{best_sonly:.4f}" if np.isfinite(best_sonly) else "∞"
                    )
                    comparison_table_data["Joint < S-only?"].append(
                        "✅"
                        if best_joint < best_sonly - 1e-8
                        else "≈ tie"
                        if abs(best_joint - best_sonly) < 1e-8
                        else "❌"
                    )

                st.subheader("📊 Joint vs S-only Comparison")
                st.table(comparison_table_data)

        st.success(
            f"Scan completed in {elapsed:.1f}s "
            f"({elapsed / (n_theta * n_restarts):.2f}s per run)",
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

        table_data: dict[str, list[str]] = {
            "θ": [],
            "Best Δθ": [],
            "Δθ_SQL": [],
            "vs SQL": [],
            "Spread": [],
            "T_H*": [],
            "Purity": [],
            "⟨M⟩": [],
            "Cov": [],
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
                f"{sql_dtheta:.4f}" if np.isfinite(sql_dtheta) else "—",
            )
            table_data["vs SQL"].append(imp_str)
            table_data["Spread"].append(spread_str)
            table_data["T_H*"].append(f"{t_h_star:.4f}")
            table_data["Purity"].append(f"{r.purity_S:.3f}")
            # Joint measurement diagnostics (always computed by the backend)
            table_data["⟨M⟩"].append(f"{r.expectation_M:.4f}")
            table_data["Cov"].append(f"{r.covariance_SA:.4f}")
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
                line={"color": "firebrick", "width": 2},
                marker={"size": 8},
            ),
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
                line={"color": "gray", "width": 2, "dash": "dash"},
            ),
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
                marker={"color": "rgba(100, 100, 100, 0.3)", "size": 4},
                showlegend=True,
            ),
        )

        fig.update_layout(
            title="Δθ vs True θ",
            xaxis_title="True θ",
            yaxis_title="Δθ",
            yaxis_type="log",
            height=450,
            legend={"yanchor": "top", "y": 0.95, "xanchor": "left", "x": 0.05},
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
                    ),
                )

            fig_params.update_layout(
                title="Optimal Parameters vs θ",
                xaxis_title="True θ",
                yaxis_title="Parameter value",
                height=400,
                legend={"yanchor": "top", "y": 0.95, "xanchor": "right", "x": 0.95},
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
                ),
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
                """,
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
                ],
            )

            if st.button("Compute Sensitivity", key="test_btn"):
                with st.spinner("Computing..."):
                    ops_test = build_two_qubit_operators()
                    M_op_test = build_joint_operator(ops_test)
                    # Build state and evolve to get diagnostics
                    psi0_test = two_qubit_state(
                        theta_S_test, phi_S_test, theta_A_test, phi_A_test
                    )
                    psi_final = evolve_full(
                        psi0_test,
                        T_BS1_test,
                        T_BS2_test,
                        T_H_test,
                        theta_test,
                        (a_xx_test, a_xz_test, a_zx_test, a_zz_test),
                        ops_test,
                    )
                    exp_S, var_S = compute_expectation_and_variance(
                        psi_final, ops_test["Jz_S"]
                    )
                    exp_M, var_M = compute_expectation_and_variance(
                        psi_final, M_op_test
                    )
                    cov_SA = compute_covariance(psi_final, ops_test)
                    purity_SA = compute_reduced_purity(psi_final)

                    # Sensitivity under both measurement operators
                    val_sonly = compute_sensitivity(
                        psi0_test,
                        T_BS1_test,
                        T_BS2_test,
                        T_H_test,
                        theta_test,
                        (a_xx_test, a_xz_test, a_zx_test, a_zz_test),
                        ops_test,
                        meas_op=None,
                    )
                    val_joint = compute_sensitivity(
                        psi0_test,
                        T_BS1_test,
                        T_BS2_test,
                        T_H_test,
                        theta_test,
                        (a_xx_test, a_xz_test, a_zx_test, a_zz_test),
                        ops_test,
                        meas_op=M_op_test,
                    )
                    sql_ref = 1.0 / T_H_test

                dcol1, dcol2, dcol3 = st.columns(3)
                with dcol1:
                    st.metric(
                        "Δθ (S-only)", f"{val_sonly:.6f}", delta=f"SQL: {sql_ref:.6f}"
                    )
                    if val_sonly < sql_ref:
                        st.success("🎯 Below SQL (S-only)")
                    elif np.isinf(val_sonly):
                        st.warning("⚠️ Fringe extremum (S-only)")
                with dcol2:
                    st.metric(
                        "Δθ (Joint M)", f"{val_joint:.6f}", delta=f"SQL: {sql_ref:.6f}"
                    )
                    if val_joint < sql_ref:
                        st.success("🎯 Below SQL (Joint)")
                    elif np.isinf(val_joint):
                        st.warning("⚠️ Fringe extremum (Joint)")
                with dcol3:
                    st.metric(
                        "Purity Tr(ρ_S²)",
                        f"{purity_SA:.4f}",
                        delta="Entangled" if purity_SA < 0.99 else "Product",
                    )

                # Detailed diagnostics
                with st.expander("Diagnostics", expanded=False):
                    diag_col1, diag_col2, diag_col3, diag_col4 = st.columns(4)
                    diag_col1.metric("⟨J_z^S⟩", f"{exp_S:.4f}")
                    diag_col1.metric("Var(J_z^S)", f"{var_S:.4f}")
                    diag_col2.metric("⟨M⟩", f"{exp_M:.4f}")
                    diag_col2.metric("Var(M)", f"{var_M:.4f}")
                    diag_col3.metric("Cov(J_z^S, J_z^A)", f"{cov_SA:.4f}")
                    diag_col3.metric("⟨J_z^A⟩", f"{exp_M - exp_S:.4f}")
                    diag_col4.metric("T_H", f"{T_H_test:.2f}")
                    diag_col4.metric("SQL ref", f"{sql_ref:.4f}")

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

        # Joint measurement diagnostics summary
        cov_vals = [
            r.covariance_SA
            for results in scan_result.all_results.values()
            for r in results
            if np.isfinite(r.covariance_SA)
        ]
        max_cov = max(abs(c) for c in cov_vals) if cov_vals else 0.0
        purity_vals = [
            r.purity_S
            for results in scan_result.all_results.values()
            for r in results
            if np.isfinite(r.purity_S)
        ]
        min_purity = min(purity_vals) if purity_vals else 1.0

        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            if improvement_best > 5.0:
                st.success(
                    rf"""
                    **{meas_label_short} measurement**: The optimiser found configurations
                    with **{improvement_best:.1f}% improvement** over the SQL baseline.
                    The optimal parameters deviate significantly from the decoupled case.
                    """,
                )
            elif improvement_best > 0.0:
                st.info(
                    rf"""
                    **{meas_label_short} measurement**: Modest improvement
                    ({improvement_best:.1f}%) over SQL observed.
                    Optimal configurations are near the decoupled limit.
                    """,
                )
            else:
                st.warning(
                    f"""
                    **{meas_label_short} measurement**: No improvement over SQL observed.
                    Try more restarts or wider search bounds.
                    """,
                )
        with summary_col2:
            if meas_op is not None:
                st.info(
                    rf"""
                    **Joint measurement diagnostics**: max |Cov| = {max_cov:.4f},
                    min purity = {min_purity:.3f}.
                    Non-zero covariance indicates S–A entanglement generated
                    via $H_{{\text{{int}}}}$.
                    """,
                )
            else:
                st.info(
                    rf"""
                    **S–A correlation**: min purity = {min_purity:.3f}.
                    Purity < 1 indicates S–A entanglement generated
                    via $H_{{\text{{int}}}}$.
                    """,
                )

    else:
        st.info("👈 Configure parameters in the sidebar and click **▶ Run θ Scan**")

    # ── Footer ───────────────────────────────────────────────────────────────────
    st.divider()
    obj_label = (
        r"\Delta\theta = \sqrt{\text{Var}(J_z^S)} / |\partial\langle J_z^S\rangle/\partial\theta|"
        if meas_choice == "S-only (J_z^S)"
        else (
            r"\Delta\theta = \sqrt{\text{Var}(J_z^S + J_z^A)} "
            r"/ |\partial\langle J_z^S + J_z^A\rangle/\partial\theta|"
        )
    )
    st.caption(
        rf"""
        **Model**: $\mathcal{{H}} = \mathbb{{C}}^2 \otimes \mathbb{{C}}^2$ |
        **Method**: Nelder–Mead (11 parameters) |
        **Objective**: ${obj_label}$ |
        🔗 See `reports/2026-05-12/2026-05-12-Ancilla-Assisted-Metrology-Optimization.md` and
        `reports/2026-05-15/2026-05-15-Ancilla-Assisted-Metrology-Joint-Measurement.md`.
        """,
    )

with tab2:
    st.subheader("α-Coefficient Scan")

    alpha_scan_mode = st.radio(
        "Scan mode",
        ["Single coefficient scan", "4D random search"],
        key="alpha_scan_mode",
    )

    if alpha_scan_mode == "Single coefficient scan":
        acol1, acol2, acol3 = st.columns(3)
        with acol1:
            alpha_name = st.selectbox(
                "α coefficient",
                ["xx", "xz", "zx", "zz"],
                key="alpha_scan_name",
            )
            alpha_min = st.slider(
                "α min",
                -5.0,
                0.0,
                -2.0,
                0.1,
                key="alpha_scan_min",
            )
            alpha_max = st.slider(
                "α max",
                0.0,
                5.0,
                2.0,
                0.1,
                key="alpha_scan_max",
            )
            n_points = st.slider(
                "Number of points",
                5,
                101,
                21,
                1,
                key="alpha_scan_npts",
            )
        with acol2:
            T_H_alpha = st.slider(
                "T_H",
                0.1,
                10.0,
                1.0,
                0.1,
                key="alpha_scan_th",
            )
            theta_true_alpha = st.slider(
                "θ_true",
                0.1,
                5.0,
                1.0,
                0.1,
                key="alpha_scan_theta",
            )
        with acol3:
            sql_ref_alpha = 1.0 / T_H_alpha if T_H_alpha > 0 else float("inf")
            st.metric("SQL Δθ = 1/T_H", f"{sql_ref_alpha:.4f}")

        if st.button("▶ Run Scan", key="alpha_scan_btn"):
            ops_alpha = build_two_qubit_operators()
            M_op_alpha = build_joint_operator(ops_alpha)

            with st.spinner("Scanning α coefficient (S-only)..."):
                result_sonly = scan_alpha_single_parameter(
                    alpha_name=alpha_name,
                    alpha_min=alpha_min,
                    alpha_max=alpha_max,
                    n_points=n_points,
                    T_H=T_H_alpha,
                    theta_true=theta_true_alpha,
                    meas_op=None,
                )

            with st.spinner("Scanning α coefficient (Joint)..."):
                result_joint = scan_alpha_single_parameter(
                    alpha_name=alpha_name,
                    alpha_min=alpha_min,
                    alpha_max=alpha_max,
                    n_points=n_points,
                    T_H=T_H_alpha,
                    theta_true=theta_true_alpha,
                    meas_op=M_op_alpha,
                )

            # Plot two traces + SQL reference
            fig_alpha = go.Figure()
            fig_alpha.add_trace(
                go.Scatter(
                    x=result_sonly.alpha_values,
                    y=result_sonly.delta_theta_values,
                    mode="lines+markers",
                    name="S-only (J_z^S)",
                    line={"color": "firebrick", "width": 2},
                ),
            )
            fig_alpha.add_trace(
                go.Scatter(
                    x=result_joint.alpha_values,
                    y=result_joint.delta_theta_values,
                    mode="lines+markers",
                    name="Joint M = J_z^S + J_z^A",
                    line={"color": "royalblue", "width": 2},
                ),
            )
            fig_alpha.add_hline(
                y=sql_ref_alpha,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"SQL = {sql_ref_alpha:.4f}",
            )
            fig_alpha.update_layout(
                title=f"Sensitivity vs α_{{{alpha_name}}}",
                xaxis_title=f"α_{alpha_name}",
                yaxis_title="Δθ",
                yaxis_type="log",
                height=450,
                legend={"yanchor": "top", "y": 0.95, "xanchor": "left", "x": 0.05},
            )
            st.plotly_chart(fig_alpha, use_container_width=True)

            # Best values summary
            best_sonly = float(np.min(result_sonly.delta_theta_values))
            best_joint = float(np.min(result_joint.delta_theta_values))
            bcol1, bcol2, bcol3 = st.columns(3)
            bcol1.metric("Best Δθ (S-only)", f"{best_sonly:.4f}")
            bcol2.metric("Best Δθ (Joint)", f"{best_joint:.4f}")
            bcol3.metric("SQL ref", f"{sql_ref_alpha:.4f}")

    else:
        # ── 4D random search ─────────────────────────────────────────────
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            n_samples = st.slider(
                "Number of samples",
                50,
                2000,
                200,
                10,
                key="alpha_rnd_nsamples",
            )
            T_H_rnd = st.slider(
                "T_H",
                0.1,
                10.0,
                1.0,
                0.1,
                key="alpha_rnd_th",
            )
            theta_true_rnd = st.slider(
                "θ_true",
                0.1,
                5.0,
                1.0,
                0.1,
                key="alpha_rnd_theta",
            )

        if st.button("▶ Run Random Search", key="alpha_rnd_btn"):
            ops_rnd = build_two_qubit_operators()
            M_op_rnd = build_joint_operator(ops_rnd)

            with st.spinner(f"Random search over {n_samples} samples (S-only)..."):
                rnd_sonly = random_search_alpha(
                    n_samples=n_samples,
                    T_H=T_H_rnd,
                    theta_true=theta_true_rnd,
                    meas_op=None,
                    seed=int(seed),
                )

            with st.spinner(f"Random search over {n_samples} samples (Joint)..."):
                rnd_joint = random_search_alpha(
                    n_samples=n_samples,
                    T_H=T_H_rnd,
                    theta_true=theta_true_rnd,
                    meas_op=M_op_rnd,
                    seed=int(seed),
                )

            sql_ref_rnd = 1.0 / T_H_rnd if T_H_rnd > 0 else float("inf")

            scol1, scol2, scol3 = st.columns(3)
            scol1.metric(
                "Best Δθ (S-only)",
                f"{rnd_sonly.best_delta_theta:.4f}",
                delta=f"SQL: {sql_ref_rnd:.4f}",
            )
            scol2.metric(
                "Best Δθ (Joint)",
                f"{rnd_joint.best_delta_theta:.4f}",
                delta=f"SQL: {sql_ref_rnd:.4f}",
            )
            scol3.metric("SQL ref", f"{sql_ref_rnd:.4f}")

            # Show best α vectors
            with st.expander("Best α configurations", expanded=False):
                scc1, scc2 = st.columns(2)
                with scc1:
                    st.markdown("**S-only best α**")
                    st.code(
                        f"α_xx = {rnd_sonly.best_alpha[0]:.4f}\n"
                        f"α_xz = {rnd_sonly.best_alpha[1]:.4f}\n"
                        f"α_zx = {rnd_sonly.best_alpha[2]:.4f}\n"
                        f"α_zz = {rnd_sonly.best_alpha[3]:.4f}",
                    )
                with scc2:
                    st.markdown("**Joint best α**")
                    st.code(
                        f"α_xx = {rnd_joint.best_alpha[0]:.4f}\n"
                        f"α_xz = {rnd_joint.best_alpha[1]:.4f}\n"
                        f"α_zx = {rnd_joint.best_alpha[2]:.4f}\n"
                        f"α_zz = {rnd_joint.best_alpha[3]:.4f}",
                    )

            # Histogram
            fig_rnd = go.Figure()
            fig_rnd.add_trace(
                go.Histogram(
                    x=rnd_sonly.delta_theta_values,
                    name="S-only",
                    opacity=0.7,
                    nbinsx=30,
                ),
            )
            fig_rnd.add_trace(
                go.Histogram(
                    x=rnd_joint.delta_theta_values,
                    name="Joint",
                    opacity=0.7,
                    nbinsx=30,
                ),
            )
            fig_rnd.add_vline(
                x=sql_ref_rnd,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"SQL = {sql_ref_rnd:.4f}",
            )
            fig_rnd.update_layout(
                title="Δθ Distribution (4D Random Search)",
                xaxis_title="Δθ",
                yaxis_title="Count",
                height=400,
                barmode="overlay",
                legend={"yanchor": "top", "y": 0.95, "xanchor": "left", "x": 0.05},
            )
            st.plotly_chart(fig_rnd, use_container_width=True)

with tab3:
    st.subheader("α-Scan with State Re-optimisation")

    rcol1, rcol2, rcol3 = st.columns(3)
    with rcol1:
        alpha_name_reopt = st.selectbox(
            "α coefficient",
            ["xx", "xz", "zx", "zz"],
            key="alpha_reopt_name",
        )
        n_points_reopt = st.slider(
            "Number of α points",
            5,
            51,
            21,
            1,
            key="alpha_reopt_npts",
        )
    with rcol2:
        n_restarts_reopt = st.slider(
            "Restarts per α",
            1,
            20,
            5,
            1,
            key="alpha_reopt_restarts",
        )
        maxiter_reopt = st.slider(
            "Nelder–Mead max iterations",
            50,
            2000,
            500,
            50,
            key="alpha_reopt_maxiter",
        )
    with rcol3:
        theta_true_reopt = st.slider(
            "θ_true",
            0.1,
            5.0,
            1.0,
            0.1,
            key="alpha_reopt_theta",
        )
        st.caption(
            "Re-optimises 7 state parameters (θ_S, φ_S, θ_A, φ_A, "
            "T_BS1, T_BS2, T_H) at each α value.",
        )

    if st.button("▶ Run Re-optimisation Scan", key="alpha_reopt_btn"):
        with st.spinner(
            f"Re-optimising over {n_points_reopt} α values × "
            f"{n_restarts_reopt} restarts = "
            f"{n_points_reopt * n_restarts_reopt * 2} runs...",
        ):
            reopt_result = scan_alpha_with_reoptimisation(
                alpha_name=alpha_name_reopt,
                alpha_values=np.linspace(-2.0, 2.0, n_points_reopt),
                theta_true=theta_true_reopt,
                n_restarts=n_restarts_reopt,
                maxiter=maxiter_reopt,
                seed=int(seed),
            )

        st.success("Re-optimisation scan complete!")

        # Comparison table
        st.subheader("📊 Joint vs S-only Comparison")
        table_reopt: dict[str, list[str]] = {
            "α": [],
            "Δθ Joint": [],
            "Δθ S-only": [],
            "Joint < S-only?": [],
        }
        for i in range(len(reopt_result.alpha_values)):
            j = reopt_result.delta_theta_joint[i]
            s = reopt_result.delta_theta_sonly[i]
            table_reopt["α"].append(f"{reopt_result.alpha_values[i]:.3f}")
            table_reopt["Δθ Joint"].append(
                f"{j:.4f}" if np.isfinite(j) else "∞",
            )
            table_reopt["Δθ S-only"].append(
                f"{s:.4f}" if np.isfinite(s) else "∞",
            )
            if np.isfinite(j) and np.isfinite(s):
                table_reopt["Joint < S-only?"].append(
                    "✅" if j < s - 1e-8 else "≈ tie" if abs(j - s) < 1e-8 else "❌"
                )
            else:
                table_reopt["Joint < S-only?"].append("—")

        st.table(table_reopt)

        # Sensitivity curves
        fig_reopt = go.Figure()
        fig_reopt.add_trace(
            go.Scatter(
                x=reopt_result.alpha_values,
                y=reopt_result.delta_theta_joint,
                mode="lines+markers",
                name="Joint M = J_z^S + J_z^A",
                line={"color": "royalblue", "width": 2},
            ),
        )
        fig_reopt.add_trace(
            go.Scatter(
                x=reopt_result.alpha_values,
                y=reopt_result.delta_theta_sonly,
                mode="lines+markers",
                name="S-only (J_z^S)",
                line={"color": "firebrick", "width": 2},
            ),
        )
        fig_reopt.update_layout(
            title=f"Sensitivity vs α_{{{alpha_name_reopt}}} (Re-optimised)",
            xaxis_title=f"α_{alpha_name_reopt}",
            yaxis_title="Δθ",
            yaxis_type="log",
            height=450,
            legend={"yanchor": "top", "y": 0.95, "xanchor": "left", "x": 0.05},
        )
        st.plotly_chart(fig_reopt, use_container_width=True)
