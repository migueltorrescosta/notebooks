"""
Streamlit page: Ancilla-Drive-Enhanced Metrology.

Implements the driven-ancilla metrology protocol from
``reports/20260518/Ancilla-Drive-Enhanced-Metrology.md``:
- System-only 50/50 BS, unknown θ on system
- Ancilla drive H_A = a_x J_x^A + a_y J_y^A + a_z J_z^A
- Ising interaction H_int = a_zz J_z^S ⊗ J_z^A
- Error propagation sensitivity via J_z^S measurement

References:
- ``reports/20260518/Ancilla-Drive-Enhanced-Metrology.md``
"""

from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.analysis.ancilla_drive_metrology import (
    build_drive_hold_hamiltonian,
    compute_drive_decoupled_baseline,
    compute_drive_sensitivity,
    drive_2d_slice,
    drive_random_search,
    evolve_drive_circuit,
    run_drive_nelder_mead,
    run_drive_theta_scan,
    system_only_bs_unitary,
)
from src.analysis.ancilla_optimization import (
    build_two_qubit_operators,
    compute_expectation_and_variance,
    validate_bs_unitarity,
    validate_operators,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ancilla Drive Metrology",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Ancilla-Drive-Enhanced Metrology")
st.markdown(
    r"""
    **Testing whether an actively driven ancilla can beat the standard quantum limit
    $\Delta\theta_{\text{SQL}} = 1/T_H$ using only a single particle in the interferometer.**

    The circuit: BS$_S$ → Hold($\theta, H_A, H_{\text{int}}$) → BS$_S$ → Measure $J_z^S$.
    """,
)

# ── Validate core infrastructure ─────────────────────────────────────────────
with st.expander("🔧 Validation Checks", expanded=False):
    vcol1, vcol2, vcol3 = st.columns(3)
    ops = build_two_qubit_operators()
    ops_ok = validate_operators(ops)
    bs_ok = validate_bs_unitarity()

    # System-only BS unitarity
    U_bs_sys = system_only_bs_unitary(np.pi / 2)
    I_4 = np.eye(4, dtype=complex)
    bs_sys_ok = bool(np.allclose(U_bs_sys @ U_bs_sys.conj().T, I_4, atol=1e-12))

    vcol1.success("✅ Operators Hermitian & SU(2)") if ops_ok else vcol1.error("❌")
    vcol2.success("✅ BS unitary") if bs_ok else vcol2.error("❌")
    vcol3.success("✅ System-only BS unitary") if bs_sys_ok else vcol3.error("❌")

    # Decoupled baseline check
    baseline = compute_drive_decoupled_baseline()
    ratio = baseline.delta_theta / baseline.sql
    st.markdown(
        rf"**Decoupled baseline**: $\Delta\theta = {baseline.delta_theta:.6f}$, "
        rf"SQL $= {baseline.sql:.6f}$, ratio $= {ratio:.6f}$ "
        + ("✅" if abs(ratio - 1.0) < 0.05 else "❌"),
    )

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Circuit Parameters")

    theta = st.number_input(
        r"$\theta$ (phase rate)",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.1,
        format="%.2f",
        help="Unknown phase rate parameter being estimated.",
    )
    T_H = st.number_input(
        r"$T_H$ (holding time)",
        min_value=0.1,
        max_value=50.0,
        value=10.0,
        step=1.0,
        format="%.1f",
        help="Holding time: SQL reference Δθ_SQL = 1/T_H.",
    )
    T_BS = st.number_input(
        r"$T_{\mathrm{BS}}$ (BS duration)",
        min_value=0.0,
        max_value=np.pi,
        value=float(np.pi / 2.0),
        step=0.01,
        format="%.3f",
        help="Beam-splitter duration: π/2 for 50/50.",
    )

    st.header("Ancilla Drive & Interaction")
    a_x = st.slider(
        r"$a_x$ (J_x^A drive)",
        -5.0,
        5.0,
        0.0,
        step=0.1,
        help="Coefficient for J_x^A (non-commuting drive component).",
    )
    a_y = st.slider(
        r"$a_y$ (J_y^A drive)",
        -5.0,
        5.0,
        0.0,
        step=0.1,
        help="Coefficient for J_y^A (non-commuting drive component).",
    )
    a_z = st.slider(
        r"$a_z$ (J_z^A drive)",
        -5.0,
        5.0,
        0.0,
        step=0.1,
        help="Coefficient for J_z^A (commuting drive component).",
    )
    a_zz = st.slider(
        r"$a_{zz}$ (Ising interaction)",
        -5.0,
        5.0,
        0.0,
        step=0.1,
        help="Coefficient for J_z^S ⊗ J_z^A interaction.",
    )

# ── Main panel: tabs ─────────────────────────────────────────────────────────
tab_single, tab_2d, tab_search, tab_theta, tab_about = st.tabs(
    [
        "🎯 Single Point",
        "📊 2D Slice",
        "🎲 Random Search",
        "📈 θ Scan",
        "📖 About",
    ]
)

# ── Tab 1: Single-point evaluation ───────────────────────────────────────────
with tab_single:
    st.subheader("Single-Point Sensitivity")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            rf"""
            **Current configuration:**
            - $\theta = {theta:.2f}$
            - $a_x = {a_x:.2f}$, $a_y = {a_y:.2f}$, $a_z = {a_z:.2f}$
            - $a_{{zz}} = {a_zz:.2f}$
            - $T_H = {T_H:.1f}$, SQL $= 1/T_H = {1.0 / T_H:.4f}$
            """,
        )

    if st.button("🚀 Compute Sensitivity", type="primary", key="btn_single"):
        with st.spinner("Computing..."):
            t0 = time.time()
            ops_local = build_two_qubit_operators()
            dtheta = compute_drive_sensitivity(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
                T_BS,
                T_H,
                theta,
                a_x,
                a_y,
                a_z,
                a_zz,
                ops_local,
            )
            t_elapsed = time.time() - t0

            # Compute diagnostics
            psi_final = evolve_drive_circuit(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
                T_BS,
                T_H,
                theta,
                a_x,
                a_y,
                a_z,
                a_zz,
                ops_local,
            )
            exp_val, var_val = compute_expectation_and_variance(
                psi_final, ops_local["Jz_S"]
            )

        sql = 1.0 / T_H
        ratio = dtheta / sql if np.isfinite(dtheta) else float("inf")

        col2.metric(
            r"$\Delta\theta$",
            f"{dtheta:.6f}" if np.isfinite(dtheta) else "∞ (fringe extremum)",
            delta=f"vs SQL: {ratio:.4f}x" if np.isfinite(dtheta) else "",
        )

        st.markdown(
            rf"""
            | Quantity | Value |
            |----------|-------|
            | $\Delta\theta$ | {dtheta:.6f} |
            | SQL $= 1/T_H$ | {sql:.6f} |
            | Ratio $\Delta\theta / \text{{SQL}}$ | {ratio:.4f} |
            | $\langle J_z^S \rangle$ | {exp_val:.6f} |
            | $\text{{Var}}(J_z^S)$ | {var_val:.10f} |
            | Computation time | {t_elapsed:.3f}s |
            """,
        )

        if np.isfinite(dtheta) and dtheta < sql:
            st.success(f"🎉 **Below SQL!** Δθ = {dtheta:.6f} < SQL = {sql:.6f}")
        elif np.isfinite(dtheta):
            st.info(f"Δθ = {dtheta:.6f} ≥ SQL = {sql:.6f}")

    # Display the hold Hamiltonian
    with st.expander("🔬 Hold Hamiltonian", expanded=False):
        ops_h = build_two_qubit_operators()
        H_full = build_drive_hold_hamiltonian(theta, a_x, a_y, a_z, a_zz, ops_h)
        st.markdown(r"$H = \theta J_z^S + H_A + H_{\mathrm{int}}$")
        st.latex(
            rf"H = {theta:.2f}\,J_z^S + ({a_x:.2f})\,J_x^A + ({a_y:.2f})\,J_y^A + ({a_z:.2f})\,J_z^A + ({a_zz:.2f})\,J_z^S \otimes J_z^A"
        )
        st.write("Matrix representation (4×4):")
        st.dataframe(np.real_if_close(H_full))

# ── Tab 2: 2D Slice Scan ─────────────────────────────────────────────────────
with tab_2d:
    st.subheader("2D Parameter Slice Scan")
    st.markdown(
        r"""
        Scan over $(a_{\text{drive}}, a_{zz})$ at fixed $\theta$ to identify
        promising parameter regions. The red contour marks the SQL bound
        $\Delta\theta = 1/T_H$. Points below this contour beat the SQL.
        """,
    )

    col_slice, col_slice_params = st.columns([1, 2])

    with col_slice_params:
        slice_type = st.radio(
            "Drive axis",
            ["a_x", "a_y"],
            horizontal=True,
            help="Which drive coefficient to scan.",
        )
        slice_theta = st.number_input(
            r"$\theta$ for slice",
            0.1,
            10.0,
            1.0,
            step=0.5,
        )
        n_grid = st.select_slider(
            "Grid points per axis",
            options=[5, 7, 11, 15, 21, 51, 101, 201],
            value=11,
        )

    with col_slice:
        if st.button("🚀 Run 2D Slice", type="primary", key="btn_2d"):
            with st.spinner("Computing 2D slice..."):
                t0 = time.time()
                result = drive_2d_slice(
                    theta=slice_theta,
                    slice_type="ax" if slice_type == "a_x" else "ay",
                    n_drive=n_grid,
                    n_azz=n_grid,
                    T_H=T_H,
                    T_BS=T_BS,
                )
                t_elapsed = time.time() - t0

            st.success(f"Computed {n_grid}×{n_grid} grid in {t_elapsed:.2f}s")

            # Find best point in the grid
            flat_idx = np.argmin(result.delta_theta_grid)
            best_i, best_j = np.unravel_index(flat_idx, result.delta_theta_grid.shape)
            best_dt = result.delta_theta_grid[best_i, best_j]
            best_drive = result.drive_values[best_i]
            best_azz = result.azz_values[best_j]

            st.markdown(
                rf"""
                **Best point**: ${slice_type}^* = {best_drive:.3f}$, $a_{{zz}}^* = {best_azz:.3f}$,
                $\Delta\theta = {best_dt:.6f}$ (SQL $= {1.0 / T_H:.4f}$)
                """,
            )

            # Plotly heatmap
            fig = go.Figure(
                data=go.Heatmap(
                    x=result.azz_values,
                    y=result.drive_values,
                    z=result.delta_theta_grid,
                    colorscale="Viridis",
                    zmin=0.0,
                    zmax=min(3.0 * result.sql, np.nanmax(result.delta_theta_grid)),
                    colorbar={"title": r"$\Delta\theta$"},
                ),
            )
            # SQL contour (approximate via scatter of near-SQL points)
            sql_mask = np.abs(result.delta_theta_grid - result.sql) < 0.02
            if np.any(sql_mask):
                sql_x, sql_y = np.meshgrid(result.azz_values, result.drive_values)
                fig.add_trace(
                    go.Scatter(
                        x=sql_x.flatten()[sql_mask.flatten()],
                        y=sql_y.flatten()[sql_mask.flatten()],
                        mode="markers",
                        marker={"color": "red", "symbol": "cross", "size": 4},
                        name=f"SQL ≈ {result.sql:.3f}",
                    ),
                )
            fig.update_layout(
                xaxis_title=r"$a_{zz}$",
                yaxis_title=rf"${slice_type}$",
                title=rf"$\Delta\theta$ at $\theta={slice_theta:.1f}$",
                width=600,
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Random Search ─────────────────────────────────────────────────────
with tab_search:
    st.subheader("4D Random Search over $(a_x, a_y, a_z, a_{zz})$")
    st.markdown(
        r"""
        Random sampling in $[-5, 5]^4$ at fixed $\theta$ to find candidates
        that beat the SQL. Follow up with Nelder--Mead refinement.
        """,
    )

    col_rs, col_rs_params = st.columns([1, 2])

    with col_rs_params:
        rs_theta = st.number_input(
            r"$\theta$ for search",
            0.1,
            10.0,
            1.0,
            step=0.5,
            key="rs_theta",
        )
        rs_n = st.number_input(
            "Samples",
            50,
            2000,
            500,
            step=50,
            key="rs_n",
        )

    with col_rs:
        if st.button("🎲 Run Random Search", type="primary", key="btn_rs"):
            with st.spinner(f"Running {rs_n} random evaluations..."):
                t0 = time.time()
                rs_result = drive_random_search(
                    theta=rs_theta,
                    n_samples=rs_n,
                    seed=42,
                    T_H=T_H,
                    T_BS=T_BS,
                )
                t_elapsed = time.time() - t0

            sql = 1.0 / T_H
            best = rs_result.best_delta_theta
            n_below_sql = int(np.sum(rs_result.delta_theta_values < sql))

            st.success(
                f"Evaluated {rs_n} points in {t_elapsed:.2f}s. "
                f"Best Δθ = {best:.6f} ({best / sql:.3f}× SQL). "
                f"{n_below_sql} / {rs_n} below SQL.",
            )

            st.markdown(
                rf"""
                **Best parameters**:
                $a_x^* = {rs_result.best_params[0]:.3f}$,
                $a_y^* = {rs_result.best_params[1]:.3f}$,
                $a_z^* = {rs_result.best_params[2]:.3f}$,
                $a_{{zz}}^* = {rs_result.best_params[3]:.3f}$
                """,
            )

            # Nelder-Mead refinement button
            if st.button("🔧 Refine Best", key="btn_refine"):
                with st.spinner("Running Nelder--Mead refinement..."):
                    nm_result = run_drive_nelder_mead(
                        theta_true=rs_theta,
                        x0=np.array(rs_result.best_params),
                        T_H=T_H,
                        T_BS=T_BS,
                        maxiter=2000,
                    )
                st.success(
                    f"Nelder--Mead: Δθ = {nm_result.delta_theta_opt:.6f} "
                    f"({nm_result.delta_theta_opt / sql:.3f}× SQL). "
                    f"nfev = {nm_result.nfev}.",
                )
                st.markdown(
                    rf"""
                    **Refined parameters**:
                    $a_x^* = {nm_result.params_opt[0]:.3f}$,
                    $a_y^* = {nm_result.params_opt[1]:.3f}$,
                    $a_z^* = {nm_result.params_opt[2]:.3f}$,
                    $a_{{zz}}^* = {nm_result.params_opt[3]:.3f}$
                    """,
                )

            # Histogram
            finite = rs_result.delta_theta_values[
                np.isfinite(rs_result.delta_theta_values)
            ]
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=finite,
                    nbinsx=40,
                    name="Δθ distribution",
                    marker_color="royalblue",
                    opacity=0.7,
                )
            )
            fig.add_vline(
                x=sql,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"SQL = {sql:.4f}",
            )
            fig.add_vline(
                x=best,
                line_dash="dot",
                line_color="red",
                annotation_text=f"Best = {best:.4f}",
            )
            fig.update_layout(
                xaxis_title=r"$\Delta\theta$",
                yaxis_title="Count",
                title=f"Random search: {len(finite)} finite points",
                width=700,
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 4: θ Scan ────────────────────────────────────────────────────────────
with tab_theta:
    st.subheader("θ Scan with Nelder--Mead Refinement")
    st.markdown(
        r"""
        For each $\theta$ value, run a 4D random search followed by
        Nelder--Mead refinement from the best points. Determine whether
        the optimal $\Delta\theta$ can beat the SQL for any $\theta$.
        """,
    )

    col_ts, col_ts_params = st.columns([1, 2])

    with col_ts_params:
        theta_vals_input = st.text_input(
            r"$\theta$ values (comma-separated)",
            "0.1, 0.5, 1.0, 2.0, 5.0",
        )
        ts_n_random = st.number_input(
            "Random samples per θ",
            50,
            1000,
            200,
            step=50,
            key="ts_nr",
        )
        ts_n_refine = st.number_input(
            "Nelder--Mead refinements per θ",
            2,
            100,
            10,
            step=5,
            key="ts_nm",
        )

    with col_ts:
        if st.button("📈 Run θ Scan", type="primary", key="btn_ts"):
            try:
                theta_list = [float(x.strip()) for x in theta_vals_input.split(",")]
            except ValueError:
                st.error("Invalid θ values. Use comma-separated numbers.")
                st.stop()

            with st.spinner(
                f"Scanning {len(theta_list)} θ values (may take a while)..."
            ):
                t0 = time.time()
                ts_result = run_drive_theta_scan(
                    theta_values=theta_list,
                    n_random=ts_n_random,
                    n_nm_refine=ts_n_refine,
                    seed=42,
                    maxiter=2000,
                    T_H=T_H,
                    T_BS=T_BS,
                )
                t_elapsed = time.time() - t0

            sql = 1.0 / T_H
            st.success(f"Completed in {t_elapsed:.1f}s")

            # Results table
            st.markdown("#### Optimal parameters per θ")
            rows = []
            for i, th in enumerate(ts_result.theta_values):
                params = ts_result.best_params_per_theta[i]
                dt = ts_result.best_delta_theta_per_theta[i]
                ratio = dt / sql if np.isfinite(dt) else float("inf")
                rows.append(
                    {
                        r"$\theta$": f"{th:.1f}",
                        r"$a_x^*$": f"{params[0]:.3f}",
                        r"$a_y^*$": f"{params[1]:.3f}",
                        r"$a_z^*$": f"{params[2]:.3f}",
                        r"$a_{zz}^*$": f"{params[3]:.3f}",
                        r"$\Delta\theta$": f"{dt:.6f}" if np.isfinite(dt) else "∞",
                        r"Δθ/SQL": f"{ratio:.4f}",
                        "Below SQL": "✅" if (np.isfinite(dt) and dt < sql) else "❌",
                    },
                )
            st.table(rows)

            # Plot with Plotly
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=ts_result.theta_values,
                    y=ts_result.best_delta_theta_per_theta,
                    mode="lines+markers",
                    name=r"$\Delta\theta$",
                    marker={"size": 10, "color": "royalblue"},
                    line={"width": 2},
                )
            )
            fig.add_hline(
                y=sql,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"SQL = {sql:.4f}",
            )
            fig.update_layout(
                xaxis_title=r"$\theta$",
                yaxis_title=r"$\Delta\theta$",
                title=r"θ-scan: best $\Delta\theta$",
                width=700,
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 5: About ─────────────────────────────────────────────────────────────
with tab_about:
    st.markdown(
        r"""
        ## About This Simulation

        This page implements the protocol described in
        `reports/20260518/Ancilla-Drive-Enhanced-Metrology.md`.

        **Circuit**:
        $$
        |\Psi_{\text{final}}\rangle = U_{\text{BS}}^{(S)} \,
        e^{-i T_H (\theta J_z^S + H_A + H_{\text{int}})} \,
        U_{\text{BS}}^{(S)} \, |00\rangle
        $$

        where $U_{\text{BS}}^{(S)} = e^{-i (\pi/2) J_x^S} \otimes \mathbb{1}_2$ is
        a 50/50 beam splitter on the system only.

        **Hypothesis**: An actively driven ancilla ($H_A \neq 0$ with at least
        one non-commuting component $a_x \neq 0$ or $a_y \neq 0$) can generate
        $\Delta\theta < 1/T_H$ (beating the SQL) even with only $N=1$ particle
        in the interferometer.

        **Key predictions**:
        1. SQL violation requires $H_A \neq 0$, $a_{zz} \neq 0$, and $\theta \neq 0$.
        2. Non-commuting drive ($a_x \neq 0$ or $a_y \neq 0$) is essential.
        3. Purely commuting drive ($a_x = a_y = 0$, $a_z \neq 0$) provides no benefit.

        **References**:
        - Report `reports/20260512/2026-05-12-Ancilla-Assisted-Metrology-Optimization` (passive ancilla)
        - Report `reports/20260515/2026-05-15-Ancilla-Assisted-Metrology-Joint-Measurement`
        - Giovannetti, Lloyd, Maccone, *Nat. Photonics* **5**, 222 (2011)

        **Numerical methods**:
        - Exact $4 \times 4$ matrix exponentiation via `scipy.linalg.expm`
        - Error propagation sensitivity via central finite differences
        - 4D Nelder--Mead optimisation via `scipy.optimize.minimize`
        """,
    )

    with st.expander("📋 Numerical Invariants"):
        st.markdown(
            r"""
            The following physical invariants are verified throughout:
            - **State normalisation**: $\||\Psi\rangle\| = 1$ to machine precision
            - **Unitarity**: $U^\dagger U = \mathbb{1}$ for BS and hold unitaries
            - **Hermiticity**: $H^\dagger = H$ for all Hamiltonians
            - **Variance positivity**: $\text{Var}(J_z^S) \geq 0$
            - **SQL baseline recovery**: $\Delta\theta = 1/T_H$ exactly at
              $(a_x, a_y, a_z, a_{zz}) = (0, 0, 0, 0)$
            """,
        )
