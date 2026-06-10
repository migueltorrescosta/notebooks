"""
Streamlit page: Ancilla-Assisted vs. Two-Particle Probe Comparison.

Compares two configurations at fixed total resource of 2 particles:
- Configuration A: 1 system particle + 1 ancilla spin-½
- Configuration B: 2 system particles, no ancilla

Shows Quantum Fisher Information and phase sensitivity ratio.
"""

from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.analysis.ancilla_comparison import (
    analytical_fq_A_zero,
    analytical_fq_B_max,
    run_comparison,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ancilla vs. System Comparison",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️ Ancilla-Assisted vs. Two-Particle Probe")
st.markdown(
    r"""
    **Fixed resource: 2 total particles.** Which configuration yields better
    phase sensitivity?
    """,
)

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Simulation Parameters")

    T_hold = st.slider(
        "Holding-time strength T_hold",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Controls how strongly the phase and interaction act. QFI scales as T_hold².",
    )

    n_samples_B = st.slider(
        "Random samples (Case B)",
        min_value=500,
        max_value=20000,
        value=5000,
        step=500,
        help="Number of random pure states for two-particle optimisation.",
    )

    n_samples_A = st.slider(
        "Random samples (Case A)",
        min_value=1000,
        max_value=20000,
        value=5000,
        step=500,
        help="Number of random states × α vectors for ancilla optimisation.",
    )

    n_alpha_samples = st.slider(
        "α vectors to test",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Number of random interaction coefficient vectors.",
    )

    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=42,
        help="For reproducible results.",
    )

    run_button = st.button("▶ Run Comparison", type="primary", use_container_width=True)

# ── Main content ─────────────────────────────────────────────────────────────

# Show analytical predictions
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Case B (2 particles, no ancilla)",
        f"F_Q = {analytical_fq_B_max(T_hold):.0f}",
        f"Δω = 1/√{analytical_fq_B_max(T_hold):.0f}",
    )

with col2:
    st.metric(
        "Case A (1 particle + ancilla, α = 0)",
        f"F_Q = {analytical_fq_A_zero(T_hold):.0f}",
        "Δω = 1",
    )

with col3:
    ratio_theory = np.sqrt(analytical_fq_B_max(T_hold) / analytical_fq_A_zero(T_hold))
    st.metric(
        "Ratio ℛ = Δω_A / Δω_B",
        f"{ratio_theory:.2f}",
        "Ancilla advantage if ℛ < 1",
    )

st.info(
    r"""
    **Theoretical prediction:** The J = 1 (2-particle) system has generator
    eigenvalues ±1, giving F_Q = 4. The J = ½ (1-particle) system has
    generator eigenvalues ±½, giving F_Q = 1 — even with non-commuting
    ancilla interactions. Thus ℛ = √(4/1) = 2, and the ancilla-assisted
    configuration **cannot outperform** the two-particle probe.
    """,
)

# ── Results table ────────────────────────────────────────────────────────────
if run_button:
    with st.spinner("Running comparison simulation..."):
        start = time.time()
        result = run_comparison(
            T_hold=T_hold,
            n_samples_B=n_samples_B,
            n_samples_A=n_samples_A,
            n_alpha_samples=n_alpha_samples,
            pure_only=True,
            seed=int(seed),
        )
        elapsed = time.time() - start

    st.success(f"Comparison completed in {elapsed:.1f}s")

    # Summary metrics
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("Case B F_Q (numerical)", f"{result.fq_B_max:.4f}")
    with mcol2:
        st.metric(
            "Case B F_Q (theory)",
            f"{analytical_fq_B_max(T_hold):.4f}",
            delta=f"{(result.fq_B_max / analytical_fq_B_max(T_hold) - 1) * 100:.2f}%",
        )
    with mcol3:
        st.metric(
            "Case A F_Q (numerical)",
            f"{result.fq_A_max:.4f}",
        )
    with mcol4:
        st.metric(
            "Ratio ℛ",
            f"{result.ratio:.4f}",
            delta=f"{'≥ 2 ✓' if result.ratio >= 2.0 else '< 2 — possible ancilla advantage'}",
            delta_color="off" if result.ratio >= 2.0 else "inverse",
        )

    # Detailed results
    with st.expander("Detailed Results", expanded=False):
        st.markdown("#### Case B (2-particle system)")
        bcol1, bcol2, bcol3 = st.columns(3)
        bcol1.metric("Max QFI", f"{result.fq_B_max:.6f}")
        bcol2.metric("⟨N⟩ (best state)", f"{result.mean_N_B:.6f}")
        bcol3.metric(
            "Pop. |N_max,N_max⟩",
            f"{result.pop_NN_B:.2e}",
        )

        st.markdown("#### Case A (ancilla-assisted)")
        acol1, acol2, acol3, acol4 = st.columns(4)
        acol1.metric("Max QFI (penalised)", f"{result.fq_A_max:.6f}")
        acol2.metric("Baseline (α = 0)", f"{result.fq_A_zero:.6f}")
        acol3.metric("⟨N⟩ (best state)", f"{result.mean_N_A:.6f}")
        acol4.metric("Pop. |0,0⟩", f"{result.pop_00_A:.2e}")

        if result.best_alphas_A is not None:
            st.markdown("**Best interaction coefficients:**")
            ac = result.best_alphas_A
            st.latex(
                rf"\alpha_{{xx}}={ac[0]:.4f},\; "
                rf"\alpha_{{xz}}={ac[1]:.4f},\; "
                rf"\alpha_{{zx}}={ac[2]:.4f},\; "
                rf"\alpha_{{zz}}={ac[3]:.4f}",
            )

        if result.fq_A_omega:
            st.markdown("**ω-dependence of QFI (Case A):**")
            omega_df = {
                "ω": list(result.fq_A_omega.keys()),
                "F_Q": [f"{v:.6f}" for v in result.fq_A_omega.values()],
            }
            st.table(omega_df)

    # Distribution plot
    st.subheader("QFI Distribution from Random Search")

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=result.fq_B_all,
            name="Case B (2 particles)",
            opacity=0.7,
            nbinsx=50,
            marker_color="royalblue",
        ),
    )
    fig.add_trace(
        go.Histogram(
            x=result.fq_A_all,
            name="Case A (ancilla)",
            opacity=0.7,
            nbinsx=50,
            marker_color="firebrick",
        ),
    )

    fig.update_layout(
        title="QFI Distribution",
        xaxis_title="QFI F_Q",
        yaxis_title="Count",
        barmode="overlay",
        height=400,
        legend={"yanchor": "top", "y": 0.95, "xanchor": "right", "x": 0.95},
    )

    st.plotly_chart(fig, use_container_width=True)

    # Conclusion
    st.subheader("Conclusion")
    if result.ratio >= 2.0:
        st.success(
            rf"""
            **ℛ = {result.ratio:.3f} ≥ 2** — The ancilla-assisted configuration
            **cannot beat** the two-particle system at fixed resource of 2
            particles. The 2-particle probe (J = 1, generator eigenvalue range
            ±1) achieves up to **F_Q = {analytical_fq_B_max(T_hold):.0f}**, while
            the ancilla-assisted probe (J = ½, generator eigenvalue range ±½)
            is bounded by **F_Q = {analytical_fq_A_zero(T_hold):.0f}**.
            """,
        )
    else:
        st.warning(
            f"ℛ = {result.ratio:.3f} < 2 — The ancilla-assisted configuration "
            "may offer an advantage in this regime.",
        )

else:
    st.info("👈 Configure parameters in the sidebar and click **▶ Run Comparison**")
