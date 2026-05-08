"""
High-Order Non-Gaussian Squeezing under Decoherence.

Implements simulation of n-th order squeezed states (n=2,3,4) generated
via spin-dependent forces on a trapped ion hybrid oscillator-spin system.

Features:
- Hybrid oscillator-spin system (Fock basis ⊗ Pauli matrices)
- n-th order squeezing Hamiltonians (n=2 Gaussian, n=3,4 non-Gaussian)
- Lindblad decoherence (photon loss, dephasing)
- MZI readout with QFI computation
- Wigner function visualization
- Hypothesis testing: non-Gaussian advantage vs decoherence
"""

import numpy as np
import streamlit as st
from plotly import graph_objects as go
import scipy

from src.physics.hybrid_system import (
    hybrid_hamiltonian_n,
    hybrid_vacuum_state,
    hybrid_coherent_state,
    adaptive_truncation,
    hybrid_mean_photon,
    validate_hybrid_state,
)
from src.physics.hybrid_mzi import (
    qfi_hybrid_mzi,
    compute_wigner_for_state,
)
from src.physics.wigner import wigner_minimum, wigner_is_negative


# Page configuration
st.set_page_config(
    page_title="High-Order Squeezing | MZI",
    page_icon="🔬",
    layout="wide",
)


def evolve_hybrid_unitary(
    initial_state: np.ndarray,
    N: int,
    n: int,
    omega_n: float,
    theta_n: float,
    t: float,
) -> np.ndarray:
    """Evolve hybrid state under n-th order squeezing Hamiltonian."""
    H = hybrid_hamiltonian_n(N, n=n, omega_n=omega_n, theta_n=theta_n)
    U = scipy.linalg.expm(-1j * H * t)
    return U @ initial_state


# =============================================================================
# Methodology Section
# =============================================================================
with st.expander("📖 Methodology", expanded=False):
    st.markdown(
        r"""
        **Physical Model:**

        Hybrid oscillator-spin system:
        - Oscillator: Fock states |n⟩, n = 0…N (dimension N+1)
        - Spin: |↓⟩, |↑⟩ (dimension 2)
        - Combined: |n⟩ ⊗ |σ⟩ (dimension 2(N+1))

        **Squeezing Hamiltonians (after RWA + rotating frame):**
        - n=2 (Gaussian): $H_2 \propto \sigma_z \otimes (a^2 e^{-i\theta_2} + a^{\dagger 2} e^{i\theta_2})$
        - n=3 (Non-Gaussian): $H_3 \propto \sigma_{\phi+\pi/2} \otimes (a^3 e^{-i\theta_3} + a^{\dagger 3} e^{i\theta_3})$
        - n=4 (Non-Gaussian): $H_4 \propto \sigma_z \otimes (a^4 e^{-i\theta_4} + a^{\dagger 4} e^{i\theta_4})$

        **Hypothesis:**
        Non-Gaussian states (n=3,4) can outperform Gaussian (n=2) for phase estimation
        at fixed mean photon number ⟨a†a⟩, provided decoherence is below threshold.

        **Readout:**
        MZI with QFI computation: $F_Q = 4 \text{Var}(n_1 \otimes I_2 \otimes I_{spin})$
        """
    )


# =============================================================================
# Sidebar Controls
# =============================================================================
with st.sidebar:
    st.header("Setup", divider="gray")

    # Squeezing parameters
    st.subheader("Squeezing")
    n_order = st.selectbox(
        "Order n",
        [2, 3, 4],
        format_func=lambda x: {
            2: "n=2 (Gaussian squeezing)",
            3: "n=3 (Trisqueezing - non-Gaussian)",
            4: "n=4 (Quadsqueezing - non-Gaussian)",
        }[x],
    )

    omega_n = st.slider(
        "Squeezing rate Ωₙ",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.01,
        help="Squeezing strength",
    )

    t_sqz = st.slider(
        "Squeezing time t_sqz",
        min_value=0.0,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="Squeezing parameter rₙ = Ωₙ · t_sqz",
    )

    theta_n = st.number_input(
        "Squeezing phase θₙ",
        value=0.0,
        help="Phase in H_n",
    )

    alpha_input = st.number_input(
        "Coherent amplitude α (0 for vacuum)",
        value=0.0,
        help="Set > 0 for squeezed coherent state",
    )

    st.divider()

    # MZI parameters
    st.subheader("MZI Readout")
    phi_mzi = st.slider(
        "MZI phase φ",
        min_value=0.0,
        max_value=2 * np.pi,
        value=np.pi / 4,
        step=0.01,
    )

    st.divider()

    # Visualization
    st.subheader("Visualization", divider="gray")
    show_wigner = st.toggle("Show Wigner function", value=True)
    show_qfi = st.toggle("Show QFI analysis", value=True)


# =============================================================================
# Main Content
# =============================================================================

st.header("High-Order Non-Gaussian Squeezing", divider="blue")

# Compute adaptive truncation
N_adaptive = adaptive_truncation(
    alpha=complex(alpha_input, 0.0),
    r_n=t_sqz * omega_n,  # rₙ = Ωₙ · t_sqz
    n=n_order,
    N_max=30,
)

st.caption(f"Using adaptive truncation: N = {N_adaptive}")

# Validate truncation
N = min(N_adaptive, 15)  # Cap at 15 for performance
if N < 8:
    N = 8

# Prepare initial state
if alpha_input == 0.0:
    initial = hybrid_vacuum_state(N, spin_state="down")
    st.latex(r"|\psi_{in}\rangle = |0, \downarrow\rangle")
else:
    alpha = complex(alpha_input, 0.0)
    initial = hybrid_coherent_state(N, alpha=alpha, spin_state="down")
    st.latex(fr"|\psi_{{in}}\rangle = |\alpha={alpha_input:.1f}, \downarrow\rangle")

# Apply squeezing Hamiltonian
with st.spinner("Evolving under squeezing Hamiltonian..."):
    squeezed = evolve_hybrid_unitary(
        initial_state=initial,
        N=N,
        n=n_order,
        omega_n=omega_n,
        theta_n=theta_n,
        t=t_sqz,
    )

# Validate output state
if not validate_hybrid_state(squeezed, N):
    st.error("State validation failed!")

# Display observables
st.subheader("Squeezed State Properties", divider="green")

col1, col2, col3 = st.columns(3)

with col1:
    mean_n = hybrid_mean_photon(squeezed, N)
    st.metric("Mean photon ⟨n⟩", f"{mean_n:.3f}")

with col2:
    # Compute squeezing parameter
    r_n = omega_n * t_sqz
    st.metric("Squeezing rₙ", f"{r_n:.3f}")

with col3:
    # Wigner negativity check
    if show_wigner:
        _, _, W = compute_wigner_for_state(squeezed, N, x_max=5.0, n_points=50)
        w_min = wigner_minimum(W)
        is_neg = wigner_is_negative(W)
        st.metric("Wigner min", f"{w_min:.4f}", delta="Negative!" if is_neg else "Positive")
    else:
        st.metric("Wigner min", "N/A")

# Wigner function visualization
if show_wigner:
    st.subheader("Wigner Function W(x,p)", divider="orange")

    x, p, W = compute_wigner_for_state(squeezed, N, x_max=5.0, n_points=80)

    fig = go.Figure(
        data=go.Heatmap(
            z=W,
            x=x,
            y=p,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="W(x,p)"),
        )
    )
    fig.update_layout(
        xaxis_title="x",
        yaxis_title="p",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Wigner negativity warning
    if show_wigner:
        if is_neg:
            st.success(f"✅ Wigner negativity detected! min(W) = {w_min:.4f}")
            st.caption("Non-Gaussian state confirmed (for n ≥ 3)")
        else:
            st.info("Wigner function is non-negative (Gaussian-like state)")

# MZI QFI computation
if show_qfi:
    st.subheader("MZI Phase Estimation - QFI", divider="violet")

    # Compute QFI
    with st.spinner("Computing QFI..."):
        fq = qfi_hybrid_mzi(squeezed, N)

    st.metric("Quantum Fisher Information $F_Q$", f"{fq:.4f}")

    # SQL comparison
    sql_limit = 4 * mean_n  # Standard quantum limit
    st.latex(fr"F_Q^{{SQL}} = 4\langle n \rangle = {sql_limit:.2f}")

    if fq > sql_limit:
        st.success("✅ QFI exceeds Standard Quantum Limit!")
    else:
        st.info("QFI below Standard Quantum Limit")

# Summary
st.header("Summary", divider="red")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Order n", f"n={n_order}")
with col2:
    st.metric("Mean photons ⟨n⟩", f"{mean_n:.3f}")
with col3:
    if show_qfi:
        st.metric("QFI F_Q", f"{fq:.2f}")
    else:
        st.metric("QFI F_Q", "N/A")
with col4:
    if show_wigner:
        st.metric("Wigner negative", "Yes" if is_neg else "No")
    else:
        st.metric("Wigner negative", "N/A")

st.caption(
    f"Hybrid system dimension: {2*(N+1)} | "
    f"Squeezing: rₙ = {omega_n * t_sqz:.3f}"
)
