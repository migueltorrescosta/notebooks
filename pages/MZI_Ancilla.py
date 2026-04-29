"""
Mach-Zehnder Interferometer with Ancilla Simulation Page

This page simulates a Mach-Zehnder interferometer with an ancilla system that
can be entangled with the interferometer for enhanced measurements.

Features:
- Selectable input states (vacuum, single photon, coherent, Fock, NOON)
- Variable beam splitter ratio
- Controllable system-ancilla coupling strength
- Visualization of interference fringes, state evolution, and density matrix
"""

import numpy as np
import pandas as pd
import streamlit as st
from plotly import graph_objects as go

from src.physics.mzi_simulation import (
    prepare_input_state,
    evolve_mzi,
    get_reduced_density_matrix,
    compute_output_probabilities,
    compute_interference_fringe,
    compute_all_stage_states,
)
from src.visualization.plotting import plot_array


# Page configuration
st.set_page_config(page_title="MZI with Ancilla", page_icon="🔬", layout="wide")


st.header("Mach-Zehnder Interferometer with Ancilla", divider="blue")


# =============================================================================
# Methodology
# =============================================================================
with st.expander("📖 Methodology", expanded=False):
    st.markdown(r"""
    **Physical Model:**
    
    A Mach-Zehnder interferometer (MZI) splits an input beam into two arms via a beam splitter,
    applies a phase shift to one arm, then recombines the beams at a second beam splitter.
    
    **With Ancilla:**
    
    An additional quantum system (ancilla) is coupled to the interferometer to create entanglement,
    which can enhance phase sensitivity beyond the standard quantum limit.
    
    **Circuit:**
    ```
    Input -> BS1 -> Phase(φ) -> [Interaction with Ancilla] -> BS2 -> Output
    ```
    
    **Key Parameters:**
    - $\theta$: Beam splitter angle (θ = π/4 for 50/50)
    - $\phi$: Phase shift in one arm
    - $g$: System-ancilla coupling strength
    - $t_{int}$: Interaction time
    
    **Input States Available:**
    - |0,0⟩: Vacuum (both modes empty)
    - |1,0⟩ or |0,1⟩: Single photon in one mode
    - |α⟩: Coherent state (Gaussian)
    - |n⟩: Fock state with n photons
    - |NOON⟩: (|N,0⟩ + |0,N⟩)/√2 for quantum enhanced metrology
    """)


# =============================================================================
# Sidebar Controls
# =============================================================================
with st.sidebar:
    st.header("Setup", divider="gray")

    # Input state selection
    st.subheader("Input State")
    state_type = st.selectbox(
        "State type",
        ["vacuum", "single_photon", "coherent", "fock", "noon"],
        format_func=lambda x: {
            "vacuum": "Vacuum |0,0⟩",
            "single_photon": "Single Photon |1,0⟩",
            "coherent": "Coherent State |α⟩",
            "fock": "Fock State |n⟩",
            "noon": "NOON State |N,0⟩+|0,N⟩",
        }[x],
    )

    # State-specific parameters
    max_photons = st.number_input("Max photons (basis size)", min_value=1, value=3)

    n_particles = 1
    alpha = 1.0 + 0.0j
    mode = 0

    if state_type == "single_photon":
        mode = st.radio("Photon mode", [0, 1], format_func=lambda x: f"Mode {x}")
    elif state_type == "fock":
        n_particles = st.number_input(
            "n (photon number)", min_value=1, value=1, max_value=max_photons
        )
    elif state_type == "coherent":
        alpha_real = st.number_input("Re(α)", value=0.5)
        alpha_imag = st.number_input("Im(α)", value=0.0)
        alpha = complex(alpha_real, alpha_imag)
    elif state_type == "noon":
        n_particles = st.number_input(
            "N (NOON state order)", min_value=1, value=2, max_value=max_photons
        )

    st.divider()

    # Interferometer parameters
    st.subheader("Interferometer")
    theta = st.slider(
        "Beam splitter angle θ",
        min_value=0.0,
        max_value=np.pi / 2,
        value=np.pi / 4,
        help="θ = π/4 gives 50/50 beam splitter",
    )
    phi_bs = st.number_input("Beam splitter phase φ_bs", value=0.0)
    phi_phase = st.slider("Phase shift φ", 0.0, 2 * np.pi, 0.0, step=0.01)

    st.divider()

    # Ancilla parameters
    st.subheader("Ancilla (Entangled Probe)")
    ancilla_dim = st.number_input("Ancilla dimension", min_value=2, value=3)
    g = st.slider("Coupling strength g", 0.0, 2.0, 0.0, step=0.01)
    interaction_time = st.number_input("Interaction time", 0.0, 10.0, 0.1, step=0.01)
    coupling_type = st.selectbox(
        "Coupling type",
        ["phase_coupling", "flip_flop"],
        format_func=lambda x: {
            "phase_coupling": "Phase coupling (n·J_z)",
            "flip_flop": "Flip-flop (a+a^dagger)·J_x",
        }[x],
    )

    st.divider()

    # Visualization options
    st.subheader("Visualization", divider="gray")
    show_fringe = st.toggle("Show interference fringe", value=True)
    show_evolution = st.toggle("Show state evolution", value=True)
    show_density = st.toggle("Show density matrix", value=True)


# =============================================================================
# Prepare Input State
# =============================================================================
# Adjust max_photons if needed for NOON/Fock states
effective_max = max(max_photons, n_particles)
initial_state = prepare_input_state(
    state_type=state_type,
    max_photons=effective_max,
    n_particles=n_particles,
    alpha=alpha,
    mode=mode,
)


# =============================================================================
# Main Display
# =============================================================================

# State information
st.subheader(f"Input: {state_type.replace('_', ' ').title()}")
st.latex(
    "|\\psi_{in}\\rangle = "
    + {
        "vacuum": "|0,0\\rangle",
        "single_photon": f"|1,{mode}\\rangle",
        "coherent": f"|\\alpha={alpha:.2f}\\rangle",
        "fock": f"|{n_particles},0\\rangle",
        "noon": f"\\frac{{|{n_particles},0\\rangle + |0,{n_particles}\\rangle}}{{\\sqrt{{2}}}}",
    }[state_type]
)


# =============================================================================
# Compute Evolution and Outputs
# =============================================================================

# Evolve the state
evolved_state = evolve_mzi(
    initial_system_state=initial_state,
    theta=theta,
    phi_bs=phi_bs,
    phi_phase=phi_phase,
    g=g,
    interaction_time=interaction_time,
    coupling_type=coupling_type,
    max_photons=effective_max,
    ancilla_dim=ancilla_dim,
)

# Output probabilities
p0, p1 = compute_output_probabilities(evolved_state, effective_max, ancilla_dim)


# =============================================================================
# Interference Fringe
# =============================================================================
if show_fringe:
    st.header("Interference Fringe", divider="orange")

    # Compute fringe over phase range
    phases = np.linspace(0, 2 * np.pi, 100)
    fringe = compute_interference_fringe(
        phase_range=phases,
        initial_system_state=initial_state,
        theta=theta,
        phi_bs=phi_bs,
        g=g,
        interaction_time=interaction_time,
        coupling_type=coupling_type,
        max_photons=effective_max,
        ancilla_dim=ancilla_dim,
    )

    # Plot
    fig_fringe = go.Figure()
    fig_fringe.add_trace(
        go.Scatter(
            x=phases,
            y=fringe,
            mode="lines",
            name="P(out0)",
            line=dict(color="blue", width=2),
        )
    )
    fig_fringe.add_trace(
        go.Scatter(
            x=[phi_phase, phi_phase],
            y=[0, 1],
            mode="lines",
            name="Current φ",
            line=dict(color="red", width=2, dash="dash"),
        )
    )
    fig_fringe.update_layout(
        xaxis_title="Phase φ (rad)",
        yaxis_title="P(out0)",
        yaxis_range=[0, 1],
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig_fringe, use_container_width=True)

    # Current point info
    st.metric("Current Output Probabilities", f"P₀ = {p0:.3f}, P₁ = {p1:.3f}")


# =============================================================================
# State Evolution Visualization
# =============================================================================
if show_evolution:
    st.header("State Evolution", divider="green")

    # Compute states at each stage
    stages = compute_all_stage_states(
        initial_system_state=initial_state,
        theta=theta,
        phi_bs=phi_bs,
        phi_phase=phi_phase,
        g=g,
        interaction_time=interaction_time,
        coupling_type=coupling_type,
        max_photons=effective_max,
        ancilla_dim=ancilla_dim,
    )

    # Display stage information
    stage_names = [
        ("initial", "Initial |ψ⟩ ⊗ |0⟩"),
        ("after_bs1", "After BS1"),
        ("after_phase", "After Phase Shift"),
        ("after_interaction", "After Ancilla Interaction"),
        ("final", "Final (After BS2)"),
    ]

    # Show probabilities at each stage
    stage_probs = []
    for key, label in stage_names:
        state = stages[key]
        p0, p1 = compute_output_probabilities(state, effective_max, ancilla_dim)
        stage_probs.append({"Stage": label, "P(out0)": p0, "P(out1)": p1})

    df_stages = pd.DataFrame(stage_probs)
    st.table(df_stages)

    # Bar chart of probabilities
    fig_ev = go.Figure(
        data=[
            go.Bar(
                name="P(out0)",
                x=[s[1] for s in stage_names],
                y=[p["P(out0)"] for p in stage_probs],
            ),
            go.Bar(
                name="P(out1)",
                x=[s[1] for s in stage_names],
                y=[p["P(out1)"] for p in stage_probs],
            ),
        ]
    )
    fig_ev.update_layout(barmode="stack", template="plotly_white", height=300)
    st.plotly_chart(fig_ev, use_container_width=True)


# =============================================================================
# Density Matrix Visualization
# =============================================================================
if show_density:
    st.header("Density Matrix", divider="violet")

    # Get reduced system density matrix
    rho_sys = get_reduced_density_matrix(
        evolved_state, effective_max, ancilla_dim, trace_out_ancilla=True
    )

    # Plot using existing plot_array function
    st.subheader("Real Part")
    plot_array(np.real(rho_sys), midpoint=0, text_auto=False)

    st.subheader("Imaginary Part")
    plot_array(np.imag(rho_sys), midpoint=0, text_auto=False)

    # Purity
    purity = np.real(np.trace(rho_sys @ rho_sys))
    st.metric("State Purity", f"{purity:.4f}")


# =============================================================================
# Summary Info
# =============================================================================
st.header("Summary", divider="red")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Input State", state_type.replace("_", " ").title())
with col2:
    st.metric("N Particles", n_particles)
with col3:
    st.metric("Ancilla Dim", ancilla_dim)
with col4:
    st.metric("Coupling g", f"{g:.2f}")

st.latex(rf"""
\begin{{array}}{{lcl}}
\theta & = & {theta:.4f} \\
\phi_{{bs}} & = & {phi_bs:.4f} \\
\phi & = & {phi_phase:.4f} \\
g & = & {g:.4f} \\
t_{{int}} & = & {interaction_time:.4f}
\end{{array}}
""")

st.divider()
st.caption(
    "MZI with Ancilla Simulation | System dimension: "
    + f"{((effective_max + 1) ** 2) * ancilla_dim} (system ⊗ ancilla)"
)
