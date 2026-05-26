"""
Unified Scaling Survey Page for Interferometry Sensitivity.

This page lets users run the scaling survey interactively to map
combinations of (input state, noise model) to their scaling exponent α
in Δφ ∝ N^α.

Features:
- Select models (input states) to include
- Select noise channels to activate
- Configure N range and operating phase
- Display table of fitted scaling exponents
- Show log-log plots of Δφ vs N
- Show heatmap/phase diagram of α across (state, noise) grid
- CSV/JSON export of survey results

This page calls into src/analysis/scaling_survey.py and does NOT contain
any physics logic itself.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import pandas as pd
import streamlit as st
from plotly import graph_objects as go

# Define stub classes for mypy static analysis. These match the interface
# of the dynamically-imported dataclasses from local.py. The TYPE_CHECKING
# block is only seen by type checkers (mypy, pyright). At runtime, the
# names are overwritten by the actual dataclasses from importlib.
if TYPE_CHECKING:

    class ModelConfig:
        """Stub matching the dataclass in reports/20260511/local.py."""

        model_id: str
        state_type: str
        noise_type: str
        entangler: str
        label: str
        custom_sensitivity_fn: Callable[[int, float], float] | None

        def __init__(
            self,
            model_id: str,
            state_type: str = "",
            noise_type: str = "none",
            entangler: str = "none",
            label: str = "",
            custom_sensitivity_fn: Callable[[int, float], float] | None = None,
        ) -> None: ...

    class SurveyConfig:
        """Stub matching the dataclass in reports/20260511/local.py."""

        N_range: tuple[int, int]
        N_points: int
        noise_levels: list[float]
        phi: float
        method: str
        seed: int

        def __init__(
            self,
            N_range: tuple[int, int] = (2, 64),
            N_points: int = 8,
            noise_levels: list[float] | None = None,
            phi: float = np.pi / 4,
            method: str = "qfi",
            seed: int = 42,
        ) -> None: ...


# Load exclusive functions from reports/20260511/local.py via importlib.
# Try multiple resolution strategies to handle both normal execution
# and AppTest (which copies the script to a temp directory).
_local_candidates = [
    Path(__file__).resolve().parent.parent / "reports" / "20260511" / "local.py",
    Path(sys.path[0]) / "reports" / "20260511" / "local.py",
    Path.cwd() / "reports" / "20260511" / "local.py",
]
_local_path = None
for _candidate in _local_candidates:
    if _candidate.exists():
        _local_path = _candidate
        break
if _local_path is None:
    raise ImportError(
        "Cannot find reports/20260511/local.py. "
        "Run 'uv run python reports/20260511/local.py --force' from the project root."
    )
_spec = importlib.util.spec_from_file_location(
    "report_scaling_survey_local", str(_local_path)
)
if _spec is None or _spec.loader is None:
    raise ImportError(
        f"Cannot load reports/20260511/local.py at {_local_path}. "
        "Run 'uv run python reports/20260511/local.py --force' first."
    )
_report_local = importlib.util.module_from_spec(_spec)
sys.modules["report_scaling_survey_local"] = _report_local
_spec.loader.exec_module(_report_local)

ModelConfig = _report_local.ModelConfig  # type: ignore[misc]
SurveyConfig = _report_local.SurveyConfig  # type: ignore[misc]
create_default_survey = _report_local.create_default_survey
fit_all_exponents = _report_local.fit_all_exponents
run_scaling_survey = _report_local.run_scaling_survey
CavityMziConfig = _report_local.CavityMziConfig
cavity_enhanced_sensitivity = _report_local.cavity_enhanced_sensitivity
DistributedMziConfig = _report_local.DistributedMziConfig
distributed_mzi_sensitivity = _report_local.distributed_mzi_sensitivity
dd_phase_sensitivity = _report_local.dd_phase_sensitivity
combined_sensitivity = _report_local.combined_sensitivity
create_thermal_config = _report_local.create_thermal_config
TTLNoiseConfig = _report_local.TTLNoiseConfig
ttl_limited_sensitivity = _report_local.ttl_limited_sensitivity

# Page configuration
st.set_page_config(
    page_title="Metro | Interferometry Scaling Survey",
    page_icon="📊",
    layout="wide",
)


# =============================================================================
# UI Helper Functions
# =============================================================================


def model_selector() -> list[ModelConfig]:
    """Let user select which models to include in the survey.

    Returns:
        List of selected ModelConfig objects.

    """
    st.subheader("Models (Input States)")

    all_models = create_default_survey()
    model_options = {m.label: m for m in all_models}

    # Add advanced scaling models (from this report) as custom-sensitivity plugins
    # Thermal model: uses noise_level as thermal_strength S
    _thermal_model = ModelConfig(
        model_id="thermal_sql",
        state_type="",
        noise_type="custom",
        label="Thermal + SQL crossover",
        custom_sensitivity_fn=lambda N, nl: combined_sensitivity(
            N,
            create_thermal_config(thermal_strength=nl),
        ),
    )
    model_options[_thermal_model.label] = _thermal_model

    # Cavity-enhanced MZI
    _cavity_config = CavityMziConfig(F=10.0)
    _cavity_model = ModelConfig(
        model_id="cavity_enhanced",
        state_type="",
        noise_type="custom",
        label="Cavity-enhanced MZI (F=10)",
        custom_sensitivity_fn=lambda N, _nl: cavity_enhanced_sensitivity(
            N,
            np.pi / 4,
            _cavity_config,
        ),
    )
    model_options[_cavity_model.label] = _cavity_model

    # Dynamical decoupling (CPMG, 8 pulses)
    _dd_model = ModelConfig(
        model_id="dd_cpmg",
        state_type="",
        noise_type="custom",
        label="Dynamical decoupling (CPMG, n=8)",
        custom_sensitivity_fn=lambda N, _nl: dd_phase_sensitivity(
            N,
            0.0,
            T=1.0,
            n_pulses=8,
        ),
    )
    model_options[_dd_model.label] = _dd_model

    # Distributed array (classical, M=4)
    _dist_config = DistributedMziConfig(M=4, entangled=False)
    _dist_model = ModelConfig(
        model_id="distributed_classical",
        state_type="",
        noise_type="custom",
        label="Distributed array (M=4, classical)",
        custom_sensitivity_fn=lambda N, _nl: distributed_mzi_sensitivity(
            N,
            0.0,
            _dist_config,
        )["delta_phi"],
    )
    model_options[_dist_model.label] = _dist_model

    # Distributed array (entangled, M=4)
    _dist_ent_config = DistributedMziConfig(M=4, entangled=True)
    _dist_ent_model = ModelConfig(
        model_id="distributed_entangled",
        state_type="",
        noise_type="custom",
        label="Distributed array (M=4, entangled)",
        custom_sensitivity_fn=lambda N, _nl: distributed_mzi_sensitivity(
            N,
            0.0,
            _dist_ent_config,
        )["delta_phi"],
    )
    model_options[_dist_ent_model.label] = _dist_ent_model

    # Tilt-to-length noise
    _ttl_config = TTLNoiseConfig()
    _ttl_model = ModelConfig(
        model_id="ttl_noise",
        state_type="",
        noise_type="custom",
        label="Tilt-to-length noise",
        custom_sensitivity_fn=lambda N, _nl: ttl_limited_sensitivity(
            N,
            1.0 / np.sqrt(N),
            _ttl_config,
        ),
    )
    model_options[_ttl_model.label] = _ttl_model

    # Multi-select
    selected_labels = st.multiselect(
        "Select states to analyze:",
        options=list(model_options.keys()),
        default=list(model_options.keys())[:4],  # First 4 by default
        help="Choose which quantum states to include in the scaling survey",
    )

    selected_models = [model_options[label] for label in selected_labels]

    # Display selected
    if selected_models:
        st.caption(f"Selected: {', '.join(m.label for m in selected_models)}")

    return selected_models


def noise_selector() -> tuple[str, list[float]]:
    """Let user select noise type and levels.

    Returns:
        Tuple of (noise_type, noise_levels).

    """
    st.subheader("Noise Channels")

    noise_types = {
        "None (ideal)": "none",
        "Phase diffusion (dephasing)": "dephasing",
        "One-body loss": "loss",
        "Two-body loss": "two_body",
        "Detection inefficiency": "detection",
    }

    selected_label = st.selectbox(
        "Noise type:",
        options=list(noise_types.keys()),
        index=1,  # Default to dephasing
        help="Choose the decoherence mechanism",
    )

    noise_type = noise_types[selected_label]

    # Noise levels
    if noise_type == "none":
        noise_levels = [0.0]
        st.caption("Ideal (noiseless) — noise level fixed at 0")
    elif noise_type == "detection":
        # For detection, noise_level = efficiency (inverted)
        # eta = 1 means perfect detection, eta = 0 means no detection
        st.markdown("**Detection efficiency η** (1.0 = perfect, 0.5 = 50% loss):")
        col1, col2, col3 = st.columns(3)
        with col1:
            eta1 = st.number_input(
                "η₁",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
            )
        with col2:
            eta2 = st.number_input(
                "η₂",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.1,
            )
        with col3:
            eta3 = st.number_input(
                "η₃",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
            )
        noise_levels = [eta for eta in [eta1, eta2, eta3] if eta > 0]
        noise_levels = sorted(set(noise_levels))  # Unique and sorted
    else:
        # Dimensionless rates for other noise types
        st.markdown("**Noise levels (γ · T, dimensionless):**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            nl1 = st.number_input(
                "γ₁",
                min_value=0.0,
                value=0.0,
                step=0.001,
                format="%.4f",
            )
        with col2:
            nl2 = st.number_input(
                "γ₂",
                min_value=0.0,
                value=0.01,
                step=0.01,
                format="%.4f",
            )
        with col3:
            nl3 = st.number_input(
                "γ₃",
                min_value=0.0,
                value=0.1,
                step=0.1,
                format="%.4f",
            )
        with col4:
            nl4 = st.number_input(
                "γ₄",
                min_value=0.0,
                value=0.5,
                step=0.1,
                format="%.4f",
            )
        noise_levels = [nl for nl in [nl1, nl2, nl3, nl4] if nl >= 0]
        noise_levels = sorted(set(noise_levels))  # Unique and sorted

    return noise_type, noise_levels


def n_range_selector() -> tuple[int, int, int]:
    """Let user select N range.

    Returns:
        Tuple of (N_min, N_max, N_points).

    """
    st.subheader("N Sweep Configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        N_min = st.number_input("N min", min_value=2, value=4, step=2)
    with col2:
        N_max = st.number_input("N max", min_value=N_min + 2, value=32, step=4)
    with col3:
        N_points = st.slider("N points", min_value=3, max_value=12, value=6)

    return int(N_min), int(N_max), int(N_points)


def phase_selector() -> float:
    """Let user select operating phase.

    Returns:
        Operating phase φ in radians.

    """
    st.subheader("Operating Phase")

    phase_deg = st.slider(
        "Phase φ",
        min_value=0,
        max_value=180,
        value=45,
        help="Operating phase for sensitivity estimation (π/4 = 45° is optimal for most states)",
    )

    phase_rad = np.radians(phase_deg)
    st.caption(f"φ = {phase_deg}° = {phase_rad:.3f} radians")

    return float(phase_rad)


# =============================================================================
# Survey Execution
# =============================================================================


def run_full_survey(
    models: list[ModelConfig],
    noise_type: str,
    noise_levels: list[float],
    N_min: int,
    N_max: int,
    N_points: int,
    phi: float,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the scaling survey with given parameters.

    Uses the unified ``run_scaling_survey`` pipeline for all models.
    Models with ``custom_sensitivity_fn`` are handled directly by the
    pipeline; standard MZI models get their noise type updated.

    Args:
        models: List of models to survey.
        noise_type: Type of noise to apply (applied only to models without
            ``custom_sensitivity_fn``).
        noise_levels: List of noise levels to sweep.
        N_min: Minimum N for sweep.
        N_max: Maximum N for sweep.
        N_points: Number of log-spaced N values.
        phi: Operating phase.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (raw_results_df, fitted_exponents_df).

    """
    # Build the final model list:
    # - Custom models (with custom_sensitivity_fn) are kept as-is.
    # - Standard MZI models get their noise_type updated.
    survey_models = []
    for m in models:
        if m.custom_sensitivity_fn is not None:
            survey_models.append(m)
        else:
            survey_models.append(
                ModelConfig(
                    model_id=m.model_id,
                    state_type=m.state_type,
                    noise_type=noise_type,
                    entangler=m.entangler,
                    label=m.label,
                ),
            )

    if not survey_models:
        return pd.DataFrame(), pd.DataFrame()

    # Create survey config
    survey_config = SurveyConfig(
        N_range=(N_min, N_max),
        N_points=N_points,
        noise_levels=noise_levels,
        phi=phi,
        method="qfi",
        seed=seed,
    )

    # Run unified survey (handles both standard and custom models)
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(curr: int, total: int) -> None:
        progress_bar.progress(curr / total)
        status_text.text(f"Computing: {curr}/{total}")

    combined_df = run_scaling_survey(survey_models, survey_config, progress_callback)
    progress_bar.empty()
    status_text.empty()

    if combined_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Fit exponents
    fit_df = fit_all_exponents(combined_df, min_N=max(4, N_min))

    return combined_df, fit_df


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_scaling_curves(
    raw_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    N_min: int,
    N_max: int,
) -> None:
    """Plot log-log Δφ vs N for selected models.

    Args:
        raw_df: Raw survey results.
        fit_df: Fitted exponents.
        N_min: Minimum N for reference lines.
        N_max: Maximum N for reference lines.

    """
    st.subheader("Δφ vs N (Log-Log)")

    if raw_df.empty:
        st.info("No data to plot")
        return

    fig = go.Figure()

    # Reference lines
    N_ref = np.array([N_min, N_max])
    delta_sql = 1.0 / np.sqrt(N_ref)
    delta_hl = 1.0 / N_ref

    fig.add_trace(
        go.Scatter(
            x=N_ref,
            y=delta_sql,
            mode="lines",
            name="SQL (1/√N, α=-0.5)",
            line={"dash": "dash", "color": "gray"},
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=N_ref,
            y=delta_hl,
            mode="lines",
            name="HL (1/N, α=-1.0)",
            line={"dash": "dot", "color": "gray"},
        ),
    )

    # Color map for models
    unique_models = raw_df["label"].unique()
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    color_map = dict(zip(unique_models, colors, strict=False))

    # Plot data
    for (label, noise_level), group in raw_df.groupby(["label", "noise_level"]):
        group_sorted = group.sort_values("N")

        # Get alpha for this combination if available
        alpha = "N/A"
        if not fit_df.empty:
            mask = (fit_df["label"] == label) & (fit_df["noise_level"] == noise_level)
            if mask.any():
                alpha_val = fit_df.loc[mask, "alpha"].iloc[0]
                alpha = f"{alpha_val:.3f}"

        fig.add_trace(
            go.Scatter(
                x=group_sorted["N"],
                y=group_sorted["delta_phi"],
                mode="lines+markers",
                name=f"{label} (γ={noise_level:.3g}, α={alpha})",
                line={"color": color_map.get(label, "blue")},
                marker={"size": 6},
            ),
        )

    fig.update_layout(
        xaxis_type="log",
        yaxis_type="log",
        xaxis_title="N (particle number)",
        yaxis_title="Δφ (phase uncertainty)",
        template="plotly_white",
        height=400,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_alpha_heatmap(fit_df: pd.DataFrame) -> None:
    """Plot heatmap of α across (state, noise) grid.

    Args:
        fit_df: Fitted exponents DataFrame.

    """
    st.subheader("Phase Diagram: α by (State, Noise)")

    if fit_df.empty:
        st.info("No fit data to display")
        return

    # Pivot the data for heatmap
    pivot = fit_df.pivot_table(
        index="label",
        columns="noise_level",
        values="alpha",
        aggfunc="first",  # Take first if duplicates
    )

    if pivot.empty or pivot.isna().all().all():
        st.info("Insufficient data for heatmap")
        return

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.astype(str),  # Convert to strings for display
            y=pivot.index,
            colorscale="RdYlBu_r",  # Red = good (negative), Blue = poor (close to 0)
            zmid=-0.75,  # Midpoint between -0.5 and -1.0
            colorbar={"title": "α exponent"},
            hoverongaps=False,
        ),
    )

    fig.update_layout(
        xaxis_title="Noise Level (γ)",
        yaxis_title="Quantum State",
        template="plotly_white",
        height=400,
    )

    # Add annotations with α values
    for _i, row_idx in enumerate(pivot.index):
        for _j, col_idx in enumerate(pivot.columns):
            val = pivot.loc[row_idx, col_idx]
            if not pd.isna(val):
                val_numeric = (
                    float(val)
                    if isinstance(val, (int, float, np.floating, np.integer))
                    else 0.0
                )
                fig.add_annotation(
                    x=str(col_idx),
                    y=row_idx,
                    text=f"{val_numeric:.2f}",
                    showarrow=False,
                    font={"color": "white" if abs(val_numeric) > 0.5 else "black"},
                )

    st.plotly_chart(fig, use_container_width=True)


def display_fit_table(fit_df: pd.DataFrame) -> None:
    """Display table of fitted exponents.

    Args:
        fit_df: Fitted exponents DataFrame.

    """
    st.subheader("Fitted Scaling Exponents")

    if fit_df.empty:
        st.info("No fit data to display")
        return

    # Select and format key columns
    display_cols = [
        "label",
        "noise_level",
        "alpha",
        "alpha_err",
        "C",
        "R_squared",
        "n_points",
        "valid",
    ]
    available_cols = [c for c in display_cols if c in fit_df.columns]

    display_df = fit_df[available_cols].copy()

    # Rename for display
    rename_map = {
        "label": "State",
        "noise_level": "Noise (γ)",
        "alpha": "α",
        "alpha_err": "α_err",
        "C": "Prefactor C",
        "R_squared": "R²",
        "n_points": "Points",
        "valid": "Valid",
    }
    display_df = display_df.rename(
        columns={k: v for k, v in rename_map.items() if k in display_df.columns},
    )

    # Format numeric columns
    st.dataframe(
        display_df.style.format(
            {
                "α": "{:.3f}",
                "α_err": "{:.4f}",
                "Prefactor C": "{:.3f}",
                "R²": "{:.3f}",
                "Noise (γ)": "{:.4f}",
            },
            na_rep="N/A",
        ),
        use_container_width=True,
    )


def export_controls(raw_df: pd.DataFrame, fit_df: pd.DataFrame) -> None:
    """Display export controls.

    Args:
        raw_df: Raw survey results.
        fit_df: Fitted exponents.

    """
    st.subheader("Export Results")

    if raw_df.empty and fit_df.empty:
        st.info("No data to export")
        return

    col1, col2 = st.columns(2)

    with col1:
        if not raw_df.empty:
            # Parquet export for raw data
            buffer = io.BytesIO()
            raw_df.to_parquet(buffer, index=False)

            st.download_button(
                label="📥 Download Raw Data (Parquet)",
                data=buffer.getvalue(),
                file_name="scaling_survey_raw.parquet",
                mime="application/octet-stream",
                help="Download raw (N, Δφ) values for all survey points",
            )

    with col2:
        if not fit_df.empty:
            # Parquet export for fit results
            buffer = io.BytesIO()
            fit_df.to_parquet(buffer, index=False)

            st.download_button(
                label="📥 Download Fit Results (Parquet)",
                data=buffer.getvalue(),
                file_name="scaling_survey_fit.parquet",
                mime="application/octet-stream",
                help="Download fitted exponents (α, C, R²) for each (state, noise) combination",
            )


# =============================================================================
# Main Page Layout
# =============================================================================


def main() -> None:
    """Main page layout and execution."""
    st.title("📊 Interferometry Sensitivity Scaling Survey")

    st.markdown(r"""
    This interactive survey maps combinations of **(quantum state, noise model)**
    to their scaling exponent $\alpha$ in $\Delta\phi \propto N^\alpha$.

    Use the sidebar to configure your survey, then click **Run Survey** to see:
    - Which combinations achieve **Heisenberg scaling** ($\alpha = -1.0$)
    - Which are limited to the **Standard Quantum Limit** ($\alpha = -0.5$)
    - Where **scaling collapse** occurs ($\alpha \to 0$)
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("Survey Configuration", divider="blue")

        st.subheader("Run Control", divider="gray")
        run_button = st.button(
            "▶️ Run Survey",
            type="primary",
            use_container_width=True,
            help="Execute the scaling survey with current parameters",
        )

        st.divider()

        # Configuration sections
        models = model_selector()
        st.divider()

        noise_type, noise_levels = noise_selector()
        st.divider()

        N_min, N_max, N_points = n_range_selector()
        st.divider()

        phi = phase_selector()

        st.divider()
        st.subheader("Reproducibility", divider="gray")
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=999999,
            value=42,
            help="For reproducible random sampling.",
        )

    # Initialize or use cached results
    if "survey_results" not in st.session_state:
        st.session_state.survey_results = None
        st.session_state.fit_results = None

    # Run survey when button is clicked
    if run_button:
        if not models:
            st.error("Please select at least one model (input state) to analyze")
        elif not noise_levels:
            st.error("Please specify at least one noise level")
        else:
            with st.spinner("Running scaling survey..."):
                raw_df, fit_df = run_full_survey(
                    models=models,
                    noise_type=noise_type,
                    noise_levels=noise_levels,
                    N_min=N_min,
                    N_max=N_max,
                    N_points=N_points,
                    phi=phi,
                    seed=int(seed),
                )

                st.session_state.survey_results = raw_df
                st.session_state.fit_results = fit_df

                st.success("Survey completed!")

    # Display results
    if st.session_state.survey_results is not None:
        raw_df = st.session_state.survey_results
        fit_df = st.session_state.fit_results
        assert fit_df is not None

        # Summary metrics
        st.header("Summary", divider="blue")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            n_models = len(raw_df["model_id"].unique()) if not raw_df.empty else 0
            st.metric("Models Analyzed", str(n_models))
        with col2:
            n_noise = len(raw_df["noise_level"].unique()) if not raw_df.empty else 0
            st.metric("Noise Levels", str(n_noise))
        with col3:
            n_n = len(raw_df["N"].unique()) if not raw_df.empty else 0
            st.metric("N Values", str(n_n))
        with col4:
            if not fit_df.empty:
                best_alpha = fit_df["alpha"].min()
                st.metric("Best α", f"{best_alpha:.3f}")
            else:
                st.metric("Best α", "N/A")

        # Visualizations
        st.header("Results", divider="green")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📈 Scaling Curves", "🔥 Phase Diagram", "📋 Fit Table", "💾 Export"],
        )

        with tab1:
            plot_scaling_curves(raw_df, fit_df, N_min, N_max)

        with tab2:
            plot_alpha_heatmap(fit_df)

        with tab3:
            display_fit_table(fit_df)

        with tab4:
            export_controls(raw_df, fit_df)

    else:
        # Welcome message when no results yet
        st.info("👈 Configure your survey in the sidebar and click **Run Survey**")


if __name__ == "__main__":
    main()
