r"""
Pedagogical Noise Comparison: Phase Diffusion vs One-Body Loss in a Single-Particle MZI.

Implements the numerical simulation for report #20260630.

Physical model
===============
- Hilbert space:  |n₁,n₂⟩ with max_photons = 1  →  dimension 4.
- Input state:    |1,0⟩.
- BS:             50/50  U_BS = exp(-i·π/4 · (a0†a1 + a1†a0)).
- Holding:        H = ω·J_z  with  J_z = (n₁ - n₂)/2,
                  plus Lindblad noise with rates γ_φ (dephasing)
                  and γ₁ (one-body loss on mode 1).
- Measurement:    ⟨J_z⟩ on the final state.
- Sensitivity:    Δω = √Var(J_z) / |∂⟨J_z⟩/∂ω|  (error propagation,
                  central finite differences).

Usage
=====
    uv run python reports/r20260630/noise_comparison_mzi.py          # run all sweeps
    uv run python reports/r20260630/noise_comparison_mzi.py --force  # re-run from scratch
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.physics.mzi_lindblad import (
    MziNoiseConfig,
    run_noisy_mzi_hamiltonian,
)
from src.physics.mzi_simulation import build_jz_operator
from src.utils.paths import report_path_fn
from src.utils.serialization import ParquetSerializable

sns.set_theme(style="whitegrid")

# ── Paths ────────────────────────────────────────────────────────────────────

_REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
_REPORT_DATE = "20260630"
_parquet_path, _fig_path = report_path_fn(_REPORTS_DIR, _REPORT_DATE)

# ── Sweep Parameters ─────────────────────────────────────────────────────────

SWEEP_A_N_POINTS: int = 200
SWEEP_A_T_HOLD_MIN: float = 0.1
SWEEP_A_T_HOLD_MAX: float = 100.0
SWEEP_A_GAMMA: float = 0.1

SWEEP_B_N_POINTS: int = 30
SWEEP_B_GAMMA_MIN: float = 1e-3
SWEEP_B_GAMMA_MAX: float = 1e0

SWEEP_C_N_POINTS: int = 40
SWEEP_C_GAMMA_MIN: float = 1e-3
SWEEP_C_GAMMA_MAX: float = 1e0

DEFAULT_OMEGA: float = 1.0
DEFAULT_T_HOLD: float = np.pi / 2.0  # mid-fringe operating point
FD_STEP: float = 1e-6  # finite-difference step for derivative

SCENARIO_CLEAN = "clean"
SCENARIO_DEPHASING = "dephasing"
SCENARIO_LOSS = "loss"
SCENARIO_BOTH = "both"

# ── Core Physics ─────────────────────────────────────────────────────────────


def run_noisy_circuit(
    omega: float,
    t_hold: float,
    gamma_phi: float,
    gamma_1: float,
) -> np.ndarray:
    r"""Evaluate the full noisy MZI circuit and return the final density matrix.

    Circuit: |1,0⟩⟨1,0| → U_BS → Lindblad(H=ω·J_z, t_hold) → U_BS → ρ_final

    Delegates the Lindblad evolution to
    :func:`src.physics.mzi_lindblad.run_noisy_mzi_hamiltonian`.

    Args:
        omega: True phase rate.
        t_hold: Holding time.
        gamma_phi: Phase diffusion rate.
        gamma_1: One-body loss rate on mode 1.

    Returns:
        Final density matrix (4×4).
    """
    # Input state |1,0⟩⟨1,0|
    rho0 = np.zeros((4, 4), dtype=complex)
    rho0[2, 2] = 1.0  # index 2 = |1,0⟩

    noise_config = MziNoiseConfig(
        gamma_phi=gamma_phi,
        gamma_1=gamma_1,
        T_decay=t_hold,
    )

    return run_noisy_mzi_hamiltonian(
        initial_state=rho0,
        max_photons=1,
        theta=np.pi / 4,
        phi_bs=0.0,
        omega=omega,
        noise_config=noise_config,
    )


def expectation_jz(rho: np.ndarray, jz: np.ndarray) -> float:
    r"""Compute ⟨J_z⟩ = Tr(ρ · J_z)."""
    return float(np.real(np.trace(rho @ jz)))


def variance_jz(rho: np.ndarray, jz: np.ndarray) -> float:
    r"""Compute Var(J_z) = ⟨J_z²⟩ - ⟨J_z⟩²."""
    jz2 = jz @ jz
    mean = float(np.real(np.trace(rho @ jz)))
    mean_sq = float(np.real(np.trace(rho @ jz2)))
    return mean_sq - mean**2


# ── Sensitivity Computation ──────────────────────────────────────────────────


def compute_sensitivity(
    omega: float,
    t_hold: float,
    gamma_phi: float,
    gamma_1: float,
    jz: np.ndarray,
    fd_step: float = FD_STEP,
) -> dict[str, float]:
    r"""Compute Δω via error propagation at a single operating point.

    Δω = √Var(J_z) / |∂⟨J_z⟩/∂ω|

    The derivative uses central finite differences: re-evaluates the full
    noisy circuit at ω ± fd_step.

    Args:
        omega: True phase rate.
        t_hold: Holding time.
        gamma_phi: Phase diffusion rate.
        gamma_1: One-body loss rate on mode 1.
        jz: 4×4 J_z operator.
        fd_step: Finite-difference step size.

    Returns:
        Dictionary with keys:
            jz_mean, jz_var, d_jz_domega, delta_omega, sql, ratio
    """
    rho = run_noisy_circuit(omega, t_hold, gamma_phi, gamma_1)
    jz_mean = expectation_jz(rho, jz)
    jz_var = variance_jz(rho, jz)

    # Central finite difference
    rho_plus = run_noisy_circuit(omega + fd_step, t_hold, gamma_phi, gamma_1)
    rho_minus = run_noisy_circuit(omega - fd_step, t_hold, gamma_phi, gamma_1)
    jz_plus = expectation_jz(rho_plus, jz)
    jz_minus = expectation_jz(rho_minus, jz)
    d_jz = (jz_plus - jz_minus) / (2.0 * fd_step)

    sql = 1.0 / t_hold if t_hold > 0 else float("inf")
    denom = abs(d_jz)
    if denom < 1e-12:
        delta_omega = float("inf")
    else:
        delta_omega = np.sqrt(max(jz_var, 0.0)) / denom

    ratio = delta_omega / sql if np.isfinite(sql) and sql > 0 else float("inf")

    return {
        "jz_mean": jz_mean,
        "jz_var": jz_var,
        "d_jz_domega": float(d_jz),
        "delta_omega": float(delta_omega),
        "sql": float(sql),
        "ratio": float(ratio),
    }


# ── Scenario Label Helper ────────────────────────────────────────────────────


def _scenario_label(gamma_phi: float, gamma_1: float) -> str:
    """Return a scenario label for given noise rates."""
    if gamma_phi == 0 and gamma_1 == 0:
        return SCENARIO_CLEAN
    if gamma_phi > 0 and gamma_1 == 0:
        return SCENARIO_DEPHASING
    if gamma_phi == 0 and gamma_1 > 0:
        return SCENARIO_LOSS
    return SCENARIO_BOTH


def _scenario_rates(scenario: str, gamma: float) -> tuple[float, float]:
    """Return (gamma_phi, gamma_1) for a given scenario and base rate."""
    if scenario == SCENARIO_CLEAN:
        return 0.0, 0.0
    if scenario == SCENARIO_DEPHASING:
        return gamma, 0.0
    if scenario == SCENARIO_LOSS:
        return 0.0, gamma
    if scenario == SCENARIO_BOTH:
        return gamma, gamma
    raise ValueError(f"Unknown scenario: {scenario}")


# ── Result Dataclass ─────────────────────────────────────────────────────────


@dataclass
class NoiseSweepResult(ParquetSerializable):
    """Result of a noise-comparison parameter sweep.

    Stores all input parameters and computed results in a single flat
    DataFrame.  Each row corresponds to one (scenario, parameter) point.

    Attributes:
        sweep_type: One of ``"t_hold_scan"``, ``"gamma_scan"``,
            ``"landscape_2d"``.
        omega: True phase rate used for all points.
        fd_step: Finite-difference step.
        gamma_base: Base noise rate for single-gamma sweeps (``"t_hold_scan"``).
            ``NaN`` for multi-gamma sweeps (``"gamma_scan"``, ``"landscape_2d"``).
        data: DataFrame with columns:

            - ``t_hold``: Holding time.
            - ``gamma_phi``: Phase diffusion rate.
            - ``gamma_1``: One-body loss rate.
            - ``scenario``: Scenario label.
            - ``jz_mean``: ⟨J_z⟩.
            - ``jz_var``: Var(J_z).
            - ``d_jz_domega``: ∂⟨J_z⟩/∂ω.
            - ``delta_omega``: Δω via error propagation.
            - ``sql``: Standard quantum limit (1/t_hold).
            - ``ratio``: Δω/SQL.
    """

    sweep_type: str
    omega: float
    fd_step: float
    gamma_base: float
    data: pd.DataFrame

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "sweep_type",
        "omega",
        "fd_step",
        "gamma_base",
        "t_hold",
        "gamma_phi",
        "gamma_1",
        "scenario",
        "jz_mean",
        "jz_var",
        "d_jz_domega",
        "delta_omega",
        "sql",
        "ratio",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        """Serialize to a fully self-describing DataFrame.

        Metadata columns (sweep_type, omega, fd_step, gamma_base) are
        broadcast to every row so the Parquet file is complete.
        """
        df = self.data.copy()
        df["sweep_type"] = self.sweep_type
        df["omega"] = self.omega
        df["fd_step"] = self.fd_step
        df["gamma_base"] = self.gamma_base
        return df

    @classmethod
    def from_parquet(cls, path: str | Path) -> NoiseSweepResult:
        """Reconstruct from a Parquet file written by ``to_dataframe``.

        Raises:
            ValueError: If required columns are missing.
        """
        df = pd.read_parquet(path)
        cls._validate_columns(df)

        return cls(
            sweep_type=str(df["sweep_type"].iloc[0]),
            omega=float(df["omega"].iloc[0]),
            fd_step=float(df["fd_step"].iloc[0]),
            gamma_base=float(df["gamma_base"].iloc[0]),
            data=df.drop(columns=["sweep_type", "omega", "fd_step", "gamma_base"]),
        )


# ── Sweep Orchestrators ──────────────────────────────────────────────────────


def sweep_t_hold(
    omega: float = DEFAULT_OMEGA,
    gamma: float = SWEEP_A_GAMMA,
    t_hold_min: float = SWEEP_A_T_HOLD_MIN,
    t_hold_max: float = SWEEP_A_T_HOLD_MAX,
    n_points: int = SWEEP_A_N_POINTS,
    fd_step: float = FD_STEP,
) -> NoiseSweepResult:
    r"""Sweep A: holding-time degradation curves.

    Four scenarios (Clean, Dephasing-only, Loss-only, Both) at fixed
    ``gamma``, scanning ``t_hold`` from ``t_hold_min`` to ``t_hold_max``.

    Returns:
        NoiseSweepResult with scenario × t_hold grid.
    """
    jz = _get_jz_operator()
    t_hold_values = np.logspace(np.log10(t_hold_min), np.log10(t_hold_max), n_points)
    scenarios = [SCENARIO_CLEAN, SCENARIO_DEPHASING, SCENARIO_LOSS, SCENARIO_BOTH]

    rows: list[dict] = []
    for scenario in scenarios:
        gamma_phi, gamma_1 = _scenario_rates(scenario, gamma)
        for t_hold in t_hold_values:
            result = compute_sensitivity(omega, t_hold, gamma_phi, gamma_1, jz, fd_step)
            rows.append(
                {
                    "t_hold": t_hold,
                    "gamma_phi": gamma_phi,
                    "gamma_1": gamma_1,
                    "scenario": scenario,
                    **result,
                }
            )

    df = pd.DataFrame(rows)
    return NoiseSweepResult(
        sweep_type="t_hold_scan",
        omega=omega,
        fd_step=fd_step,
        gamma_base=gamma,
        data=df,
    )


def sweep_gamma(
    omega: float = DEFAULT_OMEGA,
    t_hold: float = DEFAULT_T_HOLD,
    gamma_min: float = SWEEP_B_GAMMA_MIN,
    gamma_max: float = SWEEP_B_GAMMA_MAX,
    n_points: int = SWEEP_B_N_POINTS,
    fd_step: float = FD_STEP,
) -> NoiseSweepResult:
    r"""Sweep B: noise-rate scaling.

    Three noisy scenarios (Dephasing-only, Loss-only, Both) at fixed
    ``t_hold``, scanning ``gamma`` from ``gamma_min`` to ``gamma_max``.
    The Clean scenario is omitted because its ratio is identically 1.

    Returns:
        NoiseSweepResult with scenario × gamma grid.
    """
    jz = _get_jz_operator()
    gamma_values = np.logspace(np.log10(gamma_min), np.log10(gamma_max), n_points)
    scenarios = [SCENARIO_DEPHASING, SCENARIO_LOSS, SCENARIO_BOTH]

    rows: list[dict] = []
    for scenario in scenarios:
        for gamma in gamma_values:
            gamma_phi, gamma_1 = _scenario_rates(scenario, gamma)
            result = compute_sensitivity(omega, t_hold, gamma_phi, gamma_1, jz, fd_step)
            rows.append(
                {
                    "t_hold": t_hold,
                    "gamma_phi": gamma_phi,
                    "gamma_1": gamma_1,
                    "scenario": scenario,
                    **result,
                }
            )

    df = pd.DataFrame(rows)
    return NoiseSweepResult(
        sweep_type="gamma_scan",
        omega=omega,
        fd_step=fd_step,
        gamma_base=float("nan"),
        data=df,
    )


def sweep_2d(
    omega: float = DEFAULT_OMEGA,
    t_hold: float = DEFAULT_T_HOLD,
    gamma_min: float = SWEEP_C_GAMMA_MIN,
    gamma_max: float = SWEEP_C_GAMMA_MAX,
    n_points: int = SWEEP_C_N_POINTS,
    fd_step: float = FD_STEP,
) -> NoiseSweepResult:
    r"""Sweep C: 2D noise landscape.

    Heatmap of Δω/SQL over (γ_φ, γ₁) ∈ [gamma_min, gamma_max]²
    at fixed ``t_hold``.

    Returns:
        NoiseSweepResult with gamma_phi × gamma_1 grid.
    """
    jz = _get_jz_operator()
    gamma_values = np.logspace(np.log10(gamma_min), np.log10(gamma_max), n_points)

    rows: list[dict] = []
    for gamma_phi in gamma_values:
        for gamma_1 in gamma_values:
            result = compute_sensitivity(omega, t_hold, gamma_phi, gamma_1, jz, fd_step)
            rows.append(
                {
                    "t_hold": t_hold,
                    "gamma_phi": gamma_phi,
                    "gamma_1": gamma_1,
                    "scenario": _scenario_label(gamma_phi, gamma_1),
                    **result,
                }
            )

    df = pd.DataFrame(rows)
    return NoiseSweepResult(
        sweep_type="landscape_2d",
        omega=omega,
        fd_step=fd_step,
        gamma_base=float("nan"),
        data=df,
    )


# ── Operator Cache ───────────────────────────────────────────────────────────


def _get_jz_operator() -> np.ndarray:
    """Return the cached J_z operator for max_photons=1 as a full matrix."""
    if not hasattr(_get_jz_operator, "_cache"):
        _get_jz_operator._cache = np.diag(build_jz_operator(max_photons=1))  # type: ignore[attr-defined]
    return _get_jz_operator._cache  # type: ignore[attr-defined]


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_degradation_curves(
    result: NoiseSweepResult,
    save_path: Path | None = None,
) -> Path:
    """Plot Sweep A: Δω vs t_hold (log-log), four scenario curves.

    Args:
        result: Sweep A result.
        save_path: Output SVG path.  If None, auto-generate.

    Returns:
        Path to the saved SVG.
    """
    if save_path is None:
        save_path = _fig_path("degradation-curves")

    df = result.data
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        SCENARIO_CLEAN: "black",
        SCENARIO_DEPHASING: "red",
        SCENARIO_LOSS: "blue",
        SCENARIO_BOTH: "purple",
    }
    styles = {
        SCENARIO_CLEAN: "--",
        SCENARIO_DEPHASING: "-",
        SCENARIO_LOSS: "-",
        SCENARIO_BOTH: "-",
    }
    labels = {
        SCENARIO_CLEAN: "Clean (SQL)",
        SCENARIO_DEPHASING: f"Dephasing ($\\gamma_\\phi={result.gamma_base}$)",
        SCENARIO_LOSS: "Loss ($\\gamma_1=" + f"{result.gamma_base}$)",
        SCENARIO_BOTH: "Both",
    }

    for scenario in [SCENARIO_CLEAN, SCENARIO_DEPHASING, SCENARIO_LOSS, SCENARIO_BOTH]:
        mask = df["scenario"] == scenario
        subset = df[mask].sort_values("t_hold")
        ax.loglog(
            subset["t_hold"],
            subset["delta_omega"],
            color=colors[scenario],
            linestyle=styles[scenario],
            label=labels[scenario],
        )

    # SQL reference line
    t_grid = np.logspace(-1, 2, 200)
    ax.loglog(t_grid, 1.0 / t_grid, "k:", alpha=0.5, label="$1/t_{\\text{hold}}$")

    ax.set_xlabel("$t_{\\text{hold}}$")
    ax.set_ylabel("$\\Delta\\omega$")
    ax.set_title(f"Holding-Time Degradation ($\\gamma = {result.gamma_base}$)")
    ax.legend()
    ax.set_xlim(result.data["t_hold"].min(), result.data["t_hold"].max())
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_noise_rate_scaling(
    result: NoiseSweepResult,
    save_path: Path | None = None,
) -> Path:
    """Plot Sweep B: Δω/Δω_SQL vs γ (log-log), three scenario curves.

    Args:
        result: Sweep B result.
        save_path: Output SVG path.  If None, auto-generate.

    Returns:
        Path to the saved SVG.
    """
    if save_path is None:
        save_path = _fig_path("noise-rate-scaling")

    df = result.data
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        SCENARIO_DEPHASING: "red",
        SCENARIO_LOSS: "blue",
        SCENARIO_BOTH: "purple",
    }
    labels = {
        SCENARIO_DEPHASING: "Dephasing-only",
        SCENARIO_LOSS: "Loss-only",
        SCENARIO_BOTH: "Both",
    }

    # Reconstruct gamma from gamma_phi + gamma_1
    for scenario in [SCENARIO_DEPHASING, SCENARIO_LOSS, SCENARIO_BOTH]:
        mask = df["scenario"] == scenario
        subset = df[mask].sort_values(
            "gamma_phi" if scenario != SCENARIO_LOSS else "gamma_1"
        )
        gp = np.asarray(subset["gamma_phi"].values, dtype=float)
        g1 = np.asarray(subset["gamma_1"].values, dtype=float)
        gamma_vals = np.where(gp > 0, gp, g1)
        ax.loglog(
            gamma_vals,
            subset["ratio"],
            color=colors[scenario],
            label=labels[scenario],
        )

    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="SQL")
    ax.set_xlabel("$\\gamma$")
    ax.set_ylabel("$\\Delta\\omega / \\Delta\\omega_{\\text{SQL}}$")
    ax.set_title(
        "Noise-Rate Scaling ($t_{\\text{hold}} = "
        f"{result.data['t_hold'].iloc[0]:.2f}$)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_landscape_2d(
    result: NoiseSweepResult,
    save_path: Path | None = None,
    contour_levels: list[float] | None = None,
) -> Path:
    """Plot Sweep C: 2D heatmap of log₁₀(Δω/SQL) over (γ_φ, γ₁).

    Args:
        result: Sweep C result.
        save_path: Output SVG path.  If None, auto-generate.
        contour_levels: Contour levels for the ratio.  Defaults to
            [1, 2, 5, 10, 100].

    Returns:
        Path to the saved SVG.
    """
    if save_path is None:
        save_path = _fig_path("noise-landscape-2d")

    if contour_levels is None:
        contour_levels = [1, 2, 5, 10, 100]

    df = result.data
    gamma_phi_vals = np.sort(df["gamma_phi"].unique())
    gamma_1_vals = np.sort(df["gamma_1"].unique())

    # Pivot to 2D grid
    ratio_grid = df.pivot_table(
        index="gamma_phi", columns="gamma_1", values="ratio"
    ).to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6))

    log_ratio = np.log10(np.maximum(ratio_grid, 1e-15))
    extent = (
        float(np.log10(gamma_1_vals.min())),
        float(np.log10(gamma_1_vals.max())),
        float(np.log10(gamma_phi_vals.min())),
        float(np.log10(gamma_phi_vals.max())),
    )

    im = ax.imshow(
        log_ratio,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label="$\\log_{10}(\\Delta\\omega / \\text{SQL})$")

    # Contour lines
    contour_log_levels = np.log10(contour_levels)
    X, Y = np.meshgrid(np.log10(gamma_1_vals), np.log10(gamma_phi_vals))
    cs = ax.contour(
        X, Y, log_ratio, levels=contour_log_levels, colors="white", linewidths=0.8
    )
    ax.clabel(cs, fmt=lambda v: f"$10^{{{v:.0f}}}$", inline=True, fontsize=8)

    ax.set_xlabel("$\\log_{10}(\\gamma_1)$  (loss rate)")
    ax.set_ylabel("$\\log_{10}(\\gamma_\\phi)$  (dephasing rate)")
    ax.set_title(
        "2D Noise Landscape ($t_{\\text{hold}} = "
        f"{result.data['t_hold'].iloc[0]:.2f}$)"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ── Main CLI ─────────────────────────────────────────────────────────────────


def main(force: bool = False) -> None:
    """Run all three sweeps, save Parquet files, and generate figures."""
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="qutip")

    # Sweep A — Holding-time degradation
    pq_a = _parquet_path("t-hold-scan")
    if force or not pq_a.exists():
        print("Running Sweep A: holding-time degradation ...")
        result_a = sweep_t_hold()
        result_a.save_parquet(pq_a)
        print(f"  Saved {pq_a}")
    else:
        result_a = NoiseSweepResult.from_parquet(pq_a)
        print(f"  Loaded existing {pq_a}")

    fig_a = plot_degradation_curves(result_a)
    print(f"  Figure saved {fig_a}")

    # Sweep B — Noise-rate scaling
    pq_b = _parquet_path("gamma-scan")
    if force or not pq_b.exists():
        print("Running Sweep B: noise-rate scaling ...")
        result_b = sweep_gamma()
        result_b.save_parquet(pq_b)
        print(f"  Saved {pq_b}")
    else:
        result_b = NoiseSweepResult.from_parquet(pq_b)
        print(f"  Loaded existing {pq_b}")

    fig_b = plot_noise_rate_scaling(result_b)
    print(f"  Figure saved {fig_b}")

    # Sweep C — 2D noise landscape
    pq_c = _parquet_path("landscape-2d")
    if force or not pq_c.exists():
        print("Running Sweep C: 2D noise landscape ...")
        result_c = sweep_2d()
        result_c.save_parquet(pq_c)
        print(f"  Saved {pq_c}")
    else:
        result_c = NoiseSweepResult.from_parquet(pq_c)
        print(f"  Loaded existing {pq_c}")

    fig_c = plot_landscape_2d(result_c)
    print(f"  Figure saved {fig_c}")

    print("All sweeps complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pedagogical noise comparison for single-particle MZI."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all sweeps from scratch (overwrite existing Parquet files).",
    )
    args = parser.parse_args()
    main(force=args.force)
