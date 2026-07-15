"""
Generate all results and SVG figures for
Coupled-System-Ancilla Metrology Under Photon Loss (v2).

Run with:
    uv run python reports/r20260713/generate_figures.py [--force]

Produces:
    raw_data/20260713-gamma-sweep.parquet
    raw_data/20260713-omega-scan-N{N}-g{gamma}.parquet
    figures/20260713-{rqfi-heatmap,sensitivity-vs-gamma,
                     optimal-alpha-N{N},omega-dependence,
                     measurement-gap,ep-ratio}.svg
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Prevent OpenBLAS thread oversubscription (fixes eigvalsh taking 12s instead of 0.2s)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

# ── Report-local imports ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reports.r20260713.coupled_ancilla_photon_loss import (
    _trace_out_ancilla,
    build_bipartite_operators,
    build_subsystem_operators,
    compute_ep_sensitivity_from_rho,
    compute_qfi_finite_diff,
    evolve_config_a,
    evolve_config_c,
    optimise_coupling,
)

# ── Constants ─────────────────────────────────────────────────────────
T_HOLD = 10.0
FD_STEP = 1e-6
SQL_N1 = 1.0 / (np.sqrt(1) * T_HOLD)  # 0.1
DATE = "20260713"
RAW_DIR = Path(__file__).parent / "raw_data"
FIG_DIR = Path(__file__).parent / "figures"

# Parameter ranges
N_VALUES_CONFIG_A = [1, 2, 3, 4, 5, 6, 7, 8]
N_VALUES_CONFIG_C = [1, 2, 3, 4, 5, 6, 7, 8]
GAMMA_VALUES = [0.0, *list(np.logspace(-6, 6, 60))]  # 61 values: γ=0 + 60 log-spaced
OMEGA_REP = 1.0  # representative ω for coupling optimisation
OMEGA_SCAN_COUNT = 500
OMEGA_SCAN_MIN = 0.01
OMEGA_SCAN_MAX = 5.0

# Optimisation settings
N_STARTS = 5
MAX_ITER = 20
OPT_BOUNDS = (-10.0, 10.0)
OPT_SEED = 42
N_JOBS = -1  # use all available cores

sns.set_theme(style="whitegrid", font_scale=1.1)


# ============================================================================
# Data structures for sweep results
# ============================================================================


@dataclass
class GammaSweepResult:
    """Aggregated result of a γ-sweep at fixed N."""

    N: int
    gamma_values: np.ndarray
    # Config A
    delta_omega_ep_a: np.ndarray
    fq_a: np.ndarray
    delta_omega_qfi_a: np.ndarray
    # Config C (optimised α)
    delta_omega_ep_c: np.ndarray
    fq_c: np.ndarray
    delta_omega_qfi_c: np.ndarray
    alpha_opt: dict[str, np.ndarray]
    # Ratios
    r_qfi: np.ndarray  # F_Q(C) / (2 F_Q(A))
    r_ep: np.ndarray  # Δω_EP(C) / Δω_EP(A)
    r_gap: np.ndarray  # Δω_EP(C) / Δω_QFI(C)


# ============================================================================
# Simulation runners
# ============================================================================


def _run_config_a_point(
    N: int, omega: float, gamma: float
) -> tuple[float, float, float]:
    """Run Config A at one point: returns (delta_ep, fq, delta_qfi)."""
    fd = FD_STEP
    rho = evolve_config_a(N, omega, gamma, T_HOLD)
    rho_p = evolve_config_a(N, omega + fd, gamma, T_HOLD)
    rho_m = evolve_config_a(N, omega - fd, gamma, T_HOLD)

    sub = build_subsystem_operators(N)
    Jz = sub["Jz"]
    delta_ep, _, _ = compute_ep_sensitivity_from_rho(rho, Jz, rho_p, rho_m, fd)
    fq = compute_qfi_finite_diff(rho, rho_p, rho_m, fd)
    delta_qfi = 1.0 / np.sqrt(fq) if fq > 0 else float("inf")
    return delta_ep, fq, delta_qfi


def _run_config_c_point(
    N: int, omega: float, gamma: float, alpha: dict[str, float]
) -> tuple[float, float, float]:
    """Run Config C at one point: returns (delta_ep, fq, delta_qfi)."""
    fd = FD_STEP
    rho = evolve_config_c(N, omega, gamma, T_HOLD, alpha)
    rho_p = evolve_config_c(N, omega + fd, gamma, T_HOLD, alpha)
    rho_m = evolve_config_c(N, omega - fd, gamma, T_HOLD, alpha)

    ops_b = build_bipartite_operators(N)
    sub = build_subsystem_operators(N)
    dim_sub = ops_b["dim_sub"]

    rho_S = _trace_out_ancilla(rho, dim_sub)
    rho_S_p = _trace_out_ancilla(rho_p, dim_sub)
    rho_S_m = _trace_out_ancilla(rho_m, dim_sub)

    delta_ep, _, _ = compute_ep_sensitivity_from_rho(
        rho_S, sub["Jz"], rho_S_p, rho_S_m, fd
    )
    fq = compute_qfi_finite_diff(rho, rho_p, rho_m, fd)
    delta_qfi = 1.0 / np.sqrt(fq) if fq > 0 else float("inf")
    return delta_ep, fq, delta_qfi


def _eval_gamma_point_a_only(
    N: int, omega_rep: float, gamma: float
) -> tuple[float, float, float]:
    """Evaluate Config A at a single γ point. Parallelisation-safe."""
    return _run_config_a_point(N, omega_rep, gamma)


def _eval_gamma_point_both(
    N: int, omega_rep: float, gamma: float, alpha: dict[str, float]
) -> tuple[float, float, float, float, float, float]:
    """Evaluate Config A + C at a single γ point. Parallelisation-safe."""
    da, fa, dqa = _run_config_a_point(N, omega_rep, gamma)
    dc, fc, dqc = _run_config_c_point(N, omega_rep, gamma, alpha)
    return da, fa, dqa, dc, fc, dqc


def sweep_gamma_for_N(
    N: int,
    gamma_values: list[float],
    omega_rep: float = OMEGA_REP,
    n_starts: int = N_STARTS,
    max_iter: int = MAX_ITER,
    run_config_c: bool = True,
    n_jobs: int = N_JOBS,
) -> GammaSweepResult:
    """Run γ-sweep: optimise α at γ=0, then evaluate Config A and C at all γ.

    The coupling is optimised once at γ=0 (noiseless) and the same α is
    reused for all γ values. The γ loop is parallelised via joblib.

    Args:
        run_config_c: If True, run Config C (coupled) evaluations.
                      If False, run Config A only (for large N where C is too slow).
        n_jobs: Number of parallel jobs (-1 = all cores).
    """
    print(
        f"  N={N}: starting γ-sweep ({len(gamma_values)} points, "
        f"{'A+C' if run_config_c else 'A only'})..."
    )
    t0 = time.time()

    n_g = len(gamma_values)
    ep_a = np.full(n_g, np.nan)
    fq_a = np.full(n_g, np.nan)
    dqfi_a = np.full(n_g, np.nan)
    ep_c = np.full(n_g, np.nan)
    fq_c = np.full(n_g, np.nan)
    dqfi_c = np.full(n_g, np.nan)
    alpha_xx = np.zeros(n_g)
    alpha_xz = np.zeros(n_g)
    alpha_zx = np.zeros(n_g)
    alpha_zz = np.zeros(n_g)

    # Step 1: Optimise α at γ=0 (noiseless — fast)
    a_fixed: dict[str, float] = {"xx": 0.0, "xz": 0.0, "zx": 0.0, "zz": 0.0}
    if run_config_c:
        print("    Optimising α at γ=0...", end="", flush=True)
        opt = optimise_coupling(
            N,
            0.0,
            omega_rep,
            T_HOLD,
            n_starts=n_starts,
            max_iter=max_iter,
            bounds=OPT_BOUNDS,
            seed=OPT_SEED,
        )
        a_fixed = opt["alpha_opt"]
        print(
            f" done (Δω={opt['delta_ep_opt']:.6f}, "
            f"α=({a_fixed['xx']:.2f}, {a_fixed['xz']:.2f}, "
            f"{a_fixed['zx']:.2f}, {a_fixed['zz']:.2f}))"
        )

    # Step 2: Evaluate at all γ values using the fixed α (parallelised)
    if run_config_c:
        results_list = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_eval_gamma_point_both)(N, omega_rep, gamma, a_fixed)
            for gamma in gamma_values
        )
        for ig, (da, fa, dqa, dc, fc, dqc) in enumerate(results_list):
            ep_a[ig] = da
            fq_a[ig] = fa
            dqfi_a[ig] = dqa
            alpha_xx[ig] = a_fixed["xx"]
            alpha_xz[ig] = a_fixed["xz"]
            alpha_zx[ig] = a_fixed["zx"]
            alpha_zz[ig] = a_fixed["zz"]
            ep_c[ig] = dc
            fq_c[ig] = fc
            dqfi_c[ig] = dqc
    else:
        results_list = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_eval_gamma_point_a_only)(N, omega_rep, gamma)
            for gamma in gamma_values
        )
        for ig, (da, fa, dqa) in enumerate(results_list):
            ep_a[ig] = da
            fq_a[ig] = fa
            dqfi_a[ig] = dqa

    # Compute ratios
    r_qfi = np.where(fq_a > 0, fq_c / (2.0 * fq_a), np.nan)
    r_ep = np.where(
        np.isfinite(ep_a) & (ep_a > 0) & np.isfinite(ep_c),
        ep_c / ep_a,
        np.nan,
    )
    r_gap = np.where(
        np.isfinite(ep_c) & np.isfinite(dqfi_c) & (dqfi_c > 0),
        ep_c / dqfi_c,
        np.nan,
    )

    elapsed = time.time() - t0
    r_qfi_min = np.nanmin(r_qfi) if np.any(np.isfinite(r_qfi)) else float("nan")
    r_qfi_max = np.nanmax(r_qfi) if np.any(np.isfinite(r_qfi)) else float("nan")
    r_ep_min = np.nanmin(r_ep) if np.any(np.isfinite(r_ep)) else float("nan")
    r_ep_max = np.nanmax(r_ep) if np.any(np.isfinite(r_ep)) else float("nan")
    print(
        f"  N={N}: done in {elapsed:.1f}s "
        f"(R_QFI range: [{r_qfi_min:.4f}, {r_qfi_max:.4f}], "
        f"R_EP range: [{r_ep_min:.4f}, {r_ep_max:.4f}])"
    )

    return GammaSweepResult(
        N=N,
        gamma_values=np.array(gamma_values),
        delta_omega_ep_a=ep_a,
        fq_a=fq_a,
        delta_omega_qfi_a=dqfi_a,
        delta_omega_ep_c=ep_c,
        fq_c=fq_c,
        delta_omega_qfi_c=dqfi_c,
        alpha_opt={
            "xx": alpha_xx,
            "xz": alpha_xz,
            "zx": alpha_zx,
            "zz": alpha_zz,
        },
        r_qfi=r_qfi,
        r_ep=r_ep,
        r_gap=r_gap,
    )


# ============================================================================
# ω-scan runner
# ============================================================================


def omega_scan_single_point(
    N: int,
    omega: float,
    gamma: float,
    alpha: dict[str, float],
) -> dict[str, float]:
    """Evaluate Config A and C at one ω point."""
    da, fa, dqa = _run_config_a_point(N, omega, gamma)
    dc, fc, dqc = _run_config_c_point(N, omega, gamma, alpha)
    return {
        "omega": omega,
        "delta_omega_ep_a": da,
        "fq_a": fa,
        "delta_omega_qfi_a": dqa,
        "delta_omega_ep_c": dc,
        "fq_c": fc,
        "delta_omega_qfi_c": dqc,
    }


def _omega_scan_one_point(
    N: int, omega: float, gamma: float, alpha: dict[str, float]
) -> dict[str, float]:
    """Evaluate Config A only at one ω point (for large N)."""
    da, fa, dqa = _run_config_a_point(N, omega, gamma)
    return {
        "omega": omega,
        "delta_omega_ep_a": da,
        "fq_a": fa,
        "delta_omega_qfi_a": dqa,
        "delta_omega_ep_c": float("nan"),
        "fq_c": float("nan"),
        "delta_omega_qfi_c": float("nan"),
    }


# ============================================================================
# Parquet saving
# ============================================================================


def save_gamma_sweep_parquet(results: list[GammaSweepResult]) -> Path:
    """Save all γ-sweep results as a single Parquet file."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for res in results:
        n_g = len(res.gamma_values)
        rows.extend(
            {
                "N": res.N,
                "gamma": res.gamma_values[ig],
                "delta_omega_ep_a": res.delta_omega_ep_a[ig],
                "fq_a": res.fq_a[ig],
                "delta_omega_qfi_a": res.delta_omega_qfi_a[ig],
                "delta_omega_ep_c": res.delta_omega_ep_c[ig],
                "fq_c": res.fq_c[ig],
                "delta_omega_qfi_c": res.delta_omega_qfi_c[ig],
                "alpha_xx": res.alpha_opt["xx"][ig],
                "alpha_xz": res.alpha_opt["xz"][ig],
                "alpha_zx": res.alpha_opt["zx"][ig],
                "alpha_zz": res.alpha_opt["zz"][ig],
                "r_qfi": res.r_qfi[ig],
                "r_ep": res.r_ep[ig],
                "r_gap": res.r_gap[ig],
                "t_hold": T_HOLD,
            }
            for ig in range(n_g)
        )
    df = pd.DataFrame(rows)
    path = RAW_DIR / f"{DATE}-gamma-sweep.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved γ-sweep Parquet: {path} ({len(rows)} rows)")
    return path


def save_omega_scan_parquet(rows: list[dict], tag: str) -> Path:
    """Save ω-scan results as Parquet."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    path = RAW_DIR / f"{DATE}-omega-scan-{tag}.parquet"
    df.to_parquet(path, index=False)
    return path


# ============================================================================
# Figure generation
# ============================================================================


def _markevery(n: int, target: int = 25) -> int:
    return max(1, n // target)


def fig_rqfi_heatmap(results: list[GammaSweepResult]) -> Path:
    """R_QFI heatmap: γ vs N."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    N_vals = np.array([r.N for r in results])
    # Use first result's gamma values (all same)
    gamma_vals = results[0].gamma_values

    R = np.array([r.r_qfi for r in results])  # shape (n_N, n_gamma)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.pcolormesh(
        np.log2(gamma_vals + 1e-15),  # log2 scale, γ=0 maps to large negative
        N_vals,
        R,
        shading="auto",
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.5,
    )
    fig.colorbar(im, ax=ax, label=r"$\mathcal{R}_{QFI} = F_Q^{(C)} / (2 F_Q^{(A)})$")

    # Contour at R_QFI = 1
    try:
        cs = ax.contour(
            np.log2(gamma_vals + 1e-15),
            N_vals,
            R,
            levels=[1.0],
            colors="black",
            linewidths=1.5,
            linestyles="--",
        )
        ax.clabel(cs, fmt=r"$\mathcal{R}_{QFI}=1$", fontsize=9)
    except Exception:
        pass

    ax.set_xlabel(r"$\log_2(\gamma)$ (loss rate)")
    ax.set_ylabel(r"$N$ (particles per subsystem)")
    ax.set_title(r"$\mathcal{R}_{QFI}$: Coupling advantage under photon loss")
    ax.set_yticks(N_vals)

    # Mark γ=0 tick
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([np.log2(g + 1e-15) for g in gamma_vals if g == 0])
    ax2.set_xticklabels([r"$\gamma=0$"])

    fig.tight_layout()
    path = FIG_DIR / f"{DATE}-rqfi-heatmap.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig_sensitivity_vs_gamma(results: list[GammaSweepResult]) -> Path:
    """Δω_EP vs γ for Config A and C at selected N values."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot_N = [1, 3, 5] if len(results) >= 5 else [r.N for r in results[:3]]
    fig, axes = plt.subplots(
        1, len(plot_N), figsize=(5 * len(plot_N), 4.5), sharey=True
    )
    if len(plot_N) == 1:
        axes = [axes]

    for ax, N in zip(axes, plot_N, strict=False):
        res = next((r for r in results if r.N == N), None)
        if res is None:
            continue
        g = res.gamma_values
        ax.plot(
            g, res.delta_omega_ep_a, "o-", color="C0", label="Config A", linewidth=1.5
        )
        ax.plot(
            g,
            res.delta_omega_ep_c,
            "s-",
            color="C3",
            label="Config C (opt)",
            linewidth=1.5,
        )
        ax.axhline(
            y=SQL_N1 / np.sqrt(N), color="gray", linestyle="--", alpha=0.5, label="SQL"
        )
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlabel(r"$\gamma$ (loss rate)")
        ax.set_title(f"$N = {N}$")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel(r"$\Delta\omega_{EP}$")
    fig.suptitle(
        r"EP Sensitivity vs Photon Loss: Config A (system alone) vs Config C (coupled)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    path = FIG_DIR / f"{DATE}-sensitivity-vs-gamma.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig_optimal_alpha_per_N(results: list[GammaSweepResult]) -> list[Path]:
    """Optimal α components vs γ — one SVG per N value."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = [r"$\alpha_{xx}$", r"$\alpha_{xz}$", r"$\alpha_{zx}$", r"$\alpha_{zz}$"]
    colours = ["C0", "C1", "C2", "C3"]
    keys = ["xx", "xz", "zx", "zz"]
    paths: list[Path] = []

    for res in results:
        N = res.N
        fig, ax = plt.subplots(figsize=(7, 4))
        g = res.gamma_values
        for key, label, colour in zip(keys, labels, colours, strict=False):
            ax.plot(
                g,
                res.alpha_opt[key],
                "o-",
                color=colour,
                label=label,
                linewidth=1.5,
                markersize=4,
                markevery=_markevery(len(g)),
            )
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(
            y=OPT_BOUNDS[0], color="red", linestyle=":", alpha=0.3, label="Bounds"
        )
        ax.axhline(y=OPT_BOUNDS[1], color="red", linestyle=":", alpha=0.3)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlabel(r"$\gamma$ (loss rate)")
        ax.set_ylabel(r"Optimal coupling coefficient $\alpha^*$")
        ax.set_title(f"Optimal $\\alpha^*(\\gamma)$ — $N = {N}$")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        path = FIG_DIR / f"{DATE}-optimal-alpha-N{N}.svg"
        fig.savefig(path, format="svg", bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    print(f"  Saved: {len(paths)} per-N optimal-alpha SVGs")
    return paths


def fig_omega_dependence(
    omega_data: dict[tuple[int, float], pd.DataFrame],
) -> Path:
    """Δω_EP vs ω at selected (γ, N) pairs."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Select a few representative (γ, N) pairs
    keys = sorted(omega_data.keys())
    n_panels = min(len(keys), 6)
    selected = keys[:n_panels]

    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 3.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ax, key in zip(axes, selected, strict=False):
        N, gamma = key
        df = omega_data[key]
        omega = df["omega"].to_numpy()
        me = _markevery(len(omega))

        ax.plot(
            omega,
            df["delta_omega_ep_a"],
            "-",
            color="C0",
            linewidth=1.5,
            label="Config A (EP)",
            markevery=me,
            markersize=4,
        )
        ax.plot(
            omega,
            df["delta_omega_ep_c"],
            "-",
            color="C3",
            linewidth=1.5,
            label="Config C (EP, opt)",
            markevery=me,
            markersize=4,
        )
        if "delta_omega_qfi_c" in df.columns:
            valid_qfi = np.isfinite(df["delta_omega_qfi_c"].to_numpy())
            if np.any(valid_qfi):
                ax.plot(
                    omega[valid_qfi],
                    df["delta_omega_qfi_c"].to_numpy()[valid_qfi],
                    ":",
                    color="C2",
                    linewidth=1.2,
                    label=r"Config C ($\Delta\omega_{QFI}$)",
                )

        sql_n = SQL_N1 / np.sqrt(N)
        ax.axhline(
            y=sql_n, color="gray", linestyle="--", alpha=0.5, label=f"SQL ($N={N}$)"
        )
        ax.set_ylabel(r"$\Delta\omega$")
        gamma_str = f"{gamma:.4f}" if gamma < 0.01 else f"{gamma:.2f}"
        ax.set_title(f"$N={N}$, $\\gamma={gamma_str}$")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel(r"$\omega$ (phase rate)")
    fig.suptitle(
        r"$\omega$-Dependence of Sensitivity at Selected $(\gamma, N)$",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    path = FIG_DIR / f"{DATE}-omega-dependence.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig_measurement_gap(results: list[GammaSweepResult]) -> Path:
    """Measurement gap R_gap vs γ at selected N values."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot_N = [1, 3, 5] if len(results) >= 5 else [r.N for r in results[:3]]
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for N in plot_N:
        res = next((r for r in results if r.N == N), None)
        if res is None:
            continue
        g = res.gamma_values
        valid = np.isfinite(res.r_gap)
        ax.plot(
            g[valid],
            res.r_gap[valid],
            "o-",
            label=f"$N={N}$",
            linewidth=1.5,
            markersize=5,
        )

    ax.axhline(
        y=1.0,
        color="gray",
        linestyle="--",
        alpha=0.6,
        label=r"$\mathcal{R}_{gap}=1$ (S-only optimal)",
    )
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel(r"$\gamma$ (loss rate)")
    ax.set_ylabel(
        r"$\mathcal{R}_{gap} = \Delta\omega_{EP}^{(C)} / \Delta\omega_{QFI}^{(C)}$"
    )
    ax.set_title("Measurement Gap: S-only vs Optimal Joint Measurement")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0.5)

    fig.tight_layout()
    path = FIG_DIR / f"{DATE}-measurement-gap.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig_ep_ratio(results: list[GammaSweepResult]) -> Path:
    """Practical EP ratio R_EP vs γ at selected N values."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot_N = [1, 3, 5] if len(results) >= 5 else [r.N for r in results[:3]]
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for N in plot_N:
        res = next((r for r in results if r.N == N), None)
        if res is None:
            continue
        g = res.gamma_values
        valid = np.isfinite(res.r_ep)
        ax.plot(
            g[valid],
            res.r_ep[valid],
            "o-",
            label=f"$N={N}$",
            linewidth=1.5,
            markersize=5,
        )

    ax.axhline(
        y=1.0,
        color="gray",
        linestyle="--",
        alpha=0.6,
        label=r"$\mathcal{R}_{EP}=1$ (no coupling advantage)",
    )
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel(r"$\gamma$ (loss rate)")
    ax.set_ylabel(
        r"$\mathcal{R}_{EP} = \Delta\omega_{EP}^{(C)} / \Delta\omega_{EP}^{(A)}$"
    )
    ax.set_title("Practical Sensitivity Ratio: Coupled Ancilla vs System Alone")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = FIG_DIR / f"{DATE}-ep-ratio.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ============================================================================
# Main
# ============================================================================


def main(force: bool = False) -> None:
    """Run all simulations and generate all figures."""
    print("=" * 65)
    print("Coupled System-Ancilla Metrology Under Photon Loss (v2)")
    print(f"T_H={T_HOLD}, ω_rep={OMEGA_REP}, γ values={len(GAMMA_VALUES)}")
    print(f"Config A: N ∈ {N_VALUES_CONFIG_A}")
    print(f"Config C: N ∈ {N_VALUES_CONFIG_C} (optimised α at γ=0)")
    print(f"α bounds: [{OPT_BOUNDS[0]}, {OPT_BOUNDS[1]}]^4, N_starts={N_STARTS}")
    print(f"Parallelisation: n_jobs={N_JOBS}")
    print("=" * 65)

    t_total = time.time()

    # ── Phase 1: γ-sweep for all N ────────────────────────────────────
    print("\n[Phase 1] Running γ-sweep for Config A + Config C...")
    results: list[GammaSweepResult] = []
    for N in N_VALUES_CONFIG_C:
        res = sweep_gamma_for_N(N, GAMMA_VALUES, run_config_c=True, n_jobs=N_JOBS)
        results.append(res)

    # Save γ-sweep Parquet
    save_gamma_sweep_parquet(results)

    # ── Phase 2: ω-scans at selected (γ, N) pairs ─────────────────────
    print("\n[Phase 2] Running ω-scans at selected (γ, N) pairs...")
    omega_grid = np.linspace(OMEGA_SCAN_MIN, OMEGA_SCAN_MAX, OMEGA_SCAN_COUNT)

    scan_pairs: list[tuple[int, float]] = [
        (N, gamma) for N in N_VALUES_CONFIG_A for gamma in [0.0, 0.25, 1.0]
    ]

    def _run_omega_scan(
        N: int, gamma: float, omega_grid: np.ndarray, alpha: dict[str, float]
    ) -> tuple[int, float, list[dict]]:
        """Run ω-scan for one (N, γ) pair. Parallelisation-safe."""
        use_config_c = N in N_VALUES_CONFIG_C
        rows = []
        for omega in omega_grid:
            if use_config_c:
                row = omega_scan_single_point(N, omega, gamma, alpha)
            else:
                row = _omega_scan_one_point(N, omega, gamma, alpha)
            row["gamma"] = gamma
            row["N"] = N
            rows.append(row)
        return N, gamma, rows

    # Build alpha lookup from results
    alpha_lookup: dict[int, dict[str, float]] = {}
    for res in results:
        alpha_lookup[res.N] = {
            k: float(res.alpha_opt[k][0]) for k in ["xx", "xz", "zx", "zz"]
        }

    omega_results: dict[tuple[int, float], pd.DataFrame] = {}
    scan_results = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(_run_omega_scan)(
            N,
            gamma,
            omega_grid,
            alpha_lookup.get(N, {"xx": 0.0, "xz": 0.0, "zx": 0.0, "zz": 0.0}),
        )
        for N, gamma in scan_pairs
    )
    for N, gamma, rows in scan_results:
        tag = f"N{N}-g{gamma:.4f}".replace(".", "p")
        save_omega_scan_parquet(rows, tag)
        omega_results[(N, gamma)] = pd.DataFrame(rows)
    print(f"  Completed {len(scan_results)} ω-scans")

    # ── Phase 3: Generate figures ──────────────────────────────────────
    print("\n[Phase 3] Generating figures...")
    cc_results = [r for r in results if r.N in N_VALUES_CONFIG_C]

    fig_rqfi_heatmap(cc_results)
    fig_sensitivity_vs_gamma(cc_results)
    fig_optimal_alpha_per_N(cc_results)
    fig_omega_dependence(omega_results)
    fig_measurement_gap(cc_results)
    fig_ep_ratio(cc_results)

    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print(f"\n{'=' * 65}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Raw data: {RAW_DIR}")
    print(f"Figures:  {FIG_DIR}")
    print(f"\nKey results at γ=0, ω={OMEGA_REP}:")
    for res in cc_results:
        ig0 = 0  # γ=0 is first entry
        print(
            f"  N={res.N}: R_QFI={res.r_qfi[ig0]:.4f}, "
            f"R_EP={res.r_ep[ig0]:.4f}, "
            f"Δω_EP(A)={res.delta_omega_ep_a[ig0]:.6f}, "
            f"Δω_EP(C)={res.delta_omega_ep_c[ig0]:.6f}, "
            f"α*=({res.alpha_opt['xx'][ig0]:.2f}, {res.alpha_opt['xz'][ig0]:.2f}, "
            f"{res.alpha_opt['zx'][ig0]:.2f}, {res.alpha_opt['zz'][ig0]:.2f})"
        )
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Force re-generation")
    args = parser.parse_args()
    main(force=args.force)
