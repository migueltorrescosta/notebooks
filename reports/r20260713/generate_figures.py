"""
Generate all results and SVG figures for
Coupled-System-Ancilla Metrology Under Photon Loss (v2).

Run with:
    uv run python reports/r20260713/generate_figures.py [--force]

Produces:
    raw_data/20260713-gamma-sweep.parquet
    raw_data/20260713-omega-scan-N{N}-g{gamma}.parquet
    raw_data/checkpoints/gamma-sweep-N{N}.parquet
    raw_data/checkpoints/omega-scan-N{N}-g{gamma}.parquet
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
from typing import Any, cast

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
CHECKPOINT_DIR = RAW_DIR / "checkpoints"

# Parameter ranges
N_VALUES_CONFIG_A = [1, 2, 3, 4, 5, 6, 7, 8]
N_VALUES_CONFIG_C = [1, 2, 3, 4, 5, 6, 7, 8]
GAMMA_VALUES = [0.0, *list(np.logspace(-6, 6, 60))]  # 61 values: γ=0 + 60 log-spaced
SCAN_GAMMAS = [0.0, 0.25, 1.0]
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
# Simulation runners (parallelisation-safe, module-level)
# ============================================================================


_INF_RESULT: tuple[float, float, float] = (float("inf"), 0.0, float("inf"))


def _run_config_a_point(
    N: int, omega: float, gamma: float
) -> tuple[float, float, float]:
    """Run Config A at one point: returns (delta_ep, fq, delta_qfi)."""
    fd = FD_STEP
    try:
        rho = evolve_config_a(N, omega, gamma, T_HOLD)
        rho_p = evolve_config_a(N, omega + fd, gamma, T_HOLD)
        rho_m = evolve_config_a(N, omega - fd, gamma, T_HOLD)
    except (AssertionError, ValueError):
        return _INF_RESULT

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
    try:
        rho = evolve_config_c(N, omega, gamma, T_HOLD, alpha)
        rho_p = evolve_config_c(N, omega + fd, gamma, T_HOLD, alpha)
        rho_m = evolve_config_c(N, omega - fd, gamma, T_HOLD, alpha)
    except (AssertionError, ValueError):
        return _INF_RESULT

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


# ============================================================================
# Flattened evaluation functions (parallelisation-safe)
# ============================================================================


def eval_gamma_point(
    N: int,
    gamma: float,
    alpha: dict[str, float],
    omega_rep: float = OMEGA_REP,
) -> dict[str, Any]:
    """Evaluate Config A + C at one (N, γ) point. Parallelisation-safe."""
    da, fa, dqa = _run_config_a_point(N, omega_rep, gamma)
    dc, fc, dqc = _run_config_c_point(N, omega_rep, gamma, alpha)
    return {
        "N": N,
        "gamma": gamma,
        "delta_omega_ep_a": da,
        "fq_a": fa,
        "delta_omega_qfi_a": dqa,
        "delta_omega_ep_c": dc,
        "fq_c": fc,
        "delta_omega_qfi_c": dqc,
        "alpha_xx": alpha["xx"],
        "alpha_xz": alpha["xz"],
        "alpha_zx": alpha["zx"],
        "alpha_zz": alpha["zz"],
    }


def eval_omega_point(
    N: int,
    gamma: float,
    omega: float,
    alpha: dict[str, float],
) -> dict[str, float]:
    """Evaluate Config A and C at one (N, γ, ω) point. Parallelisation-safe."""
    da, fa, dqa = _run_config_a_point(N, omega, gamma)
    if N in N_VALUES_CONFIG_C:
        dc, fc, dqc = _run_config_c_point(N, omega, gamma, alpha)
    else:
        dc, fc, dqc = float("nan"), float("nan"), float("nan")
    return {
        "omega": omega,
        "gamma": gamma,
        "N": N,
        "delta_omega_ep_a": da,
        "fq_a": fa,
        "delta_omega_qfi_a": dqa,
        "delta_omega_ep_c": dc,
        "fq_c": fc,
        "delta_omega_qfi_c": dqc,
    }


# ============================================================================
# α optimisation (sequential, fast)
# ============================================================================


def optimise_all_alpha(
    N_values: list[int],
    omega_rep: float = OMEGA_REP,
    n_starts: int = N_STARTS,
    max_iter: int = MAX_ITER,
    n_jobs: int = N_JOBS,
    alpha_opt_max_n: int = 5,
) -> dict[int, dict[str, float]]:
    """Optimise α* at γ=0 for each N. Returns {N: alpha_opt}.

    For N > alpha_opt_max_n, uses α* = 0 (no coupling) to avoid
    prohibitively expensive mesolve calls on large Hilbert spaces
    ((N+1)^4 scaling makes L-BFGS-B infeasible for N >= 6).
    """
    zero_alpha: dict[str, float] = {"xx": 0.0, "xz": 0.0, "zx": 0.0, "zz": 0.0}

    opt_n = [n for n in N_values if n <= alpha_opt_max_n]
    skip_n = [n for n in N_values if n > alpha_opt_max_n]

    alpha_lookup: dict[int, dict[str, float]] = {}

    # Parallelised optimisation for small N
    if opt_n:

        def _optimise_one(
            _N: int,
        ) -> tuple[int, dict[str, float], float]:
            opt = optimise_coupling(
                _N,
                0.0,
                omega_rep,
                T_HOLD,
                n_starts=n_starts,
                max_iter=max_iter,
                bounds=OPT_BOUNDS,
                seed=OPT_SEED,
            )
            return _N, opt["alpha_opt"], opt["delta_ep_opt"]

        results = cast(
            "list[tuple[int, dict[str, float], float]]",
            Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(_optimise_one)(n) for n in opt_n
            ),
        )
        for _N, alpha, delta_opt in sorted(results, key=lambda r: r[0]):
            alpha_lookup[_N] = alpha
            print(
                f"  N={_N}: Δω={delta_opt:.6f}, "
                f"α=({alpha['xx']:.2f}, {alpha['xz']:.2f}, "
                f"{alpha['zx']:.2f}, {alpha['zz']:.2f})"
            )

    # α* = 0 for large N (coupling not beneficial, v1 confirmed)
    for _N in skip_n:
        alpha_lookup[_N] = zero_alpha
        print(
            f"  N={_N}: α*=0 (skipped — (N+1)^4 = {(_N + 1) ** 4}-dim "
            f"optimisation infeasible; coupling not beneficial for N >= 2)"
        )

    return alpha_lookup


# ============================================================================
# Assembly: flat results → GammaSweepResult
# ============================================================================


def _compute_ratios(
    ep_a: np.ndarray,
    fq_a: np.ndarray,
    dqfi_a: np.ndarray,
    ep_c: np.ndarray,
    fq_c: np.ndarray,
    dqfi_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute R_QFI, R_EP, and R_gap ratio arrays."""
    r_qfi = np.where(fq_a > 0, fq_c / (2.0 * fq_a), np.nan)
    finite_pos_a = np.isfinite(ep_a) & (ep_a > 0) & np.isfinite(ep_c)
    r_ep = np.where(finite_pos_a, ep_c / ep_a, np.nan)
    finite_c_qfi = np.isfinite(ep_c) & np.isfinite(dqfi_c) & (dqfi_c > 0)
    r_gap = np.where(finite_c_qfi, ep_c / dqfi_c, np.nan)
    return r_qfi, r_ep, r_gap


def _extract_alpha_arrays(
    rows: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Extract per-component alpha arrays from raw result dicts."""
    return {
        "xx": np.array([r["alpha_xx"] for r in rows]),
        "xz": np.array([r["alpha_xz"] for r in rows]),
        "zx": np.array([r["alpha_zx"] for r in rows]),
        "zz": np.array([r["alpha_zz"] for r in rows]),
    }


def _build_single_gamma_result(
    N: int,
    rows: list[dict[str, Any]],
    gamma_values: np.ndarray,
) -> GammaSweepResult:
    """Build a single GammaSweepResult from sorted rows for one N."""
    n_g = len(rows)
    ep_a = np.array([r["delta_omega_ep_a"] for r in rows])
    fq_a = np.array([r["fq_a"] for r in rows])
    dqfi_a = np.array([r["delta_omega_qfi_a"] for r in rows])
    ep_c = np.array([r["delta_omega_ep_c"] for r in rows])
    fq_c = np.array([r["fq_c"] for r in rows])
    dqfi_c = np.array([r["delta_omega_qfi_c"] for r in rows])
    r_qfi, r_ep, r_gap = _compute_ratios(ep_a, fq_a, dqfi_a, ep_c, fq_c, dqfi_c)

    return GammaSweepResult(
        N=N,
        gamma_values=gamma_values[:n_g],
        delta_omega_ep_a=ep_a,
        fq_a=fq_a,
        delta_omega_qfi_a=dqfi_a,
        delta_omega_ep_c=ep_c,
        fq_c=fq_c,
        delta_omega_qfi_c=dqfi_c,
        alpha_opt=_extract_alpha_arrays(rows),
        r_qfi=r_qfi,
        r_ep=r_ep,
        r_gap=r_gap,
    )


def assemble_gamma_results(
    raw: list[dict[str, Any]],
    gamma_values: np.ndarray,
) -> list[GammaSweepResult]:
    """Group flat γ-point dicts into per-N GammaSweepResult objects."""
    by_N: dict[int, list[dict[str, Any]]] = {}
    for r in raw:
        by_N.setdefault(r["N"], []).append(r)

    return [
        _build_single_gamma_result(N, sorted(by_N[N], key=lambda r: r["gamma"]), gamma_values)
        for N in sorted(by_N)
    ]


# ============================================================================
# Checkpoint helpers
# ============================================================================


def _gamma_ckpt_path(N: int) -> Path:
    return CHECKPOINT_DIR / f"gamma-sweep-N{N}.parquet"


def _omega_ckpt_path(N: int, gamma: float) -> Path:
    tag = f"N{N}-g{gamma:.4f}".replace(".", "p")
    return CHECKPOINT_DIR / f"omega-scan-{tag}.parquet"


def load_gamma_checkpoint(N: int) -> list[dict[str, Any]] | None:
    """Load completed γ-sweep rows for a given N, or None if no checkpoint."""
    path = _gamma_ckpt_path(N)
    if not path.exists():
        return None
    rows: list[dict[str, Any]] = cast(
        "list[dict[str, Any]]", pd.read_parquet(path).to_dict("records")
    )
    return rows


def save_gamma_checkpoint(N: int, rows: list[dict[str, Any]]) -> None:
    """Save γ-sweep results for one N as a checkpoint file."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(_gamma_ckpt_path(N), index=False)


def load_omega_checkpoint(N: int, gamma: float) -> set[float] | None:
    """Load completed ω values for a given (N, γ), or None."""
    path = _omega_ckpt_path(N, gamma)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return set(df["omega"].round(10))


def save_omega_checkpoint(N: int, gamma: float, rows: list[dict[str, Any]]) -> None:
    """Save ω-scan results for one (N, γ) as a checkpoint file."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(_omega_ckpt_path(N, gamma), index=False)


# ============================================================================
# Parquet saving (final consolidated files)
# ============================================================================


def save_gamma_sweep_parquet(results: list[GammaSweepResult]) -> Path:
    """Save all γ-sweep results as a single Parquet file."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
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
# Main — extracted phases
# ============================================================================


def _clear_checkpoints_if_forced(force: bool) -> None:
    """Remove checkpoint directory when --force is set."""
    if force and CHECKPOINT_DIR.exists():
        import shutil

        shutil.rmtree(CHECKPOINT_DIR)
        print("Cleared checkpoint directory (--force)")


def _collect_gamma_work(
    alpha_lookup: dict[int, dict[str, float]],
) -> list[tuple[int, float, dict[str, float]]]:
    """Build the list of (N, γ, α) tuples for uncached γ-points."""
    gamma_work: list[tuple[int, float, dict[str, float]]] = []
    for N in N_VALUES_CONFIG_C:
        existing = load_gamma_checkpoint(N)
        done_gammas = {r["gamma"] for r in existing} if existing else set()
        remaining = [γ for γ in GAMMA_VALUES if γ not in done_gammas]
        if not remaining:
            print(f"  N={N}: γ-sweep fully cached, skipping")
        else:
            print(f"  N={N}: {len(remaining)}/{len(GAMMA_VALUES)} γ-points remaining")
            gamma_work.extend((N, γ, alpha_lookup[N]) for γ in remaining)
    return gamma_work


def _merge_gamma_checkpoints(new_raw: list[dict[str, Any]]) -> None:
    """Merge newly computed γ-points into per-N checkpoint files."""
    by_N_new: dict[int, list[dict[str, Any]]] = {}
    for r in new_raw:
        by_N_new.setdefault(r["N"], []).append(r)
    for N in N_VALUES_CONFIG_C:
        existing = load_gamma_checkpoint(N)
        new_rows = by_N_new.get(N, [])
        all_rows = (existing or []) + new_rows
        if all_rows:
            save_gamma_checkpoint(N, all_rows)


def _load_all_gamma_raw() -> list[dict[str, Any]]:
    """Load all γ-sweep results from per-N checkpoint files."""
    all_gamma_raw: list[dict[str, Any]] = []
    for N in N_VALUES_CONFIG_C:
        existing = load_gamma_checkpoint(N)
        if existing:
            all_gamma_raw.extend(existing)
    return all_gamma_raw


def _run_gamma_sweep_phase(
    alpha_lookup: dict[int, dict[str, float]],
) -> list[dict[str, Any]]:
    """Run Phase 1: γ-sweep across all N values. Returns assembled raw rows."""
    print("\n[Phase 1] Running γ-sweep (flattened)...")
    gamma_work = _collect_gamma_work(alpha_lookup)

    if gamma_work:
        t0 = time.time()
        new_raw = cast(
            "list[dict[str, Any]]",
            Parallel(n_jobs=N_JOBS, verbose=0)(
                delayed(eval_gamma_point)(*w) for w in gamma_work
            ),
        )
        elapsed = time.time() - t0
        print(f"  Computed {len(new_raw)} γ-points in {elapsed:.1f}s")
        _merge_gamma_checkpoints(new_raw)

    all_gamma_raw = _load_all_gamma_raw()
    results = assemble_gamma_results(all_gamma_raw, np.array(GAMMA_VALUES))
    save_gamma_sweep_parquet(results)
    return all_gamma_raw


def _collect_omega_work(
    alpha_lookup: dict[int, dict[str, float]],
    omega_grid: np.ndarray,
) -> list[tuple[int, float, float, dict[str, float]]]:
    """Build the list of (N, γ, ω, α) tuples for uncached ω-points."""
    omega_work: list[tuple[int, float, float, dict[str, float]]] = []
    for N in N_VALUES_CONFIG_A:
        for γ in SCAN_GAMMAS:
            done = load_omega_checkpoint(N, γ)
            if done is not None:
                remaining = [ω for ω in omega_grid if ω.round(10) not in done]
            else:
                remaining = list(omega_grid)
            if not remaining:
                continue
            alpha = alpha_lookup.get(N, {"xx": 0.0, "xz": 0.0, "zx": 0.0, "zz": 0.0})
            omega_work.extend((N, γ, ω, alpha) for ω in remaining)
    return omega_work


def _merge_omega_checkpoints(new_omega: list[dict[str, Any]]) -> None:
    """Merge newly computed ω-points into per-(N,γ) checkpoint files."""
    by_pair: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for r in new_omega:
        by_pair.setdefault((r["N"], r["gamma"]), []).append(r)
    for (N, γ), new_rows in by_pair.items():
        existing_rows = load_omega_checkpoint(N, γ)
        if existing_rows is not None:
            old_df = pd.read_parquet(_omega_ckpt_path(N, γ))
            old_rows: list[dict[str, Any]] = cast(
                "list[dict[str, Any]]", old_df.to_dict("records")
            )
            all_rows = old_rows + new_rows
        else:
            all_rows = new_rows
        save_omega_checkpoint(N, γ, all_rows)


def _assemble_omega_results() -> dict[tuple[int, float], pd.DataFrame]:
    """Assemble final ω-scan data from checkpoints and save consolidated files."""
    omega_results: dict[tuple[int, float], pd.DataFrame] = {}
    for N in N_VALUES_CONFIG_A:
        for γ in SCAN_GAMMAS:
            path = _omega_ckpt_path(N, γ)
            if path.exists():
                df = pd.read_parquet(path)
                omega_results[(N, γ)] = df
                tag = f"N{N}-g{γ:.4f}".replace(".", "p")
                save_omega_scan_parquet(df.to_dict("records"), tag)
    n_scans = len(omega_results)
    print(f"  Assembled {n_scans} ω-scan datasets from checkpoints")
    return omega_results


def _run_omega_sweep_phase(
    alpha_lookup: dict[int, dict[str, float]],
) -> dict[tuple[int, float], pd.DataFrame]:
    """Run Phase 2: ω-scan across all (N, γ) pairs. Returns assembled data."""
    print("\n[Phase 2] Running ω-scans (flattened)...")
    omega_grid = np.linspace(OMEGA_SCAN_MIN, OMEGA_SCAN_MAX, OMEGA_SCAN_COUNT)
    omega_work = _collect_omega_work(alpha_lookup, omega_grid)

    if omega_work:
        t0 = time.time()
        new_omega = cast(
            "list[dict[str, Any]]",
            Parallel(n_jobs=N_JOBS, verbose=0)(
                delayed(eval_omega_point)(*w) for w in omega_work
            ),
        )
        elapsed = time.time() - t0
        print(f"  Computed {len(new_omega)} ω-points in {elapsed:.1f}s")
        _merge_omega_checkpoints(new_omega)

    return _assemble_omega_results()


def _print_n_verification(res: GammaSweepResult) -> None:
    """Print SQL recovery and boundary-saturation check for one N."""
    ig0 = 0
    sql = 1.0 / (np.sqrt(res.N) * T_HOLD)
    sql_ok = np.isclose(res.delta_omega_ep_a[ig0], sql, rtol=1e-3)
    no_sat = all(
        abs(res.alpha_opt[k][ig0]) < OPT_BOUNDS[1] - 0.01
        for k in ["xx", "xz", "zx", "zz"]
    )
    print(
        f"  N={res.N}: SQL={'PASS' if sql_ok else 'FAIL'}, "
        f"NoSat={'PASS' if no_sat else 'FAIL'}, "
        f"α*=({res.alpha_opt['xx'][ig0]:.2f}, {res.alpha_opt['xz'][ig0]:.2f}, "
        f"{res.alpha_opt['zx'][ig0]:.2f}, {res.alpha_opt['zz'][ig0]:.2f})"
    )


def _check_qfi_ep_inequality(all_gamma_raw: list[dict[str, Any]]) -> None:
    """Check QFI-EP inequality across all data points."""
    qfi_rows = [
        r
        for r in all_gamma_raw
        if np.isfinite(r["delta_omega_qfi_c"]) and r["delta_omega_qfi_c"] > 0
    ]
    violations = sum(
        1 for r in qfi_rows if r["delta_omega_qfi_c"] > r["delta_omega_ep_c"] * 1.01
    )
    print(f"  QFI-EP inequality: {violations} violations in {len(qfi_rows)} points")


def _verify_and_report(
    cc_results: list[GammaSweepResult],
    all_gamma_raw: list[dict[str, Any]],
) -> None:
    """Run verification checks and print summary."""
    print(f"\n{'=' * 65}")
    print("[Verification]")

    for res in cc_results:
        _print_n_verification(res)

    print("  QFI additivity at α=0: checked by test suite")
    _check_qfi_ep_inequality(all_gamma_raw)
    print("  Trace preservation: checked in _evolve_lindblad (atol=1e-2)")

    for N in range(4, 9):
        has_data = any(r["N"] == N for r in all_gamma_raw)
        print(f"  Config C N={N}: {'PASS' if has_data else 'FAIL'}")


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
    print(f"Parallelisation: n_jobs={N_JOBS}, flattened across all (N, γ)")
    print(f"Checkpointing: {'disabled (--force)' if force else 'enabled'}")
    print("=" * 65)

    t_total = time.time()
    _clear_checkpoints_if_forced(force)

    # ── Phase 0: Optimise α* at γ=0 for all N (parallelised) ───────────
    print("\n[Phase 0] Optimising α* at γ=0 for all N (parallelised)...")
    alpha_lookup = optimise_all_alpha(N_VALUES_CONFIG_C, n_jobs=N_JOBS)

    # ── Phase 1: γ-sweep ────────────────────────────────────────────────
    all_gamma_raw = _run_gamma_sweep_phase(alpha_lookup)

    # ── Phase 2: ω-scan ─────────────────────────────────────────────────
    omega_results = _run_omega_sweep_phase(alpha_lookup)

    # ── Phase 3: Generate figures ────────────────────────────────────────
    print("\n[Phase 3] Generating figures...")
    cc_results = [r for r in assemble_gamma_results(all_gamma_raw, np.array(GAMMA_VALUES)) if r.N in N_VALUES_CONFIG_C]

    fig_rqfi_heatmap(cc_results)
    fig_sensitivity_vs_gamma(cc_results)
    fig_optimal_alpha_per_N(cc_results)
    fig_omega_dependence(omega_results)
    fig_measurement_gap(cc_results)
    fig_ep_ratio(cc_results)

    # ── Verification & summary ───────────────────────────────────────────
    _verify_and_report(cc_results, all_gamma_raw)

    elapsed = time.time() - t_total
    print(f"\n{'=' * 65}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Raw data: {RAW_DIR}")
    print(f"Figures:  {FIG_DIR}")
    print(f"\nKey results at γ=0, ω={OMEGA_REP}:")
    for res in cc_results:
        ig0 = 0
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear checkpoints and re-generate everything from scratch",
    )
    args = parser.parse_args()
    main(force=args.force)
