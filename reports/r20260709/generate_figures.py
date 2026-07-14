"""
Figure generation for report 20260709 (Symmetric ω-Modulated Drive:
Bounded-Compound Comparison).

Reads existing Parquet data files and generates SVG figures with
correct LaTeX rendering in titles and legends.  Uses raw strings for all
mathtext to prevent the matplotlib backslash/mathtext corruption issue
observed in the originally generated SVGs.

Usage:
    uv run python reports/r20260709/generate_figures.py

    Optionally re-generate data first:
    uv run python reports/r20260709/compound_comparison.py --force
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path

# Ensure the project root is on sys.path so that ``src.*`` imports work.
# The report directory name (20260709) is not a valid Python identifier,
# so we add its directory to sys.path for the sibling import below.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
_REPORT_DIR = Path(__file__).resolve().parent
if str(_REPORT_DIR) not in _sys.path:
    _sys.path.insert(0, str(_REPORT_DIR))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

# Import from the sibling module (same directory, valid path via sys.path)
from compound_comparison import (  # noqa: E402
    CompoundRatioResult,
    ScenarioACompoundResult,
    _fig_path,
    _parquet_path,
)

from src.analysis.ancilla_drive_results import DriveOmegaScanResult  # noqa: E402

# Force non-interactive backend (safe even if already set)
mpl.use("Agg")
sns.set_theme(style="whitegrid")

# Ensure matplotlib's mathtext parser is used (not usetex) for SVG output
plt.rcParams.update(
    {
        "text.usetex": False,
        "svg.fonttype": "path",  # embed fonts as paths for portability
    }
)

# ──────────────────────────────────────────────
# Helper: load results
# ──────────────────────────────────────────────


def _load_scenario_a() -> ScenarioACompoundResult:
    return ScenarioACompoundResult.from_parquet(_parquet_path("scenario-a-omega-scan"))


def _load_scenario_b() -> DriveOmegaScanResult:
    return DriveOmegaScanResult.from_parquet(_parquet_path("scenario-b-omega-scan"))


def _load_compound_ratio() -> CompoundRatioResult:
    return CompoundRatioResult.from_parquet(_parquet_path("compound-ratio"))


# ──────────────────────────────────────────────
# Helper: adaptive marker interval
# ──────────────────────────────────────────────


def _markevery(n_pts: int, target: int = 30) -> int:
    """Compute marker interval to show roughly *target* markers.

    For 500 points shows ~25 markers; for 50 points shows ~25 markers.
    """
    return max(1, n_pts // target)


# ──────────────────────────────────────────────
# Figure 1: Scenario A ω scan
# ──────────────────────────────────────────────


def plot_scenario_a_omega_scan(
    result: ScenarioACompoundResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 7),
) -> Path:
    """Two-panel figure: Δω vs ω (top) and Δω/SQL ratio (bottom).

    Title: "Scenario A: System-only ω-modulated drive"
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    omega = result.omega_values
    delta = result.best_delta_omega_per_omega
    sql = result.sql_values[0]

    n_omega = len(omega)
    me = _markevery(n_omega)
    # ── Upper panel: Δω vs ω ──
    ax1.axhline(y=sql, color="C1", linestyle="--", alpha=0.6, label=f"SQL = {sql:.4f}")
    ax1.plot(
        omega,
        delta,
        marker="o",
        markevery=me,
        linestyle="-",
        color="C0",
        markersize=4,
        linewidth=2,
        label=r"$\Delta\omega$ (best)",
    )

    below_sql = delta < sql
    if np.any(below_sql):
        ax1.scatter(
            omega[below_sql],
            delta[below_sql],
            marker="*",
            s=150,
            color="red",
            zorder=5,
            label="Below SQL",
        )

    ax1.set_ylabel(r"$\Delta\omega$")
    ax1.set_title(r"Scenario A: System-only $\omega$-modulated drive")
    ax1.legend()

    # ── Lower panel: ratio Δω / SQL ──
    ratio = delta / sql
    ax2.plot(
        omega,
        ratio,
        marker="o",
        markevery=me,
        linestyle="-",
        color="C0",
        markersize=4,
        linewidth=2,
    )
    ax2.axhline(y=1.0, color="C1", linestyle="--", alpha=0.6, label="SQL")

    min_idx = int(np.argmin(ratio))
    min_ratio = float(ratio[min_idx])
    min_omega = float(omega[min_idx])
    ax2.annotate(
        rf"Best = {min_ratio:.3f}$\times$ at $\omega$={min_omega:.1f}",
        xy=(min_omega, min_ratio),
        xytext=(min_omega + 0.6, min_ratio + 0.15),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "gray"},
    )
    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel(r"$\Delta\omega \;/\; \mathrm{SQL}$")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# Figure 2: Scenario A optimal parameters (3D scatter)
# ──────────────────────────────────────────────


def plot_scenario_a_optimal_params_3d(
    result: ScenarioACompoundResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 8),
) -> Path:
    """3D scatter of optimal (a_x, a_y, a_z) coloured by sensitivity."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    a_x = np.array([p[0] for p in result.best_params_per_omega])
    a_y = np.array([p[1] for p in result.best_params_per_omega])
    a_z = np.array([p[2] for p in result.best_params_per_omega])
    delta = result.best_delta_omega_per_omega

    scatter = ax.scatter(
        a_x, a_y, a_z,
        c=delta, cmap="viridis", s=40, alpha=0.8,
        edgecolors="none",
    )
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(r"$\Delta\omega$")

    ax.set_xlabel(r"$a_x^*$")
    ax.set_ylabel(r"$a_y^*$")
    ax.set_zlabel(r"$a_z^*$")
    ax.set_title(r"Scenario A: Optimal $(a_x, a_y, a_z)$ coloured by $\Delta\omega$")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# Figure 3: Scenario B ω scan
# ──────────────────────────────────────────────


def plot_scenario_b_omega_scan(
    result: DriveOmegaScanResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 7),
) -> Path:
    """Two-panel figure: Δω vs ω (top) and Δω/SQL ratio (bottom).

    Title: "Scenario B: Ancilla-assisted identical ω-modulated drive"
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    omega = result.omega_values
    delta = result.best_delta_omega_per_omega
    sql = result.sql_values[0]
    me = _markevery(len(omega))

    # ── Upper panel ──
    ax1.axhline(y=sql, color="C1", linestyle="--", alpha=0.6, label=f"SQL = {sql:.4f}")
    ax1.plot(
        omega,
        delta,
        marker="o",
        markevery=me,
        linestyle="-",
        color="C0",
        markersize=4,
        linewidth=2,
        label=r"$\Delta\omega$ (best)",
    )

    below_sql = delta < sql
    if np.any(below_sql):
        ax1.scatter(
            omega[below_sql],
            delta[below_sql],
            marker="*",
            s=150,
            color="red",
            zorder=5,
            label="Below SQL",
        )

    ax1.set_ylabel(r"$\Delta\omega$")
    ax1.set_title(r"Scenario B: Ancilla-assisted identical $\omega$-modulated drive")
    ax1.legend()

    # ── Lower panel ──
    ratio = delta / sql
    ax2.plot(
        omega,
        ratio,
        marker="o",
        markevery=me,
        linestyle="-",
        color="C0",
        markersize=4,
        linewidth=2,
    )
    ax2.axhline(y=1.0, color="C1", linestyle="--", alpha=0.6, label="SQL")

    min_idx = int(np.argmin(ratio))
    min_ratio = float(ratio[min_idx])
    min_omega = float(omega[min_idx])
    ax2.annotate(
        rf"Best = {min_ratio:.3f}$\times$ at $\omega$={min_omega:.1f}",
        xy=(min_omega, min_ratio),
        xytext=(min_omega + 0.6, min_ratio + 0.15),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "gray"},
    )
    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel(r"$\Delta\omega \;/\; \mathrm{SQL}$")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# Figure 4: Scenario B optimal parameters (3D scatter)
# ──────────────────────────────────────────────


def plot_scenario_b_optimal_params_3d(
    result: DriveOmegaScanResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 8),
) -> Path:
    """3D scatter of optimal (a_x, a_z, a_zz) coloured by sensitivity.

    a_y is omitted from the visualisation — it does not contribute to the
    signal (see Analytical Bounds section).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    a_x = np.array([p[0] for p in result.best_params_per_omega])
    a_z = np.array([p[2] for p in result.best_params_per_omega])
    a_zz = np.array([p[3] for p in result.best_params_per_omega])
    delta = result.best_delta_omega_per_omega

    scatter = ax.scatter(
        a_x, a_z, a_zz,
        c=delta, cmap="viridis", s=40, alpha=0.8,
        edgecolors="none",
    )
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(r"$\Delta\omega$")

    ax.set_xlabel(r"$a_x^*$")
    ax.set_ylabel(r"$a_z^*$")
    ax.set_zlabel(r"$a_{zz}^*$")
    ax.set_title(
        r"Scenario B: Optimal $(a_x, a_z, a_{zz})$ coloured by $\Delta\omega$"
    )

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# Figure 5: Compound ratio
# ──────────────────────────────────────────────


def plot_compound_ratio(
    cr: CompoundRatioResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """R_compound = Δω_A / Δω_B vs ω.

    Title: "Symmetric ω-Modulated Drive: Scenario A vs Scenario B"
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    omega = cr.omega_values
    ratio = cr.compound_ratio
    me = _markevery(len(omega))

    ax.plot(
        omega,
        ratio,
        marker="o",
        markevery=me,
        linestyle="-",
        color="C0",
        markersize=4,
        linewidth=2,
        label=r"$\mathcal{R}_{\mathrm{compound}} = \Delta\omega_A / \Delta\omega_B$",
    )
    ax.axhline(
        y=1.0, color="gray", linestyle="--", alpha=0.7, linewidth=1.5, label="y = 1"
    )

    # Mark the best compound ratio
    valid = np.isfinite(ratio)
    if np.any(valid):
        best_idx = int(np.nanargmax(np.where(valid, ratio, 0.0)))
        best_r = float(ratio[best_idx])
        best_w = float(omega[best_idx])
        ax.annotate(
            rf"Best = {best_r:.4f}$\times$ at $\omega$={best_w:.2f}",
            xy=(best_w, best_r),
            xytext=(best_w + 0.6, best_r + 0.08),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "gray",
            },
        )

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\mathcal{R}_{\mathrm{compound}}$")
    ax.set_title(r"Symmetric $\omega$-Modulated Drive: Scenario A vs Scenario B")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# Figure 6: SQL-violation ratio comparison
# ──────────────────────────────────────────────


def plot_sql_violation_ratio(
    cr: CompoundRatioResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """R_A = SQL/Δω_A and R_B = SQL/Δω_B vs ω.

    Title: "SQL-violation ratio: Scenario A vs Scenario B"
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    omega = cr.omega_values
    r_a = cr.ratio_A_to_sql
    r_b = cr.ratio_B_to_sql
    me = _markevery(len(omega))

    ax.plot(
        omega,
        r_a,
        "o-",
        markevery=me,
        color="C0",
        markersize=4,
        linewidth=1.8,
        label=r"Scenario A: $\Delta\omega_{\mathrm{SQL}} / \Delta\omega_A$",
    )
    ax.plot(
        omega,
        r_b,
        "s-",
        markevery=me,
        color="C3",
        markersize=4,
        linewidth=1.8,
        label=r"Scenario B: $\Delta\omega_{\mathrm{SQL}} / \Delta\omega_B$",
    )
    ax.axhline(
        y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1.2, label="SQL"
    )

    # Highlight best point for each
    valid_a = np.isfinite(r_a)
    if np.any(valid_a):
        best_a_idx = int(np.nanargmax(np.where(valid_a, r_a, 0.0)))
        best_a_r = float(r_a[best_a_idx])
        best_a_w = float(omega[best_a_idx])
        ax.annotate(
            rf"Best A = {best_a_r:.2f}$\times$ at $\omega$={best_a_w:.2f}",
            xy=(best_a_w, best_a_r),
            xytext=(best_a_w + 0.5, best_a_r + 0.5),
            arrowprops={"arrowstyle": "->", "color": "C0", "lw": 1.2},
            fontsize=9,
            color="C0",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "C0"},
        )
    valid_b = np.isfinite(r_b)
    if np.any(valid_b):
        best_b_idx = int(np.nanargmax(np.where(valid_b, r_b, 0.0)))
        best_b_r = float(r_b[best_b_idx])
        best_b_w = float(omega[best_b_idx])
        ax.annotate(
            rf"Best B = {best_b_r:.2f}$\times$ at $\omega$={best_b_w:.2f}",
            xy=(best_b_w, best_b_r),
            xytext=(best_b_w + 0.5, best_b_r - 0.8),
            arrowprops={"arrowstyle": "->", "color": "C3", "lw": 1.2},
            fontsize=9,
            color="C3",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "C3"},
        )

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\Delta\omega_{\mathrm{SQL}} \;/\; \Delta\omega$")
    ax.set_title(r"SQL-violation ratio: Scenario A vs Scenario B")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────


def generate_all_figures(force: bool = False) -> None:
    """Generate all SVG figures from existing Parquet data.

    Args:
        force: If True, overwrite existing SVG files.
    """
    print("Loading data...")
    result_a = _load_scenario_a()
    result_b = _load_scenario_b()
    cr = _load_compound_ratio()

    figures = [
        ("scenario-a-omega-scan", plot_scenario_a_omega_scan, [result_a]),
        ("scenario-a-optimal-params", plot_scenario_a_optimal_params_3d, [result_a]),
        ("scenario-b-omega-scan", plot_scenario_b_omega_scan, [result_b]),
        ("scenario-b-optimal-params", plot_scenario_b_optimal_params_3d, [result_b]),
        ("compound-ratio", plot_compound_ratio, [cr]),
        ("sql-violation-ratio", plot_sql_violation_ratio, [cr]),
    ]

    for tag, plot_fn, args in figures:
        svg_path = _fig_path(tag)
        if svg_path.exists() and not force:
            print(f"  [skip] {svg_path.name} exists")
            continue
        print(f"  [plot] {svg_path.name} ...", end=" ", flush=True)
        plot_fn(*args, svg_path)
        print("done")

    print("All figures generated.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate figures for report 20260709")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing SVG files"
    )
    args = parser.parse_args()

    generate_all_figures(force=args.force)


if __name__ == "__main__":
    main()
