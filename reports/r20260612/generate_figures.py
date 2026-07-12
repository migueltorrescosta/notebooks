"""Generate figures for report #20260612 from existing checkpoint data.

Loads N=1-8 checkpoint parquet files, creates the three standard
N-scaling plots (ratio, sensitivity, optimal params) plus a
comparison plot with the fixed J_A=1/2 ancilla data from #20260611.

Usage:
    cd /home/miguel/Git/notebooks && uv run python -m reports.r20260612.generate_figures
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.n_scaling_result import NScalingResult, NScalingScanResult
from src.visualization.scaling_plots import (
    plot_n_scaling_optimal_params,
    plot_n_scaling_ratio,
    plot_n_scaling_ratio_comparison,
    plot_n_scaling_sensitivity,
)

REPORT_DATE = "20260612"
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORT_DIR = REPORTS_DIR / f"r{REPORT_DATE}"
CHECKPOINT_DIR = REPORT_DIR / "raw_data" / "checkpoints"
FIGURES_DIR = REPORT_DIR / "figures"
DATA_DIR_20260611 = REPORTS_DIR / "r20260611" / "raw_data"


def load_checkpoints() -> NScalingScanResult:
    """Load all existing N_*.parquet checkpoints into an NScalingScanResult."""
    results: list[NScalingResult] = []
    for ckpt_file in sorted(CHECKPOINT_DIR.glob("N_*.parquet")):
        df_ckpt = pd.read_parquet(ckpt_file)
        for _, row in df_ckpt.iterrows():
            n_val = int(row["N"])
            w_val = float(row["omega"])
            delta = float(row["delta_omega_opt"])
            if not np.isfinite(delta):
                continue
            results.append(
                NScalingResult(
                    N=n_val,
                    omega=w_val,
                    delta_omega_opt=delta,
                    sql=float(row["sql"]),
                    ratio=float(row["ratio"]),
                    a_x_opt=float(row["a_x_opt"]),
                    a_y_opt=float(row["a_y_opt"]),
                    a_z_opt=float(row["a_z_opt"]),
                    a_zz_opt=float(row["a_zz_opt"]),
                    expectation_Jz=float(row.get("expectation_Jz", 0.0)),
                    variance_Jz=float(row.get("variance_Jz", 0.0)),
                    success=bool(int(row.get("success", 0))),
                    nfev=int(row.get("nfev", 0)),
                ),
            )
    print(
        f"[load] {len(results)} results from {len(list(CHECKPOINT_DIR.glob('N_*.parquet')))} checkpoint files"
    )
    return NScalingScanResult(results=results)


def load_fixed_ancilla_data() -> pd.DataFrame:
    """Load the fixed J_A=1/2 ancilla data from report #20260611."""
    parquet_p = DATA_DIR_20260611 / "20260611-n-scaling-scan.parquet"
    df = pd.read_parquet(parquet_p)
    print(f"[load] Fixed-ancilla data: {len(df)} rows from {parquet_p.name}")
    return df


def main() -> None:
    """Main entry point: load data, generate plots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load multi-particle checkpoint data
    scan_result = load_checkpoints()
    df_multi = scan_result.to_dataframe()

    # Print summary
    N_range = sorted(df_multi["N"].unique())
    omega_range = sorted(df_multi["omega"].unique())
    print(f"[info] Multi-particle data: N={N_range}, ω={omega_range}")
    print(f"[info] Total data points: {len(df_multi)}")

    # 2. Save merged Parquet for future use
    merged_parquet = REPORT_DIR / "raw_data" / "20260612-n-scaling-scan.parquet"
    scan_result.save_parquet(merged_parquet)
    print(f"[save] Merged checkpoint data -> {merged_parquet}")

    # 3. Generate the three standard plots
    print()
    print("=" * 60)
    print("  Generating standard N-scaling plots (N=1-8)")
    print("=" * 60)

    # Ratio vs N
    fig_ratio = FIGURES_DIR / "20260612-ratio-vs-n.svg"
    plot_n_scaling_ratio(df_multi, fig_ratio)
    print(f"[fig]  {fig_ratio}")

    # Sensitivity vs N (log-log)
    fig_sens = FIGURES_DIR / "20260612-sensitivity-vs-n.svg"
    plot_n_scaling_sensitivity(df_multi, fig_sens)
    print(f"[fig]  {fig_sens}")

    # Optimal params vs N
    fig_params = FIGURES_DIR / "20260612-optimal-params-vs-n.svg"
    plot_n_scaling_optimal_params(df_multi, fig_params)
    print(f"[fig]  {fig_params}")

    # 4. Comparison plot: multi-particle vs fixed ancilla
    print()
    print("=" * 60)
    print("  Generating comparison plot (multi vs fixed)")
    print("=" * 60)

    df_fixed = load_fixed_ancilla_data()

    # Filter fixed data to match the N range of multi-particle data
    max_n_multi = int(df_multi["N"].max())
    df_fixed_subset = df_fixed[df_fixed["N"] <= max_n_multi].copy()

    fig_comp = FIGURES_DIR / "20260612-comparison-multi-vs-fixed.svg"
    plot_n_scaling_ratio_comparison(df_multi, df_fixed_subset, fig_comp)
    print(f"[fig]  {fig_comp}")

    # 5. Variance and derivative analysis plot
    print()
    print("=" * 60)
    print("  Generating variance/derivative diagnostic plot")
    print("=" * 60)

    fig_var = _plot_variance_derivative(
        df_multi, FIGURES_DIR / "20260612-variance-derivative.svg"
    )
    print(f"[fig]  {fig_var}")

    print()
    print("Done. All figures saved to:", FIGURES_DIR)


def _plot_variance_derivative(
    df: pd.DataFrame,
    save_path: Path,
) -> Path:
    """Plot Var(J_z^S) and derivative |d<J_z^S>/dω| vs N.

    This helps understand why the scaling exponent remains at SQL level:
    both quantities grow with N, keeping their ratio (Δω) roughly constant.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    omega_values = sorted(df["omega"].unique())
    colours = plt.colormaps["viridis"](np.linspace(0.2, 0.9, len(omega_values)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for omega_val, colour in zip(omega_values, colours, strict=False):
        sub = df[np.isclose(df["omega"], omega_val)]
        sub = sub.sort_values("N")

        # Variance in left panel
        ax1.loglog(
            sub["N"],
            sub["variance_Jz"],
            "o-",
            color=colour,
            label=rf"$\omega={omega_val:.1f}$",
            markersize=6,
            linewidth=1.5,
        )

        # Estimate derivative from Δω and variance:
        # Δω = sqrt(Var) / |d⟨O⟩/dω|  =>  |d⟨O⟩/dω| = sqrt(Var) / Δω
        deriv = np.sqrt(sub["variance_Jz"]) / sub["delta_omega_opt"]
        ax2.loglog(
            sub["N"],
            deriv,
            "o-",
            color=colour,
            label=rf"$\omega={omega_val:.1f}$"
            if omega_val == omega_values[0]
            else None,
            markersize=6,
            linewidth=1.5,
        )

    ax1.set_xlabel(r"$N$")
    ax1.set_ylabel(r"$\mathrm{Var}(J_z^S)$")
    ax1.set_title("Measurement variance")
    ax1.legend(fontsize=8)

    ax2.set_xlabel(r"$N$")
    ax2.set_ylabel(r"$|\partial\langle J_z^S\rangle/\partial\omega|$")
    ax2.set_title("Signal derivative")
    ax2.legend(fontsize=8)

    fig.suptitle("Variance and derivative scaling (diagnostic)")
    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    main()
