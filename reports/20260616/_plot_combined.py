"""
Combined comparison figure: Step 2 (J_A=1/2) vs Step 3 (J_A=N/2) scaling.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORT_DATE = "20260616"


def load_data(name: str) -> pd.DataFrame:
    p = REPORTS_DIR / REPORT_DATE / "raw_data" / f"{REPORT_DATE}-{name}.parquet"
    return pd.read_parquet(p)


def plot_combined_comparison() -> None:
    df2 = load_data("step2-n-scaling")
    df3 = load_data("step3-n-scaling")

    save_path = (
        REPORTS_DIR
        / REPORT_DATE
        / "figures"
        / f"{REPORT_DATE}-step2-vs-step3-ratio.svg"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Ratio comparison
    ax = axes[0]
    omega_vals = sorted(set(df2["omega"].unique()) & set(df3["omega"].unique()))
    colours = plt.colormaps["viridis"](np.linspace(0.2, 0.9, len(omega_vals)))

    for omega_val, colour in zip(omega_vals, colours, strict=False):
        sub2 = df2[np.isclose(df2["omega"], omega_val)].sort_values("N")
        sub3 = df3[np.isclose(df3["omega"], omega_val)].sort_values("N")
        valid2 = np.isfinite(sub2["ratio"])
        valid3 = np.isfinite(sub3["ratio"])
        ax.plot(
            sub2["N"][valid2],
            sub2["ratio"][valid2],
            "o-",
            color=colour,
            markersize=5,
            linewidth=1.2,
            label=rf"$J_A=1/2$, $\omega={omega_val:.1f}$",
        )
        ax.plot(
            sub3["N"][valid3],
            sub3["ratio"][valid3],
            "s--",
            color=colour,
            markersize=5,
            linewidth=1.2,
            label=rf"$J_A=N/2$, $\omega={omega_val:.1f}$",
        )

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="SQL (R=1)")
    ax.set_xlabel(r"$N$ (system particles)")
    ax.set_ylabel(r"$R(N) = \Delta\omega_{\mathrm{SQL}} / \Delta\omega_{\mathrm{opt}}$")
    ax.set_title("SQL-violation ratio: $J_A=1/2$ vs $J_A=N/2$")
    ax.legend(fontsize=8, ncol=2)

    # Panel 2: Sensitivity comparison (log-log)
    ax = axes[1]
    for omega_val, colour in zip(omega_vals, colours, strict=False):
        sub2 = df2[np.isclose(df2["omega"], omega_val)].sort_values("N")
        sub3 = df3[np.isclose(df3["omega"], omega_val)].sort_values("N")
        valid2 = np.isfinite(sub2["delta_omega_opt"] * sub2["delta_omega_opt"] > 0)
        valid3 = np.isfinite(sub3["delta_omega_opt"] * sub3["delta_omega_opt"] > 0)
        ax.loglog(
            sub2["N"][valid2],
            sub2["delta_omega_opt"][valid2],
            "o-",
            color=colour,
            markersize=5,
            linewidth=1.2,
            label=rf"$J_A=1/2$, $\omega={omega_val:.1f}$",
        )
        ax.loglog(
            sub3["N"][valid3],
            sub3["delta_omega_opt"][valid3],
            "s--",
            color=colour,
            markersize=5,
            linewidth=1.2,
        )

    N_range = np.linspace(1, max(*df2["N"], *df3["N"]), 100)
    sql_line = 1.0 / (np.sqrt(N_range) * 10)
    hl_line = 1.0 / (N_range * 10)
    ax.loglog(N_range, sql_line, "k--", alpha=0.7, linewidth=1.5, label="SQL")
    ax.loglog(N_range, hl_line, "k:", alpha=0.5, linewidth=1.2, label="HL")
    ax.set_xlabel(r"$N$ (system particles)")
    ax.set_ylabel(r"$\Delta\omega_{\mathrm{opt}}$")
    ax.set_title("Optimal sensitivity: $J_A=1/2$ vs $J_A=N/2$")
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {save_path}")


def plot_scaling_exponents() -> None:
    """Plot scaling exponents from the analysis."""
    try:
        df = pd.read_parquet(
            REPORTS_DIR
            / REPORT_DATE
            / "raw_data"
            / f"{REPORT_DATE}-scaling-analysis.parquet"
        )
    except FileNotFoundError:
        print("[skip] No scaling analysis data")
        return

    save_path = (
        REPORTS_DIR / REPORT_DATE / "figures" / f"{REPORT_DATE}-scaling-exponents.svg"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for scan_type, marker, label in [
        ("step2_JA_half", "o", r"$J_A=1/2$"),
        ("step3_JA_halfN", "s", r"$J_A=N/2$"),
    ]:
        sub = df[df["scan"] == scan_type].sort_values("omega")
        ax.plot(
            sub["omega"],
            sub["alpha"],
            f"{marker}-",
            markersize=8,
            linewidth=1.5,
            label=label,
        )

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5, label=r"$\alpha=0$ (flat)")
    ax.axhline(
        y=-0.5, color="gray", linestyle="--", alpha=0.5, label=r"SQL: $\alpha=-0.5$"
    )
    ax.axhline(
        y=-1.0, color="gray", linestyle="-.", alpha=0.3, label=r"HL: $\alpha=-1.0$"
    )

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"Scaling exponent $\alpha$ ($\Delta\omega \propto N^\alpha$)")
    ax.set_title(r"N-scaling exponent $\alpha$ vs $\omega$")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {save_path}")


if __name__ == "__main__":
    plot_combined_comparison()
    plot_scaling_exponents()
