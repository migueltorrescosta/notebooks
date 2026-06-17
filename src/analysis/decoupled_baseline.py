"""
Decoupled baseline verification — shared orchestration, visualisation, and
verification.

Provides ``generate_decoupled_baseline`` (cache-check → compute → save → plot
pipeline), ``verify_decoupled_baseline`` (parameterised verification that
decoupled configurations recover SQL sensitivity), and
``plot_decoupled_baseline_heatmap`` (|ratio-1| log-scale heatmap).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from src.analysis.sensitivity_metrics import sql_reference
from src.physics.n_particle_drive import compute_n_particle_decoupled_baseline
from src.utils.serialization import ParquetSerializable


def verify_decoupled_baseline(
    N_values: list[int] | None = None,
    omega_values: list[float] | None = None,
    rtol: float = 1e-10,
    *,
    compute_fn: Callable[[int, float], float] = compute_n_particle_decoupled_baseline,
    **compute_kwargs: Any,
) -> dict[tuple[int, float], bool]:
    """Verify the decoupled baseline recovers SQL sensitivity.

    At zero drive and zero interaction, the sensitivity must equal
    Δω = 1/(√N × T_HOLD) to within the given tolerance.

    The ``compute_fn`` callback allows reports with different decoupled-baseline
    computation functions (e.g., combined 4-parameter+drive models) to reuse
    the same verification loop. Extra keyword arguments are forwarded to
    ``compute_fn(N, omega, **compute_kwargs)``.

    Args:
        N_values: List of N values. Defaults to ``list(range(1, 21))``.
        omega_values: List of ω values. Defaults to
            ``[0.1, 0.2, 0.5, 1.0, 2.0]``.
        compute_fn: Function ``(N, omega) → Δω`` at decoupled parameters.
            Default is ``compute_n_particle_decoupled_baseline``.
        rtol: Relative tolerance for ``np.isclose`` comparison with SQL.
        **compute_kwargs: Forwarded to ``compute_fn``.

    Returns:
        Dict mapping ``(N, ω) → True`` (PASS) or ``False`` (FAIL).
    """
    if N_values is None:
        N_values = list(range(1, 21))
    if omega_values is None:
        omega_values = [0.1, 0.2, 0.5, 1.0, 2.0]

    results: dict[tuple[int, float], bool] = {}
    for N in N_values:
        sql_ref = sql_reference(N)
        for omega in omega_values:
            delta = compute_fn(N, omega, **compute_kwargs)
            results[(N, omega)] = bool(np.isclose(delta, sql_ref, rtol=rtol))
    return results


def generate_decoupled_baseline(
    force: bool = False,
    *,
    parquet_path: Path,
    fig_path: Path | None = None,
    compute_fn: Callable[..., Any],
    compute_args: tuple = (),
    compute_kwargs: dict | None = None,
    result_cls: type[ParquetSerializable] | None = None,
    plot_fn: Callable[..., Any] | None = None,
    label: str = "decoupled baseline",
) -> Any | None:
    """Generate the decoupled baseline verification.

    Orchestrates the standard cache-check → compute → save → plot pipeline
    used by all eight reports that define this function locally.

    Two cache behaviours are supported:

    * **With** ``result_cls`` — on cache hit the result is loaded via
      ``result_cls.from_parquet(parquet_path)`` and returned.
    * **Without** ``result_cls`` (``None``) — on cache hit ``None`` is
      returned (the caller does not need the loaded data).

    Args:
        force: Recompute even if a cached parquet exists.
        parquet_path: Path to the result parquet file.
        fig_path: Optional path for the figure.
        compute_fn: Function that computes the result.
        compute_args: Positional arguments for ``compute_fn``.
        compute_kwargs: Keyword arguments for ``compute_fn``.
        result_cls: Result dataclass with ``save_parquet`` / ``from_parquet``.
            When ``None``, the result is assumed to be a ``pd.DataFrame``
            saved via ``df.to_parquet(parquet_path)``.
        plot_fn: Optional plotting function taking ``(result, fig_path)``.
        label: Human-readable label for console output.

    Returns:
        The computed (or loaded) result, or ``None`` when ``result_cls`` is
        ``None`` and a cache hit occurs.
    """
    if parquet_path.exists() and not force:
        print(f"[skip] {parquet_path.name} exists (use --force to overwrite)")
        if result_cls is not None:
            return result_cls.from_parquet(parquet_path)
        return None

    print(f"[run]  Computing {label}...")
    compute_kwargs = compute_kwargs or {}
    result = compute_fn(*compute_args, **compute_kwargs)

    if isinstance(result, ParquetSerializable):
        result.save_parquet(parquet_path)
    elif isinstance(result, pd.DataFrame):
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(parquet_path, index=False)
    else:
        raise TypeError(
            f"Expected ParquetSerializable or pd.DataFrame, got {type(result).__name__}"
        )

    print(f"[save] {parquet_path}")

    if plot_fn is not None and fig_path is not None:
        plot_fn(result, fig_path)
        print(f"[fig]  {fig_path}")

    return result


def plot_decoupled_baseline_heatmap(
    result: Any,
    save_path: str | Path,
    *,
    figsize: tuple[float, float] = (10, 7),
    sql_label: str = r"$|\Delta\omega/\Delta\omega_{\mathrm{SQL}} - 1|$",
    title_prefix: str = "Decoupled Baseline Verification",
    omega_label: str = r"$\omega$",
    N_label: str = r"$N$ (particles per subsystem)",
    sql_ref_label: str = "SQL",
) -> Path:
    """Plot a heatmap of ``|Δω/Δω_SQL − 1|`` across (ω, N).

    The result object must provide array attributes ``omega_values``,
    ``N_values``, ``ratio``, and a scalar ``t_hold``.

    Args:
        result: Result object with the attributes listed above.
        save_path: Output SVG path.
        figsize: Figure size in inches.
        sql_label: Colorbar label.
        title_prefix: Prefix for the first title line. The ``t_hold`` value
            is appended automatically in the form ``, $t_hold = ...$``, so
            ``title_prefix`` should include the opening parenthesis and all
            content up to (but not including) the ``t_hold`` element.
        omega_label: X-axis label.
        N_label: Y-axis label.
        sql_ref_label: SQL label for the max-dev line (e.g. ``"SQL"`` or
            ``"SQL^{2N}"``).

    Returns:
        Path to the saved SVG file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    omega_vals = np.unique(result.omega_values)
    N_vals = np.unique(result.N_values)
    dev_map = np.full((len(N_vals), len(omega_vals)), np.nan, dtype=float)

    for i, omega in enumerate(omega_vals):
        for j, N_val in enumerate(N_vals):
            mask = np.isclose(result.omega_values, omega) & (result.N_values == N_val)
            if np.any(mask):
                r = float(result.ratio[mask][0])
                dev_map[j, i] = abs(r - 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    finite = dev_map[np.isfinite(dev_map)]
    if len(finite) > 0:
        vmin_raw = float(np.min(finite))
        vmax_raw = float(np.max(finite))
    else:
        vmin_raw, vmax_raw = 1e-15, 1.0

    # LogNorm requires vmin > 0 and vmax > vmin
    norm_vmin = max(vmin_raw, 1e-16)
    norm_vmax = max(vmax_raw, norm_vmin + 1e-15)

    im = ax.pcolormesh(
        omega_vals,
        N_vals,
        dev_map,
        shading="nearest",
        cmap="viridis",
        norm=LogNorm(vmin=norm_vmin, vmax=norm_vmax),
    )
    fig.colorbar(im, ax=ax, label=sql_label)

    max_dev = float(np.max(finite)) if len(finite) > 0 else 0.0
    ax.set_xlabel(omega_label)
    ax.set_ylabel(N_label)
    ax.set_title(
        f"{title_prefix}, $t_hold = {result.t_hold}$)\n"
        f"Max $|\\Delta\\omega/\\mathrm{{{sql_ref_label}}} - 1|"
        f" = {max_dev:.2e}$, points checked: {len(finite)}"
    )

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path
