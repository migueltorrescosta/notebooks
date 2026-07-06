"""
Ancilla-Assisted Metrology — Result Dataclasses.

All dataclasses with Parquet serialization, used by ancilla_optimization_scans
and by consumer pages/reports.

References:
- Giovannetti, Lloyd, Maccone, Nat. Photonics 5, 222 (2011)
- Davis et al., PRA 94, 063814 (2016)
- Kitagawa & Ueda, PRA 47, 5138 (1993)

"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from src.utils.serialization import ParquetSerializable

# ============================================================================
# Convergence Metric (operates on OptimisationResult)
# ============================================================================


def compute_convergence_metric(
    results: list[OptimisationResult],
) -> float:
    """Compute the convergence spread metric for a set of restarts.

    Returns std(Δω) / mean(Δω) for the given results. A value < 0.10
    indicates consistent convergence across restarts (Success Criterion #1).

    Args:
        results: List of OptimisationResult for a given θ (all restarts).

    Returns:
        Relative standard deviation (spread / mean). Returns 0.0 if
        fewer than 2 results are provided or mean is zero.

    """
    if len(results) < 2:
        return 0.0
    deltas = np.array([r.delta_omega_opt for r in results])
    # Filter out infinite values (fringe-extremum results)
    finite = deltas[np.isfinite(deltas)]
    if len(finite) < 2:
        return 0.0
    mean_val = float(np.mean(finite))
    if mean_val == 0.0:
        return 0.0
    return float(np.std(finite, ddof=1) / mean_val)


# ============================================================================
# Optimisation Result
# ============================================================================


@dataclass
class OptimisationResult(ParquetSerializable):
    """Result of a single Nelder–Mead run.

    Attributes:
        delta_omega_opt: Best sensitivity Δθ found (named in standard
            metrology notation; field kept for compatibility).
        meas_label: Measurement type — "S-only" or "Joint M".
        params_opt: Optimal 11-element parameter vector.
        omega_true: True ω used for this optimisation.
        success: Whether the optimiser reported success.
        nfev: Number of function evaluations.
        message: Optimiser message.
        expectation_Jz: ⟨J_z^S⟩ at the optimal operating point.
        variance_Jz: Var(J_z^S) at the optimal operating point.
        purity_S: Tr(ρ_S²) purity of the reduced system state (1 = pure,
            0.5 = maximally entangled with ancilla).
        expectation_M: ⟨J_z^S + J_z^A⟩ at the optimal operating point
            (for joint measurement).
        variance_M: Var(J_z^S + J_z^A) at the optimal operating point
            (for joint measurement).
        covariance_SA: Cov(J_z^S, J_z^A) at the optimal operating point
            (for joint measurement).
        history: Objective function values at each iteration (for
            convergence analysis). Empty list if callback not enabled.

    """

    delta_omega_opt: float
    params_opt: np.ndarray
    omega_true: float
    success: bool
    nfev: int
    message: str
    meas_label: str = "S-only"
    expectation_Jz: float = 0.0
    variance_Jz: float = 0.0
    purity_S: float = 0.0
    expectation_M: float = 0.0
    variance_M: float = 0.0
    covariance_SA: float = 0.0
    history: list[float] = field(default_factory=list)

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "delta_omega_opt",
        "meas_label",
        "omega_true",
        "success",
        "nfev",
        "message",
        "expectation_Jz",
        "variance_Jz",
        "purity_S",
        "expectation_M",
        "variance_M",
        "covariance_SA",
        "theta_S",
        "phi_S",
        "theta_A",
        "phi_A",
        "T_BS1",
        "T_BS2",
        "t_hold",
        "alpha_xx",
        "alpha_xz",
        "alpha_zx",
        "alpha_zz",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        """Single-row DataFrame with all metadata (history excluded, saved separately).

        Returns:
            DataFrame with one row containing all scalar fields and the 11
            optimisation parameters broken into named columns.

        """
        return pd.DataFrame(
            {
                "delta_omega_opt": [self.delta_omega_opt],
                "meas_label": [self.meas_label],
                "omega_true": [self.omega_true],
                "success": [int(self.success)],
                "nfev": [self.nfev],
                "message": [self.message],
                "expectation_Jz": [self.expectation_Jz],
                "variance_Jz": [self.variance_Jz],
                "purity_S": [self.purity_S],
                "expectation_M": [self.expectation_M],
                "variance_M": [self.variance_M],
                "covariance_SA": [self.covariance_SA],
                "theta_S": [float(self.params_opt[0])],
                "phi_S": [float(self.params_opt[1])],
                "theta_A": [float(self.params_opt[2])],
                "phi_A": [float(self.params_opt[3])],
                "T_BS1": [float(self.params_opt[4])],
                "T_BS2": [float(self.params_opt[5])],
                "t_hold": [float(self.params_opt[6])],
                "alpha_xx": [float(self.params_opt[7])],
                "alpha_xz": [float(self.params_opt[8])],
                "alpha_zx": [float(self.params_opt[9])],
                "alpha_zz": [float(self.params_opt[10])],
            },
        )

    def _save_sidecars(self, path: Path) -> None:
        """Save optimisation history as a sidecar Parquet file."""
        history_path = path.with_stem(path.stem + "-history")
        pd.DataFrame({"history": self.history}).to_parquet(history_path, index=False)

    @classmethod
    def from_parquet(cls, path: str | Path) -> OptimisationResult:
        """Reconstruct an OptimisationResult from a Parquet file.

        Args:
            path: Path to the Parquet file written by ``save_parquet``.

        Returns:
            Reconstructed OptimisationResult.

        Raises:
            ValueError: If the Parquet file is missing required columns.

        """
        path = Path(path)
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        params_opt = np.array(
            [
                float(df["theta_S"].iloc[0]),
                float(df["phi_S"].iloc[0]),
                float(df["theta_A"].iloc[0]),
                float(df["phi_A"].iloc[0]),
                float(df["T_BS1"].iloc[0]),
                float(df["T_BS2"].iloc[0]),
                float(df["t_hold"].iloc[0]),
                float(df["alpha_xx"].iloc[0]),
                float(df["alpha_xz"].iloc[0]),
                float(df["alpha_zx"].iloc[0]),
                float(df["alpha_zz"].iloc[0]),
            ],
        )
        history_path = path.with_stem(path.stem + "-history")
        history: list[float] = []
        if history_path.exists():
            history = pd.read_parquet(history_path)["history"].tolist()

        return cls(
            delta_omega_opt=float(df["delta_omega_opt"].iloc[0]),
            meas_label=str(df["meas_label"].iloc[0]),
            params_opt=params_opt,
            omega_true=float(df["omega_true"].iloc[0]),
            success=bool(int(df["success"].iloc[0])),
            nfev=int(df["nfev"].iloc[0]),
            message=str(df["message"].iloc[0]),
            expectation_Jz=float(df["expectation_Jz"].iloc[0]),
            variance_Jz=float(df["variance_Jz"].iloc[0]),
            purity_S=float(df["purity_S"].iloc[0]),
            expectation_M=float(df["expectation_M"].iloc[0]),
            variance_M=float(df["variance_M"].iloc[0]),
            covariance_SA=float(df["covariance_SA"].iloc[0]),
            history=history,
        )


# ============================================================================
# Omega Scan Result
# ============================================================================


@dataclass
class OmegaScanResult(ParquetSerializable):
    """Results of an ω scan with multiple Nelder–Mead restarts per ω.

    Attributes:
        results: List of OptimisationResult per ω value.
        omega_values: Array of ω values scanned.
        best_per_omega: Best Δω for each ω value.
        all_results: All results (including non-best restarts) keyed by ω.

    """

    results: list[OptimisationResult] = field(default_factory=list)
    omega_values: np.ndarray = field(default_factory=lambda: np.array([]))
    best_per_omega: np.ndarray = field(default_factory=lambda: np.array([]))
    all_results: dict[float, list[OptimisationResult]] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten the omega-scan results into a tabular DataFrame.

        One row per omega value with the best result and derived quantities.
        Works both from full results (with ``all_results`` populated) and
        from CSV-loaded summary (using ``omega_values`` / ``best_per_omega``).

        Returns:
            DataFrame with columns: omega, best_delta_omega, sql, vs_sql,
            spread, t_hold_star, covariance, expectation_M, flag.

        """
        rows: list[dict[str, float | str]] = []
        for i, omega_val in enumerate(self.omega_values):
            restarts = self.all_results.get(float(omega_val), [])
            if restarts:
                best = restarts[0]
                best_delta_omega = float(best.delta_omega_opt)
                spread = compute_convergence_metric(restarts)
                t_hold_star = float(best.params_opt[6])
                cov_val = float(best.covariance_SA)
                exp_m = float(best.expectation_M)
                vs_sql = (
                    float(best.delta_omega_opt / t_hold_star)
                    if t_hold_star > 0 and np.isfinite(1.0 / t_hold_star)
                    else float("inf")
                )
            else:
                # Summary-only mode (loaded from CSV)
                best_delta_omega = float(self.best_per_omega[i])
                spread = 0.0
                t_hold_star = 0.0
                cov_val = 0.0
                exp_m = 0.0
                vs_sql = float("inf")

            sql = 1.0 / t_hold_star if t_hold_star > 0 else float("inf")
            flag = "fringe" if abs(exp_m) < 1e-10 else "ok"

            rows.append(
                {
                    "omega": float(omega_val),
                    "best_delta_omega": best_delta_omega,
                    "sql": sql,
                    "vs_sql": vs_sql,
                    "spread": spread,
                    "t_hold_star": t_hold_star,
                    "covariance": cov_val,
                    "expectation_M": exp_m,
                    "flag": flag,
                },
            )
        return pd.DataFrame(rows)

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "omega",
        "best_delta_omega",
        "sql",
        "vs_sql",
        "spread",
        "t_hold_star",
        "covariance",
        "expectation_M",
        "flag",
    ]

    @classmethod
    def from_parquet(cls, path: str | Path) -> OmegaScanResult:
        """Reconstruct a OmegaScanResult from a Parquet file written by ``save_parquet``.

        Note: The reconstructed result contains only the summary per omega value
        (one row = one omega), not the full per-restart details.

        Args:
            path: Path to the Parquet file.

        Returns:
            A OmegaScanResult with the summary data.

        Raises:
            ValueError: If the Parquet file is missing required columns.

        """
        path = Path(path)
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        omega_values = df["omega"].to_numpy(dtype=float)
        best_per_omega = df["best_delta_omega"].to_numpy(dtype=float)
        return cls(
            results=[],
            omega_values=omega_values,
            best_per_omega=best_per_omega,
            all_results={},
        )

    _RESTART_COLUMNS: ClassVar[list[str]] = [
        "omega",
        "restart_index",
        "delta_omega_opt",
        "meas_label",
        "omega_true",
        "success",
        "nfev",
        "message",
        "expectation_Jz",
        "variance_Jz",
        "purity_S",
        "expectation_M",
        "variance_M",
        "covariance_SA",
        "theta_S",
        "phi_S",
        "theta_A",
        "phi_A",
        "T_BS1",
        "T_BS2",
        "t_hold",
        "alpha_xx",
        "alpha_xz",
        "alpha_zx",
        "alpha_zz",
        "history_json",
    ]


# ============================================================================
# Alpha Re-optimisation Scan Result
# ============================================================================


@dataclass
class AlphaReoptScanResult(ParquetSerializable):
    """Result from scanning α with state re-optimisation at each point.

    Attributes:
        alpha_values: Array of α values scanned.
        delta_omega_joint: Best Δω for joint measurement at each α.
        delta_omega_sonly: Best Δω for S-only measurement at each α.
        best_params_joint: Full 11-element optimal params (joint) at each α.
        best_params_sonly: Full 11-element optimal params (S-only) at each α.

    """

    alpha_values: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_omega_joint: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_omega_sonly: np.ndarray = field(default_factory=lambda: np.array([]))
    best_params_joint: list[np.ndarray] = field(default_factory=list)
    best_params_sonly: list[np.ndarray] = field(default_factory=list)

    _PARAM_COLS: ClassVar[list[str]] = [
        "theta_S",
        "phi_S",
        "theta_A",
        "phi_A",
        "T_BS1",
        "T_BS2",
        "t_hold",
        "alpha_xx",
        "alpha_xz",
        "alpha_zx",
        "alpha_zz",
    ]

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "alpha",
        "delta_omega_joint",
        "delta_omega_sonly",
        *(f"joint_{c}" for c in _PARAM_COLS),
        *(f"sonly_{c}" for c in _PARAM_COLS),
    ]

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten the α re-optimisation scan into a DataFrame.

        Includes all metadata: optimal parameters for both joint and S-only
        measurements are stored in named columns alongside scalar results.

        Returns:
            DataFrame with columns: alpha, delta_omega_joint,
            delta_omega_sonly, and 22 optimal-parameter columns (11 per
            measurement type).

        """
        n = len(self.alpha_values)
        data: dict[str, object] = {
            "alpha": self.alpha_values,
            "delta_omega_joint": self.delta_omega_joint,
            "delta_omega_sonly": self.delta_omega_sonly,
        }

        # Pad or truncate params lists to match n
        def _pad(p: list[np.ndarray], n_target: int) -> list[np.ndarray]:
            if len(p) < n_target:
                return p + [np.full(11, float("nan"))] * (n_target - len(p))
            return p[:n_target]

        joint_params = _pad(self.best_params_joint, n)
        sonly_params = _pad(self.best_params_sonly, n)
        for i, col in enumerate(self._PARAM_COLS):
            data[f"joint_{col}"] = [float(p[i]) for p in joint_params]
            data[f"sonly_{col}"] = [float(p[i]) for p in sonly_params]
        return pd.DataFrame(data)

    @classmethod
    def from_parquet(cls, path: str | Path) -> AlphaReoptScanResult:
        """Reconstruct from a Parquet file.

        Args:
            path: Path to the Parquet file.

        Returns:
            Reconstructed AlphaReoptScanResult.

        Raises:
            ValueError: If the Parquet file is missing required columns.

        """
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        n = len(df)
        best_params_joint: list[np.ndarray] = [
            np.array([float(df[f"joint_{col}"].iloc[i]) for col in cls._PARAM_COLS])
            for i in range(n)
        ]
        best_params_sonly: list[np.ndarray] = [
            np.array([float(df[f"sonly_{col}"].iloc[i]) for col in cls._PARAM_COLS])
            for i in range(n)
        ]
        return cls(
            alpha_values=df["alpha"].to_numpy(dtype=float),
            delta_omega_joint=df["delta_omega_joint"].to_numpy(dtype=float),
            delta_omega_sonly=df["delta_omega_sonly"].to_numpy(dtype=float),
            best_params_joint=best_params_joint,
            best_params_sonly=best_params_sonly,
        )


# ============================================================================
# Decoupled Baseline Result
# ============================================================================


@dataclass
class DecoupledBaselineResult(ParquetSerializable):
    """Result from evaluating the decoupled baseline (α=0).

    Here α refers to the Ising coupling coefficient α_{zz} (not the scaling exponent).

    Attributes:
        t_hold_values: Array of holding-time values.
        delta_omega_values: Corresponding Δω (both joint and S-only give
            the same value in the decoupled case).
        sql_values: SQL = 1/t_hold values (time-based SQL; not particle-number SQL 1/√N).

    """

    t_hold_values: np.ndarray
    delta_omega_values: np.ndarray
    sql_values: np.ndarray

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "t_hold",
        "delta_omega",
        "sql",
        "ratio",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten into a DataFrame.

        Returns:
            DataFrame with columns: t_hold, delta_omega, sql, ratio.

        """
        return pd.DataFrame(
            {
                "t_hold": self.t_hold_values,
                "delta_omega": self.delta_omega_values,
                "sql": self.sql_values,
                "ratio": np.where(
                    self.sql_values > 0,
                    self.delta_omega_values / self.sql_values,
                    np.nan,
                ),
            },
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> DecoupledBaselineResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        return cls(
            t_hold_values=df["t_hold"].to_numpy(dtype=float),
            delta_omega_values=df["delta_omega"].to_numpy(dtype=float),
            sql_values=df["sql"].to_numpy(dtype=float),
        )


# ============================================================================
# Covariance Analysis Result
# ============================================================================


@dataclass
class CovarianceAnalysisResult(ParquetSerializable):
    """Result from analysing Cov(J_z^S, J_z^A) across α coefficients.

    Attributes:
        coefficient_names: List of coefficient labels e.g. ['α_xx', ...].
        max_covariances: Max |Cov| for each coefficient over the scanned
            α range.
        covariance_signs: Sign of the max covariance (+1 or -1) for each
            coefficient.

    """

    coefficient_names: list[str]
    max_covariances: np.ndarray
    covariance_signs: np.ndarray

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "coefficient",
        "max_covariance",
        "sign",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten into a DataFrame.

        Returns:
            DataFrame with columns: coefficient, max_covariance, sign.

        """
        return pd.DataFrame(
            {
                "coefficient": self.coefficient_names,
                "max_covariance": self.max_covariances,
                "sign": self.covariance_signs,
            },
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> CovarianceAnalysisResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        return cls(
            coefficient_names=list(df["coefficient"]),
            max_covariances=df["max_covariance"].to_numpy(dtype=float),
            covariance_signs=df["sign"].to_numpy(dtype=float),
        )


# ============================================================================
# Alpha Single Scan Result
# ============================================================================


@dataclass
class AlphaSingleScanResult(ParquetSerializable):
    """Result from scanning a single α coefficient while holding others fixed.

    Attributes:
        alpha_name: Which coefficient was scanned ('xx', 'xz', 'zx', 'zz').
        alpha_values: Array of α values scanned.
        delta_omega_values: Corresponding Δω sensitivity values.
        fixed_params: Dict of the other fixed parameters used.

    """

    alpha_name: str
    alpha_values: np.ndarray
    delta_omega_values: np.ndarray
    fixed_params: dict[str, float] = field(default_factory=dict)

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "alpha_name",
        "alpha",
        "delta_omega",
        "fixed_params_json",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten the single-α scan into a DataFrame.

        Returns:
            DataFrame with columns: alpha_name, alpha, delta_omega,
            and fixed_params serialised as JSON.

        """
        return pd.DataFrame(
            {
                "alpha_name": [self.alpha_name] * len(self.alpha_values),
                "alpha": self.alpha_values,
                "delta_omega": self.delta_omega_values,
                "fixed_params_json": [json.dumps(self.fixed_params)]
                * len(self.alpha_values),
            },
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> AlphaSingleScanResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        return cls(
            alpha_name=str(df["alpha_name"].iloc[0]),
            alpha_values=df["alpha"].to_numpy(dtype=float),
            delta_omega_values=df["delta_omega"].to_numpy(dtype=float),
            fixed_params=json.loads(str(df["fixed_params_json"].iloc[0])),
        )


# ============================================================================
# Alpha Random Search Result
# ============================================================================


@dataclass
class AlphaRandomSearchResult(ParquetSerializable):
    """Result from random search over the 4D α coefficient space.

    Attributes:
        alpha_samples: Array of shape (N, 4) with sampled α values.
        delta_omega_values: Array of shape (N,) with Δω for each sample.
        best_alpha: The α = (α_xx, α_xz, α_zx, α_zz) that gave minimal Δω.
        best_delta_omega: The minimal Δω found.
        fixed_params: Dict of the other fixed parameters used.

    """

    alpha_samples: np.ndarray
    delta_omega_values: np.ndarray
    best_alpha: tuple[float, float, float, float]
    best_delta_omega: float
    fixed_params: dict[str, float] = field(default_factory=dict)

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "alpha_xx",
        "alpha_xz",
        "alpha_zx",
        "alpha_zz",
        "delta_omega",
        "fixed_params_json",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten the random search into a DataFrame.

        Returns:
            DataFrame with columns: alpha_xx, alpha_xz, alpha_zx,
            alpha_zz, delta_omega, fixed_params_json.

        """
        return pd.DataFrame(
            {
                "alpha_xx": self.alpha_samples[:, 0],
                "alpha_xz": self.alpha_samples[:, 1],
                "alpha_zx": self.alpha_samples[:, 2],
                "alpha_zz": self.alpha_samples[:, 3],
                "delta_omega": self.delta_omega_values,
                "fixed_params_json": [json.dumps(self.fixed_params)]
                * len(self.delta_omega_values),
            },
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> AlphaRandomSearchResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        alphas = df[["alpha_xx", "alpha_xz", "alpha_zx", "alpha_zz"]].to_numpy(
            dtype=float,
        )
        deltas = df["delta_omega"].to_numpy(dtype=float)
        best_idx = int(np.argmin(deltas))
        return cls(
            alpha_samples=alphas,
            delta_omega_values=deltas,
            best_alpha=(
                float(alphas[best_idx, 0]),
                float(alphas[best_idx, 1]),
                float(alphas[best_idx, 2]),
                float(alphas[best_idx, 3]),
            ),
            best_delta_omega=float(deltas[best_idx]),
            fixed_params=json.loads(str(df["fixed_params_json"].iloc[0])),
        )


# ============================================================================
# Interaction Robustness Result
# ============================================================================


@dataclass
class InteractionRobustnessResult(ParquetSerializable):
    """Result from a 2D scan over t_hold and α values.

    Attributes:
        t_hold_values: Array of t_hold holding-time values scanned.
        alpha_values: Array of α coefficient values scanned.
        delta_omega_joint: 2D array of Δω (joint measurement) for each
            (t_hold, α) pair, shape (len(t_hold_values), len(alpha_values)).
        delta_omega_sonly: 2D array of Δω (S-only measurement) for each
            (t_hold, α) pair, shape (len(t_hold_values), len(alpha_values)).

    """

    t_hold_values: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_values: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_omega_joint: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_omega_sonly: np.ndarray = field(default_factory=lambda: np.array([]))

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "t_hold",
        "alpha",
        "measurement",
        "delta_omega",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        """Melt the 2D arrays into a long-format DataFrame.

        Returns:
            DataFrame with columns: t_hold, alpha, measurement, delta_omega.

        """
        n_T = len(self.t_hold_values)
        n_a = len(self.alpha_values)
        rows: list[dict[str, float | str]] = []
        for i in range(n_T):
            for j in range(n_a):
                rows.append(
                    {
                        "t_hold": float(self.t_hold_values[i]),
                        "alpha": float(self.alpha_values[j]),
                        "measurement": "joint",
                        "delta_omega": float(self.delta_omega_joint[i, j]),
                    },
                )
                rows.append(
                    {
                        "t_hold": float(self.t_hold_values[i]),
                        "alpha": float(self.alpha_values[j]),
                        "measurement": "sonly",
                        "delta_omega": float(self.delta_omega_sonly[i, j]),
                    },
                )
        return pd.DataFrame(rows)

    @classmethod
    def from_parquet(cls, path: str | Path) -> InteractionRobustnessResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        t_hold_unique = sorted(df["t_hold"].unique())
        alpha_unique = sorted(df["alpha"].unique())
        n_T = len(t_hold_unique)
        n_a = len(alpha_unique)
        dtheta_j = np.full((n_T, n_a), np.nan)
        dtheta_s = np.full((n_T, n_a), np.nan)
        for _, row in df.iterrows():
            i = t_hold_unique.index(row["t_hold"])
            j = alpha_unique.index(row["alpha"])
            if row["measurement"] == "joint":
                dtheta_j[i, j] = row["delta_omega"]
            else:
                dtheta_s[i, j] = row["delta_omega"]
        return cls(
            t_hold_values=np.array(t_hold_unique, dtype=float),
            alpha_values=np.array(alpha_unique, dtype=float),
            delta_omega_joint=dtheta_j,
            delta_omega_sonly=dtheta_s,
        )
