"""
Ancilla-Drive-Enhanced Metrology: Beating the SQL with an Actively Driven Ancilla.

Implements the driven-ancilla metrology protocol described in
``reports/20260518/Ancilla-Drive-Enhanced-Metrology.md``.

Physical Model:
- Two qubits (system S + ancilla A), each a spin-1/2 (single-particle subspace).
- Basis: {|00⟩, |01⟩, |10⟩, |11⟩} where |0⟩ = |1,0⟩ (particle in mode 0).
- Circuit: BS_S → Hold → BS_S, where BS_S acts only on the system qubit.
- Hold Hamiltonian:
    H = θ J_z^S + H_A + H_int
    H_A = a_x J_x^A + a_y J_y^A + a_z J_z^A   (ancilla drive)
    H_int = a_zz J_z^S ⊗ J_z^A                (Ising interaction)
- Initial state: |00⟩ (both qubits in |1,0⟩).
- Measurement: J_z^S on the system qubit.
- Sensitivity: Δθ via error propagation (central finite differences).

Units:
- Dimensionless throughout. θ is the unknown phase rate.
- T_H: holding-time strength (dimensionless).
- a_x, a_y, a_z, a_zz: real coefficients.

References:
- Report ``reports/20260518/Ancilla-Drive-Enhanced-Metrology.md``
- Giovannetti, Lloyd, Maccone, Nat. Photonics 5, 222 (2011)
"""

from __future__ import annotations

import concurrent.futures
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

# Reuse shared primitives from ancilla_optimization
from src.analysis.ancilla_optimization import (
    I_2,
    bs_unitary,
    build_two_qubit_operators,
    compute_expectation_and_variance,
)

I_4 = np.eye(4, dtype=complex)


# ============================================================================
# Operator Construction
# ============================================================================


def system_only_bs_unitary(T: float) -> np.ndarray:
    """Single-qubit beam-splitter on the system, identity on the ancilla.

    U = U_BS(T) ⊗ I_2 = exp(-i T J_x^S) ⊗ I_2

    A 50/50 beam splitter corresponds to T = π/2.

    Args:
        T: Beam-splitter duration.

    Returns:
        4×4 unitary matrix.
    """
    U_sys = bs_unitary(T)
    U = np.kron(U_sys, I_2)
    assert np.allclose(U @ U.conj().T, I_4, atol=1e-12), (
        f"System-only BS unitary not unitary for T={T}"
    )
    return U


def build_ancilla_drive_hamiltonian(
    a_x: float,
    a_y: float,
    a_z: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the ancilla drive Hamiltonian.

    H_A = a_x J_x^A + a_y J_y^A + a_z J_z^A

    Args:
        a_x: Coefficient for J_x^A.
        a_y: Coefficient for J_y^A.
        a_z: Coefficient for J_z^A.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian matrix representing the ancilla drive.
    """
    H = np.zeros((4, 4), dtype=complex)
    if a_x != 0.0:
        H += a_x * ops["Jx_A"]
    if a_y != 0.0:
        H += a_y * ops["Jy_A"]
    if a_z != 0.0:
        H += a_z * ops["Jz_A"]
    # Enforce Hermiticity
    return 0.5 * (H + H.conj().T)


def build_iszz_interaction(
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the Ising-type system--ancilla interaction.

    H_int = a_zz J_z^S ⊗ J_z^A = a_zz (σ_z/2) ⊗ (σ_z/2)

    Args:
        a_zz: Interaction coupling coefficient.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian matrix.
    """
    H = np.zeros((4, 4), dtype=complex)
    if a_zz != 0.0:
        # J_z^S ⊗ J_z^A = (J_z ⊗ I_2) @ (I_2 ⊗ J_z)
        H += a_zz * (ops["Jz_S"] @ ops["Jz_A"])
    return H


def build_drive_hold_hamiltonian(
    theta: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the total holding Hamiltonian.

    H = θ J_z^S + H_A + H_int
      = θ J_z^S + (a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_zz J_z^S ⊗ J_z^A

    Args:
        theta: Unknown phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian Hamiltonian matrix.
    """
    H = theta * ops["Jz_S"]
    H += build_ancilla_drive_hamiltonian(a_x, a_y, a_z, ops)
    H += build_iszz_interaction(a_zz, ops)
    return 0.5 * (H + H.conj().T)


def drive_hold_unitary(
    T_H: float,
    theta: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Holding-time unitary for the driven-ancilla protocol.

    U_hold(T_H) = exp(-i T_H H)
    where H = θ J_z^S + H_A + H_int.

    Args:
        T_H: Holding-time strength.
        theta: True phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 unitary matrix.
    """
    H = build_drive_hold_hamiltonian(theta, a_x, a_y, a_z, a_zz, ops)
    U = expm(-1j * T_H * H)
    assert np.allclose(U @ U.conj().T, I_4, atol=1e-12), (
        f"Drive hold unitary not unitary for T_H={T_H}, θ={theta}"
    )
    return U


def evolve_drive_circuit(
    psi0: np.ndarray,
    T_BS: float,
    T_H: float,
    theta: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the full driven-ancilla MZI circuit.

    |ψ_final⟩ = U_BS_S · U_hold(T_H) · U_BS_S · |ψ₀⟩

    Args:
        psi0: Initial 4-vector (must be normalised).
        T_BS: Beam-splitter duration (both BS identical).
        T_H: Holding-time strength.
        theta: Phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators.

    Returns:
        Final normalised 4-vector state.
    """
    assert np.isclose(np.linalg.norm(psi0), 1.0), "Initial state must be normalised"

    U_bs = system_only_bs_unitary(T_BS)
    psi = U_bs @ psi0
    psi = drive_hold_unitary(T_H, theta, a_x, a_y, a_z, a_zz, ops) @ psi
    psi = U_bs @ psi

    assert np.isclose(np.linalg.norm(psi), 1.0), "Final state must be normalised"
    return psi


def compute_drive_sensitivity(
    psi0: np.ndarray,
    T_BS: float,
    T_H: float,
    theta_true: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    fd_step: float = 1e-6,
    meas_op: np.ndarray | None = None,
) -> float:
    """Compute the error-propagation sensitivity Δθ.

    Δθ = sqrt(Var(O)) / |∂⟨O⟩/∂θ|

    where O is the measurement operator (default: J_z^S).

    Args:
        psi0: Initial 4-vector (product state).
        T_BS: Beam-splitter duration.
        T_H: Holding-time strength.
        theta_true: True phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators (must contain 'Jz_S').
        fd_step: Finite-difference step size (default 1e-6).
        meas_op: Measurement operator. Defaults to ops['Jz_S'] (S-only).

    Returns:
        Sensitivity Δθ (positive float). Returns inf if derivative is zero
        (fringe extremum).

    """
    if meas_op is None:
        meas_op = ops["Jz_S"]

    # Evaluate at theta_true
    psi = evolve_drive_circuit(
        psi0,
        T_BS,
        T_H,
        theta_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    _, var = compute_expectation_and_variance(psi, meas_op)

    # Central finite difference for ∂⟨O⟩/∂θ
    psi_plus = evolve_drive_circuit(
        psi0,
        T_BS,
        T_H,
        theta_true + fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    psi_minus = evolve_drive_circuit(
        psi0,
        T_BS,
        T_H,
        theta_true - fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    exp_plus = np.real(psi_plus.conj() @ meas_op @ psi_plus)
    exp_minus = np.real(psi_minus.conj() @ meas_op @ psi_minus)
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12:
        return float("inf")

    # Zero-variance case: the state is an eigenstate of the measurement
    # operator, giving a deterministic measurement outcome.  Error propagation
    # would yield Δθ = 0 (unphysical), so flag as fringe extremum.
    if var < 1e-15:
        return float("inf")

    return float(np.sqrt(var) / abs(d_exp))


# ============================================================================
# Dataclasses
# ============================================================================


@dataclass
class DriveDecoupledBaselineResult:
    """Result from evaluating the decoupled baseline (a_x = a_y = a_z = a_zz = 0).

    Attributes:
        T_H_value: The holding-time value used.
        delta_theta: Computed Δθ at the decoupled configuration.
        sql: SQL = 1/T_H value.
    """

    T_H_value: float
    delta_theta: float
    sql: float

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "T_H": [self.T_H_value],
                "delta_theta": [self.delta_theta],
                "sql": [self.sql],
                "ratio": [
                    self.delta_theta / self.sql if self.sql > 0 else float("nan")
                ],
            },
        )

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> DriveDecoupledBaselineResult:
        df = pd.read_parquet(path)
        required = {"T_H", "delta_theta", "sql"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: "
                f"{sorted(missing)}. Regenerate the file with the current code."
            )
        return cls(
            T_H_value=float(df["T_H"].iloc[0]),
            delta_theta=float(df["delta_theta"].iloc[0]),
            sql=float(df["sql"].iloc[0]),
        )


@dataclass
class Drive2DSliceResult:
    """Result from a 2D parameter slice scan over (a_drive, a_zz).

    Attributes:
        drive_values: Array of drive coefficient values (a_x, a_y, or a_z).
        azz_values: Array of a_zz (interaction) values.
        delta_theta_grid: 2D array of Δθ values, shape
            (len(drive_values), len(azz_values)).
        theta_value: The θ value at which the scan was performed.
        slice_type: 'ax', 'ay', or 'az'.
        sql: SQL = 1/T_H reference value.
    """

    drive_values: np.ndarray
    azz_values: np.ndarray
    delta_theta_grid: np.ndarray
    theta_value: float
    slice_type: str = "ax"
    sql: float = 0.1

    def to_dataframe(self) -> pd.DataFrame:
        """Melt the 2D array into a long-format DataFrame."""
        n_d = len(self.drive_values)
        n_a = len(self.azz_values)
        rows: list[dict[str, float | str]] = [
            {
                "drive": float(self.drive_values[i]),
                "azz": float(self.azz_values[j]),
                "delta_theta": float(self.delta_theta_grid[i, j]),
                "theta_value": float(self.theta_value),
                "slice_type": str(self.slice_type),
                "sql": float(self.sql),
            }
            for i in range(n_d)
            for j in range(n_a)
        ]
        return pd.DataFrame(rows)

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> Drive2DSliceResult:
        df = pd.read_parquet(path)
        drive_unique = sorted(df["drive"].unique())
        azz_unique = sorted(df["azz"].unique())
        n_d = len(drive_unique)
        n_a = len(azz_unique)
        grid = np.full((n_d, n_a), np.nan, dtype=float)
        for _, row in df.iterrows():
            i = drive_unique.index(row["drive"])
            j = azz_unique.index(row["azz"])
            grid[i, j] = row["delta_theta"]
        # Restore metadata from CSV; fail fast if columns are missing
        required_meta = {"theta_value", "slice_type", "sql"}
        missing_meta = required_meta - set(df.columns)
        if missing_meta:
            raise ValueError(
                f"CSV at {path} is missing required columns: "
                f"{sorted(missing_meta)}. "
                "Regenerate the file with the current code."
            )
        theta_value = float(df["theta_value"].iloc[0])
        slice_type = str(df["slice_type"].iloc[0])
        sql = float(df["sql"].iloc[0])
        return cls(
            drive_values=np.array(drive_unique, dtype=float),
            azz_values=np.array(azz_unique, dtype=float),
            delta_theta_grid=grid,
            theta_value=theta_value,
            slice_type=slice_type,
            sql=sql,
        )


@dataclass
class DriveRandomSearchResult:
    """Result from a 4D random search over (a_x, a_y, a_z, a_zz).

    Attributes:
        samples: Array of shape (N, 4) with sampled parameter values.
        delta_theta_values: Array of shape (N,) with Δθ for each sample.
        best_params: The (a_x, a_y, a_z, a_zz) that gave minimal Δθ.
        best_delta_theta: The minimal Δθ found.
        theta_value: θ at which the search was performed.
        sql: SQL = 1/T_H reference.
    """

    samples: np.ndarray
    delta_theta_values: np.ndarray
    best_params: tuple[float, float, float, float]
    best_delta_theta: float
    theta_value: float = 1.0
    sql: float = 0.1
    T_H: float = 10.0

    def to_dataframe(self) -> pd.DataFrame:
        n = len(self.samples)
        return pd.DataFrame(
            {
                "a_x": self.samples[:, 0],
                "a_y": self.samples[:, 1],
                "a_z": self.samples[:, 2],
                "a_zz": self.samples[:, 3],
                "delta_theta": self.delta_theta_values,
                "theta_value": [self.theta_value] * n,
                "sql": [self.sql] * n,
                "T_H": [self.T_H] * n,
            },
        )

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> DriveRandomSearchResult:
        df = pd.read_parquet(path)
        required = {
            "a_x",
            "a_y",
            "a_z",
            "a_zz",
            "delta_theta",
            "theta_value",
            "sql",
            "T_H",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV at {path} is missing required columns: {sorted(missing)}. "
                "Regenerate the file with the current code."
            )
        samples = df[["a_x", "a_y", "a_z", "a_zz"]].to_numpy(dtype=float)
        deltas = df["delta_theta"].to_numpy(dtype=float)
        best_idx = int(np.argmin(deltas))
        return cls(
            samples=samples,
            delta_theta_values=deltas,
            best_params=(
                float(samples[best_idx, 0]),
                float(samples[best_idx, 1]),
                float(samples[best_idx, 2]),
                float(samples[best_idx, 3]),
            ),
            best_delta_theta=float(deltas[best_idx]),
            theta_value=float(df["theta_value"].iloc[0]),
            sql=float(df["sql"].iloc[0]),
            T_H=float(df["T_H"].iloc[0]),
        )


@dataclass
class DriveNelderMeadResult:
    """Result of a single Nelder--Mead run for the driven-ancilla protocol.

    Attributes:
        delta_theta_opt: Best sensitivity Δθ found.
        params_opt: Optimal 4-element parameter vector (a_x, a_y, a_z, a_zz).
        theta_true: True θ used for this optimisation.
        success: Whether the optimiser reported success.
        nfev: Number of function evaluations.
        message: Optimiser message.
        expectation_Jz: ⟨J_z^S⟩ at the optimal operating point.
        variance_Jz: Var(J_z^S) at the optimal operating point.
        history: Objective function values at each iteration.
    """

    delta_theta_opt: float
    params_opt: np.ndarray
    theta_true: float
    success: bool
    nfev: int
    message: str = ""
    expectation_Jz: float = 0.0
    variance_Jz: float = 0.0
    history: list[float] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "a_x": [float(self.params_opt[0])],
                "a_y": [float(self.params_opt[1])],
                "a_z": [float(self.params_opt[2])],
                "a_zz": [float(self.params_opt[3])],
                "delta_theta": [self.delta_theta_opt],
                "theta_true": [self.theta_true],
                "success": [int(self.success)],
                "nfev": [self.nfev],
                "expectation_Jz": [self.expectation_Jz],
                "variance_Jz": [self.variance_Jz],
            },
        )

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> DriveNelderMeadResult:
        df = pd.read_parquet(path)
        required = {
            "delta_theta",
            "a_x",
            "a_y",
            "a_z",
            "a_zz",
            "theta_true",
            "success",
            "nfev",
            "expectation_Jz",
            "variance_Jz",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: "
                f"{sorted(missing)}. Regenerate the file with the current code."
            )
        return cls(
            delta_theta_opt=float(df["delta_theta"].iloc[0]),
            params_opt=np.array(
                [
                    float(df["a_x"].iloc[0]),
                    float(df["a_y"].iloc[0]),
                    float(df["a_z"].iloc[0]),
                    float(df["a_zz"].iloc[0]),
                ]
            ),
            theta_true=float(df["theta_true"].iloc[0]),
            success=bool(int(df["success"].iloc[0])),
            nfev=int(df["nfev"].iloc[0]),
            expectation_Jz=float(df["expectation_Jz"].iloc[0]),
            variance_Jz=float(df["variance_Jz"].iloc[0]),
        )


@dataclass
class DriveThetaScanResult:
    """Results of a θ scan over driven-ancilla parameters.

    Attributes:
        theta_values: Array of θ values scanned.
        best_params_per_theta: List of optimal (a_x, a_y, a_z, a_zz) tuples.
        best_delta_theta_per_theta: Optimal Δθ for each θ value.
        sql_values: SQL = 1/T_H for each θ.
        expectation_Jz_per_theta: ⟨J_z^S⟩ at each optimal point.
        variance_Jz_per_theta: Var(J_z^S) at each optimal point.
        all_results: All Nelder-Mead results keyed by θ (for spread analysis).
    """

    theta_values: np.ndarray = field(default_factory=lambda: np.array([]))
    best_params_per_theta: list[tuple[float, float, float, float]] = field(
        default_factory=list
    )
    best_delta_theta_per_theta: np.ndarray = field(default_factory=lambda: np.array([]))
    sql_values: np.ndarray = field(default_factory=lambda: np.array([]))
    expectation_Jz_per_theta: np.ndarray = field(default_factory=lambda: np.array([]))
    variance_Jz_per_theta: np.ndarray = field(default_factory=lambda: np.array([]))
    all_results: dict[float, list[DriveNelderMeadResult]] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        for i, theta in enumerate(self.theta_values):
            sql = float(self.sql_values[i]) if i < len(self.sql_values) else 0.1
            best = (
                self.best_delta_theta_per_theta[i]
                if i < len(self.best_delta_theta_per_theta)
                else float("inf")
            )
            params = (
                self.best_params_per_theta[i]
                if i < len(self.best_params_per_theta)
                else (0.0, 0.0, 0.0, 0.0)
            )
            exp_jz = (
                float(self.expectation_Jz_per_theta[i])
                if i < len(self.expectation_Jz_per_theta)
                else 0.0
            )
            var_jz = (
                float(self.variance_Jz_per_theta[i])
                if i < len(self.variance_Jz_per_theta)
                else 0.0
            )
            rows.append(
                {
                    "theta": float(theta),
                    "best_delta_theta": best,
                    "sql": sql,
                    "ratio": best / sql
                    if np.isfinite(best) and sql > 0
                    else float("inf"),
                    "a_x": float(params[0]),
                    "a_y": float(params[1]),
                    "a_z": float(params[2]),
                    "a_zz": float(params[3]),
                    "expectation_Jz": exp_jz,
                    "variance_Jz": var_jz,
                }
            )
        return pd.DataFrame(rows)

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> DriveThetaScanResult:
        df = pd.read_parquet(path)
        required = {
            "theta",
            "best_delta_theta",
            "sql",
            "a_x",
            "a_y",
            "a_z",
            "a_zz",
            "expectation_Jz",
            "variance_Jz",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: "
                f"{sorted(missing)}. Regenerate the file with the current code."
            )
        thetas = df["theta"].to_numpy(dtype=float)
        best = df["best_delta_theta"].to_numpy(dtype=float)
        sql = df["sql"].to_numpy(dtype=float)
        exps = df["expectation_Jz"].to_numpy(dtype=float)
        vars_ = df["variance_Jz"].to_numpy(dtype=float)
        params_list: list[tuple[float, float, float, float]] = []
        for _, row in df.iterrows():
            params_list.append(
                (
                    float(row["a_x"]),
                    float(row["a_y"]),
                    float(row["a_z"]),
                    float(row["a_zz"]),
                )
            )
        return cls(
            theta_values=thetas,
            best_params_per_theta=params_list,
            best_delta_theta_per_theta=best,
            sql_values=sql,
            expectation_Jz_per_theta=exps,
            variance_Jz_per_theta=vars_,
        )


# ============================================================================
# Default configuration
# ============================================================================

# Note: Default parameters are inlined in function signatures below.
# See Global Constraints §6 — no module-level constants for defaults.


# ============================================================================
# Decoupled Baseline
# ============================================================================


def compute_drive_decoupled_baseline(
    T_H: float = 10.0,
    theta_true: float = 1.0,
) -> DriveDecoupledBaselineResult:
    """Compute the decoupled baseline sensitivity Δθ.

    At (a_x = a_y = a_z = a_zz = 0), the driving-ancilla circuit reduces
    to a standard single-qubit MZI with |1,0⟩ input and 50/50 BS,
    giving Δθ = 1/T_H.

    Args:
        T_H: Holding-time strength.
        theta_true: True phase rate.

    Returns:
        DriveDecoupledBaselineResult.
    """
    ops = build_two_qubit_operators()
    dtheta = compute_drive_sensitivity(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
        np.pi / 2.0,
        T_H,
        theta_true,
        0.0,
        0.0,
        0.0,
        0.0,
        ops,
    )
    return DriveDecoupledBaselineResult(
        T_H_value=T_H,
        delta_theta=dtheta,
        sql=1.0 / T_H,
    )


# ============================================================================
# 2D Slice Scan (ax, ay, or az)
# ============================================================================


def _drive_slice_chunk_worker(args: tuple) -> tuple[int, np.ndarray]:
    """Worker for parallel 2D slice evaluation (module-level for pickling).

    Args:
        args: Tuple (theta, drive_chunk, azz_vals, slice_type, T_H, T_BS, start_idx).

    Returns:
        Tuple (start_idx, chunk_grid) where chunk_grid has shape
        (len(drive_chunk), len(azz_vals)).
    """
    theta, drive_chunk, azz_vals, slice_type, T_H, T_BS, start_idx = args
    local_ops = build_two_qubit_operators()
    n_d = len(drive_chunk)
    n_a = len(azz_vals)
    chunk_grid = np.full((n_d, n_a), np.inf, dtype=float)
    for i, d_val in enumerate(drive_chunk):
        if slice_type == "ax":
            ax, ay, az = d_val, 0.0, 0.0
        elif slice_type == "ay":
            ax, ay, az = 0.0, d_val, 0.0
        else:
            ax, ay, az = 0.0, 0.0, d_val
        for j, a_val in enumerate(azz_vals):
            chunk_grid[i, j] = compute_drive_sensitivity(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
                T_BS,
                T_H,
                theta,
                ax,
                ay,
                az,
                a_val,
                local_ops,
            )
    return start_idx, chunk_grid


def drive_2d_slice(
    theta: float,
    drive_range: tuple[float, float] = (-5.0, 5.0),
    azz_range: tuple[float, float] = (-5.0, 5.0),
    n_drive: int = 201,
    n_azz: int = 201,
    slice_type: str = "ax",
    T_H: float = 10.0,
    T_BS: float = np.pi / 2.0,
    n_jobs: int | None = None,
) -> Drive2DSliceResult:
    """Run a 2D slice scan over (a_drive, a_zz).

    For slice_type='ax': varies a_x (with a_y = a_z = 0).
    For slice_type='ay': varies a_y (with a_x = a_z = 0).
    For slice_type='az': varies a_z (with a_x = a_y = 0).

    When n_jobs > 1, the grid is split across ``n_jobs`` worker processes
    for parallel evaluation.

    Args:
        theta: Phase rate value.
        drive_range: (min, max) for the drive coefficient.
        azz_range: (min, max) for the interaction coefficient.
        n_drive: Number of drive-coefficient points.
        n_azz: Number of a_zz points.
        slice_type: 'ax', 'ay', or 'az'.
        T_H: Holding time (default 10).
        T_BS: Beam-splitter duration (default π/2).
        n_jobs: Number of parallel workers. ``None`` (default) = sequential.
            Pass ``-1`` to use all available CPUs.

    Returns:
        Drive2DSliceResult with the sensitivity grid.
    """
    if slice_type not in ("ax", "ay", "az"):
        raise ValueError(f"slice_type must be 'ax', 'ay' or 'az', got {slice_type}")

    drive_vals = np.linspace(drive_range[0], drive_range[1], n_drive)
    azz_vals = np.linspace(azz_range[0], azz_range[1], n_azz)

    if n_jobs is None or n_jobs == 1:
        # ── Sequential path ──────────────────────────────────────────────
        ops = build_two_qubit_operators()
        grid = np.full((n_drive, n_azz), np.inf, dtype=float)

        for i, d_val in enumerate(drive_vals):
            for j, a_val in enumerate(azz_vals):
                if slice_type == "ax":
                    ax, ay, az = d_val, 0.0, 0.0
                elif slice_type == "ay":
                    ax, ay, az = 0.0, d_val, 0.0
                else:
                    ax, ay, az = 0.0, 0.0, d_val

                dtheta = compute_drive_sensitivity(
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
                    T_BS,
                    T_H,
                    theta,
                    ax,
                    ay,
                    az,
                    a_val,
                    ops,
                )
                grid[i, j] = dtheta
    else:
        # ── Parallel path ────────────────────────────────────────────────
        n_workers = max(1, os.cpu_count() or 4) if n_jobs == -1 else n_jobs
        # Split drive indices into roughly equal chunks
        drive_indices = np.arange(n_drive)
        chunks = np.array_split(drive_indices, n_workers)
        worker_args = [
            (
                theta,
                drive_vals[chunk],
                azz_vals,
                slice_type,
                T_H,
                T_BS,
                int(chunk[0]),
            )
            for chunk in chunks
        ]

        grid = np.full((n_drive, n_azz), np.inf, dtype=float)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
        ) as executor:
            futures = {
                executor.submit(_drive_slice_chunk_worker, args): args
                for args in worker_args
            }
            for future in concurrent.futures.as_completed(futures):
                start_idx, chunk_grid = future.result()
                n_chunk = chunk_grid.shape[0]
                grid[start_idx : start_idx + n_chunk, :] = chunk_grid

    return Drive2DSliceResult(
        drive_values=drive_vals,
        azz_values=azz_vals,
        delta_theta_grid=grid,
        theta_value=theta,
        slice_type=slice_type,
        sql=1.0 / T_H,
    )


# ============================================================================
# 4D Random Search
# ============================================================================


def drive_random_search(
    theta: float,
    n_samples: int = 500,
    bounds: tuple[float, float] = (-5.0, 5.0),
    T_H: float = 10.0,
    T_BS: float = np.pi / 2.0,
    seed: int | None = 42,
) -> DriveRandomSearchResult:
    """Random search over the 4D parameter space (a_x, a_y, a_z, a_zz).

    Args:
        theta: Phase rate value.
        n_samples: Number of random points to evaluate.
        bounds: (min, max) for all four coefficients.
        T_H: Holding time.
        T_BS: Beam-splitter duration.
        seed: Random seed for reproducibility.

    Returns:
        DriveRandomSearchResult with all samples and best found.
    """
    rng = np.random.default_rng(seed)
    ops = build_two_qubit_operators()
    lo, hi = bounds

    samples = rng.uniform(lo, hi, size=(n_samples, 4))
    deltas = np.full(n_samples, np.inf, dtype=float)

    for i in range(n_samples):
        ax = float(samples[i, 0])
        ay = float(samples[i, 1])
        az = float(samples[i, 2])
        azz = float(samples[i, 3])

        dtheta = compute_drive_sensitivity(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
            T_BS,
            T_H,
            theta,
            ax,
            ay,
            az,
            azz,
            ops,
        )
        deltas[i] = dtheta

    best_idx = int(np.argmin(deltas))
    best_params: tuple[float, float, float, float] = (
        float(samples[best_idx, 0]),
        float(samples[best_idx, 1]),
        float(samples[best_idx, 2]),
        float(samples[best_idx, 3]),
    )

    return DriveRandomSearchResult(
        samples=samples,
        delta_theta_values=deltas,
        best_params=best_params,
        best_delta_theta=float(deltas[best_idx]),
        theta_value=theta,
        sql=1.0 / T_H,
        T_H=T_H,
    )


# ============================================================================
# Nelder--Mead Optimisation
# ============================================================================


def drive_sensitivity_objective(
    params: np.ndarray,
    theta_true: float,
    ops: dict[str, np.ndarray],
    T_H: float = 10.0,
    T_BS: float = np.pi / 2.0,
    fd_step: float = 1e-6,
    bounds: tuple[float, float] = (-5.0, 5.0),
    penalty_scale: float = 1e6,
) -> float:
    """Objective function for minimising Δθ in the driven-ancilla protocol.

    Fixed configuration: |00⟩ initial state, fixed T_BS, fixed T_H.
    params = [a_x, a_y, a_z, a_zz] (4 elements).

    Args:
        params: 4-element parameter vector.
        theta_true: True phase rate.
        ops: Two-qubit operators.
        T_H: Holding time.
        T_BS: Beam-splitter duration.
        fd_step: Finite-difference step.
        bounds: (min, max) for all parameters.
        penalty_scale: Scale for bound-violation penalty.

    Returns:
        Δθ (plus infinite penalty if bounds violated).
    """
    ax = float(params[0])
    ay = float(params[1])
    az = float(params[2])
    azz = float(params[3])

    # Bound enforcement
    lo, hi = bounds
    penalty = 0.0
    for val in (ax, ay, az, azz):
        if val < lo:
            penalty += penalty_scale * (lo - val) ** 2
        if val > hi:
            penalty += penalty_scale * (val - hi) ** 2

    if penalty > 0.0:
        return float(1e10 + penalty)

    return compute_drive_sensitivity(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
        T_BS,
        T_H,
        theta_true,
        ax,
        ay,
        az,
        azz,
        ops,
        fd_step,
    )


def run_drive_nelder_mead(
    theta_true: float,
    x0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = 5000,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    bounds: tuple[float, float] = (-5.0, 5.0),
    T_H: float = 10.0,
    T_BS: float = np.pi / 2.0,
    track_history: bool = False,
) -> DriveNelderMeadResult:
    """Run Nelder--Mead optimisation for the driven-ancilla protocol.

    Args:
        theta_true: True phase rate parameter.
        x0: Initial 4-parameter vector [ax, ay, az, azz]. Random if None.
        seed: Random seed (used if x0 is None).
        maxiter: Maximum Nelder--Mead iterations.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function tolerance.
        adaptive: Use adaptive Nelder--Mead parameters.
        bounds: (min, max) for all four parameters.
        T_H: Holding time.
        T_BS: Beam-splitter duration.
        track_history: If True, record objective values per iteration.

    Returns:
        DriveNelderMeadResult.
    """
    ops = build_two_qubit_operators()

    if x0 is None:
        rng = np.random.default_rng(seed)
        lo, hi = bounds
        x0 = rng.uniform(lo, hi, size=4)
    else:
        x0 = np.asarray(x0, dtype=float)
        assert x0.shape == (4,), f"x0 must have 4 elements, got {x0.shape}"

    def objective(p: np.ndarray) -> float:
        return drive_sensitivity_objective(
            p,
            theta_true,
            ops,
            T_H=T_H,
            T_BS=T_BS,
            bounds=bounds,
        )

    history: list[float] = []

    def callback(_x: np.ndarray) -> None:
        if track_history:
            val = objective(_x)
            history.append(val)

    result = minimize(
        objective,
        x0=x0,
        method="Nelder-Mead",
        callback=callback if track_history else None,  # type: ignore[arg-type]
        options={  # type: ignore[call-overload]
            "maxiter": maxiter,
            "xatol": xatol,
            "fatol": fatol,
            "adaptive": adaptive,
        },
    )

    opt_params = result.x.copy()

    # Compute diagnostics at the optimal point
    psi_final = evolve_drive_circuit(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
        T_BS,
        T_H,
        theta_true,
        float(opt_params[0]),
        float(opt_params[1]),
        float(opt_params[2]),
        float(opt_params[3]),
        ops,
    )
    exp_val, var_val = compute_expectation_and_variance(psi_final, ops["Jz_S"])

    return DriveNelderMeadResult(
        delta_theta_opt=float(result.fun),
        params_opt=opt_params,
        theta_true=theta_true,
        success=bool(result.success),
        nfev=int(result.nfev),
        message=str(result.message),
        expectation_Jz=exp_val,
        variance_Jz=var_val,
        history=history.copy(),
    )


# ============================================================================
# θ Scan with Random Search + Nelder--Mead Refinement
# ============================================================================


def run_drive_theta_scan(
    theta_values: list[float] | np.ndarray,
    n_random: int = 500,
    n_nm_refine: int = 50,
    seed: int | None = 42,
    maxiter: int = 5000,
    bounds: tuple[float, float] = (-5.0, 5.0),
    T_H: float = 10.0,
    T_BS: float = np.pi / 2.0,
) -> DriveThetaScanResult:
    """Scan over θ values with 4D random search and Nelder--Mead refinement.

    For each θ:
    1. Run `n_random` random evaluations in the 4D parameter space.
    2. Select the best `n_nm_refine` points.
    3. Run Nelder--Mead refinement from each selected point.
    4. Record the best overall result.

    Args:
        theta_values: θ values to scan.
        n_random: Number of random search points per θ.
        n_nm_refine: Number of Nelder--Mead refinements per θ.
        seed: Base random seed (incremented per θ).
        maxiter: Maximum Nelder--Mead iterations.
        bounds: (min, max) for all parameters.
        T_H: Holding time.
        T_BS: Beam-splitter duration.

    Returns:
        DriveThetaScanResult with optimal parameters and sensitivities.
    """
    theta_arr = np.asarray(theta_values, dtype=float)
    base_seed = seed if seed is not None else 42

    best_params_list: list[tuple[float, float, float, float]] = []
    best_deltas: list[float] = []
    sql_vals: list[float] = []
    exp_vals: list[float] = []
    var_vals: list[float] = []
    all_results_dict: dict[float, list[DriveNelderMeadResult]] = {}

    for theta in theta_arr:
        # Stage 1: Random search
        rs_result = drive_random_search(
            theta,
            n_samples=n_random,
            bounds=bounds,
            T_H=T_H,
            T_BS=T_BS,
            seed=base_seed + int(theta * 1000),
        )

        # Sort random-search results by Δθ, take top n_nm_refine
        sorted_indices = np.argsort(rs_result.delta_theta_values)
        top_indices = sorted_indices[:n_nm_refine]

        # Stage 2: Nelder--Mead refinement from each top point
        nm_results: list[DriveNelderMeadResult] = []
        for rank, idx in enumerate(top_indices):
            x0 = rs_result.samples[idx].copy()
            nm = run_drive_nelder_mead(
                theta_true=theta,
                x0=x0,
                seed=base_seed + int(theta * 1000) + 10000 + rank,
                maxiter=maxiter,
                bounds=bounds,
                T_H=T_H,
                T_BS=T_BS,
                track_history=False,
            )
            nm_results.append(nm)

        # Sort Nelder--Mead results by Δθ
        nm_results.sort(key=lambda r: r.delta_theta_opt)
        best_nm = nm_results[0]

        best_params_list.append(
            (
                float(best_nm.params_opt[0]),
                float(best_nm.params_opt[1]),
                float(best_nm.params_opt[2]),
                float(best_nm.params_opt[3]),
            )
        )
        best_deltas.append(best_nm.delta_theta_opt)
        sql_vals.append(1.0 / T_H)
        exp_vals.append(best_nm.expectation_Jz)
        var_vals.append(best_nm.variance_Jz)
        all_results_dict[float(theta)] = nm_results

    return DriveThetaScanResult(
        theta_values=theta_arr,
        best_params_per_theta=best_params_list,
        best_delta_theta_per_theta=np.array(best_deltas, dtype=float),
        sql_values=np.array(sql_vals, dtype=float),
        expectation_Jz_per_theta=np.array(exp_vals, dtype=float),
        variance_Jz_per_theta=np.array(var_vals, dtype=float),
        all_results=all_results_dict,
    )
