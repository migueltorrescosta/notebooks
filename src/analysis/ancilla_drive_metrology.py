"""
Ancilla-Drive-Enhanced Metrology: Beating the SQL with an Actively Driven Ancilla.

Implements the driven-ancilla metrology protocol described in
``reports/r20260518/Ancilla-Drive-Enhanced-Metrology.md``.

Physical Model:
- Two qubits (system S + ancilla A), each a spin-1/2 (single-particle subspace).
- Basis: {|00⟩, |01⟩, |10⟩, |11⟩} where |0⟩ = |1,0⟩ (particle in mode 0).
- Circuit: BS_S → Hold → BS_S, where BS_S acts only on the system qubit.
- Hold Hamiltonian:
    H = ω J_z^S + H_A + H_int
    H_A = a_x J_x^A + a_y J_y^A + a_z J_z^A   (ancilla drive)
    H_int = a_zz J_z^S ⊗ J_z^A                (Ising interaction)
- Initial state: |00⟩ (both qubits in |1,0⟩).
- Measurement: J_z^S on the system qubit.
- Sensitivity: Δω via error propagation (central finite differences).

Units:
- Dimensionless throughout. ω is the unknown phase rate.
- t_hold: holding-time strength (dimensionless).
- a_x, a_y, a_z, a_zz: real coefficients.

References:
- Report ``reports/r20260518/Ancilla-Drive-Enhanced-Metrology.md``
- Giovannetti, Lloyd, Maccone, Nat. Photonics 5, 222 (2011)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

# Reuse shared primitives from ancilla_optimization
from src.analysis.ancilla_optimization import (
    I_2,
    build_two_qubit_operators,
    compute_expectation_and_variance,
    free_ancilla_initial_state,
)
from src.physics.beam_splitter import bs_qubit
from src.utils.constants import I_4

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class _DebugConfig:
    """Gated-debug configuration for operator assertions.

    Set ``verify_operators=True`` in unit tests to verify operator
    properties on every call.  When ``False`` (optimisation hot path),
    expensive assertions are skipped.
    """

    verify_operators: bool = False


# Module-level singleton.  Unit tests can set ``_debug.verify_operators = True``
# to enable cold-path assertion checks without changing call signatures.
_debug = _DebugConfig()


def _expm_hermitian(H: np.ndarray, t: float) -> np.ndarray:
    """Exponentiate a Hermitian matrix ``exp(-i t H)`` via ``eigh``.

    For a small (4×4) Hermitian matrix this is 2–3× faster than
    ``scipy.linalg.expm`` because it exploits Hermiticity.
    """
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvecs @ np.diag(np.exp(-1j * t * eigvals)) @ eigvecs.conj().T


# ============================================================================
# Operator Construction
# ============================================================================


@lru_cache(maxsize=8)
def system_only_bs_unitary(T_BS: float) -> np.ndarray:
    """Single-qubit beam-splitter on the system, identity on the ancilla.

    U = U_BS(T_BS) ⊗ I_2 = exp(-i T_BS J_x^S) ⊗ I_2

    A 50/50 beam splitter corresponds to T_BS = π/2.

    Args:
        T_BS: Beam-splitter duration.

    Returns:
        4×4 unitary matrix.
    """
    U = np.kron(bs_qubit(T_BS), I_2)
    if _debug.verify_operators:
        assert np.allclose(U @ U.conj().T, I_4, atol=1e-12), (
            f"System-only BS unitary not unitary for T_BS={T_BS}"
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


def build_phase_modulated_drive_hamiltonian(
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    r"""Build the :math:`\omega`-modulated ancilla drive Hamiltonian.

    :math:`H_A = \omega\,(a_x J_x^A + a_y J_y^A + a_z J_z^A)`

    The leading :math:`\omega` factor means
    :math:`\partial H/\partial\omega = J_z^S + H_A^{\text{norm}}`, providing an
    extra channel for :math:`\omega`-dependence via the ancilla operators.

    Args:
        omega: Unknown phase rate parameter (scales the whole drive).
        a_x: Coefficient for :math:`J_x^A`.
        a_y: Coefficient for :math:`J_y^A`.
        a_z: Coefficient for :math:`J_z^A`.
        ops: Two-qubit operators from :func:`build_two_qubit_operators`.

    Returns:
        4×4 Hermitian matrix representing the :math:`\omega`-modulated ancilla drive.
    """
    H = np.zeros((4, 4), dtype=complex)
    if a_x != 0.0:
        H += a_x * ops["Jx_A"]
    if a_y != 0.0:
        H += a_y * ops["Jy_A"]
    if a_z != 0.0:
        H += a_z * ops["Jz_A"]
    H = omega * H
    return 0.5 * (H + H.conj().T)


def build_phase_modulated_hold_hamiltonian(
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    r"""Build the total holding Hamiltonian with :math:`\omega`-modulated ancilla drive.

    :math:`H = \omega J_z^S + \omega (a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_{zz} J_z^S \otimes J_z^A`

    The :math:`\omega` factor on the drive terms means
    :math:`\partial H/\partial\omega = J_z^S + a_x J_x^A + a_y J_y^A + a_z J_z^A`,
    which includes ancilla operators.

    Args:
        omega: Unknown phase rate parameter.
        a_x: Ancilla :math:`J_x` drive coefficient.
        a_y: Ancilla :math:`J_y` drive coefficient.
        a_z: Ancilla :math:`J_z` drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators from :func:`build_two_qubit_operators`.

    Returns:
        4×4 Hermitian Hamiltonian matrix.
    """
    H = omega * ops["Jz_S"]
    H += build_phase_modulated_drive_hamiltonian(omega, a_x, a_y, a_z, ops)
    H += build_iszz_interaction(a_zz, ops)
    return 0.5 * (H + H.conj().T)


def phase_modulated_hold_unitary(
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    r"""Holding-time unitary for the :math:`\omega`-modulated ancilla protocol.

    :math:`U_{\text{hold}}(T_{\text{hold}}) = \exp(-i T_{\text{hold}} H)`
    where :math:`H = \omega J_z^S + \omega(a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_{zz} J_z^S \otimes J_z^A`.

    Args:
        t_hold: Holding-time strength.
        omega: True phase rate parameter.
        a_x: Ancilla :math:`J_x` drive coefficient.
        a_y: Ancilla :math:`J_y` drive coefficient.
        a_z: Ancilla :math:`J_z` drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators from :func:`build_two_qubit_operators`.

    Returns:
        4×4 unitary matrix.
    """
    H = build_phase_modulated_hold_hamiltonian(omega, a_x, a_y, a_z, a_zz, ops)
    U = _expm_hermitian(H, t_hold)
    if _debug.verify_operators:
        assert np.allclose(U @ U.conj().T, I_4, atol=1e-12), (
            f"Phase-modulated hold unitary not unitary for t_hold={t_hold}, ω={omega}"
        )
    return U


def evolve_phase_modulated_circuit(
    psi0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    r"""Run the full :math:`\omega`-modulated ancilla MZI circuit.

    :math:`|\psi_{\text{final}}\rangle = U_{\text{BS}}^{(S)} \,
    U_{\text{hold}}(T_{\text{hold}}) \, U_{\text{BS}}^{(S)} \, |\psi_0\rangle`

    The hold unitary uses the :math:`\omega`-modulated
    :math:`H_A = \omega (a_x J_x^A + a_y J_y^A + a_z J_z^A)`.

    Args:
        psi0: Initial 4-vector (must be normalised).
        T_BS: Beam-splitter duration (both BS identical).
        t_hold: Holding-time strength.
        omega: Phase rate parameter.
        a_x: Ancilla :math:`J_x` drive coefficient.
        a_y: Ancilla :math:`J_y` drive coefficient.
        a_z: Ancilla :math:`J_z` drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators.

    Returns:
        Final normalised 4-vector state.
    """
    if _debug.verify_operators:
        assert np.isclose(np.linalg.norm(psi0), 1.0), "Initial state must be normalised"

    U_bs = system_only_bs_unitary(T_BS)
    psi = U_bs @ psi0
    psi = phase_modulated_hold_unitary(t_hold, omega, a_x, a_y, a_z, a_zz, ops) @ psi
    psi = U_bs @ psi

    if _debug.verify_operators:
        assert np.isclose(np.linalg.norm(psi), 1.0), "Final state must be normalised"
    return psi


def compute_phase_modulated_sensitivity(
    psi0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega_true: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    fd_step: float = 1e-6,
    meas_op: np.ndarray | None = None,
) -> float:
    r"""Compute the error-propagation sensitivity :math:`\Delta\omega`.

    :math:`\Delta\omega = \sqrt{\mathrm{Var}(O)} / |\partial\langle O\rangle/\partial\omega|`

    Because :math:`\omega` appears in both :math:`H_S = \omega J_z^S` and
    :math:`H_A = \omega (a_x J_x^A + a_y J_y^A + a_z J_z^A)`, the central
    finite-difference step captures the full :math:`\omega`-dependence.

    Args:
        psi0: Initial 4-vector (product state).
        T_BS: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega_true: True phase rate parameter.
        a_x: Ancilla :math:`J_x` drive coefficient.
        a_y: Ancilla :math:`J_y` drive coefficient.
        a_z: Ancilla :math:`J_z` drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators (must contain ``'Jz_S'``).
        fd_step: Finite-difference step size (default ``1e-6``).
        meas_op: Measurement operator. Defaults to ``ops['Jz_S']`` (S-only).

    Returns:
        Sensitivity :math:`\Delta\omega` (positive float). Returns ``inf``
        if derivative is zero (fringe extremum).
    """
    if meas_op is None:
        meas_op = ops["Jz_S"]

    # Evaluate at omega_true
    psi = evolve_phase_modulated_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    _, var = compute_expectation_and_variance(psi, meas_op)

    # Central finite difference for ∂⟨O⟩/∂ω
    psi_plus = evolve_phase_modulated_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true + fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    psi_minus = evolve_phase_modulated_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true - fd_step,
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

    if var < 1e-15:
        return float("inf")

    return float(np.sqrt(var) / abs(d_exp))


def build_drive_hold_hamiltonian(
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the total holding Hamiltonian.

    H = ω J_z^S + H_A + H_int
      = ω J_z^S + (a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_zz J_z^S ⊗ J_z^A

    Args:
        omega: Unknown phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian Hamiltonian matrix.
    """
    H = omega * ops["Jz_S"]
    H += build_ancilla_drive_hamiltonian(a_x, a_y, a_z, ops)
    H += build_iszz_interaction(a_zz, ops)
    return 0.5 * (H + H.conj().T)


def drive_hold_unitary(
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Holding-time unitary for the driven-ancilla protocol.

    U_hold(t_hold) = exp(-i t_hold H)
    where H = ω J_z^S + H_A + H_int.

    Args:
        t_hold: Holding-time strength.
        omega: True phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 unitary matrix.
    """
    H = build_drive_hold_hamiltonian(omega, a_x, a_y, a_z, a_zz, ops)
    U = _expm_hermitian(H, t_hold)
    if _debug.verify_operators:
        assert np.allclose(U @ U.conj().T, I_4, atol=1e-12), (
            f"Drive hold unitary not unitary for t_hold={t_hold}, ω={omega}"
        )
    return U


def evolve_drive_circuit(
    psi0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the full driven-ancilla MZI circuit.

    |ψ_final⟩ = U_BS_S · U_hold(t_hold) · U_BS_S · |ψ₀⟩

    Args:
        psi0: Initial 4-vector (must be normalised).
        T_BS: Beam-splitter duration (both BS identical).
        t_hold: Holding-time strength.
        omega: Phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators.

    Returns:
        Final normalised 4-vector state.
    """
    if _debug.verify_operators:
        assert np.isclose(np.linalg.norm(psi0), 1.0), "Initial state must be normalised"

    U_bs = system_only_bs_unitary(T_BS)
    psi = U_bs @ psi0
    psi = drive_hold_unitary(t_hold, omega, a_x, a_y, a_z, a_zz, ops) @ psi
    psi = U_bs @ psi

    if _debug.verify_operators:
        assert np.isclose(np.linalg.norm(psi), 1.0), "Final state must be normalised"
    return psi


def compute_drive_sensitivity(
    psi0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega_true: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    fd_step: float = 1e-6,
    meas_op: np.ndarray | None = None,
) -> float:
    """Compute the error-propagation sensitivity Δω.

    Δω = sqrt(Var(O)) / |∂⟨O⟩/∂ω|

    where O is the measurement operator (default: J_z^S).

    Args:
        psi0: Initial 4-vector (product state).
        T_BS: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega_true: True phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators (must contain 'Jz_S').
        fd_step: Finite-difference step size (default 1e-6).
        meas_op: Measurement operator. Defaults to ops['Jz_S'] (S-only).

    Returns:
        Sensitivity Δω (positive float). Returns inf if derivative is zero
        (fringe extremum).

    """
    if meas_op is None:
        meas_op = ops["Jz_S"]

    # Evaluate at omega_true
    psi = evolve_drive_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    _, var = compute_expectation_and_variance(psi, meas_op)

    # Central finite difference for ∂⟨O⟩/∂omega
    psi_plus = evolve_drive_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true + fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    psi_minus = evolve_drive_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true - fd_step,
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
    # would yield Δω = 0 (unphysical), so flag as fringe extremum.
    if var < 1e-15:
        return float("inf")

    return float(np.sqrt(var) / abs(d_exp))


def compute_drive_sensitivity_with_details(
    psi0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega_true: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    fd_step: float = 1e-6,
    meas_op: np.ndarray | None = None,
) -> tuple[float, float, float, float, bool]:
    """Error-propagation sensitivity with diagnostic details.

    Same core computation as :func:`compute_drive_sensitivity` but also
    returns the intermediate expectation value, variance, derivative, and
    fringe flag alongside the sensitivity.

    Args:
        psi0: Initial 4-vector (product state).
        T_BS: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega_true: True phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators (must contain 'Jz_S').
        fd_step: Finite-difference step size (default 1e-6).
        meas_op: Measurement operator. Defaults to ops['Jz_S'] (S-only).

    Returns:
        Tuple ``(delta_omega, expectation, variance, derivative, is_fringe)``.
        ``is_fringe`` is ``True`` when sensitivity diverges (derivative near
        zero or zero-variance eigenstate).
    """
    if meas_op is None:
        meas_op = ops["Jz_S"]

    # Evaluate at omega_true
    psi = evolve_drive_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    exp_val, var_val = compute_expectation_and_variance(psi, meas_op)

    # Central finite difference for d<O>/domega
    psi_plus = evolve_drive_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true + fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    psi_minus = evolve_drive_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true - fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    exp_plus = np.real(psi_plus.conj() @ meas_op @ psi_plus)
    exp_minus = np.real(psi_minus.conj() @ meas_op @ psi_minus)
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    is_fringe = abs(d_exp) < 1e-12 or var_val < 1e-15
    if is_fringe:
        return float("inf"), exp_val, var_val, float(d_exp), True

    delta = float(np.sqrt(var_val) / abs(d_exp))
    return delta, exp_val, var_val, float(d_exp), False


def compute_free_ancilla_sensitivity(
    evolve_fn: Callable[
        [
            np.ndarray,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            dict[str, np.ndarray],
        ],
        np.ndarray,
    ],
    omega_true: float,
    theta_A: float,
    phi_A: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    *,
    t_hold: float = 10.0,
    T_BS: float = np.pi / 2.0,
    fd_step: float = 1e-6,
) -> tuple[float, float, float, float, bool]:
    """Error-propagation sensitivity with a free-ancilla initial state.

    Constructs the free-ancilla initial state :math:`|1,0\\rangle_S \\otimes
    |\\psi_A(\\theta_A,\\phi_A)\\rangle`, runs the circuit through
    *evolve_fn*, and computes :math:`\\Delta\\omega = \\sqrt{\\mathrm{Var}(O)}
    / |\\partial\\langle O\\rangle/\\partial\\omega|` with central finite
    differences.

    The *evolve_fn* callback must have the signature::

        evolve_fn(psi0, T_BS, t_hold, omega, a_x, a_y, a_z, a_zz, ops) -> psi_final

    Args:
        evolve_fn: Circuit evolution function (e.g. ``evolve_drive_circuit``
            or ``evolve_phase_modulated_circuit``).
        omega_true: True phase rate parameter.
        theta_A: Ancilla polar angle :math:`\\in [0, \\pi]`.
        phi_A: Ancilla azimuthal angle :math:`\\in [0, 2\\pi)`.
        a_x: Ancilla :math:`J_x` drive coefficient.
        a_y: Ancilla :math:`J_y` drive coefficient.
        a_z: Ancilla :math:`J_z` drive coefficient.
        a_zz: Ising interaction coefficient.
        t_hold: Holding-time strength (default 10.0).
        T_BS: Beam-splitter duration (default :math:`\\pi/2`).
        fd_step: Finite-difference step size (default 1e-6).

    Returns:
        Tuple ``(delta_omega, expectation, variance, derivative, is_fringe)``.
        ``is_fringe`` is ``True`` when sensitivity diverges (derivative near
        zero or zero-variance eigenstate).
    """
    psi0 = free_ancilla_initial_state(theta_A, phi_A)
    ops = build_two_qubit_operators()
    meas_op = ops["Jz_S"]

    # Evaluate at omega_true
    psi = evolve_fn(psi0, T_BS, t_hold, omega_true, a_x, a_y, a_z, a_zz, ops)
    exp_val, var_val = compute_expectation_and_variance(psi, meas_op)

    # Central finite difference for d<O>/domega
    psi_plus = evolve_fn(
        psi0,
        T_BS,
        t_hold,
        omega_true + fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    psi_minus = evolve_fn(
        psi0,
        T_BS,
        t_hold,
        omega_true - fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    exp_plus = float(np.real(psi_plus.conj() @ meas_op @ psi_plus))
    exp_minus = float(np.real(psi_minus.conj() @ meas_op @ psi_minus))
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12:
        return float("inf"), exp_val, var_val, 0.0, True

    if var_val < 1e-15:
        return float("inf"), exp_val, var_val, d_exp, True

    delta_omega = float(np.sqrt(var_val) / abs(d_exp))
    return delta_omega, exp_val, var_val, d_exp, False
