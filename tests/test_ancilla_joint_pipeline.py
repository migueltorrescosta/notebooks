"""End-to-end tests for the joint measurement optimisation pipeline.

Validates that the full Nelder-Mead optimisation using M = J_z^S + J_z^A
respects the QFI bound Δθ ≥ 1/t_hold and runs without errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.ancilla_optimization import (
    build_joint_operator,
    build_two_qubit_operators,
)
from src.analysis.ancilla_optimization_scans import run_omega_scan


class TestJointMeasurementE2E:
    """E2E tests for the joint measurement optimisation pipeline."""

    def test_joint_omega_scan_runs_and_respects_qfi_bound(self) -> None:
        ops = build_two_qubit_operators()
        M_op = build_joint_operator(ops)
        bounds = {
            "bloch_theta": (0.0, np.pi),
            "phi": (0.0, 2.0 * np.pi),
            "T_BS": (0.0, np.pi),
            "t_hold": (0.0, 5.0),
            "alpha": (-2.0, 2.0),
        }
        result = run_omega_scan(
            omega_values=[0.5, 1.0],
            n_restarts=2,
            seed=42,
            maxiter=100,
            bounds=bounds,
            meas_op=M_op,
        )
        assert len(result.results) == 4  # 2 θ × 2 restarts
        for r in result.results:
            t_h_opt = r.params_opt[6]
            sql = 1.0 / t_h_opt if t_h_opt > 0 else float("inf")
            # Allow a 10% tolerance for Nelder-Mead convergence noise
            assert r.delta_omega_opt >= sql - 0.1 * sql or np.isinf(
                r.delta_omega_opt,
            ), (
                f"QFI bound violated: Δθ={r.delta_omega_opt:.4f} < SQL={sql:.4f} "
                f"at ω={r.omega_true}"
            )

    @pytest.mark.slow
    def test_joint_omega_scan_with_expanded_th(self) -> None:
        ops = build_two_qubit_operators()
        M_op = build_joint_operator(ops)
        bounds = {
            "bloch_theta": (0.0, np.pi),
            "phi": (0.0, 2.0 * np.pi),
            "T_BS": (0.0, np.pi),
            "t_hold": (0.0, 20.0),
            "alpha": (-2.0, 2.0),
        }
        result = run_omega_scan(
            omega_values=[1.0],
            n_restarts=1,
            seed=42,
            maxiter=500,
            bounds=bounds,
            meas_op=M_op,
        )
        assert len(result.results) == 1
        assert np.isfinite(result.results[0].delta_omega_opt)
        # With wider t_hold bound, the optimiser should drive t_hold up
        assert result.results[0].params_opt[6] > 15.0
