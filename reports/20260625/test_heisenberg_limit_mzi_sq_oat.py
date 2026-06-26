r"""
Tests for the Heisenberg-Limit MZI: Squeezed Vacuum & OAT module (2026-06-25).

Run with:
    uv run pytest reports/20260625/test_heisenberg_limit_mzi_sq_oat.py -q --tb=short
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys as _sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from src.analysis.fisher_information import classical_fisher_information_single
from src.analysis.scaling_fit import fit_scaling_exponent
from src.physics.mzi_simulation import beam_splitter_unitary
from src.physics.mzi_states import (
    compute_jz_expectation,
    compute_jz_variance,
    input_state_factory,
    standard_twin_fock_state,
)
from src.utils.serialization import assert_roundtrip_fields

_local_path = Path(__file__).resolve().parent / "heisenberg_limit_mzi_sq_oat.py"
_spec = importlib.util.spec_from_file_location("local", str(_local_path))
assert _spec is not None
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_sys.modules["local"] = _module
_spec.loader.exec_module(_module)
del _local_path, _spec, _module

from local import (  # type: ignore[import-untyped]  # noqa: E402
    MziSensitivityDataSV,
    OATQScanResult,
    _check_truncation_convergence,
    _compute_sv_qfi,
    _compute_tmsv_qfi,
    _dicke_to_fock,
    _generate_single_resource_data,
    _make_oat_state,
    _make_two_mode_squeezed_vacuum,
    _maybe_generate_full_data,
    _maybe_plot_delta_omega_overlays,
    _oat_q_grid,
    _prepare_state,
    _resource_value_to_truncation,
    _verify_oat_q0_qfi,
    _verify_sv_qfi,
    _verify_tmsv_qfi,
    analyse_best_worst_sensitivity,
    compute_mzi_sensitivity_grid,
    generate_full_data,
    generate_single_omega_scan,
    main,
    output_number_diff_distribution,
    plot_delta_omega_overlay,
    plot_scaling,
    scan_oat_q,
    simple_mzi_evolution,
    t_hold,
)

# ============================================================================
# Two-Mode Squeezed Vacuum (TMSV) State
# ============================================================================


class TestTwoModeSqueezedVacuum:
    def test_normalized(self) -> None:
        """TMSV state must have unit norm."""
        for mean_total in [2, 4, 10, 20]:
            M = int(5 * mean_total)
            state = _make_two_mode_squeezed_vacuum(float(mean_total), M)
            assert np.isclose(np.linalg.norm(state), 1.0, rtol=1e-10), (
                f"Failed for mean_total={mean_total}"
            )

    def test_mean_photon_number(self) -> None:
        """Mean photon number should match the target value."""
        mean_total = 4.0
        M = 30
        state = _make_two_mode_squeezed_vacuum(mean_total, M)
        dim_single = M + 1
        # Compute mean photon number
        mean_n = 0.0
        for n1 in range(M + 1):
            for n2 in range(M + 1):
                idx = n1 * dim_single + n2
                prob = np.abs(state[idx]) ** 2
                mean_n += prob * (n1 + n2)
        assert np.isclose(mean_n, mean_total, rtol=1e-2), (
            f"Mean total N={mean_n}, expected {mean_total}"
        )

    def test_only_diagonal_components(self) -> None:
        """TMSV should only have |n, n⟩ components."""
        mean_total = 4.0
        M = 15
        state = _make_two_mode_squeezed_vacuum(mean_total, M)
        dim_single = M + 1
        for n1 in range(M + 1):
            for n2 in range(M + 1):
                idx = n1 * dim_single + n2
                if n1 != n2:
                    assert np.abs(state[idx]) < 1e-12, (
                        f"Non-zero component at |{n1},{n2}⟩"
                    )

    def test_jz_expectation_zero(self) -> None:
        """⟨J_z⟩ = 0 for TMSV (symmetric in both modes)."""
        for mean_total in [2, 4, 10]:
            M = int(5 * mean_total)
            state = _make_two_mode_squeezed_vacuum(float(mean_total), M)
            exp = np.real(compute_jz_expectation(state, M))
            assert np.isclose(exp, 0.0, atol=1e-12), (
                f"Failed for mean_total={mean_total}"
            )

    def test_positive_mean_N_required(self) -> None:
        """TMSV requires mean_total > 0."""
        with pytest.raises(ValueError, match="positive"):
            _make_two_mode_squeezed_vacuum(0.0, 10)


# ============================================================================
# Dicke-to-Fock Mapping
# ============================================================================


class TestDickeToFock:
    def test_normalized_mapping(self) -> None:
        """Dicke-to-Fock mapping preserves normalisation."""
        from src.algorithms.spin_squeezing import coherent_spin_state

        for N in [2, 4, 6, 10]:
            css = coherent_spin_state(N)
            fock = _dicke_to_fock(css, N)
            assert np.isclose(np.linalg.norm(fock), 1.0, rtol=1e-10), (
                f"Failed for N={N}"
            )

    def test_css_jz_variance(self) -> None:
        """CSS |J, -J⟩_x after mapping should have Var(J_z) = N/4."""
        from src.algorithms.spin_squeezing import coherent_spin_state

        for N in [2, 4, 6, 10]:
            css = coherent_spin_state(N)
            fock = _dicke_to_fock(css, N)
            var = compute_jz_variance(fock, N)
            expected = N / 4.0
            assert np.isclose(var, expected, rtol=1e-10), (
                f"N={N}: Var(J_z)={var}, expected {expected}"
            )

    def test_css_jz_expectation_zero(self) -> None:
        """CSS |J, -J⟩_x should have ⟨J_z⟩ = 0."""
        from src.algorithms.spin_squeezing import coherent_spin_state

        for N in [2, 4, 6, 10]:
            css = coherent_spin_state(N)
            fock = _dicke_to_fock(css, N)
            exp = np.real(compute_jz_expectation(fock, N))
            assert np.isclose(exp, 0.0, atol=1e-12), f"Failed for N={N}"

    def test_dimension_mismatch_raises(self) -> None:
        """Wrong Dicke state dimension must raise."""
        bad_state = np.ones(10, dtype=complex)
        with pytest.raises(ValueError, match="dimension"):
            _dicke_to_fock(bad_state, 6)

    def test_fock_state_mapping(self) -> None:
        """Mapping a Fock state |n1,n2⟩ to Dicke and back should recover it."""
        N = 6
        dim_single = N + 1
        # |3, 3⟩ = |J, 0⟩
        fock_idx = 3 * dim_single + 3
        fock = np.zeros(dim_single**2, dtype=complex)
        fock[fock_idx] = 1.0
        # Map to Dicke and back
        from src.physics.dicke_basis import to_dicke_basis

        dicke = to_dicke_basis(fock, N)
        fock_back = _dicke_to_fock(dicke, N)
        assert np.allclose(fock, fock_back), "Fock roundtrip failed"


# ============================================================================
# OAT Spin-Squeezed State
# ============================================================================


class TestOATState:
    def test_normalized(self) -> None:
        """OAT state must have unit norm."""
        for N in [2, 4, 6, 10]:
            for q in [0.0, 0.1, 1.0]:
                state = _make_oat_state(N, q)
                assert np.isclose(np.linalg.norm(state), 1.0, rtol=1e-10), (
                    f"Failed for N={N}, q={q}"
                )

    def test_css_q0_sql_variance(self) -> None:
        """OAT with q=0 (CSS) should have Var(J_z) = N/4."""
        for N in [2, 4, 6, 10]:
            state = _make_oat_state(N, 0.0)
            var = compute_jz_variance(state, N)
            expected = N / 4.0
            assert np.isclose(var, expected, rtol=1e-10), (
                f"N={N}: Var(J_z)={var}, expected {expected}"
            )

    def test_oat_changes_state(self) -> None:
        r"""OAT with q>0 changes the state (alters Dicke-basis phases).

        Note: OAT preserves :math:`\text{Var}(J_z)` because it acts as
        :math:`\exp(-i q J_z^2)` which only adds phases to each Dicke-basis
        component without changing the probability amplitudes. The QFI for
        :math:`J_z`-generated phase estimation is therefore unchanged by OAT.
        """
        N = 10
        css_state = _make_oat_state(N, 0.0)
        q_test = 0.5
        oat_state = _make_oat_state(N, q_test)
        # The state vectors must differ (phases change)
        assert not np.allclose(css_state, oat_state, rtol=1e-10), (
            f"OAT q={q_test} did not change the state"
        )

    def test_even_N_required(self) -> None:
        """OAT requires even N."""
        with pytest.raises(ValueError, match="even"):
            _make_oat_state(3, 0.5)

    def test_N_ge_2_required(self) -> None:
        """OAT requires N >= 2."""
        with pytest.raises(ValueError, match=">= 2"):
            _make_oat_state(0, 0.5)


# ============================================================================
# Standard Twin-Fock State (inline helper)
# ============================================================================


class TestStandardTwinFockState:
    def test_normalized(self) -> None:
        """|N/2, N/2⟩ must have unit norm."""
        for N in [2, 4, 6, 10]:
            state = standard_twin_fock_state(N, N)
            assert np.isclose(np.linalg.norm(state), 1.0), f"Failed for N={N}"

    def test_jz_expectation_zero(self) -> None:
        """⟨J_z⟩ = 0 for |N/2, N/2⟩."""
        for N in [2, 4, 6, 10]:
            state = standard_twin_fock_state(N, N)
            exp = np.real(compute_jz_expectation(state, N))
            assert np.isclose(exp, 0.0, atol=1e-12), f"Failed for N={N}"

    def test_jz_variance_zero(self) -> None:
        """Var(J_z) = 0 for |N/2, N/2⟩ before BS1."""
        for N in [2, 4, 6, 10]:
            state = standard_twin_fock_state(N, N)
            var = compute_jz_variance(state, N)
            assert np.isclose(var, 0.0, atol=1e-12), f"Failed for N={N}"

    def test_odd_N_raises(self) -> None:
        """Odd N must raise."""
        with pytest.raises(ValueError, match="even"):
            standard_twin_fock_state(3, 5)


# ============================================================================
# Simple MZI Evolution
# ============================================================================


class TestSimpleMziEvolution:
    @pytest.mark.parametrize("N", [1, 2, 4, 6])
    def test_norm_preserved_sv(self, N: int) -> None:
        """MZI evolution must preserve norm for SV state (skip_bs1=True)."""
        M = int(5 * N)
        state = input_state_factory(
            "squeezed_vacuum",
            N,
            M,
            r=float(np.arcsinh(np.sqrt(float(N)))),
        )
        final = simple_mzi_evolution(
            state,
            omega=1.0,
            max_photons=M,
            t_hold=t_hold,
            skip_bs1=True,
        )
        assert np.isclose(np.linalg.norm(final), 1.0, rtol=1e-10), f"Failed for N={N}"

    @pytest.mark.parametrize("N", [2, 4, 6])
    def test_norm_preserved_oat(self, N: int) -> None:
        """MZI evolution must preserve norm for OAT state."""
        state = _make_oat_state(N, 0.5)
        final = simple_mzi_evolution(
            state,
            omega=1.0,
            max_photons=N,
            t_hold=t_hold,
            skip_bs1=True,
        )
        assert np.isclose(np.linalg.norm(final), 1.0, rtol=1e-10), f"Failed for N={N}"

    @pytest.mark.parametrize("state_type", ["sv", "tmsv", "oat"])
    def test_output_variance_nonnegative(self, state_type: str) -> None:
        """Output variance must be non-negative."""
        R = 4
        if state_type == "sv":
            M = int(5 * R)
            state = input_state_factory(
                "squeezed_vacuum",
                R,
                M,
                r=float(np.arcsinh(np.sqrt(float(R)))),
            )
            skip = True
        elif state_type == "tmsv":
            M = int(5 * R)
            state = _make_two_mode_squeezed_vacuum(float(R), M)
            skip = False
        else:
            M = R
            state = _make_oat_state(R, 0.5)
            skip = True
        final = simple_mzi_evolution(
            state,
            omega=0.5,
            max_photons=M,
            t_hold=t_hold,
            skip_bs1=skip,
        )
        var = compute_jz_variance(final, M)
        assert var >= -1e-12, f"Negative variance: {var}"


# ============================================================================
# SV QFI Validation
# ============================================================================


class TestSVQFI:
    # SV state needs large M for accurate truncation (even at small ⟨N⟩,
    # the SV photon-number distribution decays slowly).
    _SV_M: ClassVar[dict[int, int]] = {1: 50, 2: 50}

    @pytest.mark.parametrize("mean_N", [1, 2])
    def test_var_jz_probe(self, mean_N: int) -> None:
        r"""Var(J_z) for SV (skip_bs1=True) should match analytical formula.

        The SV state is the probe directly: Var(J_z) = ⟨N⟩(⟨N⟩+1)/2.
        """
        M = self._SV_M[mean_N]
        r = float(np.arcsinh(np.sqrt(float(mean_N))))
        state = input_state_factory("squeezed_vacuum", mean_N, M, r=r)
        var = compute_jz_variance(state, M)
        expected_var = mean_N * (mean_N + 1) / 2.0
        assert np.isclose(var, expected_var, rtol=1e-3), (
            f"mean_N={mean_N}: Var={var}, expected {expected_var}"
        )

    @pytest.mark.parametrize("mean_N", [1, 2])
    def test_qfi_bound(self, mean_N: int) -> None:
        r"""F_Q = 2 * t_hold² * ⟨N⟩(⟨N⟩+1) for SV probe state."""
        M = self._SV_M[mean_N]
        r = float(np.arcsinh(np.sqrt(float(mean_N))))
        state = input_state_factory("squeezed_vacuum", mean_N, M, r=r)
        var = compute_jz_variance(state, M)
        fq = 4.0 * t_hold**2 * var
        expected_fq = _compute_sv_qfi(float(mean_N), t_hold)
        assert np.isclose(fq, expected_fq, rtol=1e-3), (
            f"mean_N={mean_N}: F_Q={fq}, expected {expected_fq}"
        )

    @pytest.mark.parametrize("mean_N", [1])
    def test_verify_sv_qfi_helper(self, mean_N: int) -> None:
        """Verification helper must return True for SV probe."""
        M = self._SV_M[mean_N]
        r = float(np.arcsinh(np.sqrt(float(mean_N))))
        state = input_state_factory("squeezed_vacuum", mean_N, M, r=r)
        var = compute_jz_variance(state, M)
        assert _verify_sv_qfi(float(mean_N), var), f"Failed at mean_N={mean_N}"


# ============================================================================
# TMSV QFI Validation
# ============================================================================


class TestTMSVQFI:
    _TMSV_M: ClassVar[dict[int, int]] = {2: 40, 4: 50}

    @pytest.mark.parametrize("mean_total", [2, pytest.param(4, marks=pytest.mark.slow)])
    def test_var_jz_after_bs1(self, mean_total: int) -> None:
        r"""Var(J_z) for TMSV after BS1 should match analytical formula.

        Var(J_z) = ⟨N⟩(⟨N⟩+2)/4
        """
        M = self._TMSV_M[mean_total]
        state = _make_two_mode_squeezed_vacuum(float(mean_total), M)
        bs = beam_splitter_unitary(np.pi / 4, 0.0, M)
        probe = bs @ state
        var = compute_jz_variance(probe, M)
        expected_var = mean_total * (mean_total + 2) / 4.0
        assert np.isclose(var, expected_var, rtol=1e-3), (
            f"mean_total={mean_total}: Var={var}, expected {expected_var}"
        )

    @pytest.mark.parametrize("mean_total", [2, pytest.param(4, marks=pytest.mark.slow)])
    def test_qfi_bound(self, mean_total: int) -> None:
        r"""F_Q = t_hold² * ⟨N⟩(⟨N⟩+2) for TMSV."""
        M = self._TMSV_M[mean_total]
        state = _make_two_mode_squeezed_vacuum(float(mean_total), M)
        bs = beam_splitter_unitary(np.pi / 4, 0.0, M)
        probe = bs @ state
        var = compute_jz_variance(probe, M)
        fq = 4.0 * t_hold**2 * var
        expected_fq = _compute_tmsv_qfi(float(mean_total), t_hold)
        assert np.isclose(fq, expected_fq, rtol=1e-3), (
            f"mean_total={mean_total}: F_Q={fq}, expected {expected_fq}"
        )

    @pytest.mark.parametrize("mean_total", [2])
    def test_verify_tmsv_qfi_helper(self, mean_total: int) -> None:
        """Verification helper must return True."""
        M = self._TMSV_M[mean_total]
        state = _make_two_mode_squeezed_vacuum(float(mean_total), M)
        bs = beam_splitter_unitary(np.pi / 4, 0.0, M)
        probe = bs @ state
        var = compute_jz_variance(probe, M)
        assert _verify_tmsv_qfi(float(mean_total), var), (
            f"Failed at mean_total={mean_total}"
        )


# ============================================================================
# OAT QFI Validation
# ============================================================================


class TestOATQFI:
    @pytest.mark.parametrize("N", [2, 4, 6, 10])
    def test_css_q0_sql_qfi(self, N: int) -> None:
        r"""OAT q=0 (CSS) should give Var(J_z) = N/4 (SQL)."""
        state = _make_oat_state(N, 0.0)
        var = compute_jz_variance(state, N)
        assert np.isclose(var, N / 4.0, rtol=1e-10), (
            f"N={N}: Var={var}, expected {N / 4.0}"
        )

    @pytest.mark.parametrize("N", [2, 4])
    def test_verify_oat_q0_qfi_helper(self, N: int) -> None:
        """Verification helper must return True for CSS."""
        state = _make_oat_state(N, 0.0)
        var = compute_jz_variance(state, N)
        assert _verify_oat_q0_qfi(N, var), f"Failed at N={N}"

    @pytest.mark.parametrize("N", [4, 6])
    def test_oat_variance_positive(self, N: int) -> None:
        r"""OAT with optimal q produces positive Var(J_z).

        Note: OAT preserves Var(J_z) = N/4 (SQL) for a J_z generator because
        the unitary :math:`\exp(-i q J_z^2)` only adds phases to each Dicke
        component. The QFI for :math:`J_z`-generated phase is therefore
        unchanged by OAT. Spin-squeezing enhances sensitivity only for a
        phase generator orthogonal to the squeezing axis.
        """
        q_opt_est = (6.0 / N) ** (1.0 / 3.0)
        state = _make_oat_state(N, q_opt_est)
        var = compute_jz_variance(state, N)
        assert var > 0, f"Non-positive variance at N={N}"


# ============================================================================
# MZI Sensitivity Grid Computation
# ============================================================================


class TestComputeMziSensitivityGrid:
    @pytest.mark.parametrize("state_type", ["sv", "tmsv", "oat"])
    def test_returns_required_keys(self, state_type: str) -> None:
        """Result dict must contain all expected arrays."""
        R = 4
        M = int(5 * R) if state_type != "oat" else R
        if state_type == "sv":
            r = float(np.arcsinh(np.sqrt(float(R))))
            state = input_state_factory("squeezed_vacuum", R, M, r=r)
            skip = True
        elif state_type == "tmsv":
            state = _make_two_mode_squeezed_vacuum(float(R), M)
            skip = False
        else:
            state = _make_oat_state(R, 0.5)
            skip = True
        omega_grid = np.linspace(0.1, 5.0, 5)
        result = compute_mzi_sensitivity_grid(
            state,
            omega_grid,
            M,
            t_hold=t_hold,
            skip_bs1=skip,
        )
        expected_keys = {
            "omega_values",
            "expectation_values",
            "variance_values",
            "derivative_values",
            "delta_omega_ep",
            "delta_omega_q",
            "fisher_quantum",
            "fisher_classical",
            "delta_omega_c",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )

    def test_variance_nonnegative(self) -> None:
        """Variance must be >= 0 at all ω."""
        mean_N = 1
        M = int(5 * mean_N)
        r = float(np.arcsinh(np.sqrt(float(mean_N))))
        state = input_state_factory("squeezed_vacuum", mean_N, M, r=r)
        omega_grid = np.linspace(0.1, 5.0, 5)
        result = compute_mzi_sensitivity_grid(
            state, omega_grid, M, t_hold=t_hold, skip_bs1=True
        )
        assert np.all(result["variance_values"] >= -1e-12)

    def test_cfi_positivity(self) -> None:
        """F_C(ω) must be >= 0 at all ω."""
        N = 4
        state = _make_oat_state(N, 0.5)
        omega_grid = np.linspace(0.1, 5.0, 5)
        result = compute_mzi_sensitivity_grid(
            state, omega_grid, N, t_hold=t_hold, skip_bs1=True
        )
        assert np.all(result["fisher_classical"] >= -1e-12), "Some F_C values negative"

    def test_cfi_cramer_rao_sv(self) -> None:
        """Δω_C >= Δω_Q for SV (quantum Cramér-Rao bound)."""
        mean_N = 1
        M = int(5 * mean_N)
        r = float(np.arcsinh(np.sqrt(float(mean_N))))
        state = input_state_factory("squeezed_vacuum", mean_N, M, r=r)
        omega_grid = np.linspace(0.1, 5.0, 5)
        result = compute_mzi_sensitivity_grid(
            state, omega_grid, M, t_hold=t_hold, skip_bs1=True
        )
        delta_q = result["delta_omega_q"]
        for c_val in result["delta_omega_c"]:
            if np.isfinite(c_val):
                assert c_val >= delta_q - 1e-10, (
                    f"Δω_C={c_val} < Δω_Q={delta_q} violates Cramér-Rao bound"
                )

    def test_cfi_cramer_rao_oat(self) -> None:
        """Δω_C >= Δω_Q for OAT (quantum Cramér-Rao bound)."""
        N = 4
        state = _make_oat_state(N, 0.5)
        omega_grid = np.linspace(0.1, 5.0, 5)
        result = compute_mzi_sensitivity_grid(
            state, omega_grid, N, t_hold=t_hold, skip_bs1=True
        )
        delta_q = result["delta_omega_q"]
        for c_val in result["delta_omega_c"]:
            if np.isfinite(c_val):
                assert c_val >= delta_q - 1e-10, (
                    f"Δω_C={c_val} < Δω_Q={delta_q} violates Cramér-Rao bound"
                )


# ============================================================================
# OAT q-Scan
# ============================================================================


class TestOATQScan:
    def test_q_grid_positive(self) -> None:
        """q grid must contain positive values."""
        for N in [2, 4, 10, 20]:
            q_grid = _oat_q_grid(N)
            assert np.all(q_grid > 0), f"Non-positive q values for N={N}"
            assert len(q_grid) >= 5, f"Too few q points for N={N}"

    def test_q_scan_returns_oatqscanresult(self) -> None:
        """Scan result must be an OATQScanResult."""
        N = 4
        result = scan_oat_q(N, omega=0.5, max_photons=N, n_points=5)
        assert isinstance(result, OATQScanResult), (
            f"Expected OATQScanResult, got {type(result)}"
        )
        assert isinstance(result.q_values, np.ndarray)
        assert isinstance(result.fc_values, np.ndarray)
        assert isinstance(result.q_opt, float)
        assert isinstance(result.fc_opt, float)

    def test_q_scan_fc_positive(self) -> None:
        """F_C values from q scan must be positive."""
        N = 4
        result = scan_oat_q(N, omega=0.5, max_photons=N, n_points=5)
        assert np.all(result.fc_values >= 0), "Some F_C values negative"

    def test_q_opt_is_best(self) -> None:
        """q_opt must be the q value that maximizes F_C."""
        N = 4
        result = scan_oat_q(N, omega=0.5, max_photons=N, n_points=10)
        best_idx = int(np.argmax(result.fc_values))
        assert np.isclose(result.q_opt, result.q_values[best_idx], rtol=1e-10), (
            "q_opt does not match argmax"
        )
        assert np.isclose(result.fc_opt, result.fc_values[best_idx], rtol=1e-10), (
            "fc_opt does not match max"
        )


# ============================================================================
# Truncation Convergence
# ============================================================================


class TestTruncationConvergence:
    def test_sv_truncation_sufficient(self) -> None:
        """SV analytical check: sufficient truncation passes."""
        # M values chosen so captured norm >= 0.999 for each mean_N.
        # SV converges slowly (thermal-like tail); see _compute_sv_captured_norm.
        cases: list[tuple[float, int]] = [
            (1.0, 20),
            (2.0, 30),
            (4.0, 50),
        ]
        for mean_N, M in cases:
            assert _check_truncation_convergence(
                threshold=0.999, mean_n=mean_N, max_photons=M
            ), f"truncation should pass at mean_N={mean_N}, M={M}"

    def test_sv_truncation_insufficient(self) -> None:
        """SV analytical check: insufficient truncation is detected."""
        # M=5 for mean_N=1 captures only ~95% (verified analytically).
        assert not _check_truncation_convergence(
            threshold=0.999, mean_n=1.0, max_photons=5
        ), "truncation should fail at mean_N=1, M=5"

    def test_tmsv_truncation(self) -> None:
        """TMSV state should have >99.9% norm within truncation."""
        for mean_total in [2, 4, 10]:
            M = _resource_value_to_truncation(float(mean_total), "tmsv")
            state = _make_two_mode_squeezed_vacuum(float(mean_total), M)
            assert _check_truncation_convergence(state, threshold=0.999), (
                f"truncation failed at mean_total={mean_total}, M={M}"
            )


# ============================================================================
# Output Number-Difference Distribution
# ============================================================================


class TestOutputNumberDiffDistribution:
    def test_normalized(self) -> None:
        """Sum of P(m|ω) = 1 for any output state."""
        N = 4
        state = _make_oat_state(N, 0.5)
        out = simple_mzi_evolution(
            state, omega=0.5, max_photons=N, t_hold=t_hold, skip_bs1=True
        )
        P = output_number_diff_distribution(out, N)
        assert np.isclose(np.sum(P), 1.0, rtol=1e-12), f"Sum={np.sum(P)}"

    def test_nonnegative(self) -> None:
        """All P(m) >= 0."""
        N = 4
        state = _make_oat_state(N, 0.5)
        out = simple_mzi_evolution(
            state, omega=0.5, max_photons=N, t_hold=t_hold, skip_bs1=True
        )
        P = output_number_diff_distribution(out, N)
        assert np.all(P >= -1e-15), "Some probabilities are negative"

    def test_shape(self) -> None:
        """Array shape = (2*max_photons+1,)."""
        for M in [2, 4, 6]:
            N = M
            state = _make_oat_state(N, 0.5)
            out = simple_mzi_evolution(
                state, omega=0.5, max_photons=M, t_hold=t_hold, skip_bs1=True
            )
            P = output_number_diff_distribution(out, M)
            assert P.shape == (2 * M + 1,), (
                f"max_photons={M}: shape={P.shape}, expected {(2 * M + 1,)}"
            )


# ============================================================================
# Classical Fisher Information
# ============================================================================


class TestClassicalFisher:
    def test_cfi_positivity(self) -> None:
        """F_C >= 0 for a simple test case."""
        eps = 1e-6
        P_omega = np.array([0.5, 0.0, 0.5])
        P_plus = np.array([0.5 + 5e-6, 0.0, 0.5 - 5e-6])
        P_minus = np.array([0.5 - 5e-6, 0.0, 0.5 + 5e-6])
        fc = classical_fisher_information_single(
            P_plus, P_minus, eps, p_at_theta=P_omega, prob_floor=1e-15
        )
        assert fc >= 0.0

    def test_cfi_vanishes_at_null(self) -> None:
        """When P_omega = P_plus = P_minus, F_C = 0."""
        eps = 1e-6
        P = np.array([0.5, 0.0, 0.5])
        fc = classical_fisher_information_single(
            P, P, eps, p_at_theta=P, prob_floor=1e-15
        )
        assert np.isclose(fc, 0.0, atol=1e-20), f"F_C={fc}, expected 0"


# ============================================================================
# Scaling Exponent Fitting
# ============================================================================


class TestFitScalingExponent:
    def test_sv_heisenberg_exponent(self) -> None:
        r"""SV should give α ≈ -1.0 (Heisenberg)."""
        R_vals = np.array([20, 30, 50, 80, 120], dtype=float)
        # Δω_Q = 1 / (t_hold · √(2⟨N⟩(⟨N⟩+1)))
        delta_vals = 1.0 / (t_hold * np.sqrt(2 * R_vals * (R_vals + 1)))
        result = fit_scaling_exponent(R_vals, delta_vals, min_N=4)
        assert result.valid, f"Fit invalid: {result.warnings}"
        assert np.isclose(result.alpha, -1.0, atol=0.05), (
            f"SV exponent α={result.alpha}, expected -1.0"
        )

    def test_tmsv_heisenberg_exponent(self) -> None:
        r"""TMSV should give α ≈ -1.0 (Heisenberg)."""
        R_vals = np.array([20, 30, 50, 80, 120], dtype=float)
        # Δω_Q = 1 / (t_hold · √(⟨N⟩(⟨N⟩+2)))
        delta_vals = 1.0 / (t_hold * np.sqrt(R_vals * (R_vals + 2)))
        result = fit_scaling_exponent(R_vals, delta_vals, min_N=8)
        assert result.valid, f"Fit invalid: {result.warnings}"
        assert np.isclose(result.alpha, -1.0, atol=0.05), (
            f"TMSV exponent α={result.alpha}, expected -1.0"
        )

    def test_insufficient_points(self) -> None:
        """Fewer than 3 points should return invalid result."""
        R_vals = np.array([2.0])
        delta_vals = np.array([0.1])
        result = fit_scaling_exponent(R_vals, delta_vals, min_N=2)
        assert not result.valid, "Fit should be invalid with only 1 point"


# ============================================================================
# MziSensitivityDataSV — Parquet Roundtrip
# ============================================================================


class TestMziSensitivityDataSVParquet:
    _FIELD_SPECS: ClassVar[list[tuple[str, str]]] = [
        ("state_type", "eq"),
        ("resource_type", "eq"),
        ("resource_values", "allclose"),
        ("omega_values", "allclose"),
        ("expectation_grid", "allclose"),
        ("variance_grid", "allclose"),
        ("derivative_grid", "allclose"),
        ("delta_omega_ep_grid", "allclose"),
        ("delta_omega_q_per_R", "allclose"),
        ("fisher_classical_grid", "allclose"),
        ("delta_omega_c_grid", "allclose"),
        ("t_hold", "isclose"),
        ("truncation_M_per_R", "allclose"),
        ("squeezing_q_per_R", "allclose"),
    ]

    @pytest.fixture
    def make_result(self) -> MziSensitivityDataSV:
        n_R = 3
        n_omega = 5
        rng_ep = np.random.default_rng(45)
        rng_c = np.random.default_rng(46)
        return MziSensitivityDataSV(
            state_type="sv",
            resource_type="mean_N",
            resource_values=np.array([2, 4, 6], dtype=float),
            omega_values=np.linspace(0.1, 5.0, n_omega),
            expectation_grid=np.random.default_rng(42).uniform(-1, 1, (n_R, n_omega)),
            variance_grid=np.random.default_rng(43).uniform(0, 0.5, (n_R, n_omega)),
            derivative_grid=np.random.default_rng(44).uniform(-2, 2, (n_R, n_omega)),
            delta_omega_ep_grid=rng_ep.uniform(0.01, 1, (n_R, n_omega)),
            delta_omega_q_per_R=np.array([0.05, 0.025, 0.0167]),
            fisher_classical_grid=rng_c.uniform(1, 100, (n_R, n_omega)),
            delta_omega_c_grid=rng_ep.uniform(0.01, 1, (n_R, n_omega)),
            t_hold=t_hold,
            truncation_M_per_R=np.array([10, 20, 30], dtype=float),
            squeezing_q_per_R=np.zeros(3, dtype=float),
        )

    def test_roundtrip(self, make_result: MziSensitivityDataSV, tmp_path: Path) -> None:
        p = tmp_path / "sensitivity.parquet"
        make_result.save_parquet(p)
        loaded = MziSensitivityDataSV.from_parquet(p)
        assert_roundtrip_fields(loaded, make_result, self._FIELD_SPECS)

    def test_fail_fast_missing_column(
        self, make_result: MziSensitivityDataSV, tmp_path: Path
    ) -> None:
        p = tmp_path / "bad.parquet"
        df = make_result.to_dataframe()
        df = df.drop(columns=["state_type"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            MziSensitivityDataSV.from_parquet(p)

    def test_fail_fast_missing_q_column(
        self, make_result: MziSensitivityDataSV, tmp_path: Path
    ) -> None:
        p = tmp_path / "bad_q.parquet"
        df = make_result.to_dataframe()
        df = df.drop(columns=["squeezing_q"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            MziSensitivityDataSV.from_parquet(p)


# ============================================================================
# Integration: Generate Omega Scan
# ============================================================================


class TestGenerateOmegaScan:
    def test_generate_sv_scan(self) -> None:
        """Generate a small ω scan for SV and verify basic properties."""
        omega_grid = np.linspace(0.1, 5.0, 5)
        result = generate_single_omega_scan(
            "sv",
            resource_value=4.0,
            omega_grid=omega_grid,
            max_photons=20,
            t_hold=t_hold,
        )
        assert result.state_type == "sv"
        assert result.resource_type == "mean_N"
        assert np.isclose(result.resource_values[0], 4.0)
        assert len(result.omega_values) == 5
        assert np.all(np.isfinite(result.fisher_classical_grid[0]))

    def test_generate_oat_scan(self) -> None:
        """Generate a small ω scan for OAT and verify basic properties."""
        omega_grid = np.linspace(0.1, 5.0, 5)
        result = generate_single_omega_scan(
            "oat",
            resource_value=4.0,
            omega_grid=omega_grid,
            max_photons=4,
            t_hold=t_hold,
            q=0.5,
        )
        assert result.state_type == "oat"
        assert result.resource_type == "N"

    def test_generate_sv_scan_no_max_photons(self) -> None:
        """generate_single_omega_scan works without max_photons (auto-compute)."""
        omega_grid = np.linspace(0.1, 5.0, 3)
        result = generate_single_omega_scan(
            "sv",
            resource_value=4.0,
            omega_grid=omega_grid,
            t_hold=t_hold,
        )
        assert result.state_type == "sv"
        assert len(result.omega_values) == 3
        assert np.all(np.isfinite(result.fisher_classical_grid[0]))

    def test_generate_tmsv_scan(self) -> None:
        """Generate a small ω scan for TMSV and verify basic properties."""
        omega_grid = np.linspace(0.1, 5.0, 5)
        result = generate_single_omega_scan(
            "tmsv",
            resource_value=4.0,
            omega_grid=omega_grid,
            max_photons=20,
            t_hold=t_hold,
        )
        assert result.state_type == "tmsv"
        assert result.resource_type == "mean_N"
        assert np.isclose(result.resource_values[0], 4.0)
        assert len(result.omega_values) == 5
        assert np.all(np.isfinite(result.fisher_classical_grid[0]))


# ============================================================================
# Analyse Best/Worst Sensitivity
# ============================================================================


class TestAnalyseBestWorstSensitivity:
    def test_returns_dict_keys(self) -> None:
        """analyse_best_worst_sensitivity returns expected keys."""
        R_vals = np.array([2.0, 4.0])
        omega_vals = np.linspace(0.1, 5.0, 10)
        grid = np.random.default_rng(42).uniform(0.01, 1, (2, 10))
        result = analyse_best_worst_sensitivity(R_vals, omega_vals, grid)
        expected_keys = {
            "resource_values",
            "best_sensitivity",
            "best_omega",
            "worst_sensitivity",
            "worst_omega",
        }
        assert result.keys() == expected_keys, (
            f"Missing keys: {expected_keys - result.keys()}"
        )

    def test_best_is_minimum(self) -> None:
        """Best sensitivity <= worst sensitivity at each resource."""
        R_vals = np.array([2.0, 4.0, 6.0])
        omega_vals = np.linspace(0.1, 5.0, 20)
        rng = np.random.default_rng(42)
        grid = rng.uniform(0.01, 1, (3, 20))
        result = analyse_best_worst_sensitivity(R_vals, omega_vals, grid)
        assert np.all(result["best_sensitivity"] <= result["worst_sensitivity"] + 1e-15)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    def test_truncation_oat_exact(self) -> None:
        """OAT truncation equals N."""
        for N in [2, 4, 10, 40]:
            M = _resource_value_to_truncation(float(N), "oat")
            assert M == N, f"OAT truncation: expected {N}, got {M}"

    def test_truncation_sv_scaling(self) -> None:
        """SV truncation scales with mean N."""
        for mean_N in [1, 5, 10, 20]:
            M = _resource_value_to_truncation(float(mean_N), "sv")
            assert mean_N <= M, f"SV truncation M={M} < mean_N={mean_N}"
            assert M <= 80, f"SV truncation M={M} exceeds max"

    def test_truncation_tmsv_scaling(self) -> None:
        """TMSV truncation scales with total mean N."""
        for mean_total in [2, 10, 20]:
            M = _resource_value_to_truncation(float(mean_total), "tmsv")
            assert mean_total // 2 <= M, (
                f"TMSV truncation M={M} too small for mean_total={mean_total}"
            )

    def test_sv_mean_N_1(self) -> None:
        """SV with mean_N=1 should have Var(J_z) ≈ 1 (large M for accuracy)."""
        mean_N = 1.0
        M = 40
        r = float(np.arcsinh(np.sqrt(mean_N)))
        state = input_state_factory("squeezed_vacuum", int(mean_N), M, r=r)
        var = compute_jz_variance(state, M)
        expected_var = mean_N * (mean_N + 1) / 2.0
        assert np.isclose(var, expected_var, rtol=1e-3), (
            f"mean_N=1: Var={var}, expected {expected_var}"
        )


# ============================================================================
# OAT q=None Error Handling
# ============================================================================


class TestOATQError:
    def test_oat_q_none_raises(self) -> None:
        """generate_single_omega_scan with OAT and q=None must raise."""
        omega_grid = np.linspace(0.1, 5.0, 3)
        with pytest.raises(ValueError, match="explicit q parameter"):
            generate_single_omega_scan(
                "oat",
                resource_value=4.0,
                omega_grid=omega_grid,
                max_photons=4,
                t_hold=t_hold,
                q=None,
            )


# ============================================================================
# Prepare State Dispatch (noon, twin_fock_std)
# ============================================================================


class TestPrepareStateDispatch:
    @pytest.mark.parametrize(
        ("state_type", "R"),
        [("noon", 4), ("twin_fock_std", 4)],
    )
    def test_dispatch_normalized(self, state_type: str, R: int) -> None:
        """Dispatching to noon or twin_fock_std produces a valid state."""
        N = 4
        state = _prepare_state(state_type, float(N), N)
        assert np.isclose(np.linalg.norm(state), 1.0, rtol=1e-10), (
            f"Failed for {state_type}"
        )


# ============================================================================
# TMSV Truncation Warning
# ============================================================================


class TestTMSVTruncationWarning:
    def test_truncation_warning(self) -> None:
        """Insufficient TMSV truncation must emit a warning."""
        with pytest.warns(UserWarning, match="captures only"):
            _make_two_mode_squeezed_vacuum(mean_total=20.0, max_photons=3)


# ============================================================================
# TMSV Analytical Truncation Check
# ============================================================================


class TestTMSVAnalyticalTruncation:
    def test_analytical_check_passes(self) -> None:
        """TMSV analytical truncation check passes for adequate M."""
        assert _check_truncation_convergence(
            threshold=0.999, mean_total=2.0, max_photons=20
        )

    def test_analytical_fallthrough_raises(self) -> None:
        """_check_truncation_convergence with no arguments raises ValueError."""
        with pytest.raises(ValueError, match="Must provide"):
            _check_truncation_convergence()


# ============================================================================
# Generate Single Resource Data
# ============================================================================


class TestGenerateSingleResourceData:
    def test_resource_data_sv(self) -> None:
        """_generate_single_resource_data returns a valid result for SV."""
        omega_grid = np.linspace(0.1, 5.0, 3)
        result = _generate_single_resource_data("sv", 4.0, omega_grid, t_hold=t_hold)
        assert result is not None
        assert result.state_type == "sv"
        assert len(result.omega_values) == 3

    def test_resource_data_failure_returns_none(self) -> None:
        """_generate_single_resource_data returns None on invalid input."""
        omega_grid = np.linspace(0.1, 5.0, 3)
        result = _generate_single_resource_data(
            "invalid", 4.0, omega_grid, t_hold=t_hold
        )
        assert result is None

    def test_resource_data_tmsv(self) -> None:
        """_generate_single_resource_data works for TMSV."""
        omega_grid = np.linspace(0.1, 5.0, 2)
        result = _generate_single_resource_data("tmsv", 4.0, omega_grid, t_hold=t_hold)
        assert result is not None
        assert result.state_type == "tmsv"

    def test_resource_data_oat(self) -> None:
        """_generate_single_resource_data works for OAT."""
        omega_grid = np.linspace(0.1, 5.0, 2)
        result = _generate_single_resource_data(
            "oat", 4.0, omega_grid, t_hold=t_hold, q=1.0
        )
        assert result is not None
        assert result.state_type == "oat"


# ============================================================================
# Generate Full Data
# ============================================================================


class TestGenerateFullData:
    def test_full_data_small_sv(self) -> None:
        """generate_full_data runs with a minimal resource range."""
        omega_grid = np.linspace(0.1, 5.0, 2)
        data = generate_full_data("sv", [2.0, 4.0], omega_grid, t_hold=t_hold)
        assert data.state_type == "sv"
        assert len(data.resource_values) == 2
        assert len(data.omega_values) == 2
        assert data.truncation_M_per_R is not None
        assert data.squeezing_q_per_R is not None

    def test_full_data_all_invalid_raises(self) -> None:
        """generate_full_data raises RuntimeError when all resource values fail."""
        omega_grid = np.linspace(0.1, 5.0, 2)
        with pytest.raises(RuntimeError, match="No valid resource values"):
            generate_full_data("invalid", [2.0, 4.0], omega_grid, t_hold=t_hold)


# ============================================================================
# Maybe Generate Full Data
# ============================================================================


class TestMaybeGenerateFullData:
    def test_maybe_generate_full_data_loads_existing(self) -> None:
        """_maybe_generate_full_data loads existing Parquet on second call."""
        import sys as _sys

        mod = _sys.modules["local"]
        orig_sv = mod.SV_N_RANGE
        orig_omega_range = mod.OMEGA_RANGE
        orig_omega_step = mod.OMEGA_STEP
        try:
            mod.SV_N_RANGE = [2.0]  # type: ignore[attr-defined]
            mod.OMEGA_RANGE = (0.1, 5.0)  # type: ignore[attr-defined]
            mod.OMEGA_STEP = 4.9  # type: ignore[attr-defined]
            r_range: list[float] = [2.0]
            omega_grid = np.arange(
                mod.OMEGA_RANGE[0],  # type: ignore[attr-defined]
                mod.OMEGA_RANGE[1] + mod.OMEGA_STEP / 2,  # type: ignore[attr-defined]
                mod.OMEGA_STEP,  # type: ignore[attr-defined]
            )

            # First call: generate and save
            data1 = _maybe_generate_full_data(
                "sv", r_range, "SV", omega_grid, force=True, only=None
            )
            assert data1 is not None
            assert data1.state_type == "sv"

            # Second call: load from existing Parquet
            data2 = _maybe_generate_full_data(
                "sv", r_range, "SV", omega_grid, force=False, only=None
            )
            assert data2 is not None
            assert data2.state_type == "sv"
            np.testing.assert_array_equal(data1.resource_values, data2.resource_values)
        finally:
            mod.SV_N_RANGE = orig_sv  # type: ignore[attr-defined]
            mod.OMEGA_RANGE = orig_omega_range  # type: ignore[attr-defined]
            mod.OMEGA_STEP = orig_omega_step  # type: ignore[attr-defined]


# ============================================================================
# Plots
# ============================================================================


class TestPlots:
    @pytest.fixture
    def sample_data(self) -> MziSensitivityDataSV:
        """Small MziSensitivityDataSV for plot tests."""
        n_R = 3
        n_omega = 5
        return MziSensitivityDataSV(
            state_type="sv",
            resource_type="mean_N",
            resource_values=np.array([2, 4, 6], dtype=float),
            omega_values=np.linspace(0.1, 5.0, n_omega),
            expectation_grid=np.zeros((n_R, n_omega), dtype=float),
            variance_grid=np.ones((n_R, n_omega), dtype=float),
            derivative_grid=np.ones((n_R, n_omega), dtype=float),
            delta_omega_ep_grid=np.ones((n_R, n_omega), dtype=float),
            delta_omega_q_per_R=np.array([0.1, 0.05, 0.03]),
            fisher_classical_grid=np.full((n_R, n_omega), 100.0, dtype=float),
            delta_omega_c_grid=np.ones((n_R, n_omega), dtype=float),
            t_hold=t_hold,
            truncation_M_per_R=np.array([10, 20, 30], dtype=float),
            squeezing_q_per_R=np.zeros(3, dtype=float),
        )

    def test_delta_omega_overlay_sv(
        self, sample_data: MziSensitivityDataSV, tmp_path: Path
    ) -> None:
        """plot_delta_omega_overlay creates a valid SVG."""
        svg = tmp_path / "test_overlay.svg"
        created = plot_delta_omega_overlay(sample_data, save_path=svg)
        assert svg.exists()
        assert svg.suffix == ".svg"
        assert created == svg

    def test_delta_omega_overlay_selected_R(
        self, sample_data: MziSensitivityDataSV, tmp_path: Path
    ) -> None:
        """plot_delta_omega_overlay works with explicit R selection."""
        svg = tmp_path / "test_overlay_sel.svg"
        plot_delta_omega_overlay(sample_data, selected_R=[2.0, 6.0], save_path=svg)
        assert svg.exists()

    def test_plot_scaling_single(
        self, sample_data: MziSensitivityDataSV, tmp_path: Path
    ) -> None:
        """plot_scaling works with a single data entry."""
        svg = tmp_path / "test_scaling.svg"
        created = plot_scaling([sample_data], ["SV"], save_path=svg)
        assert svg.exists()
        assert created == svg

    def test_plot_scaling_multiple(
        self, sample_data: MziSensitivityDataSV, tmp_path: Path
    ) -> None:
        """plot_scaling works with multiple data entries."""
        svg = tmp_path / "test_scaling_multi.svg"
        plot_scaling(
            [sample_data, None, sample_data],
            ["SV", "TMSV", "OAT"],
            save_path=svg,
        )
        assert svg.exists()

    def test_plot_scaling_no_data_raises(self, tmp_path: Path) -> None:
        """plot_scaling with all-None must raise."""
        with pytest.raises(ValueError, match="At least one"):
            plot_scaling([None, None], ["a", "b"], save_path=tmp_path / "none.svg")

    def test_plot_overlay_no_save_path(self, sample_data: MziSensitivityDataSV) -> None:
        """plot_delta_omega_overlay works without save_path (auto-names)."""
        # Allows SVG in the default location; clean up after.
        from local import _fig_path  # type: ignore[import-untyped]

        path = _fig_path(f"{sample_data.state_type}_delta_omega_comparison")
        try:
            created = plot_delta_omega_overlay(sample_data)
            assert created is not None
            assert created.suffix == ".svg"
        finally:
            if path.exists():
                path.unlink()

    def test_plot_overlay_non_existent_R(
        self, sample_data: MziSensitivityDataSV, tmp_path: Path
    ) -> None:
        """plot_delta_omega_overlay skips non-existent R values gracefully."""
        svg = tmp_path / "overlay_bad_R.svg"
        plot_delta_omega_overlay(sample_data, selected_R=[999.0], save_path=svg)
        assert svg.exists()

    def test_plot_scaling_no_save_path(self, sample_data: MziSensitivityDataSV) -> None:
        """plot_scaling works without save_path (auto-names)."""
        from local import _fig_path  # type: ignore[import-untyped]

        path = _fig_path("scaling_comparison")
        try:
            created = plot_scaling([sample_data], ["SV"])
            assert created is not None
            assert created.suffix == ".svg"
        finally:
            if path.exists():
                path.unlink()

    def test_plot_scaling_valid_fit(self, tmp_path: Path) -> None:
        """plot_scaling with SQL-like data triggers the fit-valid branch."""
        n_R = 10
        R_vals = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0])
        dt_c = 1.0 / np.sqrt(R_vals)  # SQL-like scaling
        dt_q = dt_c * 0.5  # QFI better than SQL
        n_omega = 5
        data = MziSensitivityDataSV(
            state_type="sv",
            resource_type="mean_N",
            resource_values=R_vals,
            omega_values=np.linspace(0.1, 5.0, n_omega),
            expectation_grid=np.zeros((n_R, n_omega), dtype=float),
            variance_grid=np.ones((n_R, n_omega), dtype=float),
            derivative_grid=np.ones((n_R, n_omega), dtype=float),
            delta_omega_ep_grid=np.ones((n_R, n_omega), dtype=float),
            delta_omega_q_per_R=dt_q,
            fisher_classical_grid=np.full((n_R, n_omega), 100.0, dtype=float),
            delta_omega_c_grid=np.tile(dt_c, (n_omega, 1)).T.astype(float),
            t_hold=t_hold,
            truncation_M_per_R=np.full(n_R, 10.0, dtype=float),
            squeezing_q_per_R=np.zeros(n_R, dtype=float),
        )
        svg = tmp_path / "scaling_fit.svg"
        plot_scaling([data], ["SV"], save_path=svg)
        assert svg.exists()


# ============================================================================
# Plot Orchestration
# ============================================================================


class TestPlotOrchestration:
    def test_maybe_plot_overlays_none_data(self) -> None:
        """_maybe_plot_delta_omega_overlays skips None results gracefully."""
        state_configs = [
            ("sv", [2.0], "SV"),
            ("tmsv", [4.0], "TMSV"),
            ("oat", [4.0], "OAT"),
        ]
        # results dict missing some keys — data.get(key) returns None
        results: dict[str, MziSensitivityDataSV] = {}
        # Should not raise even though state_configs have entries not in results
        _maybe_plot_delta_omega_overlays(results, state_configs, force=True, only=None)


# ============================================================================
# Generate All (Pipeline)
# ============================================================================


class TestGenerateAll:
    def test_generate_all_sv_only(self, tmp_path: Path) -> None:
        """generate_all with --only sv runs end-to-end for SV."""
        # Use a small omega grid and single resource value for speed.
        # Access module from sys.modules (injected at import time).
        import sys

        mod = sys.modules["local"]
        for attr in ("SV_N_RANGE", "OMEGA_RANGE", "OMEGA_STEP"):
            assert hasattr(mod, attr), f"Module missing {attr}"
        orig_sv = mod.SV_N_RANGE  # type: ignore[attr-defined]
        orig_omega_range = mod.OMEGA_RANGE  # type: ignore[attr-defined]
        orig_omega_step = mod.OMEGA_STEP  # type: ignore[attr-defined]
        try:
            mod.SV_N_RANGE = [2.0]  # type: ignore[attr-defined]
            mod.OMEGA_RANGE = (0.1, 0.5)  # type: ignore[attr-defined]
            mod.OMEGA_STEP = 0.4  # type: ignore[attr-defined]
            results = mod.generate_all(force=True, only="sv")  # type: ignore[attr-defined]
        finally:
            mod.SV_N_RANGE = orig_sv  # type: ignore[attr-defined]
            mod.OMEGA_RANGE = orig_omega_range  # type: ignore[attr-defined]
            mod.OMEGA_STEP = orig_omega_step  # type: ignore[attr-defined]

        assert "sv" in results
        data = results["sv"]
        assert data.state_type == "sv"
        assert len(data.resource_values) >= 1


# ============================================================================
# CLI
# ============================================================================


class TestCLI:
    def test_cli_help(self) -> None:
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                str(Path(__file__).resolve().parent / "heisenberg_limit_mzi_sq_oat.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()

    def test_main_direct_help(self) -> None:
        """Call main() directly with --help via sys.argv patching."""
        import sys as _sys

        old_argv = _sys.argv[:]
        try:
            _sys.argv = ["script", "--help"]
            with pytest.raises(SystemExit):
                main()
        finally:
            _sys.argv = old_argv

    def test_main_direct_run_sv(self) -> None:
        """Call main() directly with --only sv and small ranges."""
        import sys as _sys

        mod = _sys.modules["local"]
        old_argv = _sys.argv[:]
        orig_sv = mod.SV_N_RANGE  # type: ignore[attr-defined]
        orig_omega_range = mod.OMEGA_RANGE  # type: ignore[attr-defined]
        orig_omega_step = mod.OMEGA_STEP  # type: ignore[attr-defined]
        try:
            mod.SV_N_RANGE = [2.0]  # type: ignore[attr-defined]
            mod.OMEGA_RANGE = (0.1, 0.5)  # type: ignore[attr-defined]
            mod.OMEGA_STEP = 0.4  # type: ignore[attr-defined]
            _sys.argv = ["script", "--only", "sv"]
            main()
        finally:
            _sys.argv = old_argv
            mod.SV_N_RANGE = orig_sv  # type: ignore[attr-defined]
            mod.OMEGA_RANGE = orig_omega_range  # type: ignore[attr-defined]
            mod.OMEGA_STEP = orig_omega_step  # type: ignore[attr-defined]
