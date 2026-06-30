"""Tests for squeezed-vacuum analytical QFI and truncation utilities."""

import numpy as np
import pytest

from .sv_qfi import (
    check_truncation_convergence,
    compute_sv_captured_norm,
    compute_sv_qfi,
    compute_tmsv_captured_norm,
    compute_tmsv_qfi,
    verify_sv_qfi,
)


class TestComputeSvQfi:
    def test_sv_qfi_formula(self) -> None:
        """F_Q = 2 t_hold^2 ⟨N⟩(⟨N⟩+1) for ⟨N⟩=1, t_hold=10."""
        fq = compute_sv_qfi(mean_N=1.0, t_hold=10.0)
        expected = 2.0 * 100.0 * 1.0 * 2.0  # = 400
        assert fq == pytest.approx(expected)

    def test_sv_qfi_default_t_hold(self) -> None:
        """Default t_hold=10.0."""
        fq = compute_sv_qfi(mean_N=2.0)
        expected = 2.0 * 100.0 * 2.0 * 3.0  # = 1200
        assert fq == pytest.approx(expected)

    def test_sv_qfi_zero_mean_n(self) -> None:
        """Zero mean photon number gives zero QFI."""
        fq = compute_sv_qfi(mean_N=0.0)
        assert fq == pytest.approx(0.0)

    def test_sv_qfi_scales_with_n_squared(self) -> None:
        """For large ⟨N⟩, F_Q ∝ ⟨N⟩²."""
        fq_10 = compute_sv_qfi(mean_N=10.0, t_hold=1.0)
        fq_20 = compute_sv_qfi(mean_N=20.0, t_hold=1.0)
        # Ratio should be ≈ (20*21)/(10*11) ≈ 3.82
        ratio = fq_20 / fq_10
        assert ratio == pytest.approx(3.818, rel=1e-3)


class TestComputeTmsvQfi:
    def test_tmsv_qfi_formula(self) -> None:
        """F_Q = t_hold^2 ⟨N⟩(⟨N⟩+2) for ⟨N⟩=2, t_hold=10."""
        fq = compute_tmsv_qfi(mean_total=2.0, t_hold=10.0)
        expected = 100.0 * 2.0 * 4.0  # = 800
        assert fq == pytest.approx(expected)

    def test_tmsv_qfi_default_t_hold(self) -> None:
        """Default t_hold=10.0."""
        fq = compute_tmsv_qfi(mean_total=4.0)
        expected = 100.0 * 4.0 * 6.0  # = 2400
        assert fq == pytest.approx(expected)


class TestComputeSvCapturedNorm:
    def test_full_capture_for_large_truncation(self) -> None:
        """Large max_photons captures ≈100% norm."""
        captured = compute_sv_captured_norm(mean_N=1.0, max_photons=50)
        assert captured > 0.999

    def test_partial_capture_for_small_truncation(self) -> None:
        """Small max_photons gives incomplete capture."""
        captured = compute_sv_captured_norm(mean_N=10.0, max_photons=10)
        assert captured < 1.0

    def test_monotonic_in_truncation(self) -> None:
        """Captured norm increases with max_photons."""
        c5 = compute_sv_captured_norm(mean_N=5.0, max_photons=5)
        c10 = compute_sv_captured_norm(mean_N=5.0, max_photons=10)
        assert c5 < c10

    def test_decreases_with_mean_n(self) -> None:
        """Captured norm decreases with larger mean_N at fixed truncation."""
        c1 = compute_sv_captured_norm(mean_N=1.0, max_photons=10)
        c10 = compute_sv_captured_norm(mean_N=10.0, max_photons=10)
        assert c1 > c10


class TestComputeTmsvCapturedNorm:
    def test_full_capture_for_large_truncation(self) -> None:
        captured = compute_tmsv_captured_norm(mean_total=2.0, max_photons=50)
        assert captured > 0.999

    def test_partial_capture_for_small_truncation(self) -> None:
        captured = compute_tmsv_captured_norm(mean_total=10.0, max_photons=5)
        assert captured < 1.0


class TestVerifySvQfi:
    def test_verify_correct_variance(self) -> None:
        """Var(J_z) = ⟨N⟩(⟨N⟩+1)/2."""
        assert verify_sv_qfi(mean_N=1.0, var_probe=1.0)  # 1*2/2 = 1

    def test_verify_incorrect_variance(self) -> None:
        """Returns False when variance does not match."""
        assert not verify_sv_qfi(mean_N=1.0, var_probe=99.0)

    def test_verify_large_n(self) -> None:
        """Scales correctly for large ⟨N⟩."""
        mean_N = 10.0
        expected_var = mean_N * (mean_N + 1.0) / 2.0
        assert verify_sv_qfi(mean_N=mean_N, var_probe=expected_var)


class TestCheckTruncationConvergence:
    def test_sv_path_passes(self) -> None:
        assert check_truncation_convergence(mean_n=1.0, max_photons=50, threshold=0.999)

    def test_sv_path_fails(self) -> None:
        assert not check_truncation_convergence(
            mean_n=100.0, max_photons=10, threshold=0.999
        )

    def test_tmsv_path_passes(self) -> None:
        assert check_truncation_convergence(
            mean_total=2.0, max_photons=50, threshold=0.999
        )

    def test_tmsv_path_fails(self) -> None:
        assert not check_truncation_convergence(
            mean_total=100.0, max_photons=5, threshold=0.999
        )

    def test_state_fallback_passes(self) -> None:
        state = np.ones(16, dtype=complex)
        state = state / np.linalg.norm(state)
        assert check_truncation_convergence(state=state, threshold=0.5)

    def test_state_fallback_fails(self) -> None:
        state = np.ones(16, dtype=complex) * 0.01
        assert not check_truncation_convergence(state=state, threshold=0.5)

    def test_raises_without_arguments(self) -> None:
        with pytest.raises(ValueError):
            check_truncation_convergence()
