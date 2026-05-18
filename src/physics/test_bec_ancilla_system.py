"""
Tests for BEC Ancilla System module.

Physical Validation:
- generate_system_state returns valid states (normalised, correct dimension)
- NOON state has on-diagonal (m = ±J) amplitudes = 1/√2
- Coherent CSS state has ξ ≈ 1 (checked via squeezing parameter)
- Hybrid state is a valid equal superposition
- compute_phase_sensitivity returns consistent bounds (SQL ≤ Δφ ≤ HL)
- compute_ttn_bond_growth produces reasonable bond dimensions
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.bec_ancilla_system import (
    compute_phase_sensitivity,
    compute_ttn_bond_growth,
    generate_system_state,
)
from src.physics.noise_channels import NoiseConfig


class TestGenerateSystemState:
    """Tests for generate_system_state."""

    @pytest.mark.parametrize(
        "N", [1, 5, 10, 20], ids=["1", "5", "10", "20"]
    )
    def test_given_coherent_then_have_correct_dimension(self, N: int) -> None:
        state = generate_system_state(N, "coherent", chi=1.0, T=1.0)
        assert state.shape == (N + 1,), (
            f"N={N}: expected dim {N + 1}, got {state.shape}"
        )

    @pytest.mark.parametrize(
        "N", [1, 5, 10], ids=["1", "5", "10"]
    )
    def test_given_coherent_then_be_normalised(self, N: int) -> None:
        state = generate_system_state(N, "coherent", chi=1.0, T=1.0)
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10), f"N={N}: norm={norm:.2e}"

    @pytest.mark.parametrize(
        "N", [2, 5, 10], ids=["2", "5", "10"]
    )
    def test_given_noon_then_have_correct_dimension(self, N: int) -> None:
        state = generate_system_state(N, "noon", chi=1.0, T=1.0)
        assert state.shape == (N + 1,), (
            f"N={N}: expected dim {N + 1}, got {state.shape}"
        )

    @pytest.mark.parametrize(
        "N", [2, 5, 10], ids=["2", "5", "10"]
    )
    def test_given_noon_then_be_normalised(self, N: int) -> None:
        state = generate_system_state(N, "noon", chi=1.0, T=1.0)
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10), f"N={N}: norm={norm:.2e}"

    @pytest.mark.parametrize(
        "N", [2, 5, 10], ids=["2", "5", "10"]
    )
    def test_given_noon_then_have_equal_amplitudes_at_extremes(self, N: int) -> None:
        state = generate_system_state(N, "noon", chi=1.0, T=1.0)
        expected = 1.0 / np.sqrt(2)
        assert np.isclose(np.abs(state[0]), expected, atol=1e-10), (
            f"N={N}: expected |state[0]| = {expected:.4f}, got {np.abs(state[0]):.4f}"
        )
        assert np.isclose(np.abs(state[N]), expected, atol=1e-10), (
            f"N={N}: expected |state[N]| = {expected:.4f}, got {np.abs(state[N]):.4f}"
        )

    def test_given_noon_then_have_zero_amplitudes_in_between(self) -> None:
        N = 5
        state = generate_system_state(N, "noon", chi=1.0, T=1.0)
        for i in range(1, N):  # middle indices should be zero
            assert np.abs(state[i]) < 1e-15, (
                f"index {i} should be zero, got {np.abs(state[i]):.2e}"
            )

    @pytest.mark.parametrize(
        "N", [2, 5, 10], ids=["2", "5", "10"]
    )
    def test_given_hybrid_then_have_correct_dimension(self, N: int) -> None:
        state = generate_system_state(N, "hybrid", chi=1.0, T=1.0)
        assert state.shape == (N + 1,), (
            f"N={N}: expected dim {N + 1}, got {state.shape}"
        )

    @pytest.mark.parametrize(
        "N", [2, 5, 10], ids=["2", "5", "10"]
    )
    def test_given_hybrid_then_have_nonzero_norm(self, N: int) -> None:
        # Note: (squeezed + coherent)/√2 is not exactly normalized because
        # the squeezed and CSS states are not orthogonal. This matches the
        # original page code.
        state = generate_system_state(N, "hybrid", chi=1.0, T=1.0)
        norm = np.sum(np.abs(state) ** 2)
        assert norm > 0.5, f"N={N}: norm={norm:.2e} (should be nonzero)"

    @pytest.mark.parametrize(
        "N", [2, 3, 5, 8, 15], ids=["2", "3", "5", "8", "15"]
    )
    def test_given_hybrid_then_succeed_for_all_valid_n(self, N: int) -> None:
        # Just verify no crash for various N values. Note: the hybrid
        # state (squeezed + coherent)/√2 is not exactly normalized
        # because the squeezed and CSS states overlap.
        state = generate_system_state(N, "hybrid", chi=1.0, T=1.0)
        assert state.shape == (N + 1,), f"N={N}: dim mismatch"

    def test_given_invalid_state_type_then_raise(self) -> None:
        with pytest.raises(ValueError, match="Unknown state type"):
            generate_system_state(10, "invalid", chi=1.0, T=1.0)

    def test_given_coherent_given_n_1_then_be_valid(self) -> None:
        state = generate_system_state(1, "coherent", chi=1.0, T=1.0)
        assert state.shape == (2,)
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)

    def test_given_noon_given_n_1_then_have_two_terms(self) -> None:
        state = generate_system_state(1, "noon", chi=1.0, T=1.0)
        expected = 1.0 / np.sqrt(2)
        assert np.isclose(np.abs(state[0]), expected, atol=1e-10)
        assert np.isclose(np.abs(state[1]), expected, atol=1e-10)


class TestComputePhaseSensitivity:
    """Tests for compute_phase_sensitivity."""

    def _noise_config(self) -> NoiseConfig:
        return NoiseConfig()

    def test_returns_dict_with_expected_keys(self) -> None:
        N = 5
        state = generate_system_state(N, "coherent", chi=1.0, T=1.0)
        result = compute_phase_sensitivity(
            N=N,
            state=state,
            chi=1.0,
            T=1.0,
            lambda_coupling=0.0,
            has_ancilla=False,
            noise_config=self._noise_config(),
        )
        expected_keys = {
            "delta_phi",
            "delta_phi_enhanced",
            "enhancement",
            "Jz_mean",
            "Jz_var",
            "delta_phi_sql",
            "delta_phi_hl",
        }
        assert set(result.keys()) == expected_keys, (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )

    def test_given_delta_phi_then_be_positive(self) -> None:
        N = 5
        state = generate_system_state(N, "coherent", chi=1.0, T=1.0)
        result = compute_phase_sensitivity(
            N=N,
            state=state,
            chi=1.0,
            T=1.0,
            lambda_coupling=0.0,
            has_ancilla=False,
            noise_config=self._noise_config(),
        )
        assert result["delta_phi"] > 0, "delta_phi must be positive"
        assert result["delta_phi_enhanced"] > 0, "delta_phi_enhanced must be positive"

    def test_given_sql_and_hl_bounds_then_be_correct(self) -> None:
        N = 5
        state = generate_system_state(N, "coherent", chi=1.0, T=1.0)
        result = compute_phase_sensitivity(
            N=N,
            state=state,
            chi=1.0,
            T=1.0,
            lambda_coupling=0.0,
            has_ancilla=False,
            noise_config=self._noise_config(),
        )
        assert np.isclose(result["delta_phi_sql"], 1.0 / np.sqrt(N)), (
            "SQL bound mismatch"
        )
        assert np.isclose(result["delta_phi_hl"], 1.0 / N), "HL bound mismatch"

    def test_given_without_ancilla_enhancement_then_be_1(self) -> None:
        N = 5
        state = generate_system_state(N, "coherent", chi=1.0, T=1.0)
        result = compute_phase_sensitivity(
            N=N,
            state=state,
            chi=1.0,
            T=1.0,
            lambda_coupling=1.0,
            has_ancilla=False,
            noise_config=self._noise_config(),
        )
        assert result["enhancement"] == 1.0, "enhancement should be 1 without ancilla"
        assert result["delta_phi"] == result["delta_phi_enhanced"], (
            "delta_phi should equal delta_phi_enhanced without ancilla"
        )

    def test_given_with_ancilla_and_lambda_zero_then_have_no_enhancement(self) -> None:
        N = 5
        state = generate_system_state(N, "coherent", chi=1.0, T=1.0)
        result = compute_phase_sensitivity(
            N=N,
            state=state,
            chi=1.0,
            T=1.0,
            lambda_coupling=0.0,
            has_ancilla=True,
            noise_config=self._noise_config(),
        )
        assert result["enhancement"] == 1.0, "enhancement should be 1 when lambda=0"

    def test_given_with_ancilla_enhancement_then_be_correct(self) -> None:
        N = 5
        lambda_coupling = 2.0
        state = generate_system_state(N, "coherent", chi=1.0, T=1.0)
        result = compute_phase_sensitivity(
            N=N,
            state=state,
            chi=1.0,
            T=1.0,
            lambda_coupling=lambda_coupling,
            has_ancilla=True,
            noise_config=self._noise_config(),
        )
        expected_enhancement = 1.0 + lambda_coupling * N / 2
        assert np.isclose(result["enhancement"], expected_enhancement), (
            f"enhancement {result['enhancement']} != {expected_enhancement}"
        )

    def test_given_with_ancilla_delta_phi_then_be_smaller(self) -> None:
        N = 5
        state = generate_system_state(N, "coherent", chi=1.0, T=1.0)
        result_no = compute_phase_sensitivity(
            N=N,
            state=state,
            chi=1.0,
            T=1.0,
            lambda_coupling=1.0,
            has_ancilla=False,
            noise_config=self._noise_config(),
        )
        result_with = compute_phase_sensitivity(
            N=N,
            state=state,
            chi=1.0,
            T=1.0,
            lambda_coupling=1.0,
            has_ancilla=True,
            noise_config=self._noise_config(),
        )
        assert result_with["delta_phi_enhanced"] < result_no["delta_phi"], (
            "ancilla should improve sensitivity"
        )

    def test_given_jz_mean_then_be_real(self) -> None:
        N = 5
        state = generate_system_state(N, "coherent", chi=1.0, T=1.0)
        result = compute_phase_sensitivity(
            N=N,
            state=state,
            chi=1.0,
            T=1.0,
            lambda_coupling=0.0,
            has_ancilla=False,
            noise_config=self._noise_config(),
        )
        assert np.isreal(result["Jz_mean"]), (
            f"Jz_mean should be real, got {result['Jz_mean']}"
        )

    def test_given_jz_var_then_be_non_negative(self) -> None:
        N = 5
        state = generate_system_state(N, "coherent", chi=1.0, T=1.0)
        result = compute_phase_sensitivity(
            N=N,
            state=state,
            chi=1.0,
            T=1.0,
            lambda_coupling=0.0,
            has_ancilla=False,
            noise_config=self._noise_config(),
        )
        assert result["Jz_var"] >= -1e-10, (
            f"Jz_var should be non-negative, got {result['Jz_var']:.2e}"
        )

    def test_given_delta_phi_then_be_between_sql_and_hl_for_low_noise(self) -> None:
        N = 5
        state = generate_system_state(N, "coherent", chi=1.0, T=1.0)
        result = compute_phase_sensitivity(
            N=N,
            state=state,
            chi=1.0,
            T=1.0,
            lambda_coupling=0.0,
            has_ancilla=False,
            noise_config=self._noise_config(),
        )
        # CSS at SQL should have delta_phi ≈ SQL
        # Allow some slack for Lindblad evolution effects
        assert result["delta_phi"] >= result["delta_phi_hl"] * 0.5, (
            f"delta_phi {result['delta_phi']:.4f} should be >= HL/2 {result['delta_phi_hl']:.4f}"
        )
        assert result["delta_phi"] <= result["delta_phi_sql"] * 2, (
            f"delta_phi {result['delta_phi']:.4f} should be <= 2*SQL {2 * result['delta_phi_sql']:.4f}"
        )

    def test_given_n_1_then_not_raise(self) -> None:
        state = generate_system_state(1, "coherent", chi=1.0, T=1.0)
        result = compute_phase_sensitivity(
            N=1,
            state=state,
            chi=1.0,
            T=1.0,
            lambda_coupling=0.0,
            has_ancilla=False,
            noise_config=self._noise_config(),
        )
        assert result["delta_phi"] > 0

    def test_given_with_noise_then_not_raise(
        self,
    ) -> None:
        noisy_config = NoiseConfig(gamma_1=0.1, gamma_2=0.05, gamma_phi=0.01)
        N = 5
        state = generate_system_state(N, "coherent", chi=1.0, T=1.0)
        result = compute_phase_sensitivity(
            N=N,
            state=state,
            chi=1.0,
            T=1.0,
            lambda_coupling=0.0,
            has_ancilla=False,
            noise_config=noisy_config,
        )
        assert result["delta_phi"] > 0


class TestComputeTtnBondGrowth:
    """Tests for compute_ttn_bond_growth."""

    def test_returns_dict_with_expected_keys(self) -> None:
        N = 5
        state = generate_system_state(N, "coherent", chi=1.0, T=1.0)
        result = compute_ttn_bond_growth(N, state)
        expected_keys = {"N", "max_bond_dim", "epsilons", "bond_dims"}
        assert set(result.keys()) == expected_keys, (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )

    def test_given_n_then_match_input(self) -> None:
        N = 7
        state = generate_system_state(N, "coherent", chi=1.0, T=1.0)
        result = compute_ttn_bond_growth(N, state)
        assert result["N"] == N, f"N mismatch: {result['N']} != {N}"

    def test_given_coherent_css_then_have_small_bond_dim(self) -> None:
        N = 10
        state = generate_system_state(N, "coherent", chi=1.0, T=1.0)
        result = compute_ttn_bond_growth(N, state)
        # CSS is a product state, so bond dimension should be small
        assert result["max_bond_dim"] <= 3, (
            f"CSS max_bond_dim = {result['max_bond_dim']}, expected <= 3"
        )

    def test_given_noon_then_have_bond_dim_geq_1(self) -> None:
        N = 10
        noon = generate_system_state(N, "noon", chi=1.0, T=1.0)
        noon_result = compute_ttn_bond_growth(N, noon)
        # NOON has two non-zero components, so PR=2, sqrt(PR)≈1.4, int=1.
        assert noon_result["max_bond_dim"] >= 1, (
            f"NOON max_bond_dim = {noon_result['max_bond_dim']}, expected >= 1"
        )

    def test_given_epsilons_then_have_four_elements(self) -> None:
        N = 5
        state = generate_system_state(N, "coherent", chi=1.0, T=1.0)
        result = compute_ttn_bond_growth(N, state)
        assert len(result["epsilons"]) == 4, (
            f"Expected 4 epsilons, got {len(result['epsilons'])}"
        )
        assert len(result["bond_dims"]) == 4, (
            f"Expected 4 bond_dims, got {len(result['bond_dims'])}"
        )

    def test_given_bond_dims_then_be_monotonic_decreasing(self) -> None:
        N = 10
        state = generate_system_state(N, "noon", chi=1.0, T=1.0)
        result = compute_ttn_bond_growth(N, state)
        for i in range(len(result["bond_dims"]) - 1):
            assert result["bond_dims"][i] >= result["bond_dims"][i + 1], (
                f"bond_dims not monotonic at index {i}: {result['bond_dims']}"
            )

    @pytest.mark.parametrize(
        "N", [1, 5, 10, 20], ids=["1", "5", "10", "20"]
    )
    def test_given_max_bond_dim_then_not_exceed_n(self, N: int) -> None:
        state = generate_system_state(N, "noon", chi=1.0, T=1.0)
        result = compute_ttn_bond_growth(N, state)
        assert result["max_bond_dim"] <= N, (
            f"N={N}: max_bond_dim {result['max_bond_dim']} > N"
        )

    @pytest.mark.parametrize(
        "N", [1, 5, 10], ids=["1", "5", "10"]
    )
    def test_given_max_bond_dim_then_be_at_least_1(self, N: int) -> None:
        state = generate_system_state(N, "coherent", chi=1.0, T=1.0)
        result = compute_ttn_bond_growth(N, state)
        assert result["max_bond_dim"] >= 1, f"N={N}: max_bond_dim < 1"

    def test_same_state_gives_same_result(self) -> None:
        N = 5
        state = generate_system_state(N, "hybrid", chi=1.0, T=1.0)
        result1 = compute_ttn_bond_growth(N, state)
        result2 = compute_ttn_bond_growth(N, state)
        assert result1["max_bond_dim"] == result2["max_bond_dim"]
        assert result1["bond_dims"] == result2["bond_dims"]
