"""Tests for Dicke-basis state generation (src.physics.states)."""

import numpy as np
import pytest

from src.physics.states import generate_noon_state, generate_twin_fock_state


class TestGenerateTwinFockState:
    @pytest.mark.parametrize("N", [2, 4, 6, 8, 10], ids=["2", "4", "6", "8", "10"])
    def test_twin_fock_normalized(self, N: int) -> None:
        state = generate_twin_fock_state(N)
        norm = np.sum(np.abs(state) ** 2)
        assert norm == pytest.approx(1.0)

    @pytest.mark.parametrize("N", [2, 4, 6, 8, 10], ids=["2", "4", "6", "8", "10"])
    def test_twin_fock_dimension_n_plus_one(self, N: int) -> None:
        state = generate_twin_fock_state(N)
        assert state.shape == (N + 1,)

    def test_twin_fock_peak_at_middle(self) -> None:
        N = 6
        state = generate_twin_fock_state(N)
        assert state[N // 2] == pytest.approx(1.0)
        assert np.sum(np.abs(state[: N // 2]) ** 2) == pytest.approx(0.0)
        assert np.sum(np.abs(state[N // 2 + 1 :]) ** 2) == pytest.approx(0.0)

    def test_twin_fock_requires_even_n(self) -> None:
        with pytest.raises(ValueError, match="even"):
            generate_twin_fock_state(3)

    def test_twin_fock_zero_j_z(self) -> None:
        N = 6
        state = generate_twin_fock_state(N)
        # Dicke basis: index i has m = N/2 - i
        # For m=0 at index N/2, ⟨J_z⟩ = 0
        from src.physics.dicke_basis import jz_operator

        jz = jz_operator(N)
        jz_mean = np.real(np.conj(state) @ jz @ state)
        assert jz_mean == pytest.approx(0.0, abs=1e-10)

    def test_twin_fock_state_real(self) -> None:
        state = generate_twin_fock_state(8)
        assert np.all(np.isreal(state))


class TestGenerateNOONState:
    @pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 10])
    def test_noon_normalized(self, N: int) -> None:
        state = generate_noon_state(N)
        norm = np.sum(np.abs(state) ** 2)
        assert norm == pytest.approx(1.0)

    @pytest.mark.parametrize("N", [1, 2, 3, 4, 5])
    def test_noon_dimension_n_plus_one(self, N: int) -> None:
        state = generate_noon_state(N)
        assert state.shape == (N + 1,)

    @pytest.mark.parametrize("N", [1, 2, 3, 4, 5])
    def test_noon_equal_amplitudes(self, N: int) -> None:
        state = generate_noon_state(N)
        assert np.abs(state[0]) == pytest.approx(1.0 / np.sqrt(2))
        assert np.abs(state[N]) == pytest.approx(1.0 / np.sqrt(2))

    @pytest.mark.parametrize("N", [1, 2, 3, 4, 5])
    def test_noon_zero_elsewhere(self, N: int) -> None:
        state = generate_noon_state(N)
        interior = state[1:N]
        assert np.sum(np.abs(interior) ** 2) == pytest.approx(0.0, abs=1e-15)

    def test_noon_requires_positive_n(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            generate_noon_state(0)

    def test_noon_zero_j_z(self) -> None:
        N = 4
        state = generate_noon_state(N)
        from src.physics.dicke_basis import jz_operator

        jz = jz_operator(N)
        jz_mean = np.real(np.conj(state) @ jz @ state)
        assert jz_mean == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("N", [2, 4, 6])
    def test_noon_variance_n_squared_over_4(self, N: int) -> None:
        state = generate_noon_state(N)
        from src.physics.dicke_basis import jz_operator

        jz = jz_operator(N)
        jz_sq = jz @ jz
        mean = np.real(np.conj(state) @ jz @ state)
        mean_sq = np.real(np.conj(state) @ jz_sq @ state)
        var = mean_sq - mean**2
        assert var == pytest.approx(N**2 / 4, rel=1e-6)
