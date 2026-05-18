"""
Tests for Fisher Information module.

Covers:
- Pure state QFI: scaling laws, variance formula, phase sensitivity
- Mixed state QFI: first-principles verification, purity bounds, edge cases
- Classical Fisher info: positivity, CFI <= QFI bound
- Input validation: Hermiticity, NaN/Inf, dimension checks
- Physical invariants: convexity, additivity, shift invariance, unitary covariance
- Numerical stability: rank deficiency, degeneracy, large dimensions
- Noise integration: phase diffusion analytical formula
"""

from __future__ import annotations

import numpy as np
import pytest

from .fisher_information import (
    classical_fisher_information,
    generate_phase_generator,
    quantum_fisher_information,
    quantum_fisher_information_dm,
    validate_fisher_inputs,
)


def _make_qfi_first_principles(rho: np.ndarray, G: np.ndarray) -> float:
    """Textbook QFI via SLD formula: 2 * sum_{i != j} (lambda_i - lambda_j)^2 / (lambda_i + lambda_j) * |<i|G|j>|^2."""
    w, v = np.linalg.eigh(rho)
    w = w[::-1]
    v = v[:, ::-1]
    dim = len(w)
    fq = 0.0
    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            denom = w[i] + w[j]
            if denom > 1e-12:
                g_ij = np.vdot(v[:, i], G @ v[:, j])
                fq += 2.0 * (w[i] - w[j]) ** 2 / denom * abs(g_ij) ** 2
    return fq


def _make_noon_state(N: int) -> np.ndarray:
    """Create (|m=-N/2> + |m=+N/2>) / sqrt(2) in J_z basis."""
    dim = N + 1
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0 / np.sqrt(2)
    state[N] = 1.0 / np.sqrt(2)
    return state


def _make_eigenstate(N: int) -> np.ndarray:
    """Create J_z eigenstate |m=-N/2>."""
    state = np.zeros(N + 1, dtype=complex)
    state[0] = 1.0
    return state


def _make_uniform_superposition(N: int) -> np.ndarray:
    """Create uniform superposition of all J_z eigenstates."""
    return np.ones(N + 1, dtype=complex) / np.sqrt(N + 1)


def _make_uniform_qfi(N: int) -> float:
    """Analytical QFI for uniform superposition with J_z generator: F_Q = N(N+2)/3."""
    return N * (N + 2) / 3.0


class TestQuantumFisherInformationPure:
    """QFI for pure states: scaling laws, variance formula, edge cases."""

    @pytest.mark.parametrize("N", [2, 4, 8, 10])
    def test_noon_state_qfi_equals_n_squared(self, N: int) -> None:
        fq = quantum_fisher_information(
            _make_noon_state(N), generate_phase_generator(N, "Jz")
        )
        assert fq == pytest.approx(N**2, rel=0.1)

    def test_noon_qfi_exceeds_eigenstate_qfi(self) -> None:
        G = generate_phase_generator(10, "Jz")
        fq_noon = quantum_fisher_information(_make_noon_state(10), G)
        fq_eigen = quantum_fisher_information(_make_eigenstate(10), G)
        assert fq_noon > fq_eigen + 1e-5

    def test_eigenstate_of_generator_has_zero_qfi(self) -> None:
        assert quantum_fisher_information(
            _make_eigenstate(5), generate_phase_generator(5, "Jz")
        ) == pytest.approx(0.0, abs=1e-10)

    def test_qfi_equals_4_variance_for_pure_states(self) -> None:
        N = 5
        state = _make_noon_state(N)
        G = generate_phase_generator(N, "Jz")
        g_exp = np.vdot(state, G @ state).real
        g2_exp = np.vdot(state, G @ G @ state).real
        assert quantum_fisher_information(state, G) == pytest.approx(
            4.0 * (g2_exp - g_exp**2), rel=1e-5
        )

    def test_noon_delta_phi_beats_standard_quantum_limit(self) -> None:
        N = 100
        G = generate_phase_generator(N, "Jz")
        fq = quantum_fisher_information(_make_noon_state(N), G)
        assert 1.0 / np.sqrt(fq) < 1.0 / np.sqrt(N)

    @pytest.mark.parametrize("N", [10, 50, 100], ids=["N=10", "N=50", "N=100"])
    def test_uniform_superposition_qfi_matches_analytical_formula(self, N: int) -> None:
        state = _make_uniform_superposition(N)
        fq = quantum_fisher_information(state, generate_phase_generator(N, "Jz"))
        assert fq == pytest.approx(_make_uniform_qfi(N), rel=1e-10)

    def test_handles_n_1000_without_overflow(self) -> None:
        assert (
            quantum_fisher_information(
                _make_noon_state(1000), generate_phase_generator(1000, "Jz")
            )
            > 0
        )

    def test_dimension_mismatch_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            quantum_fisher_information(np.ones(5), np.eye(10))

    def test_cfi_handles_tiny_probabilities(self) -> None:
        probs = np.array([[1e-10, 1 - 1e-10], [0.5, 0.5]])
        assert np.all(np.isfinite(classical_fisher_information(probs, dphi=1e-3)))


class TestQuantumFisherInformationMixed:
    """QFI for mixed states: purity limits, first-principles verification, edge cases."""

    def test_dm_qfi_matches_pure_state_qfi(self) -> None:
        N = 5
        state = _make_noon_state(N)
        G = generate_phase_generator(N, "Jz")
        fq_pure = quantum_fisher_information(state, G)
        fq_dm = quantum_fisher_information_dm(np.outer(state, state.conj()), G)
        assert fq_pure == pytest.approx(fq_dm, rel=1e-5)

    def test_mixing_reduces_qfi(self) -> None:
        N = 5
        state = _make_noon_state(N)
        G = generate_phase_generator(N, "Jz")
        dim = N + 1
        rho_pure = np.outer(state, state.conj())
        rho_mixed = 0.5 * rho_pure + 0.5 * np.eye(dim) / dim
        assert (
            quantum_fisher_information_dm(rho_mixed, G)
            <= quantum_fisher_information_dm(rho_pure, G) + 1e-5
        )

    def test_maximally_mixed_state_has_zero_qfi(self) -> None:
        dim = 4
        G = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        G = G + G.conj().T
        assert quantum_fisher_information_dm(np.eye(dim) / dim, G) == pytest.approx(
            0.0, abs=1e-12
        )

    def test_commuting_rho_and_g_has_zero_qfi(self) -> None:
        assert quantum_fisher_information_dm(
            np.diag([0.4, 0.3, 0.2, 0.1, 0.0]), np.diag(np.arange(5.0))
        ) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize(
        "seed", range(5), ids=["seed_0", "seed_1", "seed_2", "seed_3", "seed_4"]
    )
    def test_2d_mixed_qfi_matches_sld_formula(self, seed: int) -> None:
        rng = np.random.default_rng(42 + seed)
        a = rng.uniform(0.1, 0.9)
        rho = np.diag([a, 1.0 - a])
        g11, g22 = rng.uniform(-2, 2, 2)
        g_off = rng.uniform(-1, 1) + 1j * rng.uniform(-1, 1)
        G = np.array([[g11, g_off], [np.conj(g_off), g22]], dtype=complex)
        fq = quantum_fisher_information_dm(rho, G)
        assert fq == pytest.approx(_make_qfi_first_principles(rho, G), rel=1e-10)
        assert fq >= 0.0

    @pytest.mark.parametrize("dim", [3, 5, 8])
    def test_high_dim_mixed_qfi_matches_sld_formula(self, dim: int) -> None:
        rng = np.random.default_rng(123)
        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        rho = A @ A.conj().T
        rho = rho / np.trace(rho)
        G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        G = G + G.conj().T
        fq = quantum_fisher_information_dm(rho, G)
        assert fq == pytest.approx(_make_qfi_first_principles(rho, G), rel=1e-10)
        assert fq >= 0.0

    def test_degenerate_eigenvalues_handled_correctly(self) -> None:
        dim = 3
        rho = np.diag([0.4, 0.4, 0.2])
        G = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        G = G + G.conj().T
        assert quantum_fisher_information_dm(rho, G) == pytest.approx(
            _make_qfi_first_principles(rho, G), rel=1e-10
        )

    @pytest.mark.parametrize("eps", [1e-3, 1e-6, 1e-10], ids=["1e-3", "1e-6", "1e-10"])
    def test_qfi_approaches_pure_value_as_mixedness_decreases(self, eps: float) -> None:
        dim = 4
        rng = np.random.default_rng(7)
        G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        G = G + G.conj().T
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        fq_pure = quantum_fisher_information(state, G)
        rho = (1.0 - eps) * np.outer(state, state.conj()) + eps * np.eye(dim) / dim
        assert quantum_fisher_information_dm(rho, G) <= fq_pure + 1e-10

    def test_1d_hilbert_space_has_zero_qfi(self) -> None:
        assert quantum_fisher_information_dm(
            np.ones((1, 1)), np.array([[2.0]])
        ) == pytest.approx(0.0, abs=1e-15)

    def test_real_symmetric_inputs_produce_finite_qfi(self) -> None:
        rng = np.random.default_rng(42)
        dim = 4
        A = rng.normal(size=(dim, dim))
        rho = A @ A.T / np.trace(A @ A.T)
        G = rng.normal(size=(dim, dim))
        G = G + G.T
        fq = quantum_fisher_information_dm(rho, G)
        assert np.isfinite(fq)
        assert fq >= 0.0

    def test_rank_1_rho_matches_pure_state_qfi(self) -> None:
        rng = np.random.default_rng(42)
        dim = 6
        vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        vec = vec / np.linalg.norm(vec)
        G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        G = G + G.conj().T
        assert quantum_fisher_information(vec, G) == pytest.approx(
            quantum_fisher_information_dm(np.outer(vec, vec.conj()), G), rel=1e-12
        )


class TestClassicalAndQuantumFisher:
    """Classical Fisher info: positivity, CFI <= QFI bound, phase sensitivity."""

    def test_fc_non_negative_for_sinusoidal_probability(self) -> None:
        n_phi = 10
        phis = np.linspace(0, 2 * np.pi, n_phi)
        probs = np.zeros((n_phi, 2))
        probs[:, 0] = 0.5 + 0.4 * np.sin(phis)
        probs[:, 1] = 1 - probs[:, 0]
        assert np.all(classical_fisher_information(probs, dphi=1e-3) >= 0)

    def test_fc_positive_where_prob_positive(self) -> None:
        n_phi = 20
        phis = np.linspace(0.1, 2 * np.pi - 0.1, n_phi)
        probs = np.zeros((n_phi, 2))
        probs[:, 0] = 0.5 + 0.4 * np.cos(phis)
        probs[:, 1] = 1 - probs[:, 0]
        assert np.all(
            classical_fisher_information(np.clip(probs, 1e-10, None), dphi=1e-4) >= 0
        )

    def test_fc_leq_fq_cramer_rao_bound(self) -> None:
        N = 5
        dim = N + 1
        phis = np.linspace(0, 2 * np.pi, 20)
        probs = np.zeros((20, dim))
        for i, phi in enumerate(phis):
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0 / np.sqrt(2)
            state[N] = np.exp(1j * phi) / np.sqrt(2)
            probs[i] = np.abs(state) ** 2

        G = generate_phase_generator(N, "Jz")
        fq = quantum_fisher_information(_make_noon_state(N), G)
        fc = classical_fisher_information(probs, dphi=1e-3)
        assert np.max(fc) <= fq + 1e-5

    def test_fc_leq_fq_at_each_phi(self) -> None:
        N = 5
        dim = N + 1
        phis = np.linspace(0, 2 * np.pi, 20)
        probs = np.zeros((20, dim))
        for i, phi in enumerate(phis):
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0 / np.sqrt(2)
            state[N] = np.exp(1j * phi) / np.sqrt(2)
            probs[i] = np.abs(state) ** 2

        G = generate_phase_generator(N, "Jz")
        fq = quantum_fisher_information(_make_noon_state(N), G)
        fc = classical_fisher_information(probs, dphi=1e-3)
        assert np.all(fc <= fq + 1e-5)

    def test_uniform_superposition_qfi_exceeds_n(self) -> None:
        N = 20
        probs = np.ones(N + 1) / (N + 1)
        state = np.sqrt(probs).astype(complex)
        fq = quantum_fisher_information(state, generate_phase_generator(N, "Jz"))
        assert fq > N


class TestInputValidation:
    """Input validation for both pure and mixed-state QFI paths."""

    def test_validate_positive(self) -> None:
        validate_fisher_inputs(1.0)
        validate_fisher_inputs(100.0)

    def test_validate_invalid(self) -> None:
        with pytest.raises(ValueError):
            validate_fisher_inputs(0.0)
        with pytest.raises(ValueError):
            validate_fisher_inputs(-1.0)
        with pytest.raises(ValueError):
            validate_fisher_inputs(np.nan)

    def test_non_hermitian_rho_raises(self) -> None:
        dim = 4
        rho = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        rho = rho @ rho.conj().T
        rho[0, 1] += 1.0j
        with pytest.raises(ValueError, match="Hermitian"):
            quantum_fisher_information_dm(rho, np.eye(dim, dtype=complex))

    @pytest.mark.parametrize("val", [np.nan, np.inf])
    def test_invalid_in_rho_raises(self, val: float) -> None:
        dim = 3
        rho = np.eye(dim, dtype=complex) / dim
        rho[0, 0] = val
        with pytest.raises(ValueError):
            quantum_fisher_information_dm(rho, np.eye(dim, dtype=complex))

    @pytest.mark.parametrize("val", [np.nan, np.inf])
    def test_invalid_in_generator_raises(self, val: float) -> None:
        dim = 3
        rho = np.eye(dim, dtype=complex) / dim
        G = np.eye(dim, dtype=complex)
        G[0, 0] = val
        with pytest.raises(ValueError):
            quantum_fisher_information_dm(rho, G)

    def test_non_hermitian_generator_raises(self) -> None:
        dim = 4
        G = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        with pytest.raises(ValueError, match="Hermitian"):
            quantum_fisher_information_dm(np.eye(dim, dtype=complex) / dim, G)

    def test_dimension_mismatch_generator_raises(self) -> None:
        with pytest.raises(ValueError, match=r"dimension|match"):
            quantum_fisher_information_dm(
                np.eye(3, dtype=complex) / 3, np.eye(5, dtype=complex)
            )

    def test_non_square_rho_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            quantum_fisher_information_dm(
                np.ones((3, 4), dtype=complex), np.eye(3, dtype=complex)
            )

    def test_non_square_generator_raises(self) -> None:
        with pytest.raises(ValueError, match="dimension"):
            quantum_fisher_information_dm(
                np.eye(3, dtype=complex) / 3, np.ones((3, 4), dtype=complex)
            )

    def test_zero_trace_rho_yields_finite_qfi(self) -> None:
        fq = quantum_fisher_information_dm(
            np.zeros((3, 3), dtype=complex), np.eye(3, dtype=complex)
        )
        assert np.isfinite(fq)
        assert fq >= 0.0

    @pytest.mark.parametrize("val", [np.nan, np.inf])
    def test_invalid_in_state_raises(self, val: float) -> None:
        dim = 4
        state = np.ones(dim, dtype=complex)
        state[0] = val
        with pytest.raises(ValueError):
            quantum_fisher_information(state, np.eye(dim, dtype=complex))


class TestPhysicalInvariantsDM:
    """QFI satisfies fundamental physical invariants for mixed states."""

    @pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
    def test_qfi_satisfies_convexity_inequality(self, p: float) -> None:
        rng = np.random.default_rng(42)
        dim = 5
        A1 = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        rho1 = A1 @ A1.conj().T / np.trace(A1 @ A1.conj().T)
        A2 = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        rho2 = A2 @ A2.conj().T / np.trace(A2 @ A2.conj().T)
        G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        G = G + G.conj().T

        fq1 = quantum_fisher_information_dm(rho1, G)
        fq2 = quantum_fisher_information_dm(rho2, G)
        fq_mix = quantum_fisher_information_dm(p * rho1 + (1.0 - p) * rho2, G)
        assert fq_mix <= p * fq1 + (1.0 - p) * fq2 + 1e-10

    def test_qfi_additive_over_tensor_products(self) -> None:
        rng = np.random.default_rng(42)
        d1, d2 = 3, 2
        A1 = rng.normal(size=(d1, d1)) + 1j * rng.normal(size=(d1, d1))
        rho1 = A1 @ A1.conj().T / np.trace(A1 @ A1.conj().T)
        G1 = rng.normal(size=(d1, d1)) + 1j * rng.normal(size=(d1, d1))
        G1 = G1 + G1.conj().T
        A2 = rng.normal(size=(d2, d2)) + 1j * rng.normal(size=(d2, d2))
        rho2 = A2 @ A2.conj().T / np.trace(A2 @ A2.conj().T)
        G2 = rng.normal(size=(d2, d2)) + 1j * rng.normal(size=(d2, d2))
        G2 = G2 + G2.conj().T

        expected = quantum_fisher_information_dm(
            rho1, G1
        ) + quantum_fisher_information_dm(rho2, G2)
        rho_product = np.kron(rho1, rho2)
        G_product = np.kron(G1, np.eye(d2)) + np.kron(np.eye(d1), G2)
        assert quantum_fisher_information_dm(rho_product, G_product) == pytest.approx(
            expected, rel=1e-10
        )

    @pytest.mark.parametrize(
        "c", [-2.5, 0.0, 1.0, 10.0], ids=["c_-2.5", "c_0.0", "c_1.0", "c_10.0"]
    )
    def test_qfi_invariant_under_g_shift(self, c: float) -> None:
        rng = np.random.default_rng(42)
        dim = 4
        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        rho = A @ A.conj().T / np.trace(A @ A.conj().T)
        G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        G = G + G.conj().T

        fq_base = quantum_fisher_information_dm(rho, G)
        assert fq_base == pytest.approx(
            quantum_fisher_information_dm(rho, G + c * np.eye(dim)), rel=1e-12
        )

    def test_qfi_covariant_under_unitary_transformation(self) -> None:
        rng = np.random.default_rng(42)
        dim = 4
        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        rho = A @ A.conj().T / np.trace(A @ A.conj().T)
        G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        G = G + G.conj().T
        U, _ = np.linalg.qr(
            rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        )

        fq_original = quantum_fisher_information_dm(rho, G)
        assert fq_original == pytest.approx(
            quantum_fisher_information_dm(U @ rho @ U.conj().T, U @ G @ U.conj().T),
            rel=1e-10,
        )

    @pytest.mark.parametrize("dim", [3, 5])
    def test_qfi_leq_4_variance_equality_for_pure(self, dim: int) -> None:
        rng = np.random.default_rng(42)
        G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        G = G + G.conj().T

        vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        vec = vec / np.linalg.norm(vec)
        var_g = max(
            0.0, np.real(np.vdot(vec, G @ G @ vec) - np.vdot(vec, G @ vec) ** 2)
        )
        fq_pure = quantum_fisher_information_dm(np.outer(vec, vec.conj()), G)
        assert fq_pure == pytest.approx(4.0 * var_g, rel=1e-10)

        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        rho_mixed = A @ A.conj().T / np.trace(A @ A.conj().T)
        fq_mixed = quantum_fisher_information_dm(rho_mixed, G)
        var_g_mixed = max(
            0.0, np.real(np.trace(rho_mixed @ G @ G) - np.trace(rho_mixed @ G) ** 2)
        )
        assert fq_mixed <= 4.0 * var_g_mixed + 1e-10

    def test_zero_generator_yields_zero_qfi(self) -> None:
        rng = np.random.default_rng(42)
        dim = 4
        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        rho = A @ A.conj().T / np.trace(A @ A.conj().T)
        assert quantum_fisher_information_dm(
            rho, np.zeros((dim, dim), dtype=complex)
        ) == pytest.approx(0.0, abs=1e-15)


class TestNumericalStabilityDM:
    """QFI handles numerical edge cases stably."""

    def test_rank_deficient_rho_yields_known_qfi(self) -> None:
        dim = 6
        rho = np.diag([0.5, 0.3, 0.2, 0.0, 0.0, 0.0])
        G = np.zeros((dim, dim), dtype=complex)
        G[0, 1] = G[1, 0] = 1.0
        G[2, 3] = G[3, 2] = 0.5
        fq = quantum_fisher_information_dm(rho, G)
        assert np.isfinite(fq)
        assert fq >= 0.0
        # lambda_0=0.5, lambda_1=0.3: 4*((0.5-0.3)^2/(0.5+0.3))*1^2 = 0.2
        # lambda_2=0.2, lambda_3=0.0: 4*((0.2-0)^2/(0.2+0))*0.5^2 = 0.2
        assert fq == pytest.approx(0.4, rel=1e-10)

    def test_g_couples_support_to_null_space(self) -> None:
        dim = 5
        rho = np.diag([0.6, 0.4, 0.0, 0.0, 0.0])
        G = np.zeros((dim, dim), dtype=complex)
        G[0, 3] = G[3, 0] = 2.0
        G[1, 2] = G[2, 1] = 1.0
        fq = quantum_fisher_information_dm(rho, G)
        # lambda_0=0.6, lambda_3=0.0: 4*((0.6)^2/0.6)*4 = 9.6
        # lambda_1=0.4, lambda_2=0.0: 4*((0.4)^2/0.4)*1 = 1.6
        assert fq == pytest.approx(11.2, rel=1e-10)

    @pytest.mark.parametrize(
        "eps",
        [1e-2, 1e-4, 1e-6, 1e-8, 1e-10],
        ids=["1e-2", "1e-4", "1e-6", "1e-8", "1e-10"],
    )
    def test_near_degenerate_eigenvalues_stable(self, eps: float) -> None:
        rng = np.random.default_rng(99)
        dim = 4
        G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        G = G + G.conj().T

        w = np.array([0.5 - eps / 2, 0.5 + eps / 2, 0.0, 0.0])
        rho = np.diag(w / np.sum(w))
        fq = quantum_fisher_information_dm(rho, G)
        assert np.isfinite(fq)
        assert fq >= 0.0

    def test_handles_dim_100(self) -> None:
        rng = np.random.default_rng(42)
        dim = 100
        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        rho = A @ A.conj().T / np.trace(A @ A.conj().T)
        G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        G = G + G.conj().T
        fq = quantum_fisher_information_dm(rho, G)
        assert np.isfinite(fq)
        assert fq >= 0.0
        assert fq < dim**3

    def test_diagonal_g_in_rho_eigenbasis_yields_zero_qfi(self) -> None:
        rho = np.diag([0.3, 0.25, 0.2, 0.15, 0.1])
        G = np.diag(np.random.default_rng(42).uniform(-2, 2, size=5))
        assert quantum_fisher_information_dm(rho, G) == pytest.approx(0.0, abs=1e-15)

    def test_g_proportional_to_identity_yields_zero_qfi(self) -> None:
        rng = np.random.default_rng(42)
        dim = 4
        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        rho = A @ A.conj().T / np.trace(A @ A.conj().T)
        assert quantum_fisher_information_dm(
            rho, 3.0 * np.eye(dim, dtype=complex)
        ) == pytest.approx(0.0, abs=1e-15)


def _make_phase_diffused_ghz(N: int, gamma_phi: float, t: float) -> np.ndarray:
    """Phase-diffused GHZ density matrix: F_Q(t) = N^2 * exp(-gamma * t * N^2)."""
    dim = N + 1
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 0.5
    rho[dim - 1, dim - 1] = 0.5
    decay = np.exp(-gamma_phi * t * N**2 / 2)
    rho[0, dim - 1] = decay / 2
    rho[dim - 1, 0] = decay / 2
    return rho


class TestNoiseIntegrationDM:
    """QFI correctly captures phase diffusion noise effects."""

    def test_strong_phase_noise_erases_qfi(self) -> None:
        N = 6
        G = generate_phase_generator(N, "Jz")
        assert quantum_fisher_information_dm(
            _make_phase_diffused_ghz(N, 100.0, 1.0), G
        ) == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize(
        "gamma_phi",
        [0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
        ids=["0", "0.01", "0.05", "0.1", "0.5", "1.0"],
    )
    def test_qfi_bounded_by_noiseless_value(self, gamma_phi: float) -> None:
        N = 6
        G = generate_phase_generator(N, "Jz")
        fq = quantum_fisher_information_dm(
            _make_phase_diffused_ghz(N, gamma_phi, 1.0), G
        )
        assert fq <= N**2 + 1e-10
        assert fq >= 0.0

    @pytest.mark.parametrize("N", [4, 6])
    @pytest.mark.parametrize(
        "gamma_phi",
        [0.0, 0.02, 0.05, 0.1],
        ids=["0", "0.02", "0.05", "0.1"],
    )
    def test_qfi_matches_n_squared_exp_decay(self, N: int, gamma_phi: float) -> None:
        G = generate_phase_generator(N, "Jz")
        fq = quantum_fisher_information_dm(
            _make_phase_diffused_ghz(N, gamma_phi, 1.0), G
        )
        assert fq == pytest.approx(N**2 * np.exp(-gamma_phi * N**2), rel=1e-10)

    @pytest.mark.parametrize(
        "gamma_phi",
        [1e-3, 1e-6, 1e-10],
        ids=["1e-3", "1e-6", "1e-10"],
    )
    def test_qfi_approaches_n_squared_as_gamma_t_vanishes(
        self, gamma_phi: float
    ) -> None:
        N = 8
        G = generate_phase_generator(N, "Jz")
        fq = quantum_fisher_information_dm(
            _make_phase_diffused_ghz(N, gamma_phi, 1.0), G
        )
        expected = N**2 * np.exp(-gamma_phi * N**2)
        assert fq == pytest.approx(expected, rel=1e-10)
        assert fq <= N**2 + 1e-10
