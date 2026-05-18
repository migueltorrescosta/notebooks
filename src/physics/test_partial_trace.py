from __future__ import annotations

import numpy as np
import pytest

from .partial_trace import (
    BipartiteConfig,
    build_bipartite_hamiltonian,
    build_bipartite_hamiltonian_components,
    compute_reduced_densities,
    evolve_bipartite,
    evolve_density_matrix,
    evolve_state,
    local_hamiltonian,
    partial_trace_a,
    partial_trace_b,
    validate_partial_trace,
)


class TestLocalHamiltonian:
    @pytest.mark.parametrize("dim", [2, 3, 4], ids=["dim=2", "dim=3", "dim=4"])
    def test_local_hamiltonian_is_hermitian(self, dim: int) -> None:
        H = local_hamiltonian(dim, j=1.0, u=0.5, delta=0.1)
        assert pytest.approx(H.conj().T) == H

    def test_local_hamiltonian_has_correct_shape(self) -> None:
        dim = 3
        H = local_hamiltonian(dim, 1.0, 0.0, 0.0)
        assert H.shape == (dim, dim)


class TestBipartiteHamiltonian:
    def test_bipartite_hamiltonian_is_hermitian(self) -> None:
        config = BipartiteConfig()
        H = build_bipartite_hamiltonian(config)
        assert pytest.approx(H.conj().T) == H

    def test_bipartite_hamiltonian_has_correct_dimensions(self) -> None:
        config = BipartiteConfig(dim_a=2, dim_b=3)
        H = build_bipartite_hamiltonian(config)
        assert H.shape == (6, 6)

    def test_components_have_correct_dimensions(self) -> None:
        config = BipartiteConfig()
        h_a, h_b, h_int, h_full = build_bipartite_hamiltonian_components(config)
        assert h_a.shape == (config.dim_a, config.dim_a)
        assert h_b.shape == (config.dim_b, config.dim_b)
        assert h_int.shape == (
            config.dim_a * config.dim_b,
            config.dim_a * config.dim_b,
        )
        assert h_full.shape == (
            config.dim_a * config.dim_b,
            config.dim_a * config.dim_b,
        )


class TestEvolution:
    def test_evolved_state_remains_normalized(self) -> None:
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        psi0 = np.array([1, 0], dtype=complex)
        psi_t = evolve_state(H, psi0, time=1.0)
        norm = np.sqrt(np.vdot(psi_t, psi_t).real)
        assert norm == pytest.approx(1.0)

    def test_density_matrix_evolution_preserves_trace(self) -> None:
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        rho_t = evolve_density_matrix(H, rho0, time=1.0)
        assert np.trace(rho_t) == pytest.approx(1.0)


class TestEvolveBipartite:
    def test_evolve_bipartite_zero_time_returns_initial_state(self) -> None:
        n = 4
        H = np.eye(n, dtype=complex)
        psi0 = np.zeros(n, dtype=float)
        psi0[0] = 1.0
        result = evolve_bipartite(H, psi0, time=0.0)
        assert np.allclose(result, psi0)

    def test_evolve_bipartite_matches_row_vector_convention(self) -> None:
        """Verify evolve_bipartite matches psi0 @ expm(-iHt)."""
        import scipy.linalg

        n = 4
        H = np.array(
            [
                [1.0, 0.5, 0.0, 0.0],
                [0.5, 0.0, 0.3, 0.0],
                [0.0, 0.3, -1.0, 0.2],
                [0.0, 0.0, 0.2, -0.5],
            ],
            dtype=complex,
        )
        psi0 = np.zeros(n, dtype=float)
        psi0[0] = 1.0
        t = 1.5
        expected = psi0 @ scipy.linalg.expm(-1j * t * H)
        result = evolve_bipartite(H, psi0, t)
        assert np.allclose(result, expected)

    def test_evolve_bipartite_preserves_norm(self) -> None:
        n = 4
        H = np.array(
            [
                [1.0, 0.5, 0.0, 0.0],
                [0.5, 0.0, 0.3, 0.0],
                [0.0, 0.3, -1.0, 0.2],
                [0.0, 0.0, 0.2, -0.5],
            ],
            dtype=complex,
        )
        psi0 = np.zeros(n, dtype=float)
        psi0[2] = 1.0
        result = evolve_bipartite(H, psi0, time=0.7)
        norm = np.linalg.norm(result)
        assert norm == pytest.approx(1.0)

    def test_evolve_bipartite_raises_on_norm_loss(self) -> None:
        H = np.array([[1.0, 0.0], [0.0, 0.5]], dtype=complex)
        psi0 = np.array([1.0, 0.0])
        with pytest.raises(AssertionError):
            evolve_bipartite(H * (1.0 + 0.1j), psi0, time=1.0)


class TestPartialTrace:
    def test_tracing_subsystem_b_gives_correct_dimension(self) -> None:
        dim_a, dim_b = 2, 3
        rho_full = np.ones((dim_a * dim_b, dim_a * dim_b), dtype=complex)
        rho_a = partial_trace_a(rho_full, dim_a, dim_b)
        assert rho_a.shape == (dim_a, dim_a)

    def test_tracing_subsystem_a_gives_correct_dimension(self) -> None:
        dim_a, dim_b = 2, 3
        rho_full = np.ones((dim_a * dim_b, dim_a * dim_b), dtype=complex)
        rho_b = partial_trace_b(rho_full, dim_a, dim_b)
        assert rho_b.shape == (dim_b, dim_b)

    def test_tracing_both_subsystems_preserves_trace(self) -> None:
        dim_a, dim_b = 2, 2
        psi0 = np.zeros(dim_a * dim_b, dtype=complex)
        psi0[0] = 1.0
        rho_full = np.outer(psi0, psi0.conj())

        rho_a = partial_trace_a(rho_full, dim_a, dim_b)
        rho_b = partial_trace_b(rho_full, dim_a, dim_b)

        assert np.trace(rho_a) == pytest.approx(1.0)
        assert np.trace(rho_b) == pytest.approx(1.0)


class TestComputeReducedDensities:
    def test_compute_reduced_densities_returns_all_three_matrices(self) -> None:
        config = BipartiteConfig()
        rho_a, rho_b, rho_full = compute_reduced_densities(config, time=0.0)
        assert rho_a.shape == (config.dim_a, config.dim_a)
        assert rho_b.shape == (config.dim_b, config.dim_b)
        assert rho_full.shape == (
            config.dim_a * config.dim_b,
            config.dim_a * config.dim_b,
        )

    def test_reduced_densities_satisfy_validation(self) -> None:
        config = BipartiteConfig()
        rho_a, rho_b, rho_full = compute_reduced_densities(config, time=0.0)
        assert validate_partial_trace(rho_full, rho_a, rho_b)
