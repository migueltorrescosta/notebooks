from __future__ import annotations

import numpy as np
import pytest

from .partial_trace import (
    BipartiteConfig,
    build_bipartite_hamiltonian,
    build_bipartite_hamiltonian_components,
    compute_reduced_densities,
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
