"""Unit tests for Partial Trace physics module."""

import numpy as np
import pytest

from src.partial_trace import (
    local_hamiltonian,
    BipartiteConfig,
    build_bipartite_hamiltonian,
    build_bipartite_hamiltonian_components,
    evolve_state,
    evolve_density_matrix,
    partial_trace_a,
    partial_trace_b,
    compute_reduced_densities,
    validate_partial_trace,
)


class TestLocalHamiltonian:
    def test_local_hamiltonian_hermitian(self) -> None:
        """Local Hamiltonian should be Hermitian."""
        for dim in [2, 3, 4]:
            H = local_hamiltonian(dim, j=1.0, u=0.5, delta=0.1)
            assert np.allclose(H, H.conj().T)

    def test_local_hamiltonian_shape(self) -> None:
        """Should have correct shape."""
        dim = 3
        H = local_hamiltonian(dim, 1.0, 0.0, 0.0)
        assert H.shape == (dim, dim)


class TestBipartiteHamiltonian:
    def test_bipartite_hermitian(self) -> None:
        """Bipartite Hamiltonian should be Hermitian."""
        config = BipartiteConfig()
        H = build_bipartite_hamiltonian(config)
        assert np.allclose(H, H.conj().T)

    def test_bipartite_shape(self) -> None:
        """Should have correct dimensions."""
        config = BipartiteConfig(dim_a=2, dim_b=3)
        H = build_bipartite_hamiltonian(config)
        assert H.shape == (6, 6)

    def test_components_separable(self) -> None:
        """Components should add up to full."""
        config = BipartiteConfig()
        h_a, h_b, h_int, h_full = build_bipartite_hamiltonian_components(config)
        # Check dimensions
        assert h_a.shape == (config.dim_a, config.dim_a)
        assert h_b.shape == (config.dim_b, config.dim_b)
        assert h_int.shape == (config.dim_a * config.dim_b, config.dim_a * config.dim_b)
        assert h_full.shape == (
            config.dim_a * config.dim_b,
            config.dim_a * config.dim_b,
        )


class TestEvolution:
    def test_evolve_state_normalized(self) -> None:
        """Evolved state should remain normalized."""
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        psi0 = np.array([1, 0], dtype=complex)
        psi_t = evolve_state(H, psi0, time=1.0)
        norm = np.sqrt(np.vdot(psi_t, psi_t).real)
        assert norm == pytest.approx(1.0)

    def test_evolve_density_matrix(self) -> None:
        """Density matrix evolution should preserve trace."""
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        rho_t = evolve_density_matrix(H, rho0, time=1.0)
        assert np.isclose(np.trace(rho_t), 1.0)


class TestPartialTrace:
    def test_partial_trace_a(self) -> None:
        """Tracing B should give correct dimension."""
        dim_a, dim_b = 2, 3
        rho_full = np.ones((dim_a * dim_b, dim_a * dim_b), dtype=complex)
        rho_a = partial_trace_a(rho_full, dim_a, dim_b)
        assert rho_a.shape == (dim_a, dim_a)

    def test_partial_trace_b(self) -> None:
        """Tracing A should give correct dimension."""
        dim_a, dim_b = 2, 3
        rho_full = np.ones((dim_a * dim_b, dim_a * dim_b), dtype=complex)
        rho_b = partial_trace_b(rho_full, dim_a, dim_b)
        assert rho_b.shape == (dim_b, dim_b)

    def test_partial_trace_conservation(self) -> None:
        """Tr_A[Tr_B[ρ]] = Tr[ρ]."""
        # Pure state |00⟩
        dim_a, dim_b = 2, 2
        psi0 = np.zeros(dim_a * dim_b, dtype=complex)
        psi0[0] = 1.0
        rho_full = np.outer(psi0, psi0.conj())

        rho_a = partial_trace_a(rho_full, dim_a, dim_b)
        rho_b = partial_trace_b(rho_full, dim_a, dim_b)

        # Tr_A[Tr_B[ρ]] = Tr[ρ] = 1
        assert np.isclose(np.trace(rho_a), 1.0)
        assert np.isclose(np.trace(rho_b), 1.0)


class TestComputeReducedDensities:
    def test_compute_reduced_densities(self) -> None:
        """Should return all three density matrices."""
        config = BipartiteConfig()
        rho_a, rho_b, rho_full = compute_reduced_densities(config, time=0.0)
        assert rho_a.shape == (config.dim_a, config.dim_a)
        assert rho_b.shape == (config.dim_b, config.dim_b)
        assert rho_full.shape == (
            config.dim_a * config.dim_b,
            config.dim_a * config.dim_b,
        )

    def test_validate_reduced_densities(self) -> None:
        """Reduced densities should pass validation."""
        config = BipartiteConfig()
        rho_a, rho_b, rho_full = compute_reduced_densities(config, time=0.0)
        # At t=0, initial state |00⟩ is pure
        assert validate_partial_trace(rho_full, rho_a, rho_b)
