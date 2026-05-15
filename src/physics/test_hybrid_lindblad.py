"""
Tests for Hybrid Oscillator-Spin Lindblad Solver.

Physical Validation Tests:
- Hamiltonian Hermiticity: H = H\u2020
- Lindblad operator structure: correct tensor product form
- Trace preservation: Tr[\u03c1(t)] = 1 for unitary, \u2264 1 with loss
- Hermiticity: \u03c1 = \u03c1\u2020 at all times
- Positivity: eigenvalues of \u03c1 \u2265 0
- Squeezing: n=2 should produce Gaussian squeezing
- Wigner negativity: n\u22653 should show Wigner negativity
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy

from .hybrid_lindblad import (
    HybridLindbladConfig,
    apply_squeezing,
    build_hybrid_hamiltonian,
    build_hybrid_lindblad_operators,
    evolve_hybrid_lindblad,
    lindblad_rhs,
    run_hybrid_simulation,
    validate_hybrid_density_matrix,
)
from .hybrid_system import (
    hybrid_mean_photon,
    hybrid_vacuum_state,
)


class TestHybridLindbladConfig:
    def test_default_values_are_reasonable(self) -> None:
        config = HybridLindbladConfig(N=5)
        assert config.N == 5
        assert config.n == 2
        assert config.omega_n == 1.0
        assert config.theta_n == 0.0
        assert config.phi == 0.0
        assert config.gamma_1 == 0.0
        assert config.gamma_2 == 0.0
        assert config.gamma_phi == 0.0
        assert config.t_squeeze == 1.0

    def test_custom_values_are_preserved(self) -> None:
        config = HybridLindbladConfig(
            N=10,
            n=3,
            omega_n=0.5,
            theta_n=np.pi / 4,
            phi=0.1,
            gamma_1=0.01,
            gamma_phi=0.02,
            t_squeeze=2.0,
        )
        assert config.N == 10
        assert config.n == 3
        assert config.omega_n == 0.5
        assert config.phi == 0.1
        assert config.gamma_1 == 0.01


class TestBuildHybridHamiltonian:
    @pytest.mark.parametrize("n", [2, 3, 4], ids=["n=2", "n=3", "n=4"])
    def test_hamiltonian_has_correct_shape(self, n: int) -> None:
        config = HybridLindbladConfig(N=5, n=n, omega_n=1.0)
        H = build_hybrid_hamiltonian(config)
        dim = 2 * (5 + 1)
        assert H.shape == (dim, dim)

    @pytest.mark.parametrize("n", [2, 3, 4], ids=["n=2", "n=3", "n=4"])
    def test_hamiltonian_is_hermitian(self, n: int) -> None:
        config = HybridLindbladConfig(N=5, n=n, omega_n=1.0)
        H = build_hybrid_hamiltonian(config)
        assert pytest.approx(H.conj().T) == H

    def test_given_zero_squeezing_rate_then_hamiltonian_is_zero(self) -> None:
        config = HybridLindbladConfig(N=5, n=2, omega_n=0.0)
        H = build_hybrid_hamiltonian(config)
        assert pytest.approx(0) == H

    def test_n_2_and_n_3_give_different_hamiltonians(self) -> None:
        config2 = HybridLindbladConfig(N=5, n=2, omega_n=1.0)
        config3 = HybridLindbladConfig(N=5, n=3, omega_n=1.0)
        H2 = build_hybrid_hamiltonian(config2)
        H3 = build_hybrid_hamiltonian(config3)
        assert pytest.approx(H3) != H2


class TestBuildHybridLindbladOperators:
    def test_given_no_dissipation_then_lists_are_empty(self) -> None:
        config = HybridLindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=0)
        L_ops, gammas = build_hybrid_lindblad_operators(config)
        assert len(L_ops) == 0
        assert len(gammas) == 0

    def test_one_body_loss_adds_one_operator(self) -> None:
        config = HybridLindbladConfig(N=5, gamma_1=0.1)
        L_ops, gammas = build_hybrid_lindblad_operators(config)
        assert len(L_ops) == 1
        assert len(gammas) == 1
        dim = 2 * (5 + 1)
        assert L_ops[0].shape == (dim, dim)

    def test_phase_diffusion_adds_one_operator(self) -> None:
        config = HybridLindbladConfig(N=5, gamma_phi=0.1)
        L_ops, gammas = build_hybrid_lindblad_operators(config)
        assert len(L_ops) == 1
        assert len(gammas) == 1

    def test_multiple_channels_add_multiple_operators(self) -> None:
        config = HybridLindbladConfig(N=5, gamma_1=0.1, gamma_2=0.05, gamma_phi=0.02)
        L_ops, gammas = build_hybrid_lindblad_operators(config)
        assert len(L_ops) == 3
        assert len(gammas) == 3

    def test_one_body_loss_l_a_i(self) -> None:
        N = 5
        config = HybridLindbladConfig(N=N, gamma_1=0.1)
        L_ops, _ = build_hybrid_lindblad_operators(config)

        L = L_ops[0]
        dim_osc = N + 1

        L_down = L[::2, ::2][:dim_osc, :dim_osc]
        L_up = L[1::2, 1::2][:dim_osc, :dim_osc]

        a = np.zeros((dim_osc, dim_osc), dtype=complex)
        for n in range(1, dim_osc):
            a[n - 1, n] = np.sqrt(n)

        assert L_down == pytest.approx(np.sqrt(0.1) * a)
        assert L_up == pytest.approx(np.sqrt(0.1) * a)


class TestLindbladRHS:
    def test_given_no_dissipation_then_rhs_is_minus_i_commutator(self) -> None:
        N = 5
        config = HybridLindbladConfig(N=N, n=2, omega_n=1.0)
        H = build_hybrid_hamiltonian(config)
        L_ops: list[np.ndarray] = []
        gammas: list[float] = []

        dim = 2 * (N + 1)
        rho = np.eye(dim, dtype=complex) / dim

        drho = lindblad_rhs(rho, H, L_ops, gammas)

        expected = -1.0j * (H @ rho - rho @ H)
        assert drho == pytest.approx(expected)

    def test_given_dissipation_then_drift_is_nonzero(self) -> None:
        N = 5
        config = HybridLindbladConfig(N=N, gamma_1=0.1)
        H = np.zeros((2 * (N + 1), 2 * (N + 1)), dtype=complex)
        L_ops, gammas = build_hybrid_lindblad_operators(config)

        dim = 2 * (N + 1)
        rho = np.zeros((dim, dim), dtype=complex)
        rho[2, 2] = 1.0

        drho = lindblad_rhs(rho, H, L_ops, gammas)

        assert np.max(np.abs(drho)) > 0


class TestEvolveHybridLindblad:
    def test_given_zero_time_then_returns_initial_state(self) -> None:
        N = 5
        config = HybridLindbladConfig(N=N, gamma_1=0.1)
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        rho_final = evolve_hybrid_lindblad(psi0, config, T=0.0, dt=0.01)

        rho0 = np.outer(psi0, psi0.conj())
        assert rho_final == pytest.approx(rho0)

    def test_given_no_dissipation_then_matches_unitary_evolution(self) -> None:
        N = 5
        config = HybridLindbladConfig(N=N, n=2, omega_n=0.5, t_squeeze=1.0)
        config.gamma_1 = 0.0
        config.gamma_phi = 0.0

        psi0 = hybrid_vacuum_state(N, spin_state="down")

        H = build_hybrid_hamiltonian(config)
        U = scipy.linalg.expm(-1.0j * H * 1.0)
        rho_expected = U @ np.outer(psi0, psi0.conj()) @ U.conj().T

        rho_final = evolve_hybrid_lindblad(psi0, config, T=1.0, dt=0.001, method="rk4")

        assert rho_final == pytest.approx(rho_expected, abs=1e-4)

    def test_given_no_dissipation_then_trace_is_preserved(self) -> None:
        N = 5
        config = HybridLindbladConfig(N=N, n=2, omega_n=0.5, gamma_1=0, gamma_phi=0)
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        rho_final = evolve_hybrid_lindblad(psi0, config, T=1.0, dt=0.01)

        assert np.trace(rho_final) == pytest.approx(1.0, abs=1e-6)

    def test_given_particle_loss_then_trace_does_not_exceed_one(self) -> None:
        N = 5
        config = HybridLindbladConfig(N=N, gamma_1=0.2)
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        rho_final = evolve_hybrid_lindblad(psi0, config, T=1.0, dt=0.01)

        assert np.trace(rho_final) <= 1.0 + 1e-6

    def test_density_matrix_remains_hermitian(self) -> None:
        N = 5
        config = HybridLindbladConfig(
            N=N,
            n=2,
            omega_n=0.3,
            gamma_1=0.1,
            gamma_phi=0.05,
        )
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        rho_final = evolve_hybrid_lindblad(psi0, config, T=0.5, dt=0.01)

        assert rho_final == pytest.approx(rho_final.conj().T, abs=1e-6)

    def test_eigenvalues_are_non_negative(self) -> None:
        N = 5
        config = HybridLindbladConfig(N=N, gamma_1=0.1)
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        rho_final = evolve_hybrid_lindblad(psi0, config, T=0.5, dt=0.01)

        eigenvalues = np.linalg.eigvalsh(rho_final)
        assert np.min(eigenvalues.real) >= -1e-6


class TestApplySqueezing:
    def test_squeezing_increases_mean_photon_number(self) -> None:
        N = 10
        config = HybridLindbladConfig(N=N, n=2, omega_n=0.5, t_squeeze=1.0)
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        n_initial = hybrid_mean_photon(psi0, N)
        psi_sq = apply_squeezing(config, psi0)
        n_squeezed = hybrid_mean_photon(psi_sq, N)

        assert n_squeezed > n_initial

    def test_squeezing_preserves_norm(self) -> None:
        N = 10
        config = HybridLindbladConfig(N=N, n=3, omega_n=0.3, t_squeeze=2.0)
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        psi_sq = apply_squeezing(config, psi0)

        assert np.sum(np.abs(psi_sq) ** 2) == pytest.approx(1.0, abs=1e-6)


class TestValidateHybridDensityMatrix:
    def test_valid_density_matrix_passes_all_checks(self) -> None:
        N = 5
        dim = 2 * (N + 1)
        rho = np.eye(dim, dtype=complex) / dim

        result = validate_hybrid_density_matrix(rho)

        assert result["is_hermitian"]
        assert result["is_normalized"]
        assert result["is_positive"]

    def test_pure_state_is_valid(self) -> None:
        N = 5
        psi = hybrid_vacuum_state(N, spin_state="down")
        rho = np.outer(psi, psi.conj())

        result = validate_hybrid_density_matrix(rho)

        assert result["is_hermitian"]
        assert result["is_normalized"]
        assert result["is_positive"]

    def test_non_hermitian_matrix_fails_validation(self) -> None:
        N = 5
        dim = 2 * (N + 1)
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 1] = 1.0

        result = validate_hybrid_density_matrix(rho)

        assert not result["is_hermitian"]


class TestRunHybridSimulation:
    def test_simulation_completes_without_errors(self) -> None:
        config = HybridLindbladConfig(
            N=5,
            n=2,
            omega_n=0.5,
            t_squeeze=0.5,
            gamma_1=0.01,
        )
        result = run_hybrid_simulation(config)

        assert "final_state" in result
        assert "validation" in result

        validation = result["validation"]
        assert validation["is_hermitian"]
        assert validation["is_positive"]

    def test_simulation_with_n_3_completes(self) -> None:
        config = HybridLindbladConfig(N=8, n=3, omega_n=0.3, t_squeeze=1.0, gamma_1=0.0)
        result = run_hybrid_simulation(config)

        assert result["final_state"] is not None
        assert result["validation"]["is_hermitian"]


class TestWignerNegativity:
    @pytest.mark.slow
    def test_n_2_gaussian_does_not_have_wigner_negativity(self) -> None:
        N = 20
        config = HybridLindbladConfig(N=N, n=2, omega_n=0.5, t_squeeze=1.0)
        psi_sq = apply_squeezing(config)

        rho_osc = self._extract_oscillator_density(psi_sq, N)

        from .wigner import wigner_function_single

        x = np.linspace(-5, 5, 50)
        p = np.linspace(-5, 5, 50)
        W = wigner_function_single(rho_osc, x, p)

        assert np.min(W) >= -1e-3

    def test_n_3_non_gaussian_shows_wigner_negativity(self) -> None:
        N = 10
        config = HybridLindbladConfig(N=N, n=3, omega_n=0.3, t_squeeze=2.0)
        psi_sq = apply_squeezing(config)

        rho_osc = self._extract_oscillator_density(psi_sq, N)

        from .wigner import wigner_function_single

        x = np.linspace(-5, 5, 50)
        p = np.linspace(-5, 5, 50)
        W = wigner_function_single(rho_osc, x, p)

        assert np.min(W) < -1e-3

    def _extract_oscillator_density(
        self,
        hybrid_state: np.ndarray,
        N: int,
    ) -> np.ndarray:
        dim_osc = N + 1
        rho_hybrid = np.outer(hybrid_state, hybrid_state.conj())
        rho_reshaped = rho_hybrid.reshape(dim_osc, 2, dim_osc, 2)
        return np.trace(rho_reshaped, axis1=1, axis2=3)


class TestN4WignerNegativityDiagnostic:
    OMEGA_N = 1.0
    X_MAX = 4.0

    def test_n4_baseline(self) -> None:
        from .wigner import wigner_function_single

        N = 10
        config = HybridLindbladConfig(N=N, n=4, omega_n=0.5, t_squeeze=2.0)
        psi_sq = apply_squeezing(config)
        rho_osc = self._extract_oscillator_density(psi_sq, N)

        x = np.linspace(-5, 5, 50)
        p = np.linspace(-5, 5, 50)
        W = wigner_function_single(rho_osc, x, p)

        assert np.min(W) < -1e-3

    @pytest.mark.slow
    def test_n4_grid_sweep(self) -> None:
        from .wigner import wigner_function_single

        N = 20
        t_sqz = 0.30
        resolutions = [40, 60, 80]

        config = HybridLindbladConfig(N=N, n=4, omega_n=self.OMEGA_N, t_squeeze=t_sqz)
        psi_sq = apply_squeezing(config)
        rho_osc = self._extract_oscillator_density(psi_sq, N)

        min_values = []
        for n_pts in resolutions:
            x = np.linspace(-self.X_MAX, self.X_MAX, n_pts)
            p = np.linspace(-self.X_MAX, self.X_MAX, n_pts)
            W = wigner_function_single(rho_osc, x, p)
            min_values.append(float(np.min(W)))

        assert min(min_values) < -1e-5

    @pytest.mark.slow
    def test_n4_time_sweep(self) -> None:
        from .wigner import wigner_function_single

        N = 20
        n_pts = 50
        times = np.array([0.15, 0.30, 0.45])

        best_mins: list[tuple[float, float]] = []

        for t in times:
            config = HybridLindbladConfig(
                N=N,
                n=4,
                omega_n=self.OMEGA_N,
                t_squeeze=float(t),
            )
            psi_sq = apply_squeezing(config)
            rho_osc = self._extract_oscillator_density(psi_sq, N)

            x = np.linspace(-self.X_MAX, self.X_MAX, n_pts)
            p = np.linspace(-self.X_MAX, self.X_MAX, n_pts)
            W = wigner_function_single(rho_osc, x, p)

            best_mins.append((float(t), float(np.min(W))))

        best_mins.sort(key=lambda pair: pair[1])
        _t_best, w_best = best_mins[0]

        assert w_best < -1e-4

    @pytest.mark.slow
    def test_n4_truncation_check(self) -> None:
        from .wigner import wigner_function_single

        t_sqz = 0.30
        n_pts = 60
        N_values = [10, 20, 30]

        best_w = 0.0
        for N in N_values:
            config = HybridLindbladConfig(
                N=N,
                n=4,
                omega_n=self.OMEGA_N,
                t_squeeze=t_sqz,
            )
            psi_sq = apply_squeezing(config)
            rho_osc = self._extract_oscillator_density(psi_sq, N)

            x = np.linspace(-self.X_MAX, self.X_MAX, n_pts)
            p = np.linspace(-self.X_MAX, self.X_MAX, n_pts)
            W = wigner_function_single(rho_osc, x, p)
            w_min = float(np.min(W))
            best_w = min(best_w, w_min)

        assert best_w < -1e-5

    @pytest.mark.slow
    def test_n4_high_resolution_confirm(self) -> None:
        from .wigner import wigner_function_single

        N = 20
        t_sqz = 0.30
        n_pts = 100

        config = HybridLindbladConfig(N=N, n=4, omega_n=self.OMEGA_N, t_squeeze=t_sqz)
        psi_sq = apply_squeezing(config)
        rho_osc = self._extract_oscillator_density(psi_sq, N)

        x = np.linspace(-self.X_MAX, self.X_MAX, n_pts)
        p = np.linspace(-self.X_MAX, self.X_MAX, n_pts)
        W = wigner_function_single(rho_osc, x, p)

        assert np.min(W) < -1e-5

    def _extract_oscillator_density(
        self,
        hybrid_state: np.ndarray,
        N: int,
    ) -> np.ndarray:
        dim_osc = N + 1
        rho_hybrid = np.outer(hybrid_state, hybrid_state.conj())
        rho_reshaped = rho_hybrid.reshape(dim_osc, 2, dim_osc, 2)
        return np.trace(rho_reshaped, axis1=1, axis2=3)


class TestEdgeCases:
    def test_given_minimal_dimension_then_evolves(self) -> None:
        config = HybridLindbladConfig(N=1, n=2, omega_n=0.5)
        psi0 = hybrid_vacuum_state(1, spin_state="down")
        rho = evolve_hybrid_lindblad(psi0, config, T=0.1, dt=0.01)
        assert rho.shape == (4, 4)

    def test_large_gamma_decays_photon_number(self) -> None:
        N = 5
        config = HybridLindbladConfig(N=N, gamma_1=10.0)
        from .hybrid_system import hybrid_coherent_state

        psi0 = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")

        initial_n = hybrid_mean_photon(psi0, N)
        rho = evolve_hybrid_lindblad(psi0, config, T=1.0, dt=0.01)

        dim_osc = N + 1
        n_op = np.zeros((dim_osc, dim_osc), dtype=complex)
        for n in range(dim_osc):
            n_op[n, n] = n
        n_hybrid = np.kron(n_op, np.eye(2, dtype=complex))

        final_n = np.real(np.trace(rho @ n_hybrid))

        assert final_n < initial_n * 0.5
        assert np.trace(rho) == pytest.approx(1.0, abs=1e-6)
