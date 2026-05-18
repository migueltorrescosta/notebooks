"""
Tests for Pseudomode Non-Markovian System.

Covers configuration validation, operator construction, Hamiltonian
Hermiticity, Lindblad operators, state preparation, ancilla entanglement,
partial trace, Lindblad evolution, QFI computation, metrology protocol,
density validation, pseudomode occupancy, and QFI preservation ratios.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
import scipy
import scipy.linalg
import scipy.special

from .pseudomode_system import (
    PseudomodeConfig,
    apply_ancilla_entanglement,
    build_pseudomode_hamiltonian,
    build_pseudomode_lindblad_operators,
    check_pseudomode_occupancy,
    compute_qfi_with_ancilla,
    compute_qfi_without_ancilla,
    create_pseudomode_operators,
    evolve_pseudomode,
    pseudomode_initial_state,
    pseudomode_number_operator,
    qfi_preservation_ratio,
    run_metrology_protocol,
    trace_out_pseudomode,
    trace_out_spin,
    trace_out_spin_and_pseudomode,
    tripartite_operator,
    validate_pseudomode_density,
)


def _make_dim_total(N: int, K: int) -> int:
    return 2 * (N + 1) * (K + 1)


# Configuration


class TestPseudomodeConfig:
    """Configuration dataclass defaults and validation."""

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("N", 5),
            ("K", 3),
            ("alpha", 1.0),
            ("g_sa", 1.0),
            ("tau", 0.1),
            ("g_sp", 0.5),
            ("omega_0", 0.0),
            ("lam", 1.0),
            ("T", 2.0),
            ("dt", 0.01),
        ],
    )
    def test_default_values(self, field: str, expected: object) -> None:
        assert getattr(PseudomodeConfig(N=5, K=3), field) == expected

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("N", 10),
            ("K", 5),
            ("alpha", 2.0),
            ("g_sa", 0.5),
            ("tau", 0.2),
            ("g_sp", 1.0),
            ("omega_0", 0.5),
            ("lam", 0.1),
            ("T", 5.0),
            ("dt", 0.005),
        ],
    )
    def test_custom_values(self, field: str, expected: object) -> None:
        cfg = PseudomodeConfig(
            N=10,
            K=5,
            alpha=2.0,
            g_sa=0.5,
            tau=0.2,
            g_sp=1.0,
            omega_0=0.5,
            lam=0.1,
            T=5.0,
            dt=0.005,
        )
        assert getattr(cfg, field) == expected

    def test_invalid_N_neg(self) -> None:
        with pytest.raises(ValueError):
            PseudomodeConfig(N=-1, K=3)

    def test_invalid_K_neg(self) -> None:
        with pytest.raises(ValueError):
            PseudomodeConfig(N=5, K=-1)

    def test_invalid_dt_neg(self) -> None:
        with pytest.raises(ValueError):
            PseudomodeConfig(N=5, K=3, dt=-0.1)

    def test_invalid_dt_zero(self) -> None:
        with pytest.raises(ValueError):
            PseudomodeConfig(N=5, K=3, dt=0.0)

    def test_invalid_T_neg(self) -> None:
        with pytest.raises(ValueError):
            PseudomodeConfig(N=5, K=3, T=-1.0)

    def test_invalid_tau_neg(self) -> None:
        with pytest.raises(ValueError):
            PseudomodeConfig(N=5, K=3, tau=-0.1)


# Pseudomode Operators


class TestPseudomodeOperators:
    """Ladder operator and number operator construction."""

    @pytest.mark.parametrize("K", [0, 1, 3, 10])
    def test_shape(self, K: int) -> None:
        b, bd = create_pseudomode_operators(K)
        assert b.shape == (K + 1, K + 1)
        assert bd.shape == (K + 1, K + 1)

    def test_commutation(self) -> None:
        K = 10
        b, bd = create_pseudomode_operators(K)
        comm = b @ bd - bd @ b
        expected = np.eye(K + 1, dtype=complex)
        expected[K, K] = -K  # truncation artifact
        assert comm == pytest.approx(expected, abs=1e-10)

    def test_annihilation_action(self) -> None:
        K = 5
        b, _ = create_pseudomode_operators(K)
        for k in range(K + 1):
            ket = np.zeros(K + 1, dtype=complex)
            ket[k] = 1.0
            result = b @ ket
            if k > 0:
                assert result[k - 1] == pytest.approx(np.sqrt(k))
            else:
                assert result == pytest.approx(0)

    def test_negative_K_raises(self) -> None:
        with pytest.raises(ValueError):
            create_pseudomode_operators(-1)

    # --- Number operator ---
    @pytest.mark.parametrize("K", [0, 1, 5])
    def test_number_shape(self, K: int) -> None:
        assert pseudomode_number_operator(K).shape == (K + 1, K + 1)

    def test_number_diagonal_values(self) -> None:
        K = 5
        n = pseudomode_number_operator(K)
        for k in range(K + 1):
            assert n[k, k] == pytest.approx(k)

    def test_number_consistency(self) -> None:
        K = 5
        b, bd = create_pseudomode_operators(K)
        assert bd @ b == pytest.approx(pseudomode_number_operator(K))


# Tripartite Operator


class TestTripartiteOperator:
    """Tripartite Kronecker product construction."""

    @pytest.fixture
    def cfg(self) -> tuple[int, int]:
        return (5, 3)

    def test_correct_dimensions(self) -> None:
        N, K = 5, 3
        op = tripartite_operator(np.eye(N + 1), np.eye(2), np.eye(K + 1), N, K)
        assert op.shape == (_make_dim_total(N, K), _make_dim_total(N, K))

    def test_identity_product(self) -> None:
        N, K = 5, 3
        op = tripartite_operator(np.eye(N + 1), np.eye(2), np.eye(K + 1), N, K)
        assert op == pytest.approx(np.eye(_make_dim_total(N, K)))

    def test_zero_operator(self) -> None:
        N, K = 5, 3
        op = tripartite_operator(
            np.zeros((N + 1, N + 1)), np.eye(2), np.eye(K + 1), N, K
        )
        assert op == pytest.approx(0)

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(AssertionError):
            tripartite_operator(np.eye(7), np.eye(2), np.eye(4), 5, 3)

    def test_hermitian_product(self) -> None:
        N, K = 5, 3
        A = np.random.randn(N + 1, N + 1) + 1j * np.random.randn(N + 1, N + 1)
        A = A + A.conj().T
        op = tripartite_operator(
            A,
            np.array([[1, 0], [0, -1]], dtype=complex),
            pseudomode_number_operator(K),
            N,
            K,
        )
        assert op == pytest.approx(op.conj().T, abs=1e-10)


# Hamiltonian


class TestBuildPseudomodeHamiltonian:
    """Hamiltonian construction: shape, Hermiticity, limits."""

    @pytest.fixture
    def config(self) -> PseudomodeConfig:
        return PseudomodeConfig(N=5, K=3)

    def test_shape(self, config: PseudomodeConfig) -> None:
        H = build_pseudomode_hamiltonian(config)
        assert H.shape == (
            _make_dim_total(config.N, config.K),
            _make_dim_total(config.N, config.K),
        )

    def test_hermiticity(self, config: PseudomodeConfig) -> None:
        H = build_pseudomode_hamiltonian(config)
        assert pytest.approx(H.conj().T, abs=1e-10) == H

    def test_hermiticity_no_sa(self, config: PseudomodeConfig) -> None:
        H = build_pseudomode_hamiltonian(config, include_sa=False)
        assert pytest.approx(H.conj().T, abs=1e-10) == H

    def test_zero_coupling(self) -> None:
        H = build_pseudomode_hamiltonian(
            PseudomodeConfig(N=5, K=3, g_sa=0.0, g_sp=0.0, omega_0=0.0)
        )
        assert pytest.approx(0, abs=1e-10) == H

    def test_include_vs_exclude_sa_differ(self) -> None:
        cfg = PseudomodeConfig(N=5, K=3, g_sa=1.0)
        assert build_pseudomode_hamiltonian(cfg, include_sa=True) != pytest.approx(
            build_pseudomode_hamiltonian(cfg, include_sa=False)
        )


# Lindblad Operators


class TestBuildPseudomodeLindbladOperators:
    """Lindblad operator construction."""

    @pytest.fixture
    def config(self) -> PseudomodeConfig:
        return PseudomodeConfig(N=5, K=3)

    def test_zero_lam_empty(self, config: PseudomodeConfig) -> None:
        cfg = PseudomodeConfig(N=config.N, K=config.K, lam=0.0)
        L, g = build_pseudomode_lindblad_operators(cfg)
        assert len(L) == 0
        assert len(g) == 0

    def test_negative_lam_empty(self, config: PseudomodeConfig) -> None:
        cfg = PseudomodeConfig(N=config.N, K=config.K, lam=-0.1)
        L, _ = build_pseudomode_lindblad_operators(cfg)
        assert len(L) == 0

    def test_positive_lam_adds_one(self, config: PseudomodeConfig) -> None:
        cfg = PseudomodeConfig(N=config.N, K=config.K, lam=1.0)
        L, g = build_pseudomode_lindblad_operators(cfg)
        assert len(L) == 1
        assert len(g) == 1
        assert g[0] == 1.0

    def test_correct_shape(self, config: PseudomodeConfig) -> None:
        cfg = PseudomodeConfig(N=config.N, K=config.K, lam=1.0)
        L, _ = build_pseudomode_lindblad_operators(cfg)
        assert L[0].shape == (
            _make_dim_total(cfg.N, cfg.K),
            _make_dim_total(cfg.N, cfg.K),
        )

    def test_operator_structure(self) -> None:
        N, K, lam = 3, 2, 4.0
        cfg = PseudomodeConfig(N=N, K=K, lam=lam)
        b, _ = create_pseudomode_operators(K)
        expected = np.sqrt(lam) * np.kron(np.kron(np.eye(N + 1), np.eye(2)), b)
        L, _ = build_pseudomode_lindblad_operators(cfg)
        assert L[0] == pytest.approx(expected, abs=1e-10)


# State Preparation


class TestPseudomodeInitialState:
    """Initial state preparation: shape, norm, spin/pseudomode content."""

    @pytest.fixture
    def config(self) -> PseudomodeConfig:
        return PseudomodeConfig(N=5, K=3)

    def test_shape(self, config: PseudomodeConfig) -> None:
        state = pseudomode_initial_state(config)
        assert state.shape == (_make_dim_total(config.N, config.K),)

    @pytest.mark.parametrize(
        ("alpha", "N"), [(0.0, 5), (0.5, 10), (1.0, 10), (2.0, 25)]
    )
    def test_normalization(self, alpha: float, N: int) -> None:
        state = pseudomode_initial_state(PseudomodeConfig(N=N, K=3, alpha=alpha))
        assert np.sum(np.abs(state) ** 2) == pytest.approx(1.0, abs=1e-3)

    def test_only_spin_down_s_0_components_are_non_zero(
        self, config: PseudomodeConfig
    ) -> None:
        state = pseudomode_initial_state(config)
        dim_pm = config.K + 1
        for n in range(config.N + 1):
            for k in range(dim_pm):
                assert np.abs(state[(n * 2 + 1) * dim_pm + k]) == 0

    def test_only_k_0_pseudomode_components_are_non_zero(
        self, config: PseudomodeConfig
    ) -> None:
        state = pseudomode_initial_state(config)
        dim_pm = config.K + 1
        for n in range(config.N + 1):
            for s in range(2):
                for k in range(1, dim_pm):
                    assert np.abs(state[(n * 2 + s) * dim_pm + k]) == 0

    def test_coherent_amplitudes(self) -> None:
        config = PseudomodeConfig(N=10, K=3, alpha=1.0)
        state = pseudomode_initial_state(config)
        dim_pm = config.K + 1
        for n in range(config.N + 1):
            expected = (
                config.alpha**n
                / np.sqrt(scipy.special.factorial(n))
                * np.exp(-(abs(config.alpha) ** 2) / 2)
            )
            assert state[n * 2 * dim_pm] == pytest.approx(expected, abs=1e-10)


# Ancilla Entanglement


class TestApplyAncillaEntanglement:
    """Ancilla entanglement unitary: norm, identity limits, spin population."""

    @pytest.fixture
    def config(self) -> PseudomodeConfig:
        return PseudomodeConfig(N=5, K=3, g_sa=1.0, tau=0.2)

    def test_norm_preservation(self, config: PseudomodeConfig) -> None:
        state = pseudomode_initial_state(config)
        entangled = apply_ancilla_entanglement(state, config)
        assert np.sum(np.abs(state) ** 2) == pytest.approx(
            np.sum(np.abs(entangled) ** 2), abs=1e-10
        )

    @pytest.mark.parametrize(("g_sa", "tau"), [(1.0, 0.0), (0.0, 0.2)])
    def test_zero_tau_or_zero_coupling_returns_the_original_state(
        self, g_sa: float, tau: float
    ) -> None:
        config = PseudomodeConfig(N=5, K=3, g_sa=g_sa, tau=tau)
        state = pseudomode_initial_state(config)
        assert apply_ancilla_entanglement(state, config) == pytest.approx(
            state, abs=1e-10
        )

    def test_entanglement_creates_spin_up_population_from_spin_down(
        self, config: PseudomodeConfig
    ) -> None:
        config = PseudomodeConfig(N=5, K=3, g_sa=2.0, tau=0.5)
        state = pseudomode_initial_state(config)
        entangled = apply_ancilla_entanglement(state, config)
        dim_pm = config.K + 1
        has_spin_up = any(
            np.abs(entangled[(n * 2 + 1) * dim_pm + k]) > 1e-6
            for n in range(config.N + 1)
            for k in range(dim_pm)
        )
        assert has_spin_up

    def test_wrong_dimension_raises(self) -> None:
        with pytest.raises(AssertionError):
            apply_ancilla_entanglement(
                np.zeros(10, dtype=complex), PseudomodeConfig(N=5, K=3)
            )


# Partial Trace


class TestPartialTrace:
    """Partial trace operations: dimension reduction, trace conservation."""

    def _make_test_density(self, N: int, K: int) -> np.ndarray:
        dim = _make_dim_total(N, K)
        psi = np.random.randn(dim) + 1j * np.random.randn(dim)
        psi = psi / np.linalg.norm(psi)
        return np.outer(psi, psi.conj())

    def test_trace_out_pm_shape(self) -> None:
        N, K = 5, 3
        rho = self._make_test_density(N, K)
        assert trace_out_pseudomode(rho, N, K).shape == (2 * (N + 1), 2 * (N + 1))

    def test_trace_out_spin_shape(self) -> None:
        N, K = 5, 3
        rho = self._make_test_density(N, K)
        assert trace_out_spin(rho, N, K).shape == ((N + 1) * (K + 1), (N + 1) * (K + 1))

    def test_trace_out_both_shape(self) -> None:
        N, K = 5, 3
        rho = self._make_test_density(N, K)
        assert trace_out_spin_and_pseudomode(rho, N, K).shape == (N + 1, N + 1)

    def test_sequential_trace_equals_combined_trace(self) -> None:
        N, K = 5, 3
        rho = self._make_test_density(N, K)
        rho_op = trace_out_spin(rho, N, K)
        rho_osc_a = np.trace(
            rho_op.reshape(N + 1, K + 1, N + 1, K + 1), axis1=1, axis2=3
        )
        assert rho_osc_a == pytest.approx(
            trace_out_spin_and_pseudomode(rho, N, K), abs=1e-10
        )

    def test_trace_conservation(self) -> None:
        N, K = 5, 3
        rho = self._make_test_density(N, K)
        original_trace = np.trace(rho)
        rho_reduced = trace_out_pseudomode(rho, N, K)
        assert np.trace(rho_reduced) == pytest.approx(original_trace, abs=1e-10)

    def test_hermiticity_preserved(self) -> None:
        N, K = 5, 3
        rho = self._make_test_density(N, K)
        rho_reduced = trace_out_pseudomode(rho, N, K)
        assert rho_reduced == pytest.approx(rho_reduced.conj().T, abs=1e-10)


# Lindblad Evolution


class TestEvolvePseudomode:
    """Lindblad evolution: zero time, unitary limit, trace preservation, solvers."""

    @pytest.fixture
    def config(self) -> PseudomodeConfig:
        return PseudomodeConfig(N=5, K=3)

    def test_zero_time(self, config: PseudomodeConfig) -> None:
        cfg = PseudomodeConfig(N=config.N, K=config.K, T=0.0, lam=0.5)
        psi = pseudomode_initial_state(cfg)
        rho = evolve_pseudomode(psi, cfg, method="rk4")
        assert rho == pytest.approx(np.outer(psi, psi.conj()), abs=1e-6)

    def test_unitary_evolution_no_lam(self) -> None:
        cfg = PseudomodeConfig(N=5, K=3, lam=0.0, g_sp=0.5, T=0.5)
        psi = pseudomode_initial_state(cfg)
        H = build_pseudomode_hamiltonian(cfg, include_sa=False)
        U = scipy.linalg.expm(-1.0j * H * cfg.T)
        expected = U @ np.outer(psi, psi.conj()) @ U.conj().T
        assert evolve_pseudomode(psi, cfg, method="rk4") == pytest.approx(
            expected, abs=1e-4
        )

    def test_trace_preservation(self) -> None:
        cfg = PseudomodeConfig(N=5, K=3, lam=0.5, g_sp=0.3, T=0.5)
        psi = pseudomode_initial_state(cfg)
        assert np.trace(evolve_pseudomode(psi, cfg, method="rk4")) == pytest.approx(
            1.0, abs=1e-6
        )

    def test_hermiticity(self) -> None:
        cfg = PseudomodeConfig(N=5, K=3, lam=0.5, g_sp=0.3, T=0.5)
        psi = pseudomode_initial_state(cfg)
        rho = evolve_pseudomode(psi, cfg, method="rk4")
        assert rho == pytest.approx(rho.conj().T, abs=1e-6)

    def test_positivity(self) -> None:
        cfg = PseudomodeConfig(N=5, K=3, lam=0.5, g_sp=0.3, T=0.5)
        psi = pseudomode_initial_state(cfg)
        rho = evolve_pseudomode(psi, cfg, method="rk4")
        assert np.min(np.linalg.eigvalsh(rho).real) >= -1e-6

    def test_scipy_method_matches_rk4(self) -> None:
        cfg = PseudomodeConfig(N=4, K=2, lam=0.3, g_sp=0.2, T=0.2)
        psi = pseudomode_initial_state(cfg)
        assert evolve_pseudomode(psi, cfg, method="rk4") == pytest.approx(
            evolve_pseudomode(psi, cfg, method="scipy"), abs=1e-4
        )

    def test_unknown_method_raises(self, config: PseudomodeConfig) -> None:
        with pytest.raises(ValueError):
            evolve_pseudomode(
                pseudomode_initial_state(config), config, method="invalid"
            )

    # --- Density validation ---
    def test_valid_maximally_mixed(self) -> None:
        dim = _make_dim_total(5, 3)
        rho = np.eye(dim, dtype=complex) / dim
        v = validate_pseudomode_density(rho)
        assert v["is_hermitian"] and v["is_normalized"] and v["is_positive"]

    def test_valid_pure_state(self) -> None:
        cfg = PseudomodeConfig(N=20, K=3, alpha=1.0)
        rho = np.outer(
            pseudomode_initial_state(cfg), pseudomode_initial_state(cfg).conj()
        )
        v = validate_pseudomode_density(rho, tolerance=1e-4)
        assert v["is_hermitian"] and v["is_normalized"] and v["is_positive"]

    def test_non_hermitian_fails(self) -> None:
        rho = np.zeros((10, 10), dtype=complex)
        rho[0, 1] = 1.0
        assert not validate_pseudomode_density(rho)["is_hermitian"]

    def test_not_normalized_fails(self) -> None:
        rho = np.eye(10, dtype=complex) / 10 + 0.1 * np.eye(10, dtype=complex)
        assert not validate_pseudomode_density(rho)["is_normalized"]


# QFI Computation


class TestQFIComputation:
    """QFI for the tripartite system, with and without ancilla."""

    def test_initial_nonzero(self) -> None:
        cfg = PseudomodeConfig(N=10, K=3, alpha=1.0, g_sa=0.0, g_sp=0.0, lam=0.0)
        rho = np.outer(
            pseudomode_initial_state(cfg), pseudomode_initial_state(cfg).conj()
        )
        assert compute_qfi_with_ancilla(rho, cfg.N, cfg.K) > 0

    def test_for_qfi_4_generator_a_a_i(self) -> None:
        cfg = PseudomodeConfig(N=30, K=3, alpha=2.0, g_sa=0.0, g_sp=0.0, lam=0.0)
        rho = np.outer(
            pseudomode_initial_state(cfg), pseudomode_initial_state(cfg).conj()
        )
        assert compute_qfi_with_ancilla(rho, cfg.N, cfg.K) == pytest.approx(
            4.0 * cfg.alpha**2, rel=1e-3
        )

    def test_vacuum_has_zero_qfi(self) -> None:
        cfg = PseudomodeConfig(N=5, K=3, alpha=0.0, g_sa=0.0, g_sp=0.0, lam=0.0)
        rho = np.outer(
            pseudomode_initial_state(cfg), pseudomode_initial_state(cfg).conj()
        )
        assert compute_qfi_with_ancilla(rho, cfg.N, cfg.K) == pytest.approx(
            0.0, abs=1e-10
        )

    def test_zero_gsa_matches_with_without(self) -> None:
        cfg = PseudomodeConfig(N=10, K=3, alpha=1.0, g_sa=0.0, g_sp=0.0, lam=0.0)
        rho = np.outer(
            pseudomode_initial_state(cfg), pseudomode_initial_state(cfg).conj()
        )
        assert compute_qfi_with_ancilla(rho, cfg.N, cfg.K) == pytest.approx(
            compute_qfi_without_ancilla(rho, cfg.N, cfg.K), abs=1e-10
        )

    def test_gsa_changes_qfi(self) -> None:
        cfg = PseudomodeConfig(
            N=10, K=3, alpha=1.0, g_sa=2.0, tau=0.3, g_sp=0.0, lam=0.0, T=0.0
        )
        psi = pseudomode_initial_state(cfg)
        psi_ent = apply_ancilla_entanglement(psi, cfg)
        rho = np.outer(psi_ent, psi_ent.conj())
        qfi_with = compute_qfi_with_ancilla(rho, cfg.N, cfg.K)
        qfi_without = compute_qfi_without_ancilla(rho, cfg.N, cfg.K)
        assert qfi_with >= qfi_without - 1e-10


# Metrology Protocol


class TestRunMetrologyProtocol:
    """Full metrology protocol: completion, invariants, ratios."""

    _default_kw: ClassVar[dict] = {
        "N": 5,
        "K": 3,
        "alpha": 1.0,
        "g_sa": 0.5,
        "tau": 0.2,
        "g_sp": 0.3,
        "lam": 0.5,
        "T": 0.5,
    }

    def test_protocol_completes(self) -> None:
        result = run_metrology_protocol(PseudomodeConfig(**self._default_kw))
        expected_keys = {
            "rho_final",
            "qfi_with",
            "qfi_without",
            "qfi_initial",
            "ratio_with",
            "ratio_without",
            "pm_occupancy",
            "validation",
        }
        assert expected_keys.issubset(result.keys())

    def test_ratio_between_zero_and_one(self) -> None:
        result = run_metrology_protocol(PseudomodeConfig(**self._default_kw))
        assert 0 <= result["ratio_with"] <= 1.0 + 1e-6
        assert 0 <= result["ratio_without"] <= 1.0 + 1e-6

    def test_qfi_decreases_with_time(self) -> None:
        result = run_metrology_protocol(PseudomodeConfig(**self._default_kw))
        assert result["qfi_with"] <= result["qfi_initial"] + 1e-6

    def test_validation_passes(self) -> None:
        result = run_metrology_protocol(PseudomodeConfig(**self._default_kw))
        v = result["validation"]
        assert v["is_hermitian"] and v["is_normalized"] and v["is_positive"]

    def test_ratio_with_geq_without(self) -> None:
        result = run_metrology_protocol(PseudomodeConfig(**self._default_kw))
        assert result["ratio_with"] >= result["ratio_without"] - 1e-6


# Pseudomode Occupancy


class TestCheckPseudomodeOccupancy:
    """Pseudomode truncation check."""

    def test_vacuum_occupancy_zero(self) -> None:
        config = PseudomodeConfig(N=5, K=3)
        rho = np.outer(
            pseudomode_initial_state(config), pseudomode_initial_state(config).conj()
        )
        occ, is_safe = check_pseudomode_occupancy(rho, config.N, config.K)
        assert occ == pytest.approx(0.0, abs=1e-10)
        assert is_safe

    @pytest.mark.parametrize("K", [1, 3, 5, 10])
    def test_vacuum_is_safe(self, K: int) -> None:
        config = PseudomodeConfig(N=5, K=K)
        rho = np.outer(
            pseudomode_initial_state(config), pseudomode_initial_state(config).conj()
        )
        _, is_safe = check_pseudomode_occupancy(rho, config.N, config.K)
        assert is_safe


# ... (continued after this line to finish the file)

# QFI Preservation Ratio


class TestQFIPreservationRatio:
    """QFI preservation ratio computation."""

    def test_ratio_at_T0(self) -> None:
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sa=0.5, tau=0.2, g_sp=0.0, lam=0.0, T=0.0
        )
        psi = pseudomode_initial_state(config)
        psi_ent = apply_ancilla_entanglement(psi, config)
        rho = np.outer(psi_ent, psi_ent.conj())
        fq_initial = compute_qfi_with_ancilla(rho, config.N, config.K)
        assert qfi_preservation_ratio(
            rho, fq_initial, config.N, config.K
        ) == pytest.approx(1.0, abs=1e-6)

    def test_zero_fq_initial(self) -> None:
        config = PseudomodeConfig(N=5, K=3, alpha=0.0, g_sa=0.0, g_sp=0.0, lam=0.0)
        rho = np.outer(
            pseudomode_initial_state(config), pseudomode_initial_state(config).conj()
        )
        assert qfi_preservation_ratio(rho, 0.0, config.N, config.K) == 0.0

    def test_with_and_without_ancilla(self) -> None:
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sa=0.5, tau=0.2, g_sp=0.0, lam=0.0, T=0.0
        )
        psi = pseudomode_initial_state(config)
        psi_ent = apply_ancilla_entanglement(psi, config)
        rho = np.outer(psi_ent, psi_ent.conj())
        fq_initial = compute_qfi_with_ancilla(rho, config.N, config.K)
        assert qfi_preservation_ratio(
            rho, fq_initial, config.N, config.K, with_ancilla=True
        ) == pytest.approx(1.0, abs=1e-6)
        assert (
            qfi_preservation_ratio(
                rho, fq_initial, config.N, config.K, with_ancilla=False
            )
            >= 0
        )


# Edge Cases


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_small_N_and_K(self) -> None:
        result = run_metrology_protocol(
            PseudomodeConfig(N=1, K=1, alpha=0.5, g_sa=0.5, g_sp=0.1, lam=0.1, T=0.1)
        )
        assert result["validation"]["is_normalized"]

    def test_zero_system_pseudomode_coupling_no_decoherence(self) -> None:
        result = run_metrology_protocol(
            PseudomodeConfig(
                N=5, K=3, alpha=1.0, g_sa=0.5, tau=0.2, g_sp=0.0, lam=1.0, T=0.5
            )
        )
        assert result["ratio_with"] == pytest.approx(1.0, abs=1e-4)

    def test_zero_dissipation_yields_coherent_dynamics_with_valid_state(self) -> None:
        result = run_metrology_protocol(
            PseudomodeConfig(
                N=15, K=3, alpha=1.0, g_sa=0.5, tau=0.2, g_sp=0.3, lam=0.0, T=0.5
            )
        )
        v = result["validation"]
        assert v["is_normalized"] and v["is_hermitian"]
        assert result["ratio_with"] >= 0
        assert result["qfi_with"] <= result["qfi_initial"] + 1e-6

    def test_high_alpha_large_N(self) -> None:
        cfg = PseudomodeConfig(N=40, K=5, alpha=3.0, g_sa=0.0, g_sp=0.0, lam=0.0)
        norm = np.sum(np.abs(pseudomode_initial_state(cfg)) ** 2)
        assert norm == pytest.approx(1.0, abs=1e-6)
