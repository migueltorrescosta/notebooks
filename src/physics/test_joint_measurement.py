"""Tests for the shared joint measurement operator module.

Validates both the qubit-ancilla (J_A=1/2) and bipartite (J_A=N/2)
variants of the weighted joint measurement operator M(ψ) = cosψ·J_z^S + sinψ·J_z^A.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.bipartite_operators import build_operators
from src.physics.joint_measurement import (
    build_bipartite_joint_measurement_operator,
    build_joint_measurement_operator,
)
from src.physics.n_particle_drive import build_n_particle_operators


class TestBuildJointMeasurementOperator:
    @pytest.fixture(params=[1, 2, 5, 10])
    def make_N(self, request: pytest.FixtureRequest) -> int:
        return int(request.param)

    @pytest.fixture
    def make_ops(self, make_N: int) -> dict[str, np.ndarray]:
        return build_n_particle_operators(make_N)

    def test_given_psi_zero_then_matches_Jz_S(
        self, make_N: int, make_ops: dict
    ) -> None:
        M = build_joint_measurement_operator(make_N, 0.0, make_ops)
        assert np.allclose(M, make_ops["Jz_S"], atol=1e-12)

    def test_given_psi_pi_half_then_matches_Jz_A(
        self, make_N: int, make_ops: dict
    ) -> None:
        M = build_joint_measurement_operator(make_N, np.pi / 2.0, make_ops)
        assert np.allclose(M, make_ops["Jz_A"], atol=1e-12)

    def test_given_psi_then_hermitian(self, make_N: int, make_ops: dict) -> None:
        for psi in [0.0, 0.5, 1.0, np.pi / 4]:
            M = build_joint_measurement_operator(make_N, psi, make_ops)
            assert np.allclose(M, M.conj().T, atol=1e-12)

    def test_given_psi_then_correct_dimension(
        self, make_N: int, make_ops: dict
    ) -> None:
        M = build_joint_measurement_operator(make_N, 0.5, make_ops)
        d_tot = 2 * (make_N + 1)
        assert M.shape == (d_tot, d_tot)

    def test_given_psi_pi_four_then_balanced(self, make_N: int, make_ops: dict) -> None:
        M = build_joint_measurement_operator(make_N, np.pi / 4.0, make_ops)
        expected = (make_ops["Jz_S"] + make_ops["Jz_A"]) / np.sqrt(2)
        assert np.allclose(M, expected, atol=1e-12)

    def test_given_N1_psi_pi_four_then_correct_values(self) -> None:
        ops = build_n_particle_operators(1)
        M = build_joint_measurement_operator(1, np.pi / 4.0, ops)
        expected_diag = np.array([1.0, 0.0, 0.0, -1.0]) / np.sqrt(2)
        assert np.allclose(np.diag(M), expected_diag, atol=1e-12)


class TestBuildBipartiteJointMeasurementOperator:
    @pytest.fixture(params=[1, 2, 5])
    def make_N(self, request: pytest.FixtureRequest) -> int:
        return int(request.param)

    @pytest.fixture
    def make_bipartite_ops(self, make_N: int) -> dict[str, np.ndarray]:
        return build_operators(make_N, make_N)

    def test_given_psi_zero_then_matches_Jz_S(
        self, make_N: int, make_bipartite_ops: dict
    ) -> None:
        M = build_bipartite_joint_measurement_operator(make_N, 0.0, make_bipartite_ops)
        assert np.allclose(M, make_bipartite_ops["Jz_S"], atol=1e-12)

    def test_given_psi_pi_half_then_matches_Jz_A(
        self, make_N: int, make_bipartite_ops: dict
    ) -> None:
        M = build_bipartite_joint_measurement_operator(
            make_N, np.pi / 2.0, make_bipartite_ops
        )
        assert np.allclose(M, make_bipartite_ops["Jz_A"], atol=1e-12)

    def test_given_psi_then_hermitian(
        self, make_N: int, make_bipartite_ops: dict
    ) -> None:
        for psi in [0.0, 0.5, 1.0, np.pi / 4]:
            M = build_bipartite_joint_measurement_operator(
                make_N, psi, make_bipartite_ops
            )
            assert np.allclose(M, M.conj().T, atol=1e-12)

    def test_given_psi_then_correct_dimension(
        self, make_N: int, make_bipartite_ops: dict
    ) -> None:
        M = build_bipartite_joint_measurement_operator(make_N, 0.5, make_bipartite_ops)
        d_tot = (make_N + 1) ** 2
        assert M.shape == (d_tot, d_tot)

    def test_given_psi_pi_four_then_balanced(
        self, make_N: int, make_bipartite_ops: dict
    ) -> None:
        M = build_bipartite_joint_measurement_operator(
            make_N, np.pi / 4.0, make_bipartite_ops
        )
        expected = (make_bipartite_ops["Jz_S"] + make_bipartite_ops["Jz_A"]) / np.sqrt(
            2
        )
        assert np.allclose(M, expected, atol=1e-12)
