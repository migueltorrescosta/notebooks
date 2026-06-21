"""Tests for the shared bipartite operators module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.physics.bipartite_operators import (
    build_iszz_interaction,
    build_operators,
    build_system_only_bs_unitary,
)


class TestBuildOperators:
    """Tests for build_operators."""

    @pytest.mark.parametrize(
        ("N_sys", "N_anc"),
        [(1, 1), (3, 2), (5, 4), (10, 1)],
        ids=[
            "N_sys=1,N_anc=1",
            "N_sys=3,N_anc=2",
            "N_sys=5,N_anc=4",
            "N_sys=10,N_anc=1",
        ],
    )
    def test_operator_dimensions(self, N_sys: int, N_anc: int) -> None:
        d_tot = (N_sys + 1) * (N_anc + 1)
        ops = build_operators(N_sys, N_anc)
        for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
            assert ops[key].shape == (d_tot, d_tot), (
                f"{key} has wrong shape for N_sys={N_sys}, N_anc={N_anc}"
            )

    @pytest.mark.parametrize(
        ("N_sys", "N_anc"),
        [(1, 1), (2, 3), (4, 2)],
        ids=["N_sys=1,N_anc=1", "N_sys=2,N_anc=3", "N_sys=4,N_anc=2"],
    )
    def test_operators_hermitian(self, N_sys: int, N_anc: int) -> None:
        ops = build_operators(N_sys, N_anc)
        for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
            assert np.allclose(ops[key], ops[key].conj().T, atol=1e-12), (
                f"{key} not Hermitian for N_sys={N_sys}, N_anc={N_anc}"
            )

    def test_commutation_relations_system(self) -> None:
        """[J_z^S, J_x^S] = i J_y^S"""
        ops = build_operators(3, 2)
        comm = ops["Jz_S"] @ ops["Jx_S"] - ops["Jx_S"] @ ops["Jz_S"]
        assert np.allclose(comm, 1j * ops["Jy_S"], atol=1e-10)

    def test_commutation_relations_ancilla(self) -> None:
        """[J_z^A, J_x^A] = i J_y^A"""
        ops = build_operators(3, 2)
        comm = ops["Jz_A"] @ ops["Jx_A"] - ops["Jx_A"] @ ops["Jz_A"]
        assert np.allclose(comm, 1j * ops["Jy_A"], atol=1e-10)

    def test_system_ancilla_commute(self) -> None:
        """[J_z^S, J_z^A] = 0 — cross-subsystem operators commute."""
        ops = build_operators(3, 2)
        comm = ops["Jz_S"] @ ops["Jz_A"] - ops["Jz_A"] @ ops["Jz_S"]
        assert np.allclose(comm, np.zeros_like(comm), atol=1e-10)

    def test_raises_for_bad_N_sys(self) -> None:
        with pytest.raises(ValueError, match="N_sys must be >= 1"):
            build_operators(0, 1)

    def test_raises_for_bad_N_anc(self) -> None:
        with pytest.raises(ValueError, match="N_anc must be >= 1"):
            build_operators(1, 0)


class TestBuildSystemOnlyBSUnitary:
    """Tests for build_system_only_bs_unitary."""

    @pytest.mark.parametrize(
        ("N_sys", "N_anc"),
        [(1, 1), (2, 3), (5, 2), (10, 4)],
        ids=[
            "N_sys=1,N_anc=1",
            "N_sys=2,N_anc=3",
            "N_sys=5,N_anc=2",
            "N_sys=10,N_anc=4",
        ],
    )
    def test_bs_unitary(self, N_sys: int, N_anc: int) -> None:
        d_tot = (N_sys + 1) * (N_anc + 1)
        U = build_system_only_bs_unitary(N_sys, N_anc)
        I_full = np.eye(d_tot, dtype=complex)
        assert np.allclose(U @ U.conj().T, I_full, atol=1e-12), (
            f"BS unitary not unitary for N_sys={N_sys}, N_anc={N_anc}"
        )

    def test_bs_uses_default_T_bs(self) -> None:
        """Default T_bs = pi/2 should give 50/50 beam splitter."""
        U = build_system_only_bs_unitary(1, 1)
        U_explicit = build_system_only_bs_unitary(1, 1, math.pi / 2.0)
        assert np.allclose(U, U_explicit, atol=1e-12)


class TestBuildIszzInteraction:
    """Tests for build_iszz_interaction."""

    @pytest.mark.parametrize("N_sys,N_anc", [(1, 1), (3, 2), (5, 4)])
    def test_iszz_hermitian(self, N_sys: int, N_anc: int) -> None:
        ops = build_operators(N_sys, N_anc)
        H = build_iszz_interaction(2.5, ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    @pytest.mark.parametrize("N_sys,N_anc", [(1, 1), (3, 2), (5, 4)])
    def test_iszz_zero_at_zero(self, N_sys: int, N_anc: int) -> None:
        ops = build_operators(N_sys, N_anc)
        H = build_iszz_interaction(0.0, ops)
        assert np.allclose(H, 0.0, atol=1e-12)

    @pytest.mark.parametrize("N_sys,N_anc", [(2, 3), (4, 2)])
    def test_iszz_dimension(self, N_sys: int, N_anc: int) -> None:
        d_tot = (N_sys + 1) * (N_anc + 1)
        ops = build_operators(N_sys, N_anc)
        H = build_iszz_interaction(1.0, ops)
        assert H.shape == (d_tot, d_tot), (
            f"iszz interaction has wrong shape: {H.shape} vs ({d_tot}, {d_tot})"
        )
