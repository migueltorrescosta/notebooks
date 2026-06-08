"""
Unit tests for the multi_mzi operator-construction module.

Run with:
    uv run pytest src/physics/test_multi_mzi.py -q --tb=short
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.multi_mzi import (
    dicke_single_operators,
    dual_bs_unitary,
    embed_combined_operators,
    single_bs_unitary,
)


class TestDickeSingleOperators:
    @pytest.mark.parametrize("N", [1, 2, 3, 5, 10])
    def test_correct_dimension(self, N: int) -> None:
        ops = dicke_single_operators(N)
        for name, op in ops.items():
            assert op.shape == (N + 1, N + 1), (
                f"{name} has shape {op.shape}, expected ({N + 1}, {N + 1})"
            )

    @pytest.mark.parametrize("N", [1, 2, 3, 5, 10])
    def test_hermitian(self, N: int) -> None:
        ops = dicke_single_operators(N)
        for name, op in ops.items():
            assert np.allclose(op, op.conj().T, atol=1e-12), (
                f"{name} not Hermitian for N={N}"
            )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_commutation_jz_jx(self, N: int) -> None:
        ops = dicke_single_operators(N)
        comm = ops["Jz"] @ ops["Jx"] - ops["Jx"] @ ops["Jz"]
        expected = 1j * ops["Jy"]
        assert np.allclose(comm, expected, atol=1e-12), (
            f"[Jz, Jx] = i Jy failed for N={N}"
        )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_jz_diagonal(self, N: int) -> None:
        """J_z should be diagonal in the Dicke basis."""
        Jz = dicke_single_operators(N)["Jz"]
        assert np.allclose(Jz, np.diag(np.diag(Jz)), atol=1e-12), (
            f"J_z not diagonal for N={N}"
        )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_correct_keys(self, N: int) -> None:
        ops = dicke_single_operators(N)
        assert set(ops.keys()) == {"Jz", "Jx", "Jy"}

    def test_n_zero(self) -> None:
        """N=0 should return 1×1 operators."""
        ops = dicke_single_operators(0)
        for name, op in ops.items():
            assert op.shape == (1, 1), f"{name} has shape {op.shape} for N=0"


class TestEmbedCombinedOperators:
    @pytest.mark.parametrize("N", [1, 2, 3, 5, 10])
    def test_correct_dimension(self, N: int) -> None:
        ops = embed_combined_operators(N)
        dim = (N + 1) ** 2
        for name, op in ops.items():
            assert op.shape == (dim, dim), (
                f"{name} has shape {op.shape}, expected ({dim}, {dim})"
            )

    @pytest.mark.parametrize("N", [1, 2, 3, 5, 10])
    def test_hermitian(self, N: int) -> None:
        ops = embed_combined_operators(N)
        for name in ["Jz_S", "Jz_A", "Jx_S", "Jx_A", "Jy_S", "Jy_A", "I"]:
            assert np.allclose(ops[name], ops[name].conj().T, atol=1e-12), (
                f"{name} not Hermitian for N={N}"
            )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_commutation_jz_jx_system(self, N: int) -> None:
        ops = embed_combined_operators(N)
        comm_S = ops["Jz_S"] @ ops["Jx_S"] - ops["Jx_S"] @ ops["Jz_S"]
        expected = 1j * ops["Jy_S"]
        assert np.allclose(comm_S, expected, atol=1e-12), (
            f"[Jz_S, Jx_S] = i Jy_S failed for N={N}"
        )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_commutation_jz_jx_ancilla(self, N: int) -> None:
        ops = embed_combined_operators(N)
        comm_A = ops["Jz_A"] @ ops["Jx_A"] - ops["Jx_A"] @ ops["Jz_A"]
        assert np.allclose(comm_A, 1j * ops["Jy_A"], atol=1e-12), (
            f"[Jz_A, Jx_A] = i Jy_A failed for N={N}"
        )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_identity_present_and_correct(self, N: int) -> None:
        ops = embed_combined_operators(N)
        dim = (N + 1) ** 2
        assert "I" in ops, "Missing 'I' key"
        assert ops["I"].shape == (dim, dim)
        assert np.allclose(ops["I"], np.eye(dim), atol=1e-12), (
            f"'I' is not the identity for N={N}"
        )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_kronecker_structure(self, N: int) -> None:
        """Verify J_k_S = J_k ⊗ I and J_k_A = I ⊗ J_k."""
        single = dicke_single_operators(N)
        eye = np.eye(N + 1, dtype=float)
        combined = embed_combined_operators(N)

        for k in ["Jz", "Jx", "Jy"]:
            expected_S = np.kron(single[k], eye)
            expected_A = np.kron(eye, single[k])
            assert np.allclose(combined[f"{k}_S"], expected_S, atol=1e-12), (
                f"{k}_S mismatch for N={N}"
            )
            assert np.allclose(combined[f"{k}_A"], expected_A, atol=1e-12), (
                f"{k}_A mismatch for N={N}"
            )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_correct_keys(self, N: int) -> None:
        ops = embed_combined_operators(N)
        expected_keys = {"Jz_S", "Jz_A", "Jx_S", "Jx_A", "Jy_S", "Jy_A", "I"}
        assert set(ops.keys()) == expected_keys, (
            f"Keys mismatch for N={N}: {set(ops.keys())} != {expected_keys}"
        )

    def test_n_zero(self) -> None:
        """N=0 should return 1×1 operators."""
        ops = embed_combined_operators(0)
        for name, op in ops.items():
            assert op.shape == (1, 1), f"{name} has shape {op.shape} for N=0"

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_system_ancilla_commute(self, N: int) -> None:
        """[J_k^S, J_l^A] = 0 for any k, l."""
        ops = embed_combined_operators(N)
        for k in ["Jz", "Jx", "Jy"]:
            for label in ["Jz", "Jx", "Jy"]:
                comm = (
                    ops[f"{k}_S"] @ ops[f"{label}_A"]
                    - ops[f"{label}_A"] @ ops[f"{k}_S"]
                )
                assert np.allclose(comm, 0, atol=1e-12), (
                    f"[{k}_S, {label}_A] != 0 for N={N}"
                )


class TestSingleBSUnitary:
    @pytest.mark.parametrize("N", [1, 2, 3, 5, 10])
    def test_correct_dimension(self, N: int) -> None:
        U = single_bs_unitary(N)
        assert U.shape == (N + 1, N + 1), (
            f"BS unitary shape {U.shape} != ({N + 1}, {N + 1})"
        )

    @pytest.mark.parametrize("N", [1, 2, 3, 5, 10, 20])
    def test_unitary(self, N: int) -> None:
        U = single_bs_unitary(N)
        eye = np.eye(N + 1, dtype=complex)
        assert np.allclose(U @ U.conj().T, eye, atol=1e-12), (
            f"BS unitary not unitary for N={N}"
        )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_identity_at_zero(self, N: int) -> None:
        """T=0 should give the identity."""
        U = single_bs_unitary(N, T=0.0)
        eye = np.eye(N + 1, dtype=complex)
        assert np.allclose(U, eye, atol=1e-12), f"BS(T=0) != I for N={N}"

    def test_cache_hit_returns_same_object(self) -> None:
        """Cached call with same (N, T) should return the same object."""
        U1 = single_bs_unitary(5, np.pi / 2.0)
        U2 = single_bs_unitary(5, np.pi / 2.0)
        assert U1 is U2, "Cache miss: same (N, T) returned different objects"

    def test_cache_different_T_different_object(self) -> None:
        """Different T should produce different cached objects."""
        U1 = single_bs_unitary(5, np.pi / 2.0)
        U2 = single_bs_unitary(5, 0.0)
        assert U1 is not U2, "Cache error: different T returned same object"


class TestDualBSUnitary:
    @pytest.mark.parametrize("N", [1, 2, 3, 5, 10])
    def test_correct_dimension(self, N: int) -> None:
        U = dual_bs_unitary(N)
        dim = (N + 1) ** 2
        assert U.shape == (dim, dim), (
            f"Dual BS unitary shape {U.shape} != ({dim}, {dim})"
        )

    @pytest.mark.parametrize("N", [1, 2, 3, 5, 10])
    def test_unitary(self, N: int) -> None:
        U = dual_bs_unitary(N)
        dim = (N + 1) ** 2
        eye = np.eye(dim, dtype=complex)
        assert np.allclose(U @ U.conj().T, eye, atol=1e-12), (
            f"Dual BS not unitary for N={N}"
        )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_kronecker_structure(self, N: int) -> None:
        """Verify dual BS = U_single ⊗ U_single."""
        U_single = single_bs_unitary(N)
        U_dual = dual_bs_unitary(N)
        expected = np.kron(U_single, U_single)
        assert np.allclose(U_dual, expected, atol=1e-12), (
            f"Dual BS != U_single ⊗ U_single for N={N}"
        )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_identity_at_zero(self, N: int) -> None:
        """T=0 should give the identity."""
        U = dual_bs_unitary(N, T=0.0)
        dim = (N + 1) ** 2
        eye = np.eye(dim, dtype=complex)
        assert np.allclose(U, eye, atol=1e-12), f"Dual BS(T=0) != I for N={N}"
