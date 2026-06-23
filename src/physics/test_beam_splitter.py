"""Tests for consolidated beam-splitter implementations."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.physics.beam_splitter import bs_dicke, bs_fock, bs_qubit

# ── bs_fock ──────────────────────────────────────────────────────────────────


class TestBsFock:
    """Tests for bs_fock (two-mode Fock space)."""

    @pytest.mark.parametrize("max_photons", [1, 2, 3])
    def test_correct_dimension(self, max_photons: int) -> None:
        dim = (max_photons + 1) ** 2
        U = bs_fock(np.pi / 4, 0.0, max_photons)
        assert U.shape == (dim, dim), f"Expected ({dim}, {dim}), got {U.shape}"

    @pytest.mark.parametrize("max_photons", [1, 2, 3])
    def test_unitary(self, max_photons: int) -> None:
        U = bs_fock(np.pi / 4, 0.0, max_photons)
        dim = U.shape[0]
        assert np.allclose(U @ U.conj().T, np.eye(dim), atol=1e-10)

    @pytest.mark.parametrize("max_photons", [1, 2, 3])
    def test_identity_at_zero(self, max_photons: int) -> None:
        U = bs_fock(0.0, 0.0, max_photons)
        dim = (max_photons + 1) ** 2
        assert np.allclose(U, np.eye(dim))

    def test_preserves_norm_vacuum(self) -> None:
        """Vacuum |0,0⟩ should remain normalized after BS."""
        U = bs_fock(np.pi / 4, 0.0, max_photons=2)
        vacuum = np.zeros((3**2,), dtype=complex)
        vacuum[0] = 1.0
        out = U @ vacuum
        assert np.isclose(np.linalg.norm(out), 1.0)

    def test_preserves_norm_single_photon(self) -> None:
        """|1,0⟩ should remain normalized after BS."""
        U = bs_fock(np.pi / 4, 0.0, max_photons=2)
        psi = np.zeros((3**2,), dtype=complex)
        psi[0 * 3 + 1] = 1.0  # |n1=0, n2=1⟩
        out = U @ psi
        assert np.isclose(np.linalg.norm(out), 1.0)

    def test_vacuum_remains_vacuum(self) -> None:
        """BS acting on vacuum should give vacuum."""
        U = bs_fock(np.pi / 4, 0.0, max_photons=2)
        vacuum = np.zeros((3**2,), dtype=complex)
        vacuum[0] = 1.0
        out = U @ vacuum
        assert np.isclose(out[0], 1.0)
        assert np.isclose(np.linalg.norm(out), 1.0)


# ── bs_qubit ─────────────────────────────────────────────────────────────────


class TestBsQubit:
    """Tests for bs_qubit (single-qubit space)."""

    @pytest.mark.parametrize("T_BS", [0.0, 0.5, np.pi / 4, np.pi / 2, np.pi])
    def test_shape_is_2x2(self, T_BS: float) -> None:
        U = bs_qubit(T_BS)
        assert U.shape == (2, 2)

    @pytest.mark.parametrize("T_BS", [0.0, 0.5, np.pi / 4, np.pi / 2, np.pi])
    def test_unitary(self, T_BS: float) -> None:
        U = bs_qubit(T_BS)
        assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-12)
        assert np.allclose(U.conj().T @ U, np.eye(2), atol=1e-12)

    def test_identity_at_zero(self) -> None:
        U = bs_qubit(0.0)
        assert np.allclose(U, np.eye(2))

    def test_known_matrix_at_half_pi(self) -> None:
        """U_BS(π/2) = [[1, -i], [-i, 1]] / √2."""
        U = bs_qubit(np.pi / 2)
        expected = np.array([[1, -1j], [-1j, 1]], dtype=complex) / np.sqrt(2)
        assert np.allclose(U, expected, atol=1e-12)


# ── bs_dicke ─────────────────────────────────────────────────────────────────


class TestBsDicke:
    """Tests for bs_dicke (single-subsystem Dicke basis)."""

    @pytest.mark.parametrize("N", [1, 2, 3, 5, 10])
    def test_correct_dimension(self, N: int) -> None:
        U = bs_dicke(N)
        assert U.shape == (N + 1, N + 1), f"Expected ({N + 1}, {N + 1}), got {U.shape}"

    @pytest.mark.parametrize("N", [1, 2, 3, 5, 10, 20])
    def test_unitary(self, N: int) -> None:
        U = bs_dicke(N)
        assert np.allclose(U @ U.conj().T, np.eye(N + 1), atol=1e-12)

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_identity_at_zero(self, N: int) -> None:
        U = bs_dicke(N, T_BS=0.0)
        assert np.allclose(U, np.eye(N + 1))

    def test_cache_hit_returns_same_object(self) -> None:
        U1 = bs_dicke(5, np.pi / 2)
        U2 = bs_dicke(5, np.pi / 2)
        assert U1 is U2

    def test_cache_different_T_different_object(self) -> None:
        U1 = bs_dicke(5, np.pi / 2)
        # Without caching, this would return a different object; with
        # caching the different T gives a different cache entry.
        U2 = bs_dicke(5, np.pi / 4)
        assert U1 is not U2
        assert not np.allclose(U1, U2)


# ── Hypothesis property-based tests ───────────────────────────────────────────


class TestHypothesisBsFock:
    """Property-based tests for bs_fock unitarity using hypothesis.

    These replace the need for hand-picked @pytest.mark.parametrize — hypothesis
    explores the continuous parameter space (theta, phi_bs) and discrete space
    (max_photons) automatically, ensuring the unitarity invariant holds across
    a wider range than manual parametrization could cover.
    """

    @settings(max_examples=50, deadline=5000)
    @given(
        theta=st.floats(0.0, np.pi),
        phi_bs=st.floats(0.0, 2 * np.pi),
        max_photons=st.integers(1, 5),
    )
    def test_unitary(self, theta: float, phi_bs: float, max_photons: int) -> None:
        """U @ U† = I for any beam-splitter angle, phase, and truncation."""
        U = bs_fock(theta, phi_bs, max_photons)
        dim = (max_photons + 1) ** 2
        assert U.shape == (dim, dim)
        assert np.allclose(U @ U.conj().T, np.eye(dim), atol=1e-10)

    @settings(max_examples=20, deadline=2000)
    @given(
        theta=st.floats(0.0, np.pi),
        phi_bs=st.floats(0.0, 2 * np.pi),
        max_photons=st.integers(1, 5),
    )
    def test_correct_dimension(
        self, theta: float, phi_bs: float, max_photons: int
    ) -> None:
        """Output dimension matches (max_photons + 1)^2."""
        U = bs_fock(theta, phi_bs, max_photons)
        expected_dim = (max_photons + 1) ** 2
        assert U.shape == (expected_dim, expected_dim)


class TestHypothesisBsDicke:
    """Property-based tests for bs_dicke unitarity using hypothesis."""

    @settings(max_examples=50, deadline=5000)
    @given(
        N=st.integers(1, 20),
        T_BS=st.floats(0.0, np.pi),
    )
    def test_unitary(self, N: int, T_BS: float) -> None:
        """U @ U† = I for any particle number and BS angle."""
        U = bs_dicke(N, T_BS)
        assert U.shape == (N + 1, N + 1)
        assert np.allclose(U @ U.conj().T, np.eye(N + 1), atol=1e-12)

    @settings(max_examples=20, deadline=5000)
    @given(
        N=st.integers(1, 20),
        T_BS=st.floats(0.0, np.pi),
    )
    def test_correct_dimension(self, N: int, T_BS: float) -> None:
        """Output dimension matches N + 1."""
        U = bs_dicke(N, T_BS)
        assert U.shape == (N + 1, N + 1)


class TestHypothesisBsQubit:
    """Property-based tests for bs_qubit unitarity using hypothesis."""

    @settings(max_examples=50, deadline=5000)
    @given(
        T_BS=st.floats(0.0, 2 * np.pi),
    )
    def test_unitary(self, T_BS: float) -> None:
        """U @ U† = I for any qubit BS angle."""
        U = bs_qubit(T_BS)
        assert U.shape == (2, 2)
        assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-12)
        assert np.allclose(U.conj().T @ U, np.eye(2), atol=1e-12)
