"""Tests for quimb-based bipartite tensor network representation."""

from __future__ import annotations

import time

import numpy as np
import pytest
import quimb.tensor as qtn

from .tensor_tree_network import (
    bipartite_tensor_from_state,
    compute_expectation,
    get_state_vector,
    tensor_fidelity,
)


def _make_split_and_contract(
    tensor: qtn.Tensor,
    cutoff: float = 1e-14,
    max_bond: int | None = None,
) -> np.ndarray:
    tn = tensor.split(
        "main",
        bond_ind="bond",
        cutoff=cutoff,
        cutoff_mode="rel" if cutoff > 0 else "abs",
        max_bond=max_bond,
    )
    contracted = tn.contract()
    assert contracted is not None
    return contracted.data.flatten()


def _make_split_bond_dim(
    tensor: qtn.Tensor,
    cutoff: float = 1e-14,
) -> int:
    tn = tensor.split("main", bond_ind="bond", cutoff=cutoff, cutoff_mode="rel")
    return tn.max_bond()


class TestQuimbTensor:
    """Tests for quimb Tensor creation (replaces TestTTNNode)."""

    def test_qtn_tensor_created_with_data_and_indices(self) -> None:
        tensor = np.array([[1, 0], [0, 1]], dtype=complex)
        t = qtn.Tensor(np.asarray(tensor), inds=("main", "ancilla"))
        assert t.data is not None
        assert t.shape == (2, 2)
        assert t.ndim == 2
        assert t.inds == ("main", "ancilla")

    def test_tensornetwork_supports_connected_tensors(self) -> None:
        left = qtn.Tensor(
            np.array([[1, 0], [0, 0]], dtype=complex),
            inds=("main", "bond"),
        )
        right = qtn.Tensor(
            np.array([[0, 0], [0, 1]], dtype=complex),
            inds=("bond", "ancilla"),
        )
        tn = qtn.TensorNetwork([left, right])
        assert len(tn.tensors) == 2
        left_inds = set(left.inds)
        right_inds = set(right.inds)
        assert len(left_inds & right_inds) == 1


class TestBipartiteTensor:
    """Tests for bipartite tensor creation and state roundtrip."""

    def test_bipartite_tensor_from_state_creates_correct_shape(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        t = bipartite_tensor_from_state(state, n_sites=1, local_dim=2)
        assert t.shape == (2, 2)
        assert t.inds == ("main", "ancilla")
        assert t.dtype == complex

    @pytest.mark.parametrize(
        "state",
        [
            np.array([1, 0, 0, 0], dtype=complex),
            np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
        ],
        ids=["product", "bell"],
    )
    def test_bipartite_state_roundtrip_exact(self, state: np.ndarray) -> None:
        t = bipartite_tensor_from_state(state, n_sites=1, local_dim=2)
        reconstructed = get_state_vector(t)
        assert state == pytest.approx(reconstructed)


class TestSplitContractReconstruction:
    """Tests for SVD split + contract reconstruction fidelity."""

    def _make_random_state(self, n_sites: int, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        dim = 2 ** (2 * n_sites)
        state = rng.random(dim) + 1j * rng.random(dim)
        return state / np.linalg.norm(state)

    @pytest.mark.parametrize(
        ("n_sites", "threshold"),
        [
            (2, 1 - 1e-6),
            (3, 1 - 1e-6),
            (4, 1 - 1e-4),
            (5, 1 - 0.01),
            (6, 0.9),
        ],
        ids=["n=2", "n=3", "n=4", "n=5", "n=6"],
    )
    def test_reconstruction_fidelity_meets_threshold(
        self, n_sites: int, threshold: float
    ) -> None:
        state = self._make_random_state(n_sites=n_sites, seed=42)
        tensor = bipartite_tensor_from_state(state, n_sites, local_dim=2)
        reconstructed = _make_split_and_contract(tensor, cutoff=1e-14)
        fidelity = float(abs(np.vdot(state, reconstructed)) ** 2)
        assert fidelity > threshold


class TestBondDimension:
    """Tests for bond dimension behaviour."""

    def test_bond_dimension_larger_for_entangled_states(self) -> None:
        product_state = np.array([1, 0, 0, 0], dtype=complex)
        t_prod = bipartite_tensor_from_state(product_state, n_sites=1, local_dim=2)
        bond_prod = _make_split_bond_dim(t_prod)

        entangled_state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        t_ent = bipartite_tensor_from_state(entangled_state, n_sites=1, local_dim=2)
        bond_ent = _make_split_bond_dim(t_ent)

        assert bond_ent >= bond_prod

    def test_bond_dimension_remains_tractable_for_n_5(self) -> None:
        rng = np.random.default_rng(42)
        dim = 2**10
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        t = bipartite_tensor_from_state(state, n_sites=5, local_dim=2)
        bond_dim = _make_split_bond_dim(t)
        full_dim = 2**5
        assert bond_dim <= full_dim

    def test_bond_dimension_bounded_for_n_5(self) -> None:
        rng = np.random.default_rng(42)
        dim = 2**10
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        t = bipartite_tensor_from_state(state, n_sites=5, local_dim=2)
        bond_dim = _make_split_bond_dim(t)

        assert bond_dim <= 32


class TestTruncation:
    """Tests for SVD truncation."""

    def _make_rank_deficient_state(self, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = 4
        U, _ = np.linalg.qr(rng.random((n, n)) + 1j * rng.random((n, n)))
        Vh, _ = np.linalg.qr(rng.random((n, n)) + 1j * rng.random((n, n)))
        s = np.array([1.0, 0.5, 0.1, 0.01], dtype=float)
        matrix = U @ np.diag(s) @ Vh
        matrix = matrix / np.linalg.norm(matrix, "fro")
        return matrix.flatten()

    def test_truncation_preserves_normalisation_approximately(self) -> None:
        state = self._make_rank_deficient_state()
        t = bipartite_tensor_from_state(state, n_sites=2, local_dim=2)
        tn = t.split("main", bond_ind="bond", cutoff=0.2, cutoff_mode="rel")
        contracted = tn.contract()
        assert contracted is not None
        norm = np.linalg.norm(contracted.data.flatten())
        assert norm == pytest.approx(1.0, abs=1e-2)

    def test_reconstruction_fidelity_increases_as_cutoff_decreases(self) -> None:
        state = self._make_rank_deficient_state()
        t = bipartite_tensor_from_state(state, n_sites=2, local_dim=2)
        cutoffs = [0.3, 0.1, 0.03, 0.01, 1e-8]
        fidelities = [
            float(abs(np.vdot(state, _make_split_and_contract(t, cutoff=c))) ** 2)
            for c in cutoffs
        ]
        for i in range(len(fidelities) - 1):
            assert fidelities[i] <= fidelities[i + 1] + 1e-6


class TestExpectationValues:
    """Tests for expectation value computation."""

    def test_expectation_of_identity_is_1(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        t = bipartite_tensor_from_state(state, n_sites=1, local_dim=2)
        identity = np.eye(2, dtype=complex)
        result = compute_expectation(t, [(0, identity)])
        assert result.real == pytest.approx(1.0, abs=1e-6)

    def test_z_on_main_in_00_gives_1(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        t = bipartite_tensor_from_state(state, n_sites=1, local_dim=2)
        z = np.array([[1, 0], [0, -1]], dtype=complex)
        result = compute_expectation(t, [(0, z)])
        assert result.real == pytest.approx(1.0, abs=1e-6)

    def test_z_on_ancilla_in_01_gives_negative_1(self) -> None:
        state = np.array([0, 1, 0, 0], dtype=complex)
        t = bipartite_tensor_from_state(state, n_sites=1, local_dim=2)
        z = np.array([[1, 0], [0, -1]], dtype=complex)
        result = compute_expectation(t, [(1, z)])
        assert result.real == pytest.approx(-1.0, abs=1e-6)


class TestTensorFidelity:
    """Tests for tensor fidelity computation."""

    def test_fidelity_of_identical_states_is_1(self) -> None:
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        t1 = bipartite_tensor_from_state(state, n_sites=1, local_dim=2)
        t2 = bipartite_tensor_from_state(state, n_sites=1, local_dim=2)
        assert tensor_fidelity(t1, t2) == pytest.approx(1.0)

    def test_fidelity_of_orthogonal_states_is_0(self) -> None:
        state1 = np.array([1, 0, 0, 0], dtype=complex)
        state2 = np.array([0, 0, 0, 1], dtype=complex)
        t1 = bipartite_tensor_from_state(state1, n_sites=1, local_dim=2)
        t2 = bipartite_tensor_from_state(state2, n_sites=1, local_dim=2)
        assert tensor_fidelity(t1, t2) == pytest.approx(0.0)


class TestInputValidation:
    """Tests for input validation and norm conservation."""

    def test_raises_valueerror_for_invalid_state_dimension(self) -> None:
        state = np.array([1, 0, 0], dtype=complex)
        with pytest.raises(ValueError, match="State dimension"):
            bipartite_tensor_from_state(state, n_sites=1, local_dim=2)

    def test_norm_preserved_after_split_contract(self) -> None:
        rng = np.random.default_rng(123)
        dim = 2**4
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)
        t = bipartite_tensor_from_state(state, n_sites=2, local_dim=2)
        reconstructed = _make_split_and_contract(t, cutoff=1e-14)
        exact_norm = np.vdot(state, state).real
        ttn_norm = np.vdot(reconstructed, reconstructed).real
        assert exact_norm == pytest.approx(ttn_norm, rel=1e-4)


class TestPerformance:
    """Performance tests for tensor operations."""

    def test_svd_split_contract_runs_in_100ms_for_n_8(self) -> None:
        rng = np.random.default_rng(42)
        n_sites = 4
        dim = 2 ** (2 * n_sites)
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)
        t = bipartite_tensor_from_state(state, n_sites=n_sites, local_dim=2)
        start = time.perf_counter()
        _make_split_and_contract(t, cutoff=1e-14)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1
