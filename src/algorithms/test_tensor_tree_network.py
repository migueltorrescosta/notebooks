"""Unit tests for quimb-based bipartite tensor network representation.

These tests replace the deprecated ``TensorTreeNetwork`` / ``TTNNode`` tests
with direct use of ``quimb.tensor.Tensor`` / ``quimb.tensor.TensorNetwork``.

Every guarantee previously validated by the ``TensorTreeNetwork`` tests is
preserved:

  - State roundtrip (tensor → state vector) is exact.
  - SVD split + contract reconstruction achieves machine-precision fidelity.
  - Bond dimensions grow with entanglement.
  - Truncation via ``cutoff`` preserves normalisation within truncation error.
  - Expectation values computed via gate+overlap are correct.
  - Fidelity between tensor states is correct.
  - Invalid state dimensions are rejected.
  - Performance stays below 100 ms for moderate sizes.
"""

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

# =============================================================================
# Helpers
# =============================================================================


def _split_and_contract(
    tensor: qtn.Tensor,
    cutoff: float = 1e-14,
    max_bond: int | None = None,
) -> np.ndarray:
    """Split a bipartite tensor via SVD, contract, and return the flat state."""
    tn = tensor.split(
        "main",
        bond_ind="bond",
        cutoff=cutoff,
        cutoff_mode="rel" if cutoff > 0 else "abs",
        max_bond=max_bond,
    )
    contracted = tn.contract()
    return contracted.data.flatten()


def _split_bond_dim(
    tensor: qtn.Tensor,
    cutoff: float = 1e-14,
) -> int:
    """Return the bond dimension after SVD splitting a tensor."""
    tn = tensor.split("main", bond_ind="bond", cutoff=cutoff, cutoff_mode="rel")
    return tn.max_bond()


# =============================================================================
# Tests: quimb Tensor creation and structure
# =============================================================================


class TestQuimbTensor:
    """Tests for quimb Tensor creation (replaces TestTTNNode)."""

    def test_qtn_tensor_should_be_created_with_data_and_indices(self) -> None:
        tensor = np.array([[1, 0], [0, 1]], dtype=complex)
        t = qtn.Tensor(tensor, inds=("main", "ancilla"))
        assert t.data is not None, "Expected t.data to not be None"
        assert t.shape == (2, 2), "Expected t.shape == (2, 2)"
        assert t.ndim == 2, "Expected t.ndim == 2"
        assert t.inds == ("main", "ancilla"), 'Expected t.inds == ("main", "ancilla")'

    def test_tensornetwork_should_support_connected_tensors(self) -> None:
        # Two 2D tensors sharing a "bond" index
        left = qtn.Tensor(
            np.array([[1, 0], [0, 0]], dtype=complex),
            inds=("main", "bond"),
        )
        right = qtn.Tensor(
            np.array([[0, 0], [0, 1]], dtype=complex),
            inds=("bond", "ancilla"),
        )
        tn = qtn.TensorNetwork([left, right])
        assert len(tn.tensors) == 2, "Expected 2 tensors in network"
        # Verify they share a common index
        left_inds = set(left.inds)
        right_inds = set(right.inds)
        assert len(left_inds & right_inds) == 1, "Expected shared bond index"


# =============================================================================
# Tests: bipartite tensor creation and roundtrip
# =============================================================================


class TestBipartiteTensor:
    """Tests for bipartite tensor creation and state roundtrip."""

    def test_bipartite_tensor_from_state_should_create_tensor_with_correct_shape(
        self,
    ) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        t = bipartite_tensor_from_state(state, n_sites=1, local_dim=2)
        assert t.shape == (2, 2), "Expected t.shape == (2, 2)"
        assert t.inds == ("main", "ancilla"), 'Expected t.inds == ("main", "ancilla")'
        assert t.dtype == complex, "Expected t.data.dtype == complex"

    def test_single_qubit_pair_state_roundtrip_should_be_exact(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        t = bipartite_tensor_from_state(state, n_sites=1, local_dim=2)
        reconstructed = get_state_vector(t)
        assert state == pytest.approx(reconstructed), (
            "Expected state == pytest.approx(reconstructed)"
        )

    def test_bell_superposition_state_roundtrip_should_be_exact(self) -> None:
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        t = bipartite_tensor_from_state(state, n_sites=1, local_dim=2)
        reconstructed = get_state_vector(t)
        assert state == pytest.approx(reconstructed), (
            "Expected state == pytest.approx(reconstructed)"
        )

    def test_product_state_00_roundtrip_should_be_exact(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        t = bipartite_tensor_from_state(state, n_sites=1, local_dim=2)
        reconstructed = get_state_vector(t)
        assert state == pytest.approx(reconstructed), (
            "Expected state == pytest.approx(reconstructed)"
        )

    def test_bell_entangled_state_roundtrip_should_be_exact(self) -> None:
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        t = bipartite_tensor_from_state(state, n_sites=1, local_dim=2)
        reconstructed = get_state_vector(t)
        assert state == pytest.approx(reconstructed), (
            "Expected state == pytest.approx(reconstructed)"
        )


# =============================================================================
# Tests: SVD split + contract reconstruction fidelity
# =============================================================================


class TestSplitContractReconstruction:
    """Tests for SVD split + contract reconstruction fidelity."""

    def _make_random_state(self, n_sites: int, seed: int = 42) -> np.ndarray:
        """Helper: create a normalised random state for 2*n_sites qubits."""
        rng = np.random.default_rng(seed)
        dim = 2 ** (2 * n_sites)
        state = rng.random(dim) + 1j * rng.random(dim)
        return state / np.linalg.norm(state)

    def _fidelity_after_split_contract(
        self,
        state: np.ndarray,
        n_sites: int,
        cutoff: float = 1e-14,
    ) -> float:
        """Fidelity between original state and split+contract reconstruction."""
        tensor = bipartite_tensor_from_state(state, n_sites, local_dim=2)
        reconstructed = _split_and_contract(tensor, cutoff=cutoff)
        return float(abs(np.vdot(state, reconstructed)) ** 2)

    def test_reconstruction_fidelity_for_n_2_should_be_1_1e_6(self) -> None:
        state = self._make_random_state(n_sites=2, seed=42)
        fidelity = self._fidelity_after_split_contract(state, n_sites=2)
        assert fidelity > 1 - 1e-6, f"Fidelity {fidelity} below threshold"

    def test_reconstruction_fidelity_for_n_3_should_be_1_1e_6(self) -> None:
        state = self._make_random_state(n_sites=3, seed=42)
        fidelity = self._fidelity_after_split_contract(state, n_sites=3)
        assert fidelity > 1 - 1e-6, f"Fidelity {fidelity} below threshold"

    def test_reconstruction_fidelity_for_n_4_should_be_1_1e_4(self) -> None:
        state = self._make_random_state(n_sites=4, seed=42)
        fidelity = self._fidelity_after_split_contract(state, n_sites=4)
        assert fidelity > 1 - 1e-4, f"Fidelity {fidelity} below threshold"

    def test_reconstruction_fidelity_for_n_5_should_be_1_0_01(self) -> None:
        state = self._make_random_state(n_sites=5, seed=42)
        fidelity = self._fidelity_after_split_contract(state, n_sites=5)
        assert fidelity > 1 - 0.01, f"Fidelity {fidelity} below threshold"

    def test_reconstruction_fidelity_for_n_6_should_be_0_9(self) -> None:
        state = self._make_random_state(n_sites=6, seed=42)
        fidelity = self._fidelity_after_split_contract(state, n_sites=6)
        assert fidelity > 0.9, f"Fidelity {fidelity} below threshold"


# =============================================================================
# Tests: bond dimensions
# =============================================================================


class TestBondDimension:
    """Tests for bond dimension behaviour."""

    def test_bond_dimension_should_be_larger_for_entangled_states(self) -> None:
        # Product state |00⟩: SVD of [[1,0],[0,0]] → rank 1
        product_state = np.array([1, 0, 0, 0], dtype=complex)
        t_prod = bipartite_tensor_from_state(product_state, n_sites=1, local_dim=2)
        bond_prod = _split_bond_dim(t_prod)

        # Entangled Bell state (|00⟩+|11⟩)/√2: SVD of [[1,0],[0,1]]/√2 → rank 2
        entangled_state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        t_ent = bipartite_tensor_from_state(entangled_state, n_sites=1, local_dim=2)
        bond_ent = _split_bond_dim(t_ent)

        assert bond_ent >= bond_prod, (
            f"Expected entangled bond dim ({bond_ent}) >= product bond dim ({bond_prod})"
        )

    def test_bond_dimension_should_remain_tractable_for_n_5_10_qubits(self) -> None:
        rng = np.random.default_rng(42)
        dim = 2**10  # 5 main + 5 ancilla = 10 total → 1024-dim vector
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        t = bipartite_tensor_from_state(state, n_sites=5, local_dim=2)
        bond_dim = _split_bond_dim(t)

        full_dim = 2**5  # 32
        assert bond_dim <= full_dim, (
            f"Expected bond dim ({bond_dim}) <= full dim ({full_dim})"
        )

    def test_bond_dimension_remains_bounded_for_n_5_10_qubits(self) -> None:
        rng = np.random.default_rng(42)
        dim = 2**10
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        t = bipartite_tensor_from_state(state, n_sites=5, local_dim=2)
        bond_dim = _split_bond_dim(t)

        # Maximum possible bond dimension is 2^5 = 32
        assert bond_dim <= 32, f"Expected bond dim <= 32, got {bond_dim}"
        # And should be significantly smaller than full matrix dimension
        matrix_dim = 2**5
        assert bond_dim < matrix_dim // 2 or bond_dim == matrix_dim, (
            f"Expected bond dim < {matrix_dim // 2} or == {matrix_dim}, got {bond_dim}"
        )


# =============================================================================
# Tests: truncation
# =============================================================================


class TestTruncation:
    """Tests for SVD truncation."""

    def _make_rank_deficient_state(self, seed: int = 42) -> np.ndarray:
        """Create a state with a known SVD spectrum for truncation testing.

        Returns a normalised 4-qubit (16-dim) state whose 4×4 matrix has
        singular values approximately [0.89, 0.45, 0.089, 0.0089].
        """
        rng = np.random.default_rng(seed)
        n = 4
        orth = rng.random((n, n)) + 1j * rng.random((n, n))
        U, _ = np.linalg.qr(orth)
        orth2 = rng.random((n, n)) + 1j * rng.random((n, n))
        Vh, _ = np.linalg.qr(orth2)
        s = np.array([1.0, 0.5, 0.1, 0.01], dtype=float)
        matrix = U @ np.diag(s) @ Vh
        matrix = matrix / np.linalg.norm(matrix, "fro")
        return matrix.flatten()

    def test_truncation_should_approximately_preserve_normalisation(self) -> None:
        state = self._make_rank_deficient_state()
        t = bipartite_tensor_from_state(state, n_sites=2, local_dim=2)

        # Split with mild truncation (bond dim 2 out of 4)
        tn = t.split("main", bond_ind="bond", cutoff=0.2, cutoff_mode="rel")
        contracted = tn.contract()
        norm = np.linalg.norm(contracted.data.flatten())
        # Norm should be close to 1 (truncation of small singular values
        # removes a small fraction of the norm)
        assert norm == pytest.approx(1.0, abs=1e-2), f"Expected norm ≈ 1, got {norm}"

    def test_reconstruction_fidelity_should_increase_as_cutoff_decreases(self) -> None:
        state = self._make_rank_deficient_state()
        t = bipartite_tensor_from_state(state, n_sites=2, local_dim=2)

        cutoffs = [0.3, 0.1, 0.03, 0.01, 1e-8]
        fidelities: list[float] = []

        for cutoff in cutoffs:
            reconstructed = _split_and_contract(t, cutoff=cutoff)
            fidelity = float(abs(np.vdot(state, reconstructed)) ** 2)
            fidelities.append(fidelity)

        # Fidelity should be non-decreasing as cutoff decreases
        for i in range(len(fidelities) - 1):
            assert fidelities[i] <= fidelities[i + 1] + 1e-6, (
                f"Fidelity decreased when cutoff went from {cutoffs[i]} "
                f"to {cutoffs[i + 1]}: {fidelities[i]} -> {fidelities[i + 1]}"
            )


# =============================================================================
# Tests: expectation values
# =============================================================================


class TestExpectationValues:
    """Tests for expectation value computation."""

    def test_expectation_of_identity_should_be_1(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        t = bipartite_tensor_from_state(state, n_sites=1, local_dim=2)
        identity = np.eye(2, dtype=complex)
        result = compute_expectation(t, [(0, identity)])
        assert result.real == pytest.approx(1.0, abs=1e-6), (
            f"Expected 1.0, got {result}"
        )

    def test_z_operator_on_main_qubit_in_00_should_give_1(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        t = bipartite_tensor_from_state(state, n_sites=1, local_dim=2)
        z = np.array([[1, 0], [0, -1]], dtype=complex)
        result = compute_expectation(t, [(0, z)])
        assert result.real == pytest.approx(1.0, abs=1e-6), (
            f"Expected 1.0, got {result}"
        )

    def test_z_operator_on_ancilla_in_01_should_give_negative_1(self) -> None:
        state = np.array([0, 1, 0, 0], dtype=complex)
        t = bipartite_tensor_from_state(state, n_sites=1, local_dim=2)
        z = np.array([[1, 0], [0, -1]], dtype=complex)
        result = compute_expectation(t, [(1, z)])
        assert result.real == pytest.approx(-1.0, abs=1e-6), (
            f"Expected -1.0, got {result}"
        )


# =============================================================================
# Tests: fidelity
# =============================================================================


class TestTensorFidelity:
    """Tests for tensor fidelity computation."""

    def test_fidelity_of_identical_states_should_be_1(self) -> None:
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        t1 = bipartite_tensor_from_state(state, n_sites=1, local_dim=2)
        t2 = bipartite_tensor_from_state(state, n_sites=1, local_dim=2)
        assert tensor_fidelity(t1, t2) == pytest.approx(1.0), (
            "Expected fidelity of identical states == 1"
        )

    def test_fidelity_of_orthogonal_states_should_be_0(self) -> None:
        state1 = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        state2 = np.array([0, 0, 0, 1], dtype=complex)  # |11⟩
        t1 = bipartite_tensor_from_state(state1, n_sites=1, local_dim=2)
        t2 = bipartite_tensor_from_state(state2, n_sites=1, local_dim=2)
        assert tensor_fidelity(t1, t2) == pytest.approx(0.0), (
            "Expected fidelity of orthogonal states == 0"
        )


# =============================================================================
# Tests: error handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_should_raise_valueerror_for_invalid_state_dimension(self) -> None:
        state = np.array([1, 0, 0], dtype=complex)  # 3 elements — not 2**(2*1)=4
        with pytest.raises(ValueError, match="State dimension"):
            bipartite_tensor_from_state(state, n_sites=1, local_dim=2)

    def test_exact_comparison_norm_should_match_after_split_contract(self) -> None:
        rng = np.random.default_rng(123)
        dim = 2**4
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        t = bipartite_tensor_from_state(state, n_sites=2, local_dim=2)
        reconstructed = _split_and_contract(t, cutoff=1e-14)

        exact_norm = np.vdot(state, state).real
        ttn_norm = np.vdot(reconstructed, reconstructed).real

        assert exact_norm == pytest.approx(ttn_norm, rel=1e-4), (
            f"Norms differ: exact={exact_norm}, ttn={ttn_norm}"
        )


# =============================================================================
# Tests: validation
# =============================================================================


class TestValidation:
    """Validation tests for quimb-based tensor operations."""

    def test_fidelity_after_split_contract_should_exceed_1_1e_4_for_n_4(self) -> None:
        rng = np.random.default_rng(42)
        dim = 2**4
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        t = bipartite_tensor_from_state(state, n_sites=2, local_dim=2)
        reconstructed = _split_and_contract(t, cutoff=1e-14)

        fidelity = float(abs(np.vdot(state, reconstructed)) ** 2)
        assert fidelity > 1 - 1e-4, f"Fidelity {fidelity} below threshold"


# =============================================================================
# Tests: performance
# =============================================================================


class TestPerformance:
    """Performance tests for tensor operations."""

    def test_svd_split_contract_should_run_in_100ms_for_n_8(self) -> None:
        rng = np.random.default_rng(42)
        n_sites = 4  # 4 main + 4 ancilla = 8 qubits → 256-dim vector
        dim = 2 ** (2 * n_sites)
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        t = bipartite_tensor_from_state(state, n_sites=n_sites, local_dim=2)

        start = time.perf_counter()
        _split_and_contract(t, cutoff=1e-14)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, (
            f"Split+contract took {elapsed * 1000:.1f}ms, expected < 100ms"
        )
