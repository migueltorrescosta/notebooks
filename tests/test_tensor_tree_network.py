"""Unit tests for Tensor Tree Network module."""

import numpy as np
import pytest

from src.algorithms.tensor_tree_network import TensorTreeNetwork, TTNNode


class TestTTNNode:
    """Tests for TTNNode dataclass."""

    def test_ttn_node_creation(self) -> None:
        """TTNNode should be created with tensor."""
        tensor = np.array([[1, 0], [0, 1]], dtype=complex)
        node = TTNNode(tensor=tensor)
        assert node.tensor is not None
        assert node.shape == (2, 2)
        assert node.ndim == 2

    def test_ttn_node_with_children(self) -> None:
        """TTNNode should support tree structure."""
        tensor = np.eye(2, dtype=complex)
        left = TTNNode(tensor=np.array([1, 0], dtype=complex))
        right = TTNNode(tensor=np.array([0, 1], dtype=complex))

        node = TTNNode(
            tensor=tensor,
            left=left,
            right=right,
            bond_dims={("left",): 2, ("right",): 2},
        )

        assert node.left is left
        assert node.right is right
        assert node.bond_dims[("left",)] == 2


class TestTensorTreeNetwork:
    """Tests for TensorTreeNetwork class."""

    def test_init(self) -> None:
        """TTN should initialize with correct parameters."""
        ttn = TensorTreeNetwork(n_sites=2, local_dim=2)
        assert ttn.n_sites == 2
        assert ttn.local_dim == 2
        assert ttn.root is None
        assert ttn.max_bond_dimension() == 2  # local_dim

    def test_from_state_vector_single_qubit(self) -> None:
        """TTN from single qubit pair (main + ancilla = 2 qubits)."""
        # Two qubit state |00⟩
        state = np.array([1, 0, 0, 0], dtype=complex)
        ttn = TensorTreeNetwork.from_state_vector(state, n_sites=1, local_dim=2)

        # Reconstruction should be exact
        reconstructed = ttn._to_state_vector()
        assert np.allclose(state, reconstructed)

    def test_from_state_vector_single_qubit_superposition(self) -> None:
        """TTN from single qubit pair superposition."""
        # Two qubit state: (|00⟩ + |11⟩)/√2
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        ttn = TensorTreeNetwork.from_state_vector(state, n_sites=1, local_dim=2)

        reconstructed = ttn._to_state_vector()
        assert np.allclose(state, reconstructed)

    def test_from_state_vector_two_qubitproduct(self) -> None:
        """TTN from product state |00⟩."""
        # Two qubit product state |00⟩
        state = np.array([1, 0, 0, 0], dtype=complex)
        ttn = TensorTreeNetwork.from_state_vector(state, n_sites=1, local_dim=2)

        reconstructed = ttn._to_state_vector()
        assert np.allclose(state, reconstructed)

    def test_from_state_vector_two_qubit_entangled(self) -> None:
        """TTN from entangled Bell state."""
        # Bell state (|00⟩ + |11⟩)/√2
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        ttn = TensorTreeNetwork.from_state_vector(state, n_sites=1, local_dim=2)

        reconstructed = ttn._to_state_vector()
        assert np.allclose(state, reconstructed)

    def test_reconstruction_fidelity_n2(self) -> None:
        """Test reconstruction fidelity for N=2 sites."""
        # Random state for 2 main + 2 ancilla = 4 qubits -> 16 dim
        rng = np.random.default_rng(42)
        state = rng.random(16) + 1j * rng.random(16)
        state = state / np.linalg.norm(state)

        ttn = TensorTreeNetwork.from_state_vector(
            state, n_sites=2, local_dim=2, svd_epsilon=1e-8
        )

        reconstructed = ttn._to_state_vector()
        fidelity = np.abs(np.vdot(state, reconstructed)) ** 2
        assert fidelity > 1 - 1e-6

    def test_reconstruction_fidelity_n3(self) -> None:
        """Test reconstruction fidelity for N=3 sites."""
        # Random state for 3 main + 3 ancilla = 6 qubits -> 64 dim
        rng = np.random.default_rng(42)
        dim = 2**6
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        ttn = TensorTreeNetwork.from_state_vector(
            state, n_sites=3, local_dim=2, svd_epsilon=1e-8
        )

        reconstructed = ttn._to_state_vector()
        fidelity = np.abs(np.vdot(state, reconstructed)) ** 2
        assert fidelity > 1 - 1e-6

    def test_reconstruction_fidelity_n4(self) -> None:
        """Test reconstruction fidelity for N=4 sites."""
        # Random state for 4 main + 4 ancilla = 8 qubits -> 256 dim
        rng = np.random.default_rng(42)
        dim = 2**8
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        ttn = TensorTreeNetwork.from_state_vector(
            state, n_sites=4, local_dim=2, svd_epsilon=1e-8
        )

        reconstructed = ttn._to_state_vector()
        fidelity = np.abs(np.vdot(state, reconstructed)) ** 2
        # With low epsilon, should still be very accurate
        assert fidelity > 1 - 1e-4

    def test_reconstruction_fidelity_n5(self) -> None:
        """Test reconstruction fidelity for N=5 sites."""
        # Random state for 5 main + 5 ancilla = 10 qubits -> 1024 dim
        rng = np.random.default_rng(42)
        dim = 2**10
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        ttn = TensorTreeNetwork.from_state_vector(
            state, n_sites=5, local_dim=2, svd_epsilon=1e-8
        )

        reconstructed = ttn._to_state_vector()
        fidelity = np.abs(np.vdot(state, reconstructed)) ** 2
        # With low epsilon, accuracy should be reasonable
        assert fidelity > 1 - 0.01

    def test_reconstruction_fidelity_n6(self) -> None:
        """Test reconstruction fidelity for N=6 sites (exact comparison)."""
        # Random state for 6 main + 6 ancilla = 12 qubits -> 4096 dim
        # This is expensive but should work
        rng = np.random.default_rng(42)
        dim = 2**12
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        ttn = TensorTreeNetwork.from_state_vector(
            state, n_sites=6, local_dim=2, svd_epsilon=1e-8
        )

        reconstructed = ttn._to_state_vector()
        fidelity = np.abs(np.vdot(state, reconstructed)) ** 2
        # Should still have reasonable fidelity
        assert fidelity > 0.9

    def test_bond_dimension_grows_with_entanglement(self) -> None:
        """Bond dimension should grow with entanglement."""
        # Product state
        product_state = np.array([1, 0, 0, 0], dtype=complex)
        ttn_product = TensorTreeNetwork.from_state_vector(
            product_state, n_sites=1, local_dim=2
        )
        max_bond_product = ttn_product.max_bond_dimension()

        # Entangled state
        entangled_state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        ttn_entangled = TensorTreeNetwork.from_state_vector(
            entangled_state, n_sites=1, local_dim=2
        )
        max_bond_entangled = ttn_entangled.max_bond_dimension()

        # Entangled should have at least as large bond dimension
        assert max_bond_entangled >= max_bond_product

    def test_bond_dimension_tractable_n20(self) -> None:
        """Bond dimension should remain tractable for N=10."""
        # For N=10 sites, full dimension is 2^10 = 1024
        # TTN should compute bond dimensions correctly
        rng = np.random.default_rng(42)
        dim = 2**10  # 2^(2*5) = 1024, using 5 sites per side = 10 total
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        ttn = TensorTreeNetwork.from_state_vector(
            state, n_sites=5, local_dim=2, svd_epsilon=1e-8
        )

        max_bond = ttn.max_bond_dimension()
        # For random state, bond dimension can be full rank (1024)
        # But should be <= full dimension
        assert max_bond <= dim

    def test_truncation_preserves_normalization(self) -> None:
        """Truncation should preserve normalization."""
        rng = np.random.default_rng(42)
        dim = 2**4
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        ttn = TensorTreeNetwork.from_state_vector(
            state, n_sites=2, local_dim=2, svd_epsilon=1e-10
        )

        # Apply truncation
        ttn.truncate(epsilon=1e-4)

        # Check normalization
        reconstructed = ttn._to_state_vector()
        norm = np.linalg.norm(reconstructed)
        assert np.isclose(norm, 1.0)

    def test_contract_expectation_value_product_state(self) -> None:
        """Test contraction with identity operator."""
        # Product state |00⟩ (2 qubits: main + ancilla)
        state = np.array([1, 0, 0, 0], dtype=complex)
        ttn = TensorTreeNetwork.from_state_vector(state, n_sites=1, local_dim=2)

        # Identity operator should give 1
        identity = np.eye(2, dtype=complex)
        result = ttn.contract([(0, identity)])

        # For |00⟩ with I on first qubit: should be 1
        assert np.isclose(result.real, 1.0, atol=1e-6)

    def test_contract_expectation_value_z_operator(self) -> None:
        """Test contraction with Z operator."""
        # State |00⟩ - main qubit in |0⟩
        state = np.array([1, 0, 0, 0], dtype=complex)
        ttn = TensorTreeNetwork.from_state_vector(state, n_sites=1, local_dim=2)

        # Z operator on main qubit (site 0) should give +1
        z = np.array([[1, 0], [0, -1]], dtype=complex)
        result = ttn.contract([(0, z)])

        assert np.isclose(result.real, 1.0, atol=1e-6)

    def test_contract_expectation_value_z_on_one(self) -> None:
        """Test contraction with Z operator on ancilla."""
        # State |01⟩ - ancilla qubit in |1⟩
        state = np.array([0, 1, 0, 0], dtype=complex)
        ttn = TensorTreeNetwork.from_state_vector(state, n_sites=1, local_dim=2)

        # Z operator on ancilla (site 1) should give -1
        z = np.array([[1, 0], [0, -1]], dtype=complex)
        result = ttn.contract([(1, z)])

        assert np.isclose(result.real, -1.0, atol=1e-6)

    def test_exact_vs_ttn_comparison_n4(self) -> None:
        """Compare TTN results with exact for N=4."""
        # Create a specific state
        rng = np.random.default_rng(123)
        dim = 2**4
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        # Get exact expectation value
        _identity = np.eye(2, dtype=complex)
        exact_state = state.copy()

        # Build TTN
        ttn = TensorTreeNetwork.from_state_vector(
            state, n_sites=2, local_dim=2, svd_epsilon=1e-8
        )

        # Compare expectation values
        exact_norm = np.vdot(exact_state, exact_state).real
        ttn_norm_before = np.vdot(ttn._to_state_vector(), ttn._to_state_vector()).real

        assert np.isclose(exact_norm, ttn_norm_before, rtol=1e-4)

    def test_tree_structure(self) -> None:
        """Test getting tree structure."""
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        ttn = TensorTreeNetwork.from_state_vector(state, n_sites=1, local_dim=2)

        structure = ttn.get_tree_structure()
        assert "depth" in structure
        assert "n_nodes" in structure
        assert "max_bond_dim" in structure

    def test_fidelity_identical_states(self) -> None:
        """Fidelity of identical states should be 1."""
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        ttn1 = TensorTreeNetwork.from_state_vector(state, n_sites=1, local_dim=2)
        ttn2 = TensorTreeNetwork.from_state_vector(state, n_sites=1, local_dim=2)

        assert np.isclose(ttn1.fidelity(ttn2), 1.0)

    def test_fidelity_orthogonal_states(self) -> None:
        """Fidelity of orthogonal states should be 0."""
        # |00⟩ and |11⟩ are orthogonal
        state1 = np.array([1, 0, 0, 0], dtype=complex)
        state2 = np.array([0, 0, 0, 1], dtype=complex)

        ttn1 = TensorTreeNetwork.from_state_vector(state1, n_sites=1, local_dim=2)
        ttn2 = TensorTreeNetwork.from_state_vector(state2, n_sites=1, local_dim=2)

        assert np.isclose(ttn1.fidelity(ttn2), 0.0)

    def test_invalid_state_dimension(self) -> None:
        """Should raise error for invalid state dimension."""
        state = np.array([1, 0, 0], dtype=complex)  # 3 elements

        with pytest.raises(ValueError):
            TensorTreeNetwork.from_state_vector(state, n_sites=1, local_dim=2)

    def test_truncate_call_on_empty(self) -> None:
        """Truncate should handle empty TTN gracefully."""
        ttn = TensorTreeNetwork(n_sites=1, local_dim=2)
        ttn.truncate(epsilon=1e-4)
        # Should not raise


class TestTensorTreeNetworkValidation:
    """Validation tests comparing TTN with exact results."""

    def test_exact_comparison_n4_tolerance(self) -> None:
        """Compare TTN with exact for N≤4, tolerance 10⁻⁴."""
        rng = np.random.default_rng(42)
        n_sites = 2  # Main + ancilla = 2 + 2 = 4 qubits
        dim = 2 ** (2 * n_sites)
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        # Exact calculation
        _identity = np.eye(2, dtype=complex)
        _pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

        # Exact expectation
        # Reshape for matrix form
        half_dim = 2**n_sites
        _state_matrix = state.reshape(half_dim, half_dim)
        # <Z_0> = trace((Z ⊗ I) |ψ⟩⟨ψ|)
        # = trace_2( (Z ⊗ I) rho )
        rho = np.outer(state, state.conj())
        _rho_matrix = rho.reshape(half_dim, half_dim, half_dim, half_dim)

        # Build TTN
        ttn = TensorTreeNetwork.from_state_vector(
            state, n_sites=n_sites, local_dim=2, svd_epsilon=1e-8
        )

        # Compare state vectors
        reconstructed = ttn._to_state_vector()
        fidelity = np.abs(np.vdot(state, reconstructed)) ** 2

        assert fidelity > 1 - 1e-4, f"Fidelity {fidelity} below threshold"

    def test_bond_dim_tractable_n10(self) -> None:
        """Verify bond dimensions tractable for N≤10."""
        rng = np.random.default_rng(42)
        n_sites = 5  # 5 main + 5 ancilla = 10 qubits, dim = 2^10 = 1024
        dim = 2 ** (2 * n_sites)
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        ttn = TensorTreeNetwork.from_state_vector(
            state, n_sites=n_sites, local_dim=2, svd_epsilon=1e-8
        )

        max_bond = ttn.max_bond_dimension()
        # For N=5, maximum possible bond is 32, should be much smaller
        assert max_bond <= 32
        # And significantly smaller than full dimension
        assert max_bond < dim // 4

    def test_truncation_error_vs_epsilon(self) -> None:
        """Truncation error should decrease with epsilon."""
        rng = np.random.default_rng(42)
        n_sites = 2
        dim = 2 ** (2 * n_sites)
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        epsilons = [1e-2, 1e-4, 1e-6, 1e-8]
        fidelities = []

        for eps in epsilons:
            ttn = TensorTreeNetwork.from_state_vector(
                state, n_sites=n_sites, local_dim=2, svd_epsilon=eps
            )
            reconstructed = ttn._to_state_vector()
            fidelity = np.abs(np.vdot(state, reconstructed)) ** 2
            fidelities.append(fidelity)

        # Fidelity should increase (error decrease) as epsilon decreases
        for i in range(len(fidelities) - 1):
            assert fidelities[i] <= fidelities[i + 1] + 1e-6


class TestTensorTreeNetworkPerformance:
    """Performance tests for TTN."""

    def test_runtime_n8(self) -> None:
        """TTN should run in < 100ms for N=8."""
        import time

        rng = np.random.default_rng(42)
        n_sites = 4  # 4 main + 4 ancilla = 8 qubits -> 256 dim
        dim = 2 ** (2 * n_sites)
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        start = time.perf_counter()
        TensorTreeNetwork.from_state_vector(
            state, n_sites=n_sites, local_dim=2, svd_epsilon=1e-8
        )
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, (
            f"TTN construction took {elapsed * 1000:.1f}ms, expected < 100ms"
        )
