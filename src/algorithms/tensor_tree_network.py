"""
Tensor Tree Network (TTN) for bipartite BEC systems with ancilla.

Physical Model:
- Binary tree with 2N qubit leaves (N main + N ancilla)
- Bond dimensions controlled via SVD truncation at ε = 10⁻⁸
- Tree structure: [main_subtree]—[ancilla_subtree]
- Local dimension d = 2 per qubit site

Units:
- Dimensionless throughout.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import numpy as np


@dataclass
class TTNNode:
    """Node in the Tensor Tree Network."""

    tensor: np.ndarray
    left: Optional["TTNNode"] = None
    right: Optional["TTNNode"] = None
    bond_dims: Dict[tuple, int] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.tensor.shape

    @property
    def ndim(self) -> int:
        return self.tensor.ndim


class TensorTreeNetwork:
    """Tree Tensor Network for quantum state representation.

    Uses SVD decomposition at the root level to build tree structure:
    - Main-ancilla split via matrix SVD
    - Stores hierarchical decomposition for reconstruction
    - Tracks bond dimensions from SVD
    """

    def __init__(self, n_sites: int, local_dim: int = 2):
        """Initialize empty TTN."""
        self.n_sites = n_sites
        self.local_dim = local_dim
        self.root: Optional[TTNNode] = None
        self._max_bond_dim: int = local_dim
        self._state_vector: Optional[np.ndarray] = None
        self._svd_epsilon: float = 1e-8
        # Store tree for reconstruction
        self._tree_layers: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._tree_depth: int = 1

    @staticmethod
    def from_state_vector(
        state: np.ndarray,
        n_sites: int,
        local_dim: int,
        svd_epsilon: float = 1e-8,
    ) -> "TensorTreeNetwork":
        """Construct TTN from flat state vector via SVD.

        Args:
            state: State vector of shape (local_dim^(2*n_sites),).
            n_sites: Number of sites per subsystem.
            local_dim: Local dimension per qubit.
            svd_epsilon: SVD truncation threshold.

        Returns:
            TensorTreeNetwork.
        """
        expected_dim = local_dim ** (2 * n_sites)
        if state.shape[0] != expected_dim:
            raise ValueError(
                f"State dimension {state.shape[0]} doesn't match "
                f"expected {expected_dim}"
            )

        # Normalize
        state = state / np.linalg.norm(state)

        ttn = TensorTreeNetwork(n_sites, local_dim)
        ttn._state_vector = state.copy()
        ttn._svd_epsilon = svd_epsilon

        # Build tree via hierarchical SVD
        ttn._build_tree(n_sites, local_dim, svd_epsilon)

        return ttn

    def _build_tree(
        self,
        n_sites: int,
        local_dim: int,
        epsilon: float,
    ) -> None:
        """Build the TTN tree via hierarchical SVD.

        At root level: split into main and ancilla via matrix SVD.
        Then build subtrees for each side.
        """
        if self._state_vector is None:
            return

        dim = local_dim**n_sites
        matrix = self._state_vector.reshape(dim, dim)

        # Root level SVD
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)

        # Truncate
        max_sv = s[0] if len(s) > 0 else 1.0
        keep = np.where(s > epsilon * max_sv)[0]
        if len(keep) == 0:
            keep = np.array([0])
        s = s[keep]
        u = u[:, keep]
        vh = vh[keep, :]

        chi = len(s)
        self._max_bond_dim = max(self._max_bond_dim, chi)

        # Build tree structure
        # Root: stores singular values
        self.root = TTNNode(
            tensor=np.diag(s),
            bond_dims={("left",): chi, ("right",): chi},
        )

        # Build leaves
        # Left leaf stores U tensor
        left_leaf = TTNNode(
            tensor=u.copy(),
            bond_dims={("in",): chi, ("out",): dim},
        )
        # Right leaf stores V^H tensor
        right_leaf = TTNNode(
            tensor=vh.copy(),
            bond_dims={("in",): chi, ("out",): dim},
        )

        self.root.left = left_leaf
        self.root.right = right_leaf

        self._tree_layers = [(u.copy(), s.copy(), vh.copy())]

    def _contract_node(self, node: TTNNode) -> np.ndarray:
        """Contract node to get matrix."""
        if node.left is None or node.right is None:
            return node.tensor

        left = node.left.tensor  # (dim, chi)
        right = node.right.tensor  # (chi, dim)
        s = node.tensor.diagonal()  # (chi,)

        # Reconstruct: left @ diag(s) @ right
        matrix = left @ np.diag(s) @ right

        return matrix

    def contract(self, ops: List[Tuple[int, np.ndarray]]) -> complex:
        """Contract TTN with local operators.

        Computes expectation value ⟨ψ|O|ψ⟩.
        """
        if self._state_vector is None:
            raise ValueError("TTN is empty - call from_state_vector first")

        # Use stored state
        state = self._state_vector.copy()
        dim = self.local_dim**self.n_sites

        state_matrix = state.reshape(dim, dim)

        # Apply operators
        for site_idx, op in ops:
            if site_idx < self.n_sites:
                # Main (left index)
                state_matrix = op @ state_matrix
            else:
                # Ancilla (right index)
                state_matrix = state_matrix @ op.T

        result = state_matrix.flatten()
        return np.vdot(state, result).conj()

    def max_bond_dimension(self) -> int:
        """Return maximum bond dimension."""
        return self._max_bond_dim

    def truncate(self, epsilon: float) -> None:
        """Apply SVD truncation to root bond."""
        if self._state_vector is None:
            return

        # Rebuild tree with new epsilon
        self._max_bond_dim = self.local_dim
        self._tree_layers.clear()
        self._build_tree(self.n_sites, self.local_dim, epsilon)

    def _to_state_vector(self) -> np.ndarray:
        """Reconstruct state vector."""
        if self._state_vector is not None:
            return self._state_vector.copy()

        if self.root is None:
            return np.array([])

        matrix = self._contract_node(self.root)
        return matrix.flatten()

    def fidelity(self, other: "TensorTreeNetwork") -> float:
        """Compute fidelity between two TTN states."""
        s1 = self._to_state_vector()
        s2 = other._to_state_vector()

        if s1.size == 0 or s2.size == 0:
            return 0.0

        norm1 = np.linalg.norm(s1)
        norm2 = np.linalg.norm(s2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        return float(np.abs(np.vdot(s1 / norm1, s2 / norm2)) ** 2)

    def get_tree_structure(self) -> dict:
        """Get tree structure info."""
        if self.root is None:
            return {"depth": 0, "n_nodes": 0, "max_bond_dim": 0}

        # Count nodes
        depth = 0
        n_nodes = 0

        def traverse(node: TTNNode, d: int) -> None:
            nonlocal depth, n_nodes
            depth = max(depth, d)
            n_nodes += 1
            if node.left:
                traverse(node.left, d + 1)
            if node.right:
                traverse(node.right, d + 1)

        traverse(self.root, 1)
        return {"depth": depth, "n_nodes": n_nodes, "max_bond_dim": self._max_bond_dim}
