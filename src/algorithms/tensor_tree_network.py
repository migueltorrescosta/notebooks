"""
Tensor Network utilities for bipartite BEC systems with ancilla.

This module replaces the deprecated ``TensorTreeNetwork`` / ``TTNNode`` classes
with direct use of **quimb** (``quimb.tensor.Tensor``,
``quimb.tensor.TensorNetwork``).  All functions accept and return quimb types
directly — no custom wrapper classes.

State representation:
    A bipartite state is stored as a ``qtn.Tensor`` with two indices
    ``('main', 'ancilla')``, each of dimension ``local_dim ** n_sites``.
    This is the same representation used by the TDVP module.

    The bipartite structure naturally captures the **main** and **ancilla**
    subsystems of the Mach–Zehnder interferometer model.

Units:
    Dimensionless throughout (ℏ = 1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import quimb.tensor as qtn

if TYPE_CHECKING:
    from collections.abc import Sequence


def bipartite_tensor_from_state(
    state: np.ndarray,
    n_sites: int,
    local_dim: int = 2,
    normalize: bool = True,
) -> qtn.Tensor:
    """Create a quimb ``Tensor`` from a bipartite state vector.

    The state is reshaped to ``(local_dim**n_sites, local_dim**n_sites)``
    and stored as a 2-index tensor with indices ``('main', 'ancilla')``.

    This function validates dimensions, normalises the input, and manages
    index naming — it is **not** a thin wrapper around a quimb constructor.

    Args:
        state: Flat state vector of shape ``(local_dim ** (2 * n_sites),)``.
        n_sites: Number of sites per subsystem.
        local_dim: Local dimension per qubit (default 2).
        normalize: Whether to normalise *state* before storing (default True).

    Returns:
        ``qtn.Tensor`` with shape ``(D, D)`` where ``D = local_dim ** n_sites``
        and indices ``('main', 'ancilla')``.

    Raises:
        ValueError: If *state* dimension does not match ``local_dim ** (2 * n_sites)``.
    """
    expected_dim = local_dim ** (2 * n_sites)
    if state.shape[0] != expected_dim:
        raise ValueError(
            f"State dimension {state.shape[0]} doesn't match "
            f"expected {expected_dim} "
            f"(= {local_dim} ** (2 * {n_sites}))",
        )

    if normalize:
        state = state / np.linalg.norm(state)

    dim = local_dim**n_sites
    matrix = state.reshape(dim, dim).astype(complex)

    return qtn.Tensor(np.asarray(matrix), inds=("main", "ancilla"))


def get_state_vector(tensor: qtn.Tensor) -> np.ndarray:
    """Extract the flat state vector from a bipartite tensor.

    Args:
        tensor: A ``qtn.Tensor`` (typically with indices ``('main', 'ancilla')``).

    Returns:
        1-D complex array of shape ``(D * D,)`` where ``D`` is the dimension
        of each index.
    """
    return tensor.data.flatten()


def compute_expectation(
    tensor: qtn.Tensor,
    operators: Sequence[tuple[int, np.ndarray]],
) -> complex:
    """Compute expectation value of local operators on a bipartite tensor.

    ``⟨ψ|O₀ ⊗ O₁ ⊗ …|ψ⟩`` where site index 0 maps to the ``'main'`` index
    and site index 1 (and above) maps to ``'ancilla'``.

    This is **not** a thin wrapper — it maps physical site indices to quimb
    index names and applies each operator as a gate on the correct tensor leg.

    Args:
        tensor: The bipartite state tensor.
        operators: Sequence of ``(site_index, operator_matrix)`` pairs.

    Returns:
        Complex expectation value.
    """
    state_copy = tensor.copy()
    for site_idx, op in operators:
        which = "main" if site_idx == 0 else "ancilla"
        state_copy = state_copy.gate(op, which)
    return complex(tensor.overlap(state_copy))


def tensor_fidelity(t1: qtn.Tensor, t2: qtn.Tensor) -> float:
    """Compute fidelity :math:`F = |⟨ψ₁|ψ₂⟩|²` between two tensor states.

    Both tensors are normalised before computing the overlap.

    Args:
        t1: First tensor.
        t2: Second tensor.

    Returns:
        Fidelity in :math:`[0, 1]`.
    """
    v1 = t1.data.flatten()
    v2 = t2.data.flatten()

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0

    return float(abs(np.vdot(v1 / n1, v2 / n2)) ** 2)
