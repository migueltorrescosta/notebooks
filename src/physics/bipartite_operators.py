"""
Shared operators for bipartite system--ancilla Hilbert spaces.

Provides canonical operator construction in a tensor-product space
:math:`\\mathcal{H}_S \\otimes \\mathcal{H}_A` where both subsystems
are multi-particle two-mode bosonic systems in the Dicke basis.

Used by reports that vary the system and ancilla particle counts
independently (e.g., #20260619, #20260620).
"""

from __future__ import annotations

import math

import numpy as np

from src.physics.beam_splitter import bs_dicke
from src.physics.dicke_basis import jx_operator, jy_operator, jz_operator
from src.utils.enums import OperatorBasis


def build_operators(N_sys: int, N_anc: int) -> dict[str, np.ndarray]:
    """Build operators in the :math:`(N_\\text{sys}+1)(N_\\text{anc}+1)`-dimensional
    total Hilbert space.

    Total space: :math:`\\mathcal{H}_S \\otimes \\mathcal{H}_A` with dimension
    :math:`(N_\\text{sys}+1)(N_\\text{anc}+1)`. The system has
    :math:`J_S = N_\\text{sys}/2` (dim :math:`N_\\text{sys}+1`), the ancilla has
    :math:`J_A = N_\\text{anc}/2` (dim :math:`N_\\text{anc}+1`).

    Basis ordering: :math:`\\{|m_S\\rangle_S \\otimes |m_A\\rangle_A\\}`
    where :math:`m_S` and :math:`m_A` each descend from :math:`+J` to :math:`-J`.
    Basis index: :math:`i = m_S^{\\text{idx}} \\cdot (N_\\text{anc}+1)
    + m_A^{\\text{idx}}`.

    Args:
        N_sys: Number of system particles (:math:`N_\\text{sys} \\ge 1`).
        N_anc: Number of ancilla particles (:math:`N_\\text{anc} \\ge 1`).

    Returns:
        Dict with keys ``'Jz_S'``, ``'Jx_S'``, ``'Jy_S'``, ``'Jz_A'``,
        ``'Jx_A'``, ``'Jy_A'``, ``'I_S'``, ``'I_A'``, ``'I_full'``.

    Raises:
        ValueError: If *N_sys* or *N_anc* < 1.
    """
    if N_sys < 1:
        raise ValueError(f"N_sys must be >= 1, got {N_sys}")
    if N_anc < 1:
        raise ValueError(f"N_anc must be >= 1, got {N_anc}")

    d_sys = N_sys + 1
    d_anc = N_anc + 1
    d_tot = d_sys * d_anc

    # System operators in Dicke basis
    Jz_sys = jz_operator(N_sys, basis=OperatorBasis.DICKE)
    Jx_sys = jx_operator(N_sys, basis=OperatorBasis.DICKE)
    Jy_sys = jy_operator(N_sys, basis=OperatorBasis.DICKE)

    # Ancilla operators in Dicke basis
    Jz_anc = jz_operator(N_anc, basis=OperatorBasis.DICKE)
    Jx_anc = jx_operator(N_anc, basis=OperatorBasis.DICKE)
    Jy_anc = jy_operator(N_anc, basis=OperatorBasis.DICKE)

    I_S = np.eye(d_sys, dtype=complex)
    I_A = np.eye(d_anc, dtype=complex)

    # Embed via Kronecker products
    ops: dict[str, np.ndarray] = {
        "Jz_S": np.kron(Jz_sys, I_A).astype(complex),
        "Jx_S": np.kron(Jx_sys, I_A).astype(complex),
        "Jy_S": np.kron(Jy_sys, I_A).astype(complex),
        "Jz_A": np.kron(I_S, Jz_anc).astype(complex),
        "Jx_A": np.kron(I_S, Jx_anc).astype(complex),
        "Jy_A": np.kron(I_S, Jy_anc).astype(complex),
        "I_S": I_S,
        "I_A": I_A,
        "I_full": np.eye(d_tot, dtype=complex),
    }

    # Validate dimensions
    for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
        assert ops[key].shape == (d_tot, d_tot), (
            f"{key} has shape {ops[key].shape}, expected ({d_tot}, {d_tot})"
        )

    # Validate Hermiticity
    for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
        assert np.allclose(ops[key], ops[key].conj().T, atol=1e-12), (
            f"{key} is not Hermitian for N_sys={N_sys}, N_anc={N_anc}"
        )

    # Validate commutation: [J_z^S, J_x^S] = i J_y^S
    comm_s = ops["Jz_S"] @ ops["Jx_S"] - ops["Jx_S"] @ ops["Jz_S"]
    assert np.allclose(comm_s, 1j * ops["Jy_S"], atol=1e-10), (
        f"[J_z^S, J_x^S] = i J_y^S violated for N_sys={N_sys}"
    )

    # Validate commutation: [J_z^A, J_x^A] = i J_y^A
    comm_a = ops["Jz_A"] @ ops["Jx_A"] - ops["Jx_A"] @ ops["Jz_A"]
    assert np.allclose(comm_a, 1j * ops["Jy_A"], atol=1e-10), (
        f"[J_z^A, J_x^A] = i J_y^A violated for N_anc={N_anc}"
    )

    return ops


def build_system_only_bs_unitary(
    N_sys: int,
    N_anc: int,
    T_bs: float = math.pi / 2.0,
) -> np.ndarray:
    """System-only beam-splitter unitary in the total Hilbert space.

    :math:`U_{\\text{BS}}^{(S)} = \\exp(-i T_{\\text{bs}} J_x(N_\\text{sys}))
    \\otimes \\mathbb{1}_{N_\\text{anc}+1}`

    Args:
        N_sys: Number of system particles.
        N_anc: Number of ancilla particles.
        T_bs: Beam-splitter duration (default :math:`\\pi/2` for 50/50).

    Returns:
        :math:`(N_\\text{sys}+1)(N_\\text{anc}+1) \\times
        (N_\\text{sys}+1)(N_\\text{anc}+1)` unitary matrix.
    """
    d_tot = (N_sys + 1) * (N_anc + 1)
    bs_sys = bs_dicke(N_sys, T_bs)
    I_A = np.eye(N_anc + 1, dtype=complex)
    U = np.kron(bs_sys, I_A).astype(complex)
    assert np.allclose(U @ U.conj().T, np.eye(d_tot, dtype=complex), atol=1e-12), (
        f"BS unitary not unitary for N_sys={N_sys}, N_anc={N_anc}, T_bs={T_bs}"
    )
    return U


def build_iszz_interaction(
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the Ising-type system--ancilla interaction.

    :math:`H_{\\text{int}} = a_{zz} J_z^S \\otimes J_z^A`

    Args:
        a_zz: Interaction coupling coefficient.
        ops: Operators from :func:`build_operators`.

    Returns:
        Hermitian matrix representing the interaction.
    """
    d_tot = ops["Jz_A"].shape[0]
    H = np.zeros((d_tot, d_tot), dtype=complex)
    if a_zz != 0.0:
        H += a_zz * (ops["Jz_S"] @ ops["Jz_A"])
    H = 0.5 * (H + H.conj().T)
    assert np.allclose(H, H.conj().T, atol=1e-12), "Ising interaction not Hermitian"
    return H
