"""
Symbolic verification of the Scenario B block-diagonalisation.

Constructs the 4×4 Hamiltonian in the computational basis, transforms to the
Bell basis, extracts the 3×3 block H_3, and verifies:

1. The antisymmetric state |-⟩_m decouples entirely.
2. After shifting by a_zz/4, the middle diagonal element is −a_zz/2 (not zero).
3. H_3' = H_3 − (a_zz/4)𝟙₃ is NOT proportional to ω — eigenvectors depend on ω.
4. The correct characteristic polynomial of H_3' is
       μ³ − (a_zz/2)μ² − ω²r²μ − ω²a_z²a_zz/2 = 0
   which differs from the report's claimed depressed cubic μ³ − a_z²μ + a_z r_⊥² = 0.
5. Special case a_zz = 0: eigenvalues {0, ±ωr}, eigenvectors ω-independent.
"""

from __future__ import annotations

import sympy as sp


def build_hamiltonian_4x4() -> sp.Matrix:
    """Build the full 4×4 Hamiltonian in computational basis {|00⟩,|01⟩,|10⟩,|11⟩}."""
    ax, ay, az, azz, omega = sp.symbols("a_x a_y a_z a_zz omega", real=True)

    # Pauli matrices (spin-1/2: J_k = sigma_k / 2)
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])
    I2 = sp.eye(2)

    # System operators: J_k^S = (sigma_k / 2) ⊗ I
    JxS = sp.Rational(1, 2) * sp.kronecker_product(sx, I2)
    JyS = sp.Rational(1, 2) * sp.kronecker_product(sy, I2)
    JzS = sp.Rational(1, 2) * sp.kronecker_product(sz, I2)

    # Ancilla operators: J_k^A = I ⊗ (sigma_k / 2)
    JxA = sp.Rational(1, 2) * sp.kronecker_product(I2, sx)
    JyA = sp.Rational(1, 2) * sp.kronecker_product(I2, sy)
    JzA = sp.Rational(1, 2) * sp.kronecker_product(I2, sz)

    # Full Hamiltonian: ω-modulated drive on both subsystems + Ising interaction
    H = (
        omega * (ax * JxS + ay * JyS + az * JzS)
        + omega * (ax * JxA + ay * JyA + az * JzA)
        + azz * JzS * JzA
    )
    return H


def bell_transformation() -> sp.Matrix:
    """Transformation matrix P: computational → Bell basis.

    Columns are {|00⟩, |+⟩_m, |−⟩_m, |11⟩} expressed in computational basis.
    """
    # |+⟩_m = (|01⟩ + |10⟩)/√2,  |−⟩_m = (|01⟩ − |10⟩)/√2
    return sp.Matrix(
        [
            [1, 0, 0, 0],
            [0, 1 / sp.sqrt(2), 1 / sp.sqrt(2), 0],
            [0, 1 / sp.sqrt(2), -1 / sp.sqrt(2), 0],
            [0, 0, 0, 1],
        ]
    )


def verify_antisymmetric_decoupling(H_bell: sp.Matrix) -> None:
    """Verify |-⟩_m decouples: ⟨−|H|00⟩ = ⟨−|H|11⟩ = 0."""
    # |-⟩_m is index 2 (0-indexed) in the Bell basis
    row_minus = H_bell[2, :]
    assert row_minus[0] == 0, f"⟨−|H|00⟩ = {row_minus[0]}, expected 0"
    assert row_minus[3] == 0, f"⟨−|H|11⟩ = {row_minus[3]}, expected 0"
    col_minus = H_bell[:, 2]
    assert col_minus[0] == 0, f"⟨00|H|−⟩ = {col_minus[0]}, expected 0"
    assert col_minus[3] == 0, f"⟨11|H|−⟩ = {col_minus[3]}, expected 0"
    print("  [PASS] Antisymmetric state |-⟩_m decouples entirely.")


def verify_H3_elements(H_bell: sp.Matrix) -> tuple[sp.Matrix, dict]:
    """Extract H_3 (3×3 block) and verify its matrix elements."""
    ax, ay, az, azz, omega = sp.symbols("a_x a_y a_z a_zz omega", real=True)

    # 3×3 block in {|00⟩, |+⟩_m, |11⟩} (indices 0,1,3 in 4×4)
    H3 = H_bell.extract([0, 1, 3], [0, 1, 3])

    # Expected diagonal
    exp_diag = [
        omega * az + azz / 4,
        -azz / 4,
        -omega * az + azz / 4,
    ]
    for i, (got, exp) in enumerate(zip(H3.diagonal(), exp_diag)):
        diff = sp.simplify(got - exp)
        assert diff == 0, f"H3 diagonal [{i}]: got {got}, expected {exp}, diff={diff}"

    # Expected off-diagonal (adjacent coupling)
    # H3[0,1] = ⟨00|H|+⟩_m and H3[1,2] = ⟨+|H|11⟩ (upper)
    # H3[1,0] = ⟨+|H|00⟩ and H3[2,1] = ⟨11|H|+⟩ (lower)
    exp_01 = omega / sp.sqrt(2) * (ax - sp.I * ay)
    exp_10 = omega / sp.sqrt(2) * (ax + sp.I * ay)
    exp_12 = omega / sp.sqrt(2) * (ax - sp.I * ay)
    exp_21 = omega / sp.sqrt(2) * (ax + sp.I * ay)

    assert sp.simplify(H3[0, 1] - exp_01) == 0, f"H3[0,1]: {H3[0, 1]}"
    assert sp.simplify(H3[1, 0] - exp_10) == 0, f"H3[1,0]: {H3[1, 0]}"
    assert sp.simplify(H3[0, 2]) == 0, f"H3[0,2] should be 0, got {H3[0, 2]}"
    assert sp.simplify(H3[1, 2] - exp_12) == 0, f"H3[1,2]: {H3[1, 2]}"
    assert sp.simplify(H3[2, 1] - exp_21) == 0, f"H3[2,1]: {H3[2, 1]}"

    print("  [PASS] H_3 matrix elements correct.")
    return H3, {"ax": ax, "ay": ay, "az": az, "azz": azz, "omega": omega}


def verify_shift_middle_element(H3: sp.Matrix, symbols: dict) -> None:
    """After shifting by a_zz/4, verify middle element is -a_zz/2 (not zero)."""
    azz = symbols["azz"]
    I3 = sp.eye(3)
    H3_shifted = H3 - sp.Rational(1, 4) * azz * I3

    middle = H3_shifted[1, 1]
    expected = -azz / 2
    diff = sp.simplify(middle - expected)
    assert diff == 0, f"Shifted middle element: got {middle}, expected {expected}"

    # Verify it is NOT zero (the error in the original report)
    assert middle != 0, "Middle element should NOT be zero after shift"
    print(f"  [PASS] After shift by a_zz/4: middle element = {middle} (NOT zero).")


def verify_not_proportional_to_omega(H3: sp.Matrix, symbols: dict) -> None:
    """Verify H_3' is NOT ω·H_0 — the middle element is ω-independent."""
    az, azz, omega = symbols["az"], symbols["azz"], symbols["omega"]
    I3 = sp.eye(3)
    H3_prime = H3 - sp.Rational(1, 4) * azz * I3

    # Check: can H3_prime be written as omega * H0?
    # The (1,1) element is -azz/2, which does NOT contain omega.
    middle = H3_prime[1, 1]
    has_omega = middle.coeff(omega)
    assert has_omega == 0, (
        f"Middle element {middle} should be ω-independent, "
        f"but has ω coefficient {has_omega}"
    )

    # Off-diagonal (0,1) IS proportional to ω
    offdiag = H3_prime[0, 1]
    assert offdiag.coeff(omega) != 0, f"Off-diagonal should depend on ω, got {offdiag}"

    print(
        "  [PASS] H_3' has ω-dependent off-diagonals but ω-independent "
        "middle diagonal → H_3' ≠ ω·H_0."
    )


def verify_eigenvectors_depend_on_omega(symbols: dict) -> None:
    """Verify eigenvectors of H_3' depend on ω for a_zz ≠ 0 (numerical check)."""
    import numpy as np

    az, azz, omega = symbols["az"], symbols["azz"], symbols["omega"]

    I3 = sp.eye(3)
    # Reconstruct H3
    ax_val, ay_val = 2, 3
    az_val, azz_val, omega_val1, omega_val2 = 1, 5, 1, 2

    H3_num = sp.Matrix(
        [
            [
                omega * az + azz / 4,
                omega / sp.sqrt(2) * (ax_val + sp.I * ay_val),
                0,
            ],
            [
                omega / sp.sqrt(2) * (ax_val - sp.I * ay_val),
                -azz / 4,
                omega / sp.sqrt(2) * (ax_val + sp.I * ay_val),
            ],
            [
                0,
                omega / sp.sqrt(2) * (ax_val - sp.I * ay_val),
                -omega * az + azz / 4,
            ],
        ]
    )

    H3p1 = np.array(
        H3_num.subs([(az, az_val), (azz, azz_val), (omega, omega_val1)]),
        dtype=complex,
    )
    H3p2 = np.array(
        H3_num.subs([(az, az_val), (azz, azz_val), (omega, omega_val2)]),
        dtype=complex,
    )
    H3p1 = H3p1 - azz_val / 4 * np.eye(3)
    H3p2 = H3p2 - azz_val / 4 * np.eye(3)

    _, v1 = np.linalg.eigh(H3p1)
    _, v2 = np.linalg.eigh(H3p2)

    # Eigenvectors should differ if they depend on ω
    # Sort eigenvalues to align eigenvectors
    order1 = np.argsort(np.real(np.linalg.eigvalsh(H3p1)))
    order2 = np.argsort(np.real(np.linalg.eigvalsh(H3p2)))
    v1_sorted = v1[:, order1]
    v2_sorted = v2[:, order2]

    # Check if any eigenvector pair has changed
    max_overlap = max(abs(np.vdot(v1_sorted[:, k], v2_sorted[:, k])) for k in range(3))
    assert max_overlap < 0.999, (
        f"Eigenvectors appear ω-independent (max overlap = {max_overlap:.6f}) "
        f"for a_zz = {azz_val} ≠ 0 — this should NOT happen"
    )
    print(
        f"  [PASS] Eigenvectors of H_3' depend on ω for a_zz = {azz_val} ≠ 0 "
        f"(max overlap = {max_overlap:.6f})."
    )


def verify_characteristic_polynomial(H3: sp.Matrix, symbols: dict) -> None:
    """Compute the correct characteristic polynomial and compare with the report's claim."""
    az, azz, omega = symbols["az"], symbols["azz"], symbols["omega"]
    mu = sp.Symbol("mu")
    I3 = sp.eye(3)
    H3_prime = H3 - sp.Rational(1, 4) * azz * I3

    # Correct characteristic polynomial of H_3'
    char_poly_correct = H3_prime.charpoly(mu)
    p_correct = char_poly_correct.as_expr()

    # Expected: μ³ − (a_zz/2)μ² − ω²r²μ − ω²a_z²a_zz/2
    r_sq = symbols["ax"] ** 2 + symbols["ay"] ** 2 + az**2
    p_expected = (
        mu**3 + (azz / 2) * mu**2 - omega**2 * r_sq * mu - omega**2 * az**2 * azz / 2
    )

    diff = sp.simplify(p_correct - p_expected)
    assert diff == 0, (
        f"Characteristic polynomial mismatch!\n"
        f"  Correct:    {p_correct}\n"
        f"  Expected:   {p_expected}\n"
        f"  Difference: {diff}"
    )
    print("  [PASS] Correct characteristic polynomial verified:")
    print("         μ³ + (a_zz/2)μ² − ω²r²μ − ω²a_z²a_zz/2 = 0")

    # Now verify the REPORT'S claimed polynomial is WRONG
    p_report_claimed = (
        mu**3 - az**2 * mu + az * (symbols["ax"] ** 2 + symbols["ay"] ** 2)
    )
    diff_report = sp.simplify(p_correct - p_report_claimed)
    assert diff_report != 0, (
        "Report's claimed polynomial should NOT match the correct one!"
    )
    print("  [PASS] Report's claimed μ³ − a_z²μ + a_z r_⊥² = 0 is INCORRECT.")


def verify_special_case_az_zero(symbols: dict) -> None:
    """Verify a_zz = 0 special case: eigenvalues {0, ±ωr}, eigenvectors ω-independent."""
    import numpy as np

    az, azz, omega = symbols["az"], symbols["azz"], symbols["omega"]

    # At a_zz = 0, H_3' should have eigenvalues 0, ±ωr
    # where r = sqrt(ax² + ay² + az²)
    ax_val, ay_val, az_val = 2, 3, 1
    r = np.sqrt(ax_val**2 + ay_val**2 + az_val**2)

    for omega_val in [0.5, 1.0, 2.0, 5.0]:
        H3p = np.array(
            [
                [
                    omega_val * az_val,
                    omega_val / np.sqrt(2) * (ax_val + 1j * ay_val),
                    0,
                ],
                [
                    omega_val / np.sqrt(2) * (ax_val - 1j * ay_val),
                    0,
                    omega_val / np.sqrt(2) * (ax_val + 1j * ay_val),
                ],
                [
                    0,
                    omega_val / np.sqrt(2) * (ax_val - 1j * ay_val),
                    -omega_val * az_val,
                ],
            ],
            dtype=complex,
        )

        eigenvalues = sorted(np.linalg.eigvalsh(H3p).real, key=abs)
        expected = sorted([-omega_val * r, 0, omega_val * r], key=abs)

        for got, exp in zip(eigenvalues, expected):
            assert abs(got - exp) < 1e-10, (
                f"a_zz=0, ω={omega_val}: eigenvalue {got} ≠ expected {exp}"
            )

    # Eigenvectors should be ω-independent when a_zz = 0
    H3p_1 = np.array(
        [
            [1 * az_val, 1 / np.sqrt(2) * (ax_val + 1j * ay_val), 0],
            [
                1 / np.sqrt(2) * (ax_val - 1j * ay_val),
                0,
                1 / np.sqrt(2) * (ax_val + 1j * ay_val),
            ],
            [0, 1 / np.sqrt(2) * (ax_val - 1j * ay_val), -1 * az_val],
        ],
        dtype=complex,
    )
    H3p_2 = np.array(
        [
            [2 * az_val, 2 / np.sqrt(2) * (ax_val + 1j * ay_val), 0],
            [
                2 / np.sqrt(2) * (ax_val - 1j * ay_val),
                0,
                2 / np.sqrt(2) * (ax_val + 1j * ay_val),
            ],
            [0, 2 / np.sqrt(2) * (ax_val - 1j * ay_val), -2 * az_val],
        ],
        dtype=complex,
    )

    _, v1 = np.linalg.eigh(H3p_1)
    _, v2 = np.linalg.eigh(H3p_2)

    order1 = np.argsort(np.real(np.linalg.eigvalsh(H3p_1)))
    order2 = np.argsort(np.real(np.linalg.eigvalsh(H3p_2)))
    v1_sorted = v1[:, order1]
    v2_sorted = v2[:, order2]

    max_overlap = max(abs(np.vdot(v1_sorted[:, k], v2_sorted[:, k])) for k in range(3))
    assert max_overlap > 0.999, (
        f"a_zz=0: eigenvectors should be ω-independent, "
        f"but max overlap = {max_overlap:.6f}"
    )
    print(
        "  [PASS] Special case a_zz = 0: eigenvalues {0, ±ωr}, "
        "eigenvectors ω-independent."
    )


def verify_H3_det() -> None:
    """Verify det(H_3') = ω² a_z² a_zz / 2."""

    ax, ay, az, azz, omega = sp.symbols("a_x a_y a_z a_zz omega", real=True)

    H3 = sp.Matrix(
        [
            [omega * az + azz / 4, omega / sp.sqrt(2) * (ax + sp.I * ay), 0],
            [
                omega / sp.sqrt(2) * (ax - sp.I * ay),
                -azz / 4,
                omega / sp.sqrt(2) * (ax + sp.I * ay),
            ],
            [0, omega / sp.sqrt(2) * (ax - sp.I * ay), -omega * az + azz / 4],
        ]
    )

    I3 = sp.eye(3)
    H3p = H3 - sp.Rational(1, 4) * azz * I3
    det_val = sp.simplify(H3p.det())

    expected_det = omega**2 * az**2 * azz / 2
    diff = sp.simplify(det_val - expected_det)
    assert diff == 0, f"det(H_3') = {det_val}, expected {expected_det}, diff = {diff}"
    print("  [PASS] det(H_3') = ω² a_z² a_zz / 2.")


def main() -> None:
    print("=" * 70)
    print("Symbolic Verification of Scenario B Block-Diagonalisation")
    print("=" * 70)

    print("\n1. Building 4×4 Hamiltonian in computational basis...")
    H = build_hamiltonian_4x4()

    print("2. Transforming to Bell basis...")
    P = bell_transformation()
    H_bell = sp.simplify(P.T * H * P)

    print("3. Verifying antisymmetric decoupling...")
    verify_antisymmetric_decoupling(H_bell)

    print("4. Extracting and verifying H_3 matrix elements...")
    H3, symbols = verify_H3_elements(H_bell)

    print("5. Verifying shift by a_zz/4 leaves middle element = −a_zz/2...")
    verify_shift_middle_element(H3, symbols)

    print("6. Verifying H_3' is NOT proportional to ω...")
    verify_not_proportional_to_omega(H3, symbols)

    print("7. Verifying eigenvectors depend on ω for a_zz ≠ 0...")
    verify_eigenvectors_depend_on_omega(symbols)

    print("8. Verifying correct characteristic polynomial...")
    verify_characteristic_polynomial(H3, symbols)

    print("9. Verifying det(H_3') = ω² a_z² a_zz / 2...")
    verify_H3_det()

    print("10. Verifying special case a_zz = 0...")
    verify_special_case_az_zero(symbols)

    print("\n" + "=" * 70)
    print("ALL 10 CHECKS PASSED")
    print("=" * 70)
    print("\nSummary of corrections needed in the report:")
    print("  - Line 243: H_3' is NOT ω·H_0 (middle element −a_zz/2 is ω-independent)")
    print("  - Line 243: Eigenvectors DO depend on ω for a_zz ≠ 0")
    print("  - Line 247: Depressed cubic μ³ − a_z²μ + a_z r_⊥² = 0 is WRONG")
    print("  - Line 257: Derivative formula using ω-independent V_3 is invalid")
    print("  - Correct char. poly: μ³ + (a_zz/2)μ² − ω²r²μ − ω²a_z²a_zz/2 = 0")


if __name__ == "__main__":
    main()
