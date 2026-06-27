r"""
Tests for the Squeezed-Vacuum MZI with Parity Measurement module (2026-06-25).

Run with:
    uv run pytest reports/20260625/test_squeezed_vacuum_parity.py -q --tb=short
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys as _sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from src.analysis.scaling_fit import fit_scaling_exponent
from src.analysis.sensitivity_metrics import (
    analyse_best_worst_sensitivity,
)
from src.physics.hilbert_space import resource_value_to_truncation
from src.physics.mzi_simulation import (
    beam_splitter_unitary,
    compute_mzi_sensitivity_grid,
    simple_mzi_evolution,
)
from src.physics.mzi_states import (
    input_state_factory,
)
from src.physics.sv_qfi import (
    check_truncation_convergence,
    compute_sv_captured_norm,
    compute_sv_qfi,
    verify_sv_qfi,
)
from src.utils.serialization import assert_roundtrip_fields

_local_path = Path(__file__).resolve().parent / "squeezed_vacuum_parity.py"
# Insert the report directory so that ``from _shared import ...`` resolves.
_sys.path.insert(0, str(Path(__file__).resolve().parent))
_spec = importlib.util.spec_from_file_location("squeezed_vacuum_parity", str(_local_path))
assert _spec is not None
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_sys.modules["squeezed_vacuum_parity"] = _module
_spec.loader.exec_module(_module)
del _local_path, _spec, _module

from squeezed_vacuum_parity import (  # type: ignore[import-untyped]  # noqa: E402
    MziSensitivityDataSV,
    _fig_path,
    _generate_single_resource_data,
    _maybe_generate_full_data,
    compute_parity_distribution,
    compute_parity_sensitivity_grid,
    generate_all,
    generate_full_data,
    generate_single_omega_scan,
    main,
    plot_delta_omega_overlay,
    plot_scaling,
    t_hold,
)

# ============================================================================
# Parity Distribution
# ============================================================================


class TestParityDistribution:
    def test_normalized(self) -> None:
        """Sum of P(+1) + P(-1) = 1 for any output state."""
        M = 10
        r = float(np.arcsinh(np.sqrt(2.0)))
        state = input_state_factory("squeezed_vacuum", 2, M, r=r)
        out = simple_mzi_evolution(
            state, omega=0.5, max_photons=M, t_hold=t_hold, skip_bs1=True
        )
        P = compute_parity_distribution(out, M)
        assert np.isclose(np.sum(P), 1.0, rtol=1e-12), f"Sum={np.sum(P)}"

    def test_nonnegative(self) -> None:
        """All parity probabilities >= 0."""
        M = 10
        r = float(np.arcsinh(np.sqrt(2.0)))
        state = input_state_factory("squeezed_vacuum", 2, M, r=r)
        out = simple_mzi_evolution(
            state, omega=0.5, max_photons=M, t_hold=t_hold, skip_bs1=True
        )
        P = compute_parity_distribution(out, M)
        assert np.all(P >= -1e-15), "Some probabilities are negative"

    def test_shape(self) -> None:
        """Array shape is (2,)."""
        M = 10
        r = float(np.arcsinh(np.sqrt(2.0)))
        state = input_state_factory("squeezed_vacuum", 2, M, r=r)
        out = simple_mzi_evolution(
            state, omega=0.5, max_photons=M, t_hold=t_hold, skip_bs1=True
        )
        P = compute_parity_distribution(out, M)
        assert P.shape == (2,), f"Shape={P.shape}, expected (2,)"

    def test_parity_expectation_range(self) -> None:
        """⟨Π⟩ must be in [-1, 1] for all ω."""
        M = 15
        r = float(np.arcsinh(np.sqrt(3.0)))
        state = input_state_factory("squeezed_vacuum", 3, M, r=r)
        for omega in np.linspace(0.1, 5.0, 10):
            out = simple_mzi_evolution(
                state, omega=omega, max_photons=M, t_hold=t_hold, skip_bs1=True
            )
            P = compute_parity_distribution(out, M)
            parity_exp = P[0] - P[1]
            assert -1.0 - 1e-12 <= parity_exp <= 1.0 + 1e-12, (
                f"⟨Π⟩={parity_exp} outside [-1, 1] at ω={omega}"
            )

    def test_vacuum_state_parity_plus_one(self) -> None:
        """Vacuum state |0,0⟩ should give P(+1) = 1 (n2=0 is even)."""
        M = 5
        from src.physics.mzi_simulation import prepare_input_state

        state = prepare_input_state("vacuum", max_photons=M)
        # Vacuum already has no photons, so any MZI evolution preserves parity=+1
        out = simple_mzi_evolution(
            state, omega=0.5, max_photons=M, t_hold=t_hold, skip_bs1=True
        )
        P = compute_parity_distribution(out, M)
        assert np.isclose(P[0], 1.0, atol=1e-12), f"P(+1)={P[0]}, expected 1.0"
        assert np.isclose(P[1], 0.0, atol=1e-12), f"P(-1)={P[1]}, expected 0.0"

    def test_consistent_with_expectation(self) -> None:
        """⟨Π⟩ = P(+1) - P(-1) must be consistent."""
        M = 10
        r = float(np.arcsinh(np.sqrt(2.0)))
        state = input_state_factory("squeezed_vacuum", 2, M, r=r)
        out = simple_mzi_evolution(
            state, omega=0.3, max_photons=M, t_hold=t_hold, skip_bs1=True
        )
        P = compute_parity_distribution(out, M)
        parity_exp = P[0] - P[1]
        # Also compute via explicit Π operator
        dim = M + 1
        parity_op = np.zeros((dim * dim, dim * dim), dtype=complex)
        for n1 in range(dim):
            for n2 in range(dim):
                idx = n1 * dim + n2
                parity_op[idx, idx] = (-1.0) ** n2
        exp_from_op = float(np.real(np.conj(out) @ parity_op @ out))
        assert np.isclose(parity_exp, exp_from_op, atol=1e-12), (
            f"⟨Π⟩ from distribution={parity_exp}, from operator={exp_from_op}"
        )


# ============================================================================
# Parity Sensitivity Grid
# ============================================================================


class TestParitySensitivityGrid:
    def test_returns_required_keys(self) -> None:
        """Result dict must contain all expected arrays."""
        mean_N = 2.0
        M = 20
        r = float(np.arcsinh(np.sqrt(mean_N)))
        state = input_state_factory("squeezed_vacuum", int(mean_N), M, r=r)
        omega_grid = np.linspace(0.1, 5.0, 5)
        result = compute_parity_sensitivity_grid(
            state, omega_grid, M, t_hold=t_hold, skip_bs1=True
        )
        expected_keys = {
            "omega_values",
            "expectation_values",
            "variance_values",
            "derivative_values",
            "delta_omega_ep",
            "delta_omega_q",
            "fisher_quantum",
            "fisher_classical",
            "delta_omega_c",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )

    def test_variance_nonnegative(self) -> None:
        """Var(Π) = 1 - ⟨Π⟩² must be >= 0 at all ω."""
        mean_N = 2.0
        M = 20
        r = float(np.arcsinh(np.sqrt(mean_N)))
        state = input_state_factory("squeezed_vacuum", int(mean_N), M, r=r)
        omega_grid = np.linspace(0.1, 5.0, 5)
        result = compute_parity_sensitivity_grid(
            state, omega_grid, M, t_hold=t_hold, skip_bs1=True
        )
        assert np.all(result["variance_values"] >= -1e-12)

    def test_cfi_positivity(self) -> None:
        """F_C^Π(ω) must be >= 0 at all ω."""
        mean_N = 2.0
        M = 20
        r = float(np.arcsinh(np.sqrt(mean_N)))
        state = input_state_factory("squeezed_vacuum", int(mean_N), M, r=r)
        omega_grid = np.linspace(0.1, 5.0, 5)
        result = compute_parity_sensitivity_grid(
            state, omega_grid, M, t_hold=t_hold, skip_bs1=True
        )
        assert np.all(result["fisher_classical"] >= -1e-12), (
            "Some F_C values negative"
        )

    def test_cfi_cramer_rao(self) -> None:
        """Δω_C >= Δω_Q for SV with parity (quantum Cramér-Rao bound)."""
        mean_N = 2.0
        M = 20
        r = float(np.arcsinh(np.sqrt(mean_N)))
        state = input_state_factory("squeezed_vacuum", int(mean_N), M, r=r)
        omega_grid = np.linspace(0.1, 5.0, 10)
        result = compute_parity_sensitivity_grid(
            state, omega_grid, M, t_hold=t_hold, skip_bs1=False
        )
        delta_q = float(result["delta_omega_q"])
        for c_val in np.asarray(result["delta_omega_c"]):
            if np.isfinite(c_val):
                assert c_val >= delta_q - 1e-8, (
                    f"Δω_C={c_val} < Δω_Q={delta_q} violates Cramér-Rao bound"
                )

    def test_qfi_bound_saturation(self) -> None:
        """At optimal ω, F_C^Π / F_Q should be close to 1 for moderate ⟨N⟩."""
        mean_N = 4.0
        M = 30
        r = float(np.arcsinh(np.sqrt(mean_N)))
        state = input_state_factory("squeezed_vacuum", int(mean_N), M, r=r)
        omega_grid = np.linspace(0.1, 5.0, 50)
        result = compute_parity_sensitivity_grid(
            state, omega_grid, M, t_hold=t_hold, skip_bs1=False
        )
        fq = float(result["fisher_quantum"])
        fc_max = float(np.nanmax(result["fisher_classical"]))
        if fq > 0:
            ratio = fc_max / fq
            # With skip_bs1=False, parity CFI saturates the QFI bound
            assert ratio > 0.9, (
                f"F_C^Π/F_Q = {ratio:.4f} at best ω, expected > 0.9"
            )

    def test_parity_near_qfi(self) -> None:
        """Parity CFI saturates QFI; number-diff CFI is significantly below."""
        mean_N = 4.0
        M = 30
        r = float(np.arcsinh(np.sqrt(mean_N)))
        state = input_state_factory("squeezed_vacuum", int(mean_N), M, r=r)
        omega_grid = np.linspace(0.1, 5.0, 30)

        # Parity CFI
        parity_result = compute_parity_sensitivity_grid(
            state, omega_grid, M, t_hold=t_hold, skip_bs1=False
        )

        # Number-difference CFI
        nd_result = compute_mzi_sensitivity_grid(
            state, omega_grid, M, t_hold=t_hold, skip_bs1=False
        )

        fq = float(parity_result["fisher_quantum"])
        fc_parity_max = float(np.nanmax(parity_result["fisher_classical"]))
        fc_nd_max = float(np.nanmax(nd_result["fisher_classical"]))

        # Parity CFI should be close to QFI; number-diff should be noticeably below
        parity_ratio = fc_parity_max / fq
        nd_ratio = fc_nd_max / fq
        assert parity_ratio > 0.9, (
            f"Parity F_C/F_Q={parity_ratio:.4f}, expected > 0.9"
        )
        assert nd_ratio < 0.99, (
            f"Number-diff F_C/F_Q={nd_ratio:.4f}, expected < 0.99"
        )

    def test_precomputed_bs_works(self) -> None:
        """Pre-computed BS matrix can be passed in."""
        mean_N = 2.0
        M = 20
        r = float(np.arcsinh(np.sqrt(mean_N)))
        state = input_state_factory("squeezed_vacuum", int(mean_N), M, r=r)
        bs = beam_splitter_unitary(np.pi / 4, 0.0, M)
        omega_grid = np.linspace(0.1, 5.0, 5)
        result = compute_parity_sensitivity_grid(
            state, omega_grid, M, t_hold=t_hold, skip_bs1=True, bs=bs
        )
        assert np.any(np.isfinite(result["fisher_classical"]))


# ============================================================================
# Analytical SV QFI
# ============================================================================


class TestSVQFI:
    def test_analytical_qfi(self) -> None:
        """Analytical SV QFI formula must match."""
        for mean_N in [1, 2, 4, 10]:
            fq = compute_sv_qfi(float(mean_N), t_hold)
            expected = 2.0 * t_hold**2 * mean_N * (mean_N + 1.0)
            assert np.isclose(fq, expected, rtol=1e-12), (
                f"mean_N={mean_N}: F_Q={fq}, expected {expected}"
            )

    def test_analytical_vs_numerical_var(self) -> None:
        """Var(J_z) from SV probe must match analytical formula."""
        mean_N = 2.0
        M = 80  # Large enough to capture the SV tail for mean_N=2
        r = float(np.arcsinh(np.sqrt(mean_N)))
        state = input_state_factory("squeezed_vacuum", int(mean_N), M, r=r)
        from src.physics.mzi_simulation import build_jz_operator

        jz_diag = build_jz_operator(M)
        mean_probe = np.sum(np.abs(state) ** 2 * jz_diag)
        mean_sq_probe = np.sum(np.abs(state) ** 2 * jz_diag ** 2)
        var_probe = float(np.real(mean_sq_probe - mean_probe**2))
        assert verify_sv_qfi(mean_N, var_probe), (
            f"Var(J_z)={var_probe} does not match analytical"
        )


# ============================================================================
# Truncation Convergence
# ============================================================================


class TestTruncationConvergence:
    def test_sv_truncation_sufficient(self) -> None:
        """SV analytical check: sufficient truncation passes."""
        cases: list[tuple[float, int]] = [
            (1.0, 20),
            (2.0, 30),
            (4.0, 50),
        ]
        for mean_N, M in cases:
            assert check_truncation_convergence(
                threshold=0.999, mean_n=mean_N, max_photons=M
            ), f"truncation should pass at mean_N={mean_N}, M={M}"

    def test_sv_truncation_insufficient(self) -> None:
        """SV analytical check: insufficient truncation is detected."""
        assert not check_truncation_convergence(
            threshold=0.999, mean_n=1.0, max_photons=5
        ), "truncation should fail at mean_N=1, M=5"

    def test_truncation_fallthrough_raises(self) -> None:
        """check_truncation_convergence with no arguments raises ValueError."""
        with pytest.raises(ValueError, match="Must provide"):
            check_truncation_convergence()

    def test_compute_captured_norm(self) -> None:
        """compute_sv_captured_norm returns value in [0, 1]."""
        for mean_N in [0.5, 1.0, 2.0, 5.0, 10.0]:
            M = min(int(5 * mean_N), 80)
            captured = compute_sv_captured_norm(mean_N, M)
            assert 0.0 <= captured <= 1.0, (
                f"Captured norm {captured} outside [0, 1] for ⟨N⟩={mean_N}, M={M}"
            )


# ============================================================================
# MziSensitivityDataSV — Parquet Roundtrip
# ============================================================================


class TestMziSensitivityDataSVParquet:
    _FIELD_SPECS: ClassVar[list[tuple[str, str]]] = [
        ("state_type", "eq"),
        ("resource_type", "eq"),
        ("resource_values", "allclose"),
        ("omega_values", "allclose"),
        ("expectation_grid", "allclose"),
        ("variance_grid", "allclose"),
        ("derivative_grid", "allclose"),
        ("delta_omega_ep_grid", "allclose"),
        ("delta_omega_q_per_R", "allclose"),
        ("fisher_classical_grid", "allclose"),
        ("delta_omega_c_grid", "allclose"),
        ("t_hold", "isclose"),
        ("truncation_M_per_R", "allclose"),
        ("squeezing_q_per_R", "allclose"),
    ]

    @pytest.fixture
    def make_result(self) -> MziSensitivityDataSV:
        n_R = 3
        n_omega = 5
        return MziSensitivityDataSV(
            state_type="sv_parity",
            resource_type="mean_N",
            resource_values=np.array([2, 4, 6], dtype=float),
            omega_values=np.linspace(0.1, 5.0, n_omega),
            expectation_grid=np.random.default_rng(42).uniform(-1, 1, (n_R, n_omega)),
            variance_grid=np.random.default_rng(43).uniform(0, 0.5, (n_R, n_omega)),
            derivative_grid=np.random.default_rng(44).uniform(-2, 2, (n_R, n_omega)),
            delta_omega_ep_grid=np.random.default_rng(45).uniform(0.01, 1, (n_R, n_omega)),
            delta_omega_q_per_R=np.array([0.05, 0.025, 0.0167]),
            fisher_classical_grid=np.random.default_rng(46).uniform(1, 100, (n_R, n_omega)),
            delta_omega_c_grid=np.random.default_rng(47).uniform(0.01, 1, (n_R, n_omega)),
            t_hold=t_hold,
            truncation_M_per_R=np.array([10, 20, 30], dtype=float),
            squeezing_q_per_R=np.zeros(3, dtype=float),
        )

    def test_roundtrip(self, make_result: MziSensitivityDataSV, tmp_path: Path) -> None:
        p = tmp_path / "sensitivity.parquet"
        make_result.save_parquet(p)
        loaded = MziSensitivityDataSV.from_parquet(p)
        assert_roundtrip_fields(loaded, make_result, self._FIELD_SPECS)

    def test_fail_fast_missing_column(
        self, make_result: MziSensitivityDataSV, tmp_path: Path
    ) -> None:
        p = tmp_path / "bad.parquet"
        df = make_result.to_dataframe()
        df = df.drop(columns=["state_type"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            MziSensitivityDataSV.from_parquet(p)

    def test_fail_fast_missing_q_column(
        self, make_result: MziSensitivityDataSV, tmp_path: Path
    ) -> None:
        p = tmp_path / "bad_q.parquet"
        df = make_result.to_dataframe()
        df = df.drop(columns=["squeezing_q"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            MziSensitivityDataSV.from_parquet(p)


# ============================================================================
# Generate Omega Scan (Single Resource Value)
# ============================================================================


class TestGenerateOmegaScan:
    def test_generate_sv_scan(self) -> None:
        """Generate a small ω scan and verify basic properties."""
        omega_grid = np.linspace(0.1, 5.0, 5)
        result = generate_single_omega_scan(
            resource_value=4.0,
            omega_grid=omega_grid,
            max_photons=20,
            t_hold=t_hold,
        )
        assert result.state_type == "sv_parity"
        assert result.resource_type == "mean_N"
        assert np.isclose(result.resource_values[0], 4.0)
        assert len(result.omega_values) == 5
        assert np.all(np.isfinite(result.fisher_classical_grid[0]))

    def test_generate_scan_no_max_photons(self) -> None:
        """generate_single_omega_scan works without max_photons (auto-compute)."""
        omega_grid = np.linspace(0.1, 5.0, 3)
        result = generate_single_omega_scan(
            resource_value=3.0,
            omega_grid=omega_grid,
            t_hold=t_hold,
        )
        assert result.state_type == "sv_parity"
        assert len(result.omega_values) == 3
        assert np.all(np.isfinite(result.fisher_classical_grid[0]))


# ============================================================================
# Generate Single Resource Data
# ============================================================================


class TestGenerateSingleResourceData:
    def test_resource_data(self) -> None:
        """_generate_single_resource_data returns a valid result."""
        omega_grid = np.linspace(0.1, 5.0, 3)
        result = _generate_single_resource_data(4.0, omega_grid, t_hold=t_hold)
        assert result is not None
        assert result.state_type == "sv_parity"
        assert len(result.omega_values) == 3

    def test_resource_data_fast(self) -> None:
        """_generate_single_resource_data returns valid result with small M."""
        omega_grid = np.linspace(0.1, 5.0, 2)
        result = _generate_single_resource_data(2.0, omega_grid, t_hold=t_hold)
        assert result is not None
        assert result.state_type == "sv_parity"

    def test_resource_data_failure_caught(self) -> None:
        """Invalid parameters return None (not crash)."""
        # This tests the try/except in _generate_single_resource_data
        # with a very small resource value that might be problematic
        omega_grid = np.linspace(0.1, 5.0, 2)
        result = _generate_single_resource_data(0.5, omega_grid, t_hold=t_hold)
        # This might succeed or return None, but should not crash
        if result is not None:
            assert result.state_type == "sv_parity"

    def test_malloc_trim_exception(self) -> None:
        """malloc_trim failure is caught and does not crash the pipeline."""
        from unittest.mock import patch

        with patch("ctypes.CDLL", side_effect=Exception("mock malloc_trim failure")):
            omega_grid = np.linspace(0.1, 5.0, 2)
            result = _generate_single_resource_data(
                2.0, omega_grid, t_hold=t_hold
            )
        assert result is not None
        assert result.state_type == "sv_parity"


# ============================================================================
# Generate Full Data
# ============================================================================


class TestGenerateFullData:
    def test_full_data_small(self) -> None:
        """generate_full_data runs with a minimal resource range."""
        omega_grid = np.linspace(0.1, 5.0, 2)
        data = generate_full_data([2.0, 4.0], omega_grid, t_hold=t_hold)
        assert data.state_type == "sv_parity"
        assert len(data.resource_values) == 2
        assert len(data.omega_values) == 2
        assert data.truncation_M_per_R is not None

    def test_full_data_all_invalid_raises(self) -> None:
        """generate_full_data raises RuntimeError when all resource values fail."""
        omega_grid = np.linspace(0.1, 5.0, 2)
        # Monkey-patch _generate_single_resource_data to always return None
        # We can do this by patching at module level
        import squeezed_vacuum_parity as svp_mod

        original = svp_mod._generate_single_resource_data
        svp_mod._generate_single_resource_data = lambda *a, **kw: None
        try:
            with pytest.raises(RuntimeError, match="No valid resource values"):
                generate_full_data([2.0], omega_grid, t_hold=t_hold)
        finally:
            svp_mod._generate_single_resource_data = original


# ============================================================================
# Maybe Generate Full Data
# ============================================================================


class TestMaybeGenerateFullData:
    def test_maybe_generate_creates_and_loads(
        self, tmp_path: Path,
    ) -> None:
        """_maybe_generate_full_data generates and can reload from Parquet."""

        omega_grid = np.arange(0.1, 5.1, 4.9)
        pq_path = tmp_path / "test_parity.parquet"

        # First call: generate and save
        data1 = _maybe_generate_full_data(
            [2.0], "SV+Parity", omega_grid, force=True,
            override_pq_path=pq_path,
        )
        assert data1 is not None
        assert data1.state_type == "sv_parity"

        # Second call: load from existing Parquet
        data2 = _maybe_generate_full_data(
            [2.0], "SV+Parity", omega_grid, force=False,
            override_pq_path=pq_path,
        )
        assert data2 is not None
        assert data2.state_type == "sv_parity"
        np.testing.assert_array_equal(data1.resource_values, data2.resource_values)

    def test_maybe_generate_force(
        self, tmp_path: Path,
    ) -> None:
        """_maybe_generate_full_data with force=True re-generates data."""
        omega_grid = np.arange(0.1, 5.1, 4.9)
        pq_path = tmp_path / "test_force.parquet"
        data = _maybe_generate_full_data(
            [3.0], "SV+Parity", omega_grid, force=True,
            override_pq_path=pq_path,
        )
        assert data is not None
        assert data.state_type == "sv_parity"


# ============================================================================
# Truncation Helper
# ============================================================================


class TestTruncationHelper:
    def test_resource_value_to_truncation_sv(self) -> None:
        """Truncation scales with mean N for SV."""
        for mean_N in [1, 5, 10, 20]:
            M = resource_value_to_truncation(float(mean_N), "sv")
            assert mean_N <= M, f"SV truncation M={M} < mean_N={mean_N}"
            assert M <= 80, f"SV truncation M={M} exceeds max"

    def test_truncation_returns_int(self) -> None:
        """Truncation must be an int."""
        M = resource_value_to_truncation(4.0, "sv")
        assert isinstance(M, int)


# ============================================================================
# Plots
# ============================================================================


class TestPlots:
    @pytest.fixture
    def sample_data(self) -> MziSensitivityDataSV:
        """Small MziSensitivityDataSV for plot tests."""
        n_R = 3
        n_omega = 5
        return MziSensitivityDataSV(
            state_type="sv_parity",
            resource_type="mean_N",
            resource_values=np.array([2, 4, 6], dtype=float),
            omega_values=np.linspace(0.1, 5.0, n_omega),
            expectation_grid=np.zeros((n_R, n_omega), dtype=float),
            variance_grid=np.ones((n_R, n_omega), dtype=float),
            derivative_grid=np.ones((n_R, n_omega), dtype=float),
            delta_omega_ep_grid=np.ones((n_R, n_omega), dtype=float),
            delta_omega_q_per_R=np.array([0.1, 0.05, 0.03]),
            fisher_classical_grid=np.full((n_R, n_omega), 100.0, dtype=float),
            delta_omega_c_grid=np.ones((n_R, n_omega), dtype=float),
            t_hold=t_hold,
            truncation_M_per_R=np.array([10, 20, 30], dtype=float),
            squeezing_q_per_R=np.zeros(3, dtype=float),
        )

    def test_delta_omega_overlay(
        self, sample_data: MziSensitivityDataSV, tmp_path: Path
    ) -> None:
        """plot_delta_omega_overlay creates a valid SVG."""
        svg = tmp_path / "test_overlay.svg"
        created = plot_delta_omega_overlay(sample_data, save_path=svg)
        assert svg.exists()
        assert svg.suffix == ".svg"
        assert created == svg

    def test_overlay_selected_R(
        self, sample_data: MziSensitivityDataSV, tmp_path: Path
    ) -> None:
        """plot_delta_omega_overlay works with explicit R selection."""
        svg = tmp_path / "test_overlay_sel.svg"
        plot_delta_omega_overlay(sample_data, selected_R=[2.0, 6.0], save_path=svg)
        assert svg.exists()

    def test_overlay_non_existent_R(
        self, sample_data: MziSensitivityDataSV, tmp_path: Path
    ) -> None:
        """plot_delta_omega_overlay skips non-existent R values gracefully."""
        svg = tmp_path / "overlay_bad_R.svg"
        plot_delta_omega_overlay(sample_data, selected_R=[999.0], save_path=svg)
        assert svg.exists()

    def test_plot_scaling(
        self, sample_data: MziSensitivityDataSV, tmp_path: Path
    ) -> None:
        """plot_scaling creates a valid SVG."""
        svg = tmp_path / "test_scaling.svg"
        created = plot_scaling(sample_data, save_path=svg)
        assert svg.exists()
        assert created == svg

    def test_plot_scaling_with_fit(self, tmp_path: Path) -> None:
        """plot_scaling with SQL-like data triggers the fit-valid branch."""
        n_R = 10
        R_vals = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0])
        dt_c = 1.0 / (t_hold * np.sqrt(2 * R_vals * (R_vals + 1)))  # SV QFI bound
        dt_q = dt_c  # QFI same as CFI (optimal)
        n_omega = 5
        data = MziSensitivityDataSV(
            state_type="sv_parity",
            resource_type="mean_N",
            resource_values=R_vals,
            omega_values=np.linspace(0.1, 5.0, n_omega),
            expectation_grid=np.zeros((n_R, n_omega), dtype=float),
            variance_grid=np.ones((n_R, n_omega), dtype=float),
            derivative_grid=np.ones((n_R, n_omega), dtype=float),
            delta_omega_ep_grid=np.ones((n_R, n_omega), dtype=float),
            delta_omega_q_per_R=dt_q,
            fisher_classical_grid=np.full((n_R, n_omega), 100.0, dtype=float),
            delta_omega_c_grid=np.tile(dt_c, (n_omega, 1)).T.astype(float),
            t_hold=t_hold,
            truncation_M_per_R=np.full(n_R, 10.0, dtype=float),
            squeezing_q_per_R=np.zeros(n_R, dtype=float),
        )
        svg = tmp_path / "scaling_fit.svg"
        plot_scaling(data, save_path=svg)
        assert svg.exists()

    def test_overlay_no_save_path(self, sample_data: MziSensitivityDataSV) -> None:
        """plot_delta_omega_overlay works without save_path (auto-names)."""
        path = _fig_path("sv_parity_delta_omega_comparison")
        try:
            created = plot_delta_omega_overlay(sample_data)
            assert created is not None
            assert created.suffix == ".svg"
        finally:
            if path.exists():
                path.unlink()

    def test_scaling_no_save_path(self, sample_data: MziSensitivityDataSV) -> None:
        """plot_scaling works without save_path (auto-names)."""
        path = _fig_path("sv_parity_scaling")
        try:
            created = plot_scaling(sample_data)
            assert created is not None
            assert created.suffix == ".svg"
        finally:
            if path.exists():
                path.unlink()


# ============================================================================
# Analyse Best/Worst Sensitivity
# ============================================================================


class TestAnalyseBestWorstSensitivity:
    def test_returns_dict_keys(self) -> None:
        """analyse_best_worst_sensitivity returns expected keys."""
        R_vals = np.array([2.0, 4.0])
        omega_vals = np.linspace(0.1, 5.0, 10)
        grid = np.random.default_rng(42).uniform(0.01, 1, (2, 10))
        result = analyse_best_worst_sensitivity(R_vals, omega_vals, grid)
        expected_keys = {
            "resource_values",
            "best_sensitivity",
            "best_omega",
            "worst_sensitivity",
            "worst_omega",
        }
        assert result.keys() == expected_keys, (
            f"Missing keys: {expected_keys - result.keys()}"
        )

    def test_best_is_minimum(self) -> None:
        """Best sensitivity <= worst sensitivity at each resource."""
        R_vals = np.array([2.0, 4.0, 6.0])
        omega_vals = np.linspace(0.1, 5.0, 20)
        rng = np.random.default_rng(42)
        grid = rng.uniform(0.01, 1, (3, 20))
        result = analyse_best_worst_sensitivity(R_vals, omega_vals, grid)
        assert np.all(result["best_sensitivity"] <= result["worst_sensitivity"] + 1e-15)


# ============================================================================
# Scaling Exponent Fitting
# ============================================================================


class TestFitScalingExponent:
    def test_sv_heisenberg_exponent(self) -> None:
        r"""SV QFI bound should give α ≈ -1.0 (Heisenberg)."""
        R_vals = np.array([20, 30, 50, 80, 120], dtype=float)
        delta_vals = 1.0 / (t_hold * np.sqrt(2 * R_vals * (R_vals + 1)))
        result = fit_scaling_exponent(R_vals, delta_vals, min_N=4)
        assert result.valid, f"Fit invalid: {result.warnings}"
        assert np.isclose(result.alpha, -1.0, atol=0.05), (
            f"SV exponent α={result.alpha}, expected -1.0"
        )


# ============================================================================
# CLI Smoke Test
# ============================================================================


class TestCLI:
    def test_main_help(self) -> None:
        """CLI help must display."""
        result = subprocess.run(
            [
                _sys.executable,
                str(Path(__file__).resolve().parent / "squeezed_vacuum_parity.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "Squeezed-Vacuum MZI with Parity Measurement" in result.stdout

    def test_main_direct_force(self, tmp_path: Path) -> None:
        """Call main() directly with --force via sys.argv patching."""
        import sys as _sys

        import squeezed_vacuum_parity as svp_mod

        old_argv = _sys.argv[:]
        orig_sv = svp_mod.SV_N_RANGE
        orig_omega_range = svp_mod.OMEGA_RANGE
        orig_omega_step = svp_mod.OMEGA_STEP
        try:
            svp_mod.SV_N_RANGE = [2.0]
            svp_mod.OMEGA_RANGE = (0.1, 0.5)
            svp_mod.OMEGA_STEP = 0.4
            _sys.argv = ["script", "--force"]
            main()
        finally:
            _sys.argv = old_argv
            svp_mod.SV_N_RANGE = orig_sv
            svp_mod.OMEGA_RANGE = orig_omega_range
            svp_mod.OMEGA_STEP = orig_omega_step

    def test_mpbackend_not_set(self) -> None:
        """When MPLBACKEND is unset, the module sets it to Agg."""
        import subprocess as _sub

        code = """
import os
os.environ.pop("MPLBACKEND", None)
import sys
sys.path.insert(0, "reports/20260625")
import squeezed_vacuum_parity
assert os.environ["MPLBACKEND"] == "Agg", f"Got {os.environ.get('MPLBACKEND')}"
print("OK")
"""
        result = _sub.run(
            [_sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr={result.stderr}"
        assert "OK" in result.stdout


# ============================================================================
# Generate All (Pipeline)
# ============================================================================


class TestGenerateAll:
    def test_generate_all_small(self, tmp_path: Path) -> None:
        """generate_all runs end-to-end with a small parameter range."""
        import squeezed_vacuum_parity as svp_mod

        orig_sv = svp_mod.SV_N_RANGE
        orig_omega_range = svp_mod.OMEGA_RANGE
        orig_omega_step = svp_mod.OMEGA_STEP
        try:
            svp_mod.SV_N_RANGE = [2.0]
            svp_mod.OMEGA_RANGE = (0.1, 0.5)
            svp_mod.OMEGA_STEP = 0.4
            data = generate_all(force=True, override_pq_path=tmp_path / "test.parquet")
        finally:
            svp_mod.SV_N_RANGE = orig_sv
            svp_mod.OMEGA_RANGE = orig_omega_range
            svp_mod.OMEGA_STEP = orig_omega_step
        assert data.state_type == "sv_parity"
        assert len(data.resource_values) == 1
        assert len(data.omega_values) == 2


# ============================================================================
# Integration: Simple MZI Evolution with Parity
# ============================================================================


class TestSimpleMziEvolutionParity:
    @pytest.mark.parametrize("N", [1, 2, 4])
    def test_norm_preserved_sv(self, N: int) -> None:
        """MZI evolution must preserve norm for SV state (skip_bs1=True)."""
        M = int(5 * N)
        state = input_state_factory(
            "squeezed_vacuum",
            N,
            M,
            r=float(np.arcsinh(np.sqrt(float(N)))),
        )
        final = simple_mzi_evolution(
            state,
            omega=1.0,
            max_photons=M,
            t_hold=t_hold,
            skip_bs1=True,
        )
        assert np.isclose(np.linalg.norm(final), 1.0, rtol=1e-10), f"Failed for N={N}"


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    def test_mean_N_1_parity(self) -> None:
        """SV with mean_N=1 should have finite parity CFI (unlike number-diff)."""
        mean_N = 1.0
        M = 30
        r = float(np.arcsinh(np.sqrt(mean_N)))
        state = input_state_factory("squeezed_vacuum", int(mean_N), M, r=r)
        omega_grid = np.linspace(0.1, 5.0, 10)
        result = compute_parity_sensitivity_grid(
            state, omega_grid, M, t_hold=t_hold, skip_bs1=False
        )
        fc_max = float(np.nanmax(result["fisher_classical"]))
        assert np.isfinite(fc_max), "F_C not finite at mean_N=1"
        assert fc_max > 0, f"F_C={fc_max} should be positive at mean_N=1"
