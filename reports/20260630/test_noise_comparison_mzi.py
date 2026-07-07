r"""
Tests for the Pedagogical Noise Comparison Single-Particle MZI module (2026-06-30).

Run with:
    uv run pytest reports/20260630/test_noise_comparison_mzi.py -q --tb=short
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.serialization import assert_roundtrip_fields

_m = importlib.import_module("reports.20260630.noise_comparison_mzi")

NoiseSweepResult = _m.NoiseSweepResult
compute_sensitivity = _m.compute_sensitivity
run_noisy_circuit = _m.run_noisy_circuit
sweep_t_hold = _m.sweep_t_hold
sweep_gamma = _m.sweep_gamma
sweep_2d = _m.sweep_2d
plot_degradation_curves = _m.plot_degradation_curves
plot_noise_rate_scaling = _m.plot_noise_rate_scaling
plot_landscape_2d = _m.plot_landscape_2d

SCENARIO_CLEAN = _m.SCENARIO_CLEAN
SCENARIO_DEPHASING = _m.SCENARIO_DEPHASING
SCENARIO_LOSS = _m.SCENARIO_LOSS
SCENARIO_BOTH = _m.SCENARIO_BOTH

JZ = _m._get_jz_operator()


# ============================================================================
# Input State
# ============================================================================


class TestInputState:
    def test_one_zero_state(self) -> None:
        """|1,0⟩ circuit has the correct J_z expectation at ω=0."""
        # Analytical: ⟨J_z⟩ = -(1/2)cos(ω·t_hold), at ω=0 → -0.5
        rho = run_noisy_circuit(omega=0.0, t_hold=1.0, gamma_phi=0.0, gamma_1=0.0)
        assert np.isclose(np.trace(rho), 1.0, atol=1e-10)
        jz_mean = float(np.real(np.trace(rho @ JZ)))
        expected = -0.5 * np.cos(0.0)
        assert np.isclose(jz_mean, expected, atol=1e-10), (
            f"⟨J_z⟩ = {jz_mean}, expected {expected}"
        )


# ============================================================================
# Operator Construction
# ============================================================================


class TestOperators:
    def test_jz_eigenvalues(self) -> None:
        r"""J_z eigenvalues must be {0, ±0.5}."""
        eigvals = np.sort(np.linalg.eigvalsh(JZ))
        expected = np.sort(np.array([-0.5, 0.0, 0.0, 0.5]))
        assert np.allclose(eigvals, expected, atol=1e-10), f"J_z eigvals = {eigvals}"

    def test_jz_hermitian(self) -> None:
        """J_z must be Hermitian."""
        assert np.allclose(JZ, JZ.conj().T, atol=1e-10), "J_z not Hermitian"


# ============================================================================
# Circuit Invariants
# ============================================================================


class TestCircuitInvariants:
    @pytest.mark.parametrize(
        ("gamma_phi", "gamma_1"),
        [
            (0.0, 0.0),
            (0.1, 0.0),
            (0.0, 0.1),
            (0.1, 0.1),
            (1.0, 0.0),
            (0.0, 1.0),
        ],
    )
    def test_trace_preserved(self, gamma_phi: float, gamma_1: float) -> None:
        """Circuit must preserve trace for all noise configurations."""
        rho = run_noisy_circuit(1.0, 1.0, gamma_phi, gamma_1)
        trace = float(np.real(np.trace(rho)))
        assert np.isclose(trace, 1.0, atol=1e-7), (
            f"Trace not preserved: Tr(ρ) = {trace}"
        )

    @pytest.mark.parametrize(
        ("gamma_phi", "gamma_1"),
        [
            (0.0, 0.0),
            (0.1, 0.0),
            (0.0, 0.1),
            (0.1, 0.1),
        ],
    )
    def test_hermiticity(self, gamma_phi: float, gamma_1: float) -> None:
        """Final ρ must be Hermitian."""
        rho = run_noisy_circuit(1.0, 1.0, gamma_phi, gamma_1)
        assert np.allclose(rho, rho.conj().T, atol=1e-8), "ρ not Hermitian"

    @pytest.mark.parametrize(
        ("gamma_phi", "gamma_1"),
        [
            (0.0, 0.0),
            (0.1, 0.0),
            (0.0, 0.1),
            (0.1, 0.1),
        ],
    )
    def test_positivity(self, gamma_phi: float, gamma_1: float) -> None:
        """Final ρ must be positive semidefinite within numerical tolerance."""
        rho = run_noisy_circuit(1.0, 1.0, gamma_phi, gamma_1)
        eigvals = np.linalg.eigvalsh(rho)
        assert np.min(eigvals) >= -1e-6, f"Negative eigenvalue: {np.min(eigvals)}"


# ============================================================================
# Clean Baseline
# ============================================================================


class TestCleanBaseline:
    @pytest.mark.parametrize("t_hold", [0.1, 1.0, 10.0, 100.0])
    def test_noiseless_delta_omega(self, t_hold: float) -> None:
        r"""At γ=0, Δω must equal 1/t_hold."""
        result = compute_sensitivity(1.0, t_hold, 0.0, 0.0, JZ)
        assert np.isclose(result["delta_omega"], 1.0 / t_hold, rtol=1e-5), (
            f"t_hold={t_hold}: Δω={result['delta_omega']}, expected {1.0 / t_hold}"
        )

    @pytest.mark.parametrize("t_hold", [0.1, 1.0, 10.0])
    def test_jz_mean_analytical(self, t_hold: float) -> None:
        r"""⟨J_z⟩ must match -(1/2) cos(ω·t_hold) at γ=0."""
        omega = 1.0
        result = compute_sensitivity(omega, t_hold, 0.0, 0.0, JZ)
        expected = -0.5 * np.cos(omega * t_hold)
        assert np.isclose(result["jz_mean"], expected, atol=1e-10), (
            f"⟨J_z⟩ = {result['jz_mean']}, expected {expected}"
        )


# ============================================================================
# Asymptotic Limits
# ============================================================================


class TestAsymptoticLimits:
    def test_strong_dephasing(self) -> None:
        """Large γ_φ must strongly degrade sensitivity."""
        result = compute_sensitivity(1.0, 1.0, gamma_phi=100.0, gamma_1=0.0, jz=JZ)
        # At γ_φ=100, t_hold=1, the coherence is almost completely destroyed
        assert result["delta_omega"] > 10.0, (
            f"Δω = {result['delta_omega']}, expected >> 1"
        )

    def test_strong_loss(self) -> None:
        """Large γ₁ must strongly degrade sensitivity."""
        result = compute_sensitivity(1.0, 1.0, gamma_phi=0.0, gamma_1=100.0, jz=JZ)
        assert result["delta_omega"] > 10.0, (
            f"Δω = {result['delta_omega']}, expected >> 1"
        )

    def test_weak_noise_recovers_sql(self) -> None:
        """Very small γ must recover Δω ≈ SQL."""
        result = compute_sensitivity(1.0, 1.0, gamma_phi=1e-6, gamma_1=0.0, jz=JZ)
        assert np.isclose(result["delta_omega"], 1.0, rtol=1e-3), (
            f"Δω = {result['delta_omega']}, expected ~1.0"
        )


# ============================================================================
# Numerical Derivative
# ============================================================================


class TestNumericalDerivative:
    def test_fd_agrees_with_analytical_noiseless(self) -> None:
        """Finite-difference derivative must match analytical at γ=0."""
        t_hold = 1.0
        omega = 1.0
        result = compute_sensitivity(omega, t_hold, 0.0, 0.0, JZ, fd_step=1e-6)
        expected_d = 0.5 * t_hold * np.sin(omega * t_hold)
        assert np.isclose(result["d_jz_domega"], expected_d, rtol=1e-6), (
            f"d⟨J_z⟩/dω = {result['d_jz_domega']}, expected {expected_d}"
        )

    def test_fd_step_size_sweep(self) -> None:
        """FD step must be in stable range (1e-8 to 1e-4)."""
        t_hold = 1.0
        omega = 1.0
        values = []
        for step in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
            result = compute_sensitivity(omega, t_hold, 0.0, 0.0, JZ, fd_step=step)
            values.append(result["d_jz_domega"])
        expected = 0.5 * t_hold * np.sin(omega * t_hold)
        # All steps should be within 1% of the analytical value
        assert all(np.isclose(v, expected, rtol=1e-2) for v in values), (
            f"FD values: {values}, expected {expected}"
        )


# ============================================================================
# Sweep A — Holding-Time Degradation
# ============================================================================


@pytest.mark.slow
class TestSweepA:
    @pytest.fixture(scope="class")
    def result_a(self) -> NoiseSweepResult:
        """Small sweep A for testing (20 points, fewer scenarios)."""
        return sweep_t_hold(n_points=20)

    def test_sweep_type(self, result_a: NoiseSweepResult) -> None:
        assert result_a.sweep_type == "t_hold_scan"

    def test_gamma_base(self, result_a: NoiseSweepResult) -> None:
        assert result_a.gamma_base == _m.SWEEP_A_GAMMA

    def test_shape(self, result_a: NoiseSweepResult) -> None:
        """4 scenarios × 20 t_hold points = 80 rows."""
        assert len(result_a.data) == 4 * 20, f"Rows: {len(result_a.data)}"

    def test_all_scenarios_present(self, result_a: NoiseSweepResult) -> None:
        scenarios = set(result_a.data["scenario"])
        expected = {SCENARIO_CLEAN, SCENARIO_DEPHASING, SCENARIO_LOSS, SCENARIO_BOTH}
        assert scenarios == expected, f"Missing: {expected - scenarios}"

    def test_clean_ratio_is_one(self, result_a: NoiseSweepResult) -> None:
        """Clean scenario must have ratio ≈ 1 for all t_hold."""
        clean = result_a.data[result_a.data["scenario"] == SCENARIO_CLEAN]
        assert np.allclose(clean["ratio"], 1.0, rtol=1e-5), "Clean ratio ≠ 1"

    def test_dephasing_more_damaging_than_loss(
        self, result_a: NoiseSweepResult
    ) -> None:
        """Dephasing ratio > Loss ratio at large t_hold."""
        large_t = result_a.data[result_a.data["t_hold"] > 10.0]
        deph = large_t[large_t["scenario"] == SCENARIO_DEPHASING]["ratio"].mean()
        loss = large_t[large_t["scenario"] == SCENARIO_LOSS]["ratio"].mean()
        assert deph > loss, (
            f"Dephasing mean ratio {deph:.3f} ≤ Loss mean ratio {loss:.3f}"
        )

    def test_both_worst(self, result_a: NoiseSweepResult) -> None:
        """Both scenario must have worst ratio at selected t_hold values."""
        target_vals = [0.5, 1.0, 5.0, 20.0]
        available = np.sort(result_a.data["t_hold"].unique())
        for t_val in target_vals:
            # Find the closest available t_hold
            nearest = available[np.argmin(np.abs(available - t_val))]
            subset = result_a.data[np.isclose(result_a.data["t_hold"], nearest)]
            both_ratio = subset[subset["scenario"] == SCENARIO_BOTH][
                "ratio"
            ].to_numpy()[0]
            max_others = subset[subset["scenario"] != SCENARIO_BOTH]["ratio"].max()
            assert both_ratio >= max_others - 1e-12, (
                f"t≈{nearest}: Both ratio {both_ratio:.3f} < others max {max_others:.3f}"
            )


# ============================================================================
# Sweep B — Noise-Rate Scaling
# ============================================================================


@pytest.mark.slow
class TestSweepB:
    @pytest.fixture(scope="class")
    def result_b(self) -> NoiseSweepResult:
        """Small sweep B for testing (10 points)."""
        return sweep_gamma(n_points=10)

    def test_sweep_type(self, result_b: NoiseSweepResult) -> None:
        assert result_b.sweep_type == "gamma_scan"

    def test_shape(self, result_b: NoiseSweepResult) -> None:
        """3 scenarios × 10 gamma points = 30 rows."""
        assert len(result_b.data) == 3 * 10, f"Rows: {len(result_b.data)}"

    def test_no_clean_scenario(self, result_b: NoiseSweepResult) -> None:
        assert SCENARIO_CLEAN not in set(result_b.data["scenario"])

    def test_ratio_monotonic(self, result_b: NoiseSweepResult) -> None:
        """Ratio must be non-decreasing with gamma for each scenario."""
        for scenario in [SCENARIO_DEPHASING, SCENARIO_LOSS, SCENARIO_BOTH]:
            subset = result_b.data[result_b.data["scenario"] == scenario].sort_values(
                "gamma_phi" if scenario != SCENARIO_LOSS else "gamma_1"
            )
            diffs = np.diff(subset["ratio"].values)
            assert np.all(diffs >= -1e-12), (
                f"Ratio decreasing for {scenario}: min diff = {np.min(diffs)}"
            )


# ============================================================================
# Sweep C — 2D Noise Landscape
# ============================================================================


@pytest.mark.slow
class TestSweepC:
    @pytest.fixture(scope="class")
    def result_c(self) -> NoiseSweepResult:
        """Small sweep C for testing (5 × 5 = 25 points)."""
        return sweep_2d(n_points=5)

    def test_sweep_type(self, result_c: NoiseSweepResult) -> None:
        assert result_c.sweep_type == "landscape_2d"

    def test_shape(self, result_c: NoiseSweepResult) -> None:
        """5 × 5 = 25 rows."""
        assert len(result_c.data) == 25, f"Rows: {len(result_c.data)}"

    def test_unique_gamma_values(self, result_c: NoiseSweepResult) -> None:
        """Must have 5 unique gamma_phi and 5 unique gamma_1 values."""
        assert len(result_c.data["gamma_phi"].unique()) == 5
        assert len(result_c.data["gamma_1"].unique()) == 5

    def test_clean_corner(self, result_c: NoiseSweepResult) -> None:
        """Corner (γ_φ→0, γ₁→0) must have ratio ≈ 1."""
        corner = result_c.data[
            (result_c.data["gamma_phi"] == result_c.data["gamma_phi"].min())
            & (result_c.data["gamma_1"] == result_c.data["gamma_1"].min())
        ]
        corner_ratio = float(corner["ratio"].to_numpy()[0])
        assert np.isclose(corner_ratio, 1.0, atol=1e-2), (
            f"Corner ratio = {corner_ratio}"
        )

    def test_max_noise_corner(self, result_c: NoiseSweepResult) -> None:
        """Corner (max γ_φ, max γ₁) must have ratio > clean corner."""
        min_corner = float(
            result_c.data[
                (result_c.data["gamma_phi"] == result_c.data["gamma_phi"].min())
                & (result_c.data["gamma_1"] == result_c.data["gamma_1"].min())
            ]["ratio"].to_numpy()[0]
        )
        max_corner_data = result_c.data[
            (result_c.data["gamma_phi"] == result_c.data["gamma_phi"].max())
            & (result_c.data["gamma_1"] == result_c.data["gamma_1"].max())
        ]
        max_corner_ratio = float(max_corner_data["ratio"].to_numpy()[0])
        assert max_corner_ratio > min_corner, (
            f"Max-noise corner ratio = {max_corner_ratio}, min corner = {min_corner}"
        )


# ============================================================================
# NoiseSweepResult — Parquet Roundtrip
# ============================================================================


class TestNoiseSweepResultParquet:
    _FIELD_SPECS: ClassVar[list[tuple[str, str]]] = [
        ("sweep_type", "eq"),
        ("omega", "isclose"),
        ("fd_step", "isclose"),
        ("gamma_base", "isclose"),
    ]

    @pytest.fixture
    def make_result(self) -> NoiseSweepResult:
        df = pd.DataFrame(
            {
                "t_hold": [0.1, 1.0, 10.0],
                "gamma_phi": [0.0, 0.1, 0.0],
                "gamma_1": [0.0, 0.0, 0.1],
                "scenario": ["clean", "dephasing", "loss"],
                "jz_mean": [-0.5, -0.3, -0.4],
                "jz_var": [0.25, 0.20, 0.22],
                "d_jz_domega": [0.5, 0.4, 0.45],
                "delta_omega": [10.0, 12.0, 11.5],
                "sql": [10.0, 1.0, 0.1],
                "ratio": [1.0, 12.0, 115.0],
            }
        )
        return NoiseSweepResult(
            sweep_type="t_hold_scan",
            omega=1.0,
            fd_step=1e-6,
            gamma_base=0.1,
            data=df,
        )

    def test_roundtrip(self, make_result: NoiseSweepResult, tmp_path: Path) -> None:
        p = tmp_path / "sweep.parquet"
        make_result.save_parquet(p)
        loaded = NoiseSweepResult.from_parquet(p)
        assert_roundtrip_fields(loaded, make_result, self._FIELD_SPECS)
        # Check data roundtrip
        assert len(loaded.data) == len(make_result.data)
        assert np.allclose(
            loaded.data["delta_omega"].values,
            make_result.data["delta_omega"].values,
        )

    def test_fail_fast_missing_column(
        self, make_result: NoiseSweepResult, tmp_path: Path
    ) -> None:
        p = tmp_path / "bad.parquet"
        df = make_result.to_dataframe()
        df = df.drop(columns=["scenario"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            NoiseSweepResult.from_parquet(p)

    def test_fail_fast_missing_delta_omega(
        self, make_result: NoiseSweepResult, tmp_path: Path
    ) -> None:
        p = tmp_path / "bad_col.parquet"
        df = make_result.to_dataframe()
        df = df.drop(columns=["delta_omega"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            NoiseSweepResult.from_parquet(p)

    def test_sweep_type_preserved(
        self, make_result: NoiseSweepResult, tmp_path: Path
    ) -> None:
        """sweep_type metadata must survive roundtrip."""
        p = tmp_path / "sweep_type.parquet"
        make_result.save_parquet(p)
        loaded = NoiseSweepResult.from_parquet(p)
        assert loaded.sweep_type == make_result.sweep_type


# ============================================================================
# Plot Generation
# ============================================================================


@pytest.mark.slow
class TestPlots:
    @pytest.fixture(scope="class")
    def result_a(self) -> NoiseSweepResult:
        return sweep_t_hold(n_points=20)

    @pytest.fixture(scope="class")
    def result_b(self) -> NoiseSweepResult:
        return sweep_gamma(n_points=10)

    @pytest.fixture(scope="class")
    def result_c(self) -> NoiseSweepResult:
        return sweep_2d(n_points=5)

    def test_degradation_curves(
        self, result_a: NoiseSweepResult, tmp_path: Path
    ) -> None:
        svg = tmp_path / "degradation.svg"
        created = plot_degradation_curves(result_a, save_path=svg)
        assert svg.exists()
        assert created == svg

    def test_degradation_curves_auto_path(
        self,
        result_a: NoiseSweepResult,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(_m, "_fig_path", lambda name: tmp_path / f"{name}.svg")
        path = plot_degradation_curves(result_a)  # save_path=None → auto-generate
        assert path == tmp_path / "degradation-curves.svg"
        assert path.exists()

    def test_noise_rate_scaling(
        self, result_b: NoiseSweepResult, tmp_path: Path
    ) -> None:
        svg = tmp_path / "scaling.svg"
        created = plot_noise_rate_scaling(result_b, save_path=svg)
        assert svg.exists()
        assert created == svg

    def test_noise_rate_scaling_auto_path(
        self,
        result_b: NoiseSweepResult,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(_m, "_fig_path", lambda name: tmp_path / f"{name}.svg")
        path = plot_noise_rate_scaling(result_b)  # save_path=None → auto-generate
        assert path == tmp_path / "noise-rate-scaling.svg"
        assert path.exists()

    def test_landscape_2d(self, result_c: NoiseSweepResult, tmp_path: Path) -> None:
        svg = tmp_path / "landscape.svg"
        created = plot_landscape_2d(result_c, save_path=svg)
        assert svg.exists()
        assert created == svg

    def test_landscape_2d_auto_path(
        self,
        result_c: NoiseSweepResult,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(_m, "_fig_path", lambda name: tmp_path / f"{name}.svg")
        path = plot_landscape_2d(result_c)  # save_path=None → auto-generate
        assert path == tmp_path / "noise-landscape-2d.svg"
        assert path.exists()


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    def test_zero_t_hold(self) -> None:
        """t_hold=0 must give Δω = ∞."""
        result = compute_sensitivity(1.0, 0.0, 0.0, 0.0, JZ)
        assert not np.isfinite(result["delta_omega"])

    def test_negative_gamma_phi_raises(self) -> None:
        """Negative gamma_phi must raise via run_noisy_circuit."""
        with pytest.raises(ValueError, match="non-negative"):
            run_noisy_circuit(1.0, 1.0, gamma_phi=-0.1, gamma_1=0.0)

    def test_negative_gamma_1_raises(self) -> None:
        """Negative gamma_1 must raise via run_noisy_circuit."""
        with pytest.raises(ValueError, match="non-negative"):
            run_noisy_circuit(1.0, 1.0, gamma_phi=0.0, gamma_1=-0.1)

    def test_scenario_label_unknown(self) -> None:
        """Unknown scenario must raise in _scenario_rates."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            _m._scenario_rates("invalid", 0.1)

    @pytest.mark.parametrize(
        ("gamma_phi", "gamma_1", "expected"),
        [
            (0.0, 0.0, SCENARIO_CLEAN),
            (0.1, 0.0, SCENARIO_DEPHASING),
            (0.0, 0.1, SCENARIO_LOSS),
            (0.1, 0.2, SCENARIO_BOTH),
        ],
    )
    def test_scenario_label_all_branches(
        self, gamma_phi: float, gamma_1: float, expected: str
    ) -> None:
        """_scenario_label must map all four (γ_φ, γ₁) regimes correctly."""
        label = _m._scenario_label(gamma_phi, gamma_1)
        assert label == expected, (
            f"({gamma_phi}, {gamma_1}) → {label}, expected {expected}"
        )
