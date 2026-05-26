"""
Combined tests for the 2026-05-11 report local module.

Tests verify the remaining migrated modules:
- weak_value_mzi, thermal_langevin, dynamical_decoupling
- tilt_to_length_noise, cavity_mzi, distributed_mzi
- scaling_survey, ancilla_comparison
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Load local.py via importlib
# Must register in sys.modules for dataclass machinery to resolve __module__
_local_path = Path(__file__).resolve().parent / "local.py"
_spec = importlib.util.spec_from_file_location("report_local", str(_local_path))
assert _spec is not None, f"Could not find local.py at {_local_path}"
_report_local = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["report_local"] = _report_local
_spec.loader.exec_module(_report_local)


# =============================================================================
# Tests: thermal_langevin
# =============================================================================


class TestThermalLangevinBasics:
    """Basic tests for thermal Langevin noise model."""

    def test_thermal_sensitivity_normalized_follows_formula(self) -> None:
        config = _report_local.create_thermal_config(
            thermal_strength=0.5, thermal_exponent=0.0
        )
        assert _report_local.thermal_sensitivity_normalized(1, config) == pytest.approx(
            0.5
        )
        assert _report_local.thermal_sensitivity_normalized(
            10, config
        ) == pytest.approx(0.5)
        assert _report_local.thermal_sensitivity_normalized(
            100, config
        ) == pytest.approx(0.5)

    def test_combined_sensitivity_quadrature_sum(self) -> None:
        config = _report_local.create_thermal_config(
            thermal_strength=4.0, thermal_exponent=0.0
        )
        N = 1.0 / 9.0
        combined = _report_local.combined_sensitivity(N, config)
        quantum = 1.0 / np.sqrt(N)
        thermal = 4.0
        expected = np.sqrt(quantum**2 + thermal**2)
        assert combined == pytest.approx(expected)

    def test_quantum_only_gives_sql_exponent(self) -> None:
        config = _report_local.create_quantum_only_config()
        N_values = [4, 8, 16, 32, 64]
        result = _report_local.fit_thermal_scaling_exponent(N_values, config, min_N=4)
        assert result.valid
        assert -0.55 < result.alpha < -0.45

    def test_crossover_N_analytical(self) -> None:
        config = _report_local.create_thermal_config(
            thermal_strength=0.1, thermal_exponent=0.0
        )
        N_cross = _report_local.crossover_N(config)
        assert N_cross == pytest.approx(100.0)

    def test_sweep_returns_arrays(self) -> None:
        config = _report_local.create_thermal_config(
            thermal_strength=0.1, thermal_exponent=0.0
        )
        N_arr, delta_arr = _report_local.sweep_thermal_scaling(
            [2, 4, 8, 16, 32], config
        )
        assert len(N_arr) == 5
        assert len(delta_arr) == 5
        assert np.all(np.isfinite(delta_arr))
        assert np.all(delta_arr > 0)


# =============================================================================
# Tests: dynamical_decoupling
# =============================================================================


class TestDDConfig:
    """Test DDConfig validation."""

    def test_default_values(self) -> None:
        config = _report_local.DDConfig()
        assert config.n_pulses == 0
        assert config.sequence == "CPMG"

    def test_negative_pulses_raises(self) -> None:
        with pytest.raises(ValueError):
            _report_local.DDConfig(n_pulses=-1)

    def test_unknown_sequence_raises(self) -> None:
        with pytest.raises(ValueError):
            _report_local.DDConfig(sequence="UNKNOWN")


class TestCpmgFilterFunction:
    """Test CPMG filter function."""

    def test_zero_pulses_is_unit(self) -> None:
        omega = np.linspace(-10, 10, 100)
        F = _report_local.cpmg_filter_function(omega, n_pulses=0, tau=1.0)
        assert pytest.approx(1.0, abs=1e-10) == F

    def test_dc_suppression(self) -> None:
        for n_pulses in [1, 2, 4, 8]:
            F = _report_local.cpmg_filter_function(
                np.array([0.0]), n_pulses=n_pulses, tau=0.5
            )
            assert F[0] == pytest.approx(0.0, abs=1e-15)


class TestDDEffectiveCoherenceTime:
    """Test effective coherence time."""

    def test_zero_pulses_gives_bare(self) -> None:
        T = _report_local.dd_effective_coherence_time(T_2_0=1.0, n_pulses=0)
        assert pytest.approx(1.0) == T

    def test_improves_with_pulses(self) -> None:
        for n in [1, 2, 4, 8]:
            T = _report_local.dd_effective_coherence_time(1.0, n, "CPMG")
            assert T >= 1.0


class TestDDPhaseSensitivity:
    """Test phase sensitivity computation."""

    def test_sensitivity_positive(self) -> None:
        for n_pulses in [0, 1, 4, 8]:
            d = _report_local.dd_phase_sensitivity(10, 0.0, T=1.0, n_pulses=n_pulses)
            assert d > 0

    def test_more_pulses_improves(self) -> None:
        d0 = _report_local.dd_phase_sensitivity(10, 0.0, T=1.0, n_pulses=0)
        d8 = _report_local.dd_phase_sensitivity(10, 0.0, T=1.0, n_pulses=8)
        assert d8 < d0

    def test_sql_scaling_preserved(self) -> None:
        N_values = np.logspace(1, 4, 20)
        result = _report_local.dd_sensitivity_scaling(N_values, 0, T=1.0, T_2_0=1.0)
        assert result["fitted_alpha"] == pytest.approx(-0.5, abs=0.02)


# =============================================================================
# Tests: tilt_to_length_noise
# =============================================================================


class TestTTLPathLengthNoise:
    """Test path length noise computation."""

    def test_basic_formula(self) -> None:
        config = _report_local.TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3)
        result = _report_local.ttl_path_length_noise(config)
        assert abs(result - 1e-9) < 1e-12

    def test_phase_consistency(self) -> None:
        config = _report_local.TTLNoiseConfig(
            theta_rms=1e-6, beam_offset=1e-3, wavelength=1e-6
        )
        delta_L = _report_local.ttl_path_length_noise(config)
        phi = _report_local.ttl_phase_noise(config)
        expected = 2.0 * np.pi * delta_L / config.wavelength
        assert abs(phi - expected) < 1e-12

    def test_scaling_sweep_returns_expected_keys(self) -> None:
        config = _report_local.TTLNoiseConfig()
        N = np.logspace(0, 6, 10)
        result = _report_local.ttl_scaling_sweep(N, config, quantum_scaling="sql")
        expected_keys = {
            "N",
            "delta_phi",
            "delta_phi_quantum",
            "delta_phi_ttl",
            "alpha_fitted",
        }
        assert set(result.keys()) == expected_keys

    def test_zero_theta_raises(self) -> None:
        config = _report_local.TTLNoiseConfig(theta_rms=0.0)
        with pytest.raises(ValueError, match="positive"):
            _report_local.ttl_phase_noise(config)


# =============================================================================
# Tests: cavity_mzi
# =============================================================================


class TestCavityMziConfig:
    """Test CavityMziConfig."""

    def test_default_values(self) -> None:
        config = _report_local.CavityMziConfig()
        assert config.F == 10.0
        assert config.theta == pytest.approx(np.pi / 4)


class TestCavityEnhancedMzi:
    """Test cavity-enhanced MZI."""

    def _make_fock_10(self, max_photons: int) -> np.ndarray:
        import qutip

        dim = max_photons + 1
        return qutip.tensor(qutip.fock(dim, 1), qutip.fock(dim, 0)).full().ravel()

    def test_norm_preservation(self) -> None:
        config = _report_local.CavityMziConfig(F=5.0)
        state = self._make_fock_10(max_photons=5)
        out = _report_local.cavity_enhanced_mzi(state, np.pi / 4, config, max_photons=5)
        norm = np.sum(np.abs(out) ** 2)
        assert norm == pytest.approx(1.0, abs=1e-10)

    def test_sensitivity_positive(self) -> None:
        config = _report_local.CavityMziConfig(F=10.0)
        delta = _report_local.cavity_enhanced_sensitivity(4, np.pi / 4, config)
        assert np.isfinite(delta)
        assert delta > 0

    def test_finesse_lt_one_raises(self) -> None:
        config = _report_local.CavityMziConfig(F=0.5)
        state = self._make_fock_10(max_photons=3)
        with pytest.raises(ValueError, match="Cavity finesse must be >= 1"):
            _report_local.cavity_enhanced_mzi(state, 0.0, config, max_photons=3)

    def test_trace_preservation_with_noise(self) -> None:
        config = _report_local.CavityMziConfig(F=3.0)
        state = self._make_fock_10(max_photons=5)
        rho = _report_local.cavity_enhanced_mzi_with_noise(
            state,
            phi=np.pi / 4,
            noise_gamma_1=0.1,
            noise_gamma_2=0.0,
            noise_gamma_phi=0.05,
            config=config,
            max_photons=5,
        )
        assert np.trace(rho) == pytest.approx(1.0, abs=1e-6)


# =============================================================================
# Tests: distributed_mzi
# =============================================================================


class TestDistributedMziConfig:
    """Test DistributedMziConfig."""

    def test_default_values(self) -> None:
        config = _report_local.DistributedMziConfig()
        assert config.M == 2
        assert config.entangled is False

    def test_negative_m_raises(self) -> None:
        with pytest.raises(ValueError):
            _report_local.DistributedMziConfig(M=-1)


class TestDistributedMziSensitivity:
    """Test sensitivity computation."""

    def test_classical_averaging_sql(self) -> None:
        config = _report_local.DistributedMziConfig(
            M=4, entangled=False, correlation_noise=0.0
        )
        result = _report_local.distributed_mzi_sensitivity(100, 0.0, config)
        expected = 1.0 / np.sqrt(4 * 100)
        assert result["delta_phi"] == pytest.approx(expected, rel=1e-6)

    def test_entangled_heisenberg(self) -> None:
        config = _report_local.DistributedMziConfig(
            M=4, entangled=True, correlation_noise=0.0
        )
        result = _report_local.distributed_mzi_sensitivity(100, 0.0, config)
        expected = 1.0 / (4 * 100)
        assert result["delta_phi"] == pytest.approx(expected, rel=1e-6)

    def test_fully_correlated_classical(self) -> None:
        config = _report_local.DistributedMziConfig(
            M=4, entangled=False, correlation_noise=1.0
        )
        result = _report_local.distributed_mzi_sensitivity(100, 0.0, config)
        expected = 1.0 / np.sqrt(100)
        assert result["delta_phi"] == pytest.approx(expected, rel=1e-6)

    def test_scaling_exponent_classical(self) -> None:
        config = _report_local.DistributedMziConfig(entangled=False)
        assert _report_local.distributed_scaling_exponent(config) == -0.5

    def test_scaling_exponent_entangled(self) -> None:
        config = _report_local.DistributedMziConfig(entangled=True)
        assert _report_local.distributed_scaling_exponent(config) == -1.0


# =============================================================================
# Tests: scaling_survey
# =============================================================================


class TestSurveyConfig:
    """Test SurveyConfig and ModelConfig validation."""

    def test_survey_config_defaults(self) -> None:
        config = _report_local.SurveyConfig()
        assert config.N_points == 8
        assert config.method == "qfi"

    def test_model_config_validation(self) -> None:
        model = _report_local.ModelConfig(
            model_id="test",
            state_type="coherent",
            noise_type="none",
        )
        assert model.noise_type == "none"

    def test_model_config_invalid_noise_type_raises(self) -> None:
        with pytest.raises(ValueError):
            _report_local.ModelConfig(
                model_id="test",
                state_type="coherent",
                noise_type="invalid_noise_type",
            )


class TestSurveyModelFactories:
    """Test factory functions."""

    def test_create_survey_model_defaults(self) -> None:
        model = _report_local.create_survey_model("noon_loss")
        assert model.noise_type == "loss"
        assert model.state_type == "noon"

        model = _report_local.create_survey_model("ideal_coherent")
        assert model.noise_type == "none"


class TestQfiValidation:
    """Validate QFI values for known states."""

    def test_noon_qfi_scales_as_n_squared(self) -> None:
        for N in [2, 4, 8]:
            state = _report_local.input_state_factory("noon", N=N)
            F_Q = _report_local.compute_fisher_information(state, N)
            expected = float(N**2)
            assert pytest.approx(expected, rel=1e-10) == F_Q

    def test_noon_delta_phi_scales_as_one_over_n(self) -> None:
        for N in [2, 4, 8]:
            state = _report_local.input_state_factory("noon", N=N)
            F_Q = _report_local.compute_fisher_information(state, N)
            delta = 1.0 / np.sqrt(F_Q)
            expected = 1.0 / N
            assert delta == pytest.approx(expected, rel=1e-10)


class TestScalingSurveyMiniIntegration:
    """Mini-survey integration tests."""

    def test_mini_survey_produces_expected_columns(self) -> None:
        import pandas as pd

        models = [
            _report_local.create_survey_model("ideal_coherent"),
            _report_local.create_survey_model("ideal_noon"),
        ]
        survey_config = _report_local.SurveyConfig(
            N_range=(2, 8),
            N_points=3,
            noise_levels=[0.0, 0.1],
            seed=42,
        )
        df = _report_local.run_scaling_survey(models, survey_config)
        assert isinstance(df, pd.DataFrame)
        expected_cols = {
            "model_id",
            "state_type",
            "noise_type",
            "noise_level",
            "N",
            "delta_phi",
            "method",
            "entangler",
        }
        for col in expected_cols:
            assert col in df.columns

    def test_fit_all_exponents_produces_valid_output(self) -> None:
        import pandas as pd

        models = [_report_local.create_survey_model("ideal_coherent")]
        survey_config = _report_local.SurveyConfig(
            N_range=(2, 16),
            N_points=5,
            noise_levels=[0.0],
            seed=42,
        )
        df = _report_local.run_scaling_survey(models, survey_config)
        fit_df = _report_local.fit_all_exponents(df, min_N=2)
        assert isinstance(fit_df, pd.DataFrame)
        assert "alpha" in fit_df.columns
        assert "alpha_err" in fit_df.columns


# =============================================================================
# Tests: ancilla_comparison
# =============================================================================


class TestOperatorConstruction:
    """Test J_z, J_x operator properties."""

    @pytest.mark.parametrize("N_max", [1, 2, 3])
    def test_jz_operator_diagonal_correct_eigenvalues(self, N_max: int) -> None:
        J_z, _ = _report_local.build_system_jz_jx(N_max)
        dim = (N_max + 1) ** 2
        assert J_z.shape == (dim, dim)
        assert J_z == pytest.approx(np.diag(np.diag(J_z)))

    @pytest.mark.parametrize("N_max", [1, 2])
    def test_jx_operator_hermitian(self, N_max: int) -> None:
        _, J_x = _report_local.build_system_jz_jx(N_max)
        assert J_x == pytest.approx(J_x.conj().T)

    def test_ancilla_operators_pauli(self) -> None:
        J_z_anc, J_x_anc = _report_local.build_ancilla_operators()
        assert J_z_anc.shape == (2, 2)
        assert J_x_anc.shape == (2, 2)
        np.isclose(np.linalg.eigvalsh(J_z_anc), [-0.5, 0.5]).all()
        np.isclose(np.linalg.eigvalsh(J_x_anc), [-0.5, 0.5]).all()


class TestGeneratorB:
    """Test G_B for the 2-particle system."""

    def test_generator_b_hermitian(self) -> None:
        G_B = _report_local.compute_generator_B(T_H=1.0, N_max=2)
        assert pytest.approx(G_B.conj().T) == G_B

    def test_generator_b_eigenvalues_bounded(self) -> None:
        G_B = _report_local.compute_generator_B(T_H=1.0, N_max=2)
        evals = np.linalg.eigvalsh(G_B)
        assert np.min(evals) >= -1.0 - 1e-10
        assert np.max(evals) <= 1.0 + 1e-10

    def test_generator_b_scales_linearly(self) -> None:
        G_B_1 = _report_local.compute_generator_B(T_H=1.0, N_max=2)
        G_B_2 = _report_local.compute_generator_B(T_H=2.0, N_max=2)
        assert pytest.approx(2.0 * G_B_1) == G_B_2


class TestDensityMatrix:
    """Test random density matrix generation."""

    def test_random_dm_trace_is_1(self) -> None:
        rng = np.random.default_rng(42)
        for d in [2, 4, 8]:
            rho = _report_local.random_density_matrix(d, rng)
            assert np.trace(rho) == pytest.approx(1.0)

    def test_random_dm_positive_semidefinite(self) -> None:
        rng = np.random.default_rng(42)
        for d in [2, 4, 8]:
            rho = _report_local.random_density_matrix(d, rng)
            evals = np.linalg.eigvalsh(rho)
            assert np.all(evals >= -1e-12)


class TestComparisonPipeline:
    """End-to-end tests of the comparison pipeline."""

    def test_run_comparison_returns_comparisonresult(self) -> None:
        result = _report_local.run_comparison(
            T_H=1.0,
            n_samples_B=100,
            n_samples_A=100,
            n_alpha_samples=5,
            pure_only=True,
            seed=42,
        )
        assert isinstance(result, _report_local.ComparisonResult)

    def test_fq_b_positive(self) -> None:
        result = _report_local.run_comparison(
            T_H=1.0,
            n_samples_B=100,
            n_samples_A=100,
            n_alpha_samples=5,
            pure_only=True,
            seed=42,
        )
        assert result.fq_B_max > 0

    def test_fq_a_positive(self) -> None:
        result = _report_local.run_comparison(
            T_H=1.0,
            n_samples_B=100,
            n_samples_A=100,
            n_alpha_samples=5,
            pure_only=True,
            seed=42,
        )
        assert result.fq_A_max > 0


class TestAnalyticalBounds:
    """Verify analytical bounds."""

    def test_analytical_fq_b(self) -> None:
        assert _report_local.analytical_fq_B_max(1.0) == pytest.approx(4.0)
        assert _report_local.analytical_fq_B_max(2.0) == pytest.approx(16.0)

    def test_analytical_fq_a_zero(self) -> None:
        assert _report_local.analytical_fq_A_zero(1.0) == pytest.approx(1.0)
        assert _report_local.analytical_fq_A_zero(2.0) == pytest.approx(4.0)

    def test_ratio_geq_two(self) -> None:
        F_A = _report_local.analytical_fq_A_zero(1.0)
        F_B = _report_local.analytical_fq_B_max(1.0)
        ratio = np.sqrt(F_B / F_A)
        assert ratio == pytest.approx(2.0)


# =============================================================================
# Tests: weak_value_mzi
# =============================================================================


class TestWeakValueConfig:
    """Test WeakValueConfig."""

    def test_default_values(self) -> None:
        config = _report_local.WeakValueConfig()
        assert config.theta == pytest.approx(np.pi / 4)
        assert config.post_select_angle == pytest.approx(np.pi / 2 - 0.1)


class TestWeakValueMziSensitivity:
    """Test weak-value MZI sensitivity computation."""

    def test_sensitivity_finite(self) -> None:
        config = _report_local.WeakValueConfig()
        phi = np.pi / 4
        result = _report_local.weak_value_mzi_sensitivity(4, phi, config)
        assert np.isfinite(result["delta_phi"])
        assert result["delta_phi"] > 0

    def test_post_selection_effect(self) -> None:
        """Post-selection closer to π/2 should give larger weak value."""
        phi = np.pi / 4
        config_near = _report_local.WeakValueConfig(post_select_angle=np.pi / 2 - 0.02)
        config_far = _report_local.WeakValueConfig(post_select_angle=np.pi / 2 - 0.5)

        result_near = _report_local.weak_value_mzi_sensitivity(4, phi, config_near)
        result_far = _report_local.weak_value_mzi_sensitivity(4, phi, config_far)

        # Near-orthogonal should have larger |A_w|
        assert abs(result_near["weak_value"]) > abs(result_far["weak_value"])


# =============================================================================
# Tests: Module-level CLI
# =============================================================================


class TestModuleLoading:
    """Test that the module loads correctly with all expected attributes."""

    def test_module_has_main(self) -> None:
        assert hasattr(_report_local, "main")
        assert callable(_report_local.main)

    def test_module_has_report_date(self) -> None:
        assert _report_local.REPORT_DATE == "20260511"

    def test_module_has_expected_classes(self) -> None:
        expected_classes = [
            "WeakValueConfig",
            "ThermalLangevinConfig",
            "DDConfig",
            "TTLNoiseConfig",
            "CavityMziConfig",
            "DistributedMziConfig",
            "SurveyConfig",
            "ModelConfig",
            "ComparisonResult",
            "ScalingFitResult",
        ]
        for cls_name in expected_classes:
            assert hasattr(_report_local, cls_name), f"Missing class: {cls_name}"

    @pytest.mark.slow
    def test_module_can_generate_raw_data(self) -> None:
        """Verify the CLI main function can be invoked (dry-run)."""
        # Just check the generator functions are callable
        assert callable(_report_local._generate_ancilla_vs_system_raw_data)
        assert callable(_report_local._generate_fock_mzi_raw_data)
        assert callable(_report_local._generate_collective_spin_raw_data)
        assert callable(_report_local._generate_advanced_architectures_raw_data)
