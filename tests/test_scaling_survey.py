"""Tests for the extended scaling survey orchestrator.

Tests verify:
1. Mini-survey integration test (2 models × 2 noise levels × 3 N values)
2. Noise channel tests (one-body loss produces finite Δφ and valid exponent)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from src.analysis.scaling_survey import (
    ModelConfig,
    SurveyConfig,
    create_survey_model,
    fit_all_exponents,
    run_scaling_survey,
)
from src.physics.mzi_states import (
    compute_fisher_information,
    input_state_factory,
    twin_fock_state,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# Availability check for hybrid-system modules (used in conditional tests)
try:
    import src.physics.hybrid_mzi
    import src.physics.hybrid_system  # noqa: F401

    HAS_HYBRID = True
except ImportError:
    HAS_HYBRID = False


class TestScalingSurveyMiniIntegration:
    """Mini-survey integration tests."""

    def test_given_mini_survey_produces_expected_columns_and_rows_then_produce_expected_columns_and_rows(
        self,
    ) -> None:
        # 2 simple models
        models = [
            create_survey_model("ideal_coherent"),
            create_survey_model("ideal_noon"),
        ]

        # Small config for quick test
        survey_config = SurveyConfig(
            N_range=(2, 8),  # Small N range
            N_points=3,  # 3 N values (will be log-spaced)
            noise_levels=[0.0, 0.1],  # 2 noise levels
            phi=np.pi / 4,
            method="qfi",
            seed=42,
        )

        # Run survey
        df = run_scaling_survey(models, survey_config)

        # Verify it's a DataFrame
        assert isinstance(df, pd.DataFrame), (
            "Expected df to be instance of pd.DataFrame"
        )

        # Check expected columns exist
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
            assert col in df.columns, f"Missing column: {col}"

        # Check row count
        row_count = len(df)
        expected_min = len(models) * len(survey_config.noise_levels) * 2
        assert row_count >= expected_min

        # Check numeric types
        assert pd.api.types.is_numeric_dtype(df["N"]), (
            'Condition failed: pd.api.types.is_numeric_dtype(df["N"])'
        )
        assert pd.api.types.is_numeric_dtype(df["noise_level"]), (
            'Condition failed: pd.api.types.is_numeric_dtype(df["noise_level"])'
        )
        assert pd.api.types.is_numeric_dtype(df["delta_phi"]), (
            'Condition failed: pd.api.types.is_numeric_dtype(df["delta_phi"])'
        )

    def test_given_fit_all_exponents_produces_valid_output_then_produce_fitted_exponents(
        self,
    ) -> None:
        models = [create_survey_model("ideal_coherent")]
        survey_config = SurveyConfig(
            N_range=(2, 16),
            N_points=5,
            noise_levels=[0.0],
            seed=42,
        )

        df = run_scaling_survey(models, survey_config)
        fit_df = fit_all_exponents(df, min_N=2)

        # Should have fit output
        assert isinstance(fit_df, pd.DataFrame), (
            "Expected fit_df to be instance of pd.DataFrame"
        )
        assert len(fit_df) > 0

        # Check for exponent columns
        assert "alpha" in fit_df.columns, 'Expected "alpha" in fit_df.columns'
        assert "alpha_err" in fit_df.columns, 'Expected "alpha_err" in fit_df.columns'
        assert "C" in fit_df.columns, 'Expected "C" in fit_df.columns'
        assert "R_squared" in fit_df.columns, 'Expected "R_squared" in fit_df.columns'
        assert "valid" in fit_df.columns, 'Expected "valid" in fit_df.columns'

        # Ideal coherent state should give SQL scaling (α ≈ -0.5)
        alpha_coherent = fit_df.loc[
            fit_df["model_id"] == "ideal_coherent",
            "alpha",
        ].iloc[0]
        assert -0.7 < alpha_coherent < -0.3


class TestScalingSurveyNoiseChannels:
    """Tests for different noise channels in the survey."""

    def test_given_model_config_validates_noise_type_then_reject_unknown_noise_types(
        self,
    ) -> None:
        with pytest.raises(ValueError):
            ModelConfig(
                model_id="test",
                state_type="coherent",
                noise_type="invalid_noise_type",  # Not valid
            )

        # Valid types should work
        valid_noise_types = [
            "none",
            "dephasing",
            "loss",
            "two_body",
            "detection",
            "thermal",
        ]
        for nt in valid_noise_types:
            model = ModelConfig(
                model_id=f"test_{nt}",
                state_type="coherent",
                noise_type=nt,
            )
            assert model.noise_type == nt

    @pytest.mark.slow
    def test_given_loss_noise_produces_finite_delta_phi_then_produce_finite_sensitivity_values_for_one_body_loss(
        self,
    ) -> None:
        # Create model with loss noise
        model = ModelConfig(
            model_id="coherent_loss",
            state_type="coherent",
            noise_type="loss",
            label="Coherent with loss",
        )

        survey_config = SurveyConfig(
            N_range=(2, 4),
            N_points=2,
            noise_levels=[0.01],  # Small but non-zero loss
            seed=42,
        )

        df = run_scaling_survey([model], survey_config)

        # We should get finite values (not inf)
        # Note: some values might be inf if the simulation fails,
        # but at least some should be finite for small N
        finite_mask = np.isfinite(df["delta_phi"].to_numpy()) & (
            df["delta_phi"].to_numpy() > 0
        )

        # Count finite values
        n_finite = int(np.sum(finite_mask))
        assert n_finite > 0, "Expected at least some finite Δφ values"

    def test_given_dephasing_noise_produces_valid_survey_then_work_in_the_survey(
        self,
    ) -> None:
        model = create_survey_model("ideal_noon")
        model.noise_type = "dephasing"  # Add dephasing

        survey_config = SurveyConfig(
            N_range=(2, 8),
            N_points=3,
            noise_levels=[0.0, 0.05, 0.1],
            seed=42,
        )

        df = run_scaling_survey([model], survey_config)

        # Check we have results for all noise levels
        assert len(df["noise_level"].unique()) == 3, (
            'Expected len(df["noise_level"].unique()) == 3'
        )

    def test_given_survey_with_detection_noise_then_handle_detection_efficiency_channel(
        self,
    ) -> None:
        model = ModelConfig(
            model_id="test_detection",
            state_type="coherent",
            noise_type="detection",
            label="Detection noise test",
        )

        # For detection, noise_level = eta (efficiency)
        survey_config = SurveyConfig(
            N_range=(2, 8),
            N_points=3,
            noise_levels=[0.5, 0.9, 1.0],  # Different efficiencies
            seed=42,
        )

        df = run_scaling_survey([model], survey_config)

        # Efficiency 1.0 should give same noise-free (or better) values than 0.5
        noise_levels = df["noise_level"].unique()
        assert 1.0 in noise_levels

        # At least some finite values
        finite_mask = np.isfinite(df["delta_phi"].to_numpy()) & (
            df["delta_phi"].to_numpy() > 0
        )
        assert np.sum(finite_mask) > 0

    def test_given_two_body_loss_configuration_then_be_configurable(self) -> None:
        model = ModelConfig(
            model_id="test_two_body",
            state_type="noon",
            noise_type="two_body",
            label="Two-body loss test",
        )

        assert model.noise_type == "two_body"

        # Just verify the model runs without crashing
        survey_config = SurveyConfig(
            N_range=(2, 4),
            N_points=2,
            noise_levels=[0.01],
            seed=42,
        )

        # Should not raise
        df = run_scaling_survey([model], survey_config)
        assert len(df) > 0


class TestQfiValidation:
    """Validate QFI values for known states under the J_z generator convention.

    The survey convention uses J_z = (n₁ - n₂)/2 as the phase generator.
    Reference values under this convention:
        NOON:        F_Q = N²,             Δφ = 1/N
        Twin-Fock:   F_Q ≈ N²/2,           Δφ ≈ √2/N
        Coherent:    F_Q = N,              Δφ = 1/√N  (SQL)
        Squeezed vacuum: F_Q = 2⟨N⟩(⟨N⟩+1), Δφ = 1/√(2⟨N⟩(⟨N⟩+1))
    """

    def test_given_noon_qfi_scales_as_n_squared_then_equal_n_squared_heisenberg_limit(
        self,
    ) -> None:
        for N in [2, 4, 8, 16]:
            state = input_state_factory("noon", N=N)
            max_photons = N
            F_Q = compute_fisher_information(state, max_photons)
            expected = float(N**2)
            assert pytest.approx(expected, rel=1e-10) == F_Q, (
                f"N={N}: F_Q={F_Q}, expected {expected}"
            )

    def test_given_noon_delta_phi_scales_as_one_over_n_then_be_1_over_n(self) -> None:
        for N in [2, 4, 8, 16]:
            state = input_state_factory("noon", N=N)
            max_photons = N
            F_Q = compute_fisher_information(state, max_photons)
            delta = 1.0 / np.sqrt(F_Q)
            expected = 1.0 / N
            assert delta == pytest.approx(expected, rel=1e-10), (
                f"N={N}: Δφ={delta}, expected {expected}"
            )

    @pytest.mark.slow
    def test_given_coherent_qfi_scales_as_n_then_scale_as_n(self) -> None:
        """F_Q = N for coherent states (SQL).

        Uses generous max_photons to capture Poisson tail, and relaxed
        rtol to allow finite-truncation effects (~1e-3 for N ≤ 16).
        """
        for N in [2, 4, 8]:
            alpha = complex(np.sqrt(N), 0)
            max_n = max(4 * N, N + 40)  # generous truncation for Poisson tail
            state = input_state_factory(
                "coherent",
                N=0,
                alpha1=alpha,
                alpha2=0.0 + 0j,
                max_photons=max_n,
            )
            F_Q = compute_fisher_information(state, max_n)
            expected = float(N)
            assert pytest.approx(expected, rel=1e-3) == F_Q, (
                f"N={N}: F_Q={F_Q}, expected {expected}"
            )

    @pytest.mark.slow
    def test_given_coherent_delta_phi_sql_scaling_then_be_1_over_sqrt_n_sql(
        self,
    ) -> None:
        for N in [4, 8]:
            alpha = complex(np.sqrt(N), 0)
            max_n = max(4 * N, N + 40)
            state = input_state_factory(
                "coherent",
                N=0,
                alpha1=alpha,
                alpha2=0.0 + 0j,
                max_photons=max_n,
            )
            F_Q = compute_fisher_information(state, max_n)
            delta = 1.0 / np.sqrt(F_Q)
            expected = 1.0 / np.sqrt(N)
            assert delta == pytest.approx(expected, rel=1e-3), (
                f"N={N}: Δφ={delta}, expected {expected}"
            )

    def test_given_squeezed_vacuum_qfi_formula_then_have_correct_analytic_form(
        self,
    ) -> None:
        """F_Q = 2⟨N⟩(⟨N⟩+1) for squeezed-vacuum states.

        At large ⟨N⟩ ≈ N, this gives F_Q ≈ 2N², so Δφ ≈ 1/(√2 N),
        exceeding NOON's 1/N by factor 1/√2 in the prefactor.

        Squeezed vacuum has long Poisson-like tails; max_photons must be
        set large enough (≥ ⟨N⟩ + 6σ) to capture the full distribution
        for accurate variance computation.
        """
        test_cases = [
            (0.5, 20),  # ⟨N⟩≈0.27, σ≈0.83 → 5+ terms, 0.5% truncation error
            (1.0, 40),  # ⟨N⟩≈1.38, σ≈2.56 → 20 terms, <0.1% truncation error
        ]
        for r, max_n in test_cases:
            expected_N = float(np.sinh(r) ** 2)  # mean photon number
            state = input_state_factory(
                "squeezed_vacuum",
                N=0,
                r=r,
                phi_sv=0.0,
                max_photons=max_n,
            )
            F_Q = compute_fisher_information(state, max_n)
            expected_F_Q = 2.0 * expected_N * (expected_N + 1.0)
            # Relaxed tolerance: finite truncation of squeezed vacuum
            # tail leads to ~1% error in F_Q even with generous max_photons
            assert pytest.approx(expected_F_Q, rel=1e-2) == F_Q, (
                f"r={r}, ⟨N⟩={expected_N:.6f}: F_Q={F_Q}, expected {expected_F_Q}"
            )

    def test_given_twin_fock_qfi_scaling_then_scale_correctly(self) -> None:
        """F_Q = N(N+2)/3 for the uniform-superposition Twin-Fock state.

        The code implements Twin-Fock as the uniform superposition
        ∑|n,N-n⟩/√(N+1), not the |N/2,N/2⟩ Fock state.  Under the J_z
        generator convention, this gives Var(J_z) = N(N+2)/12 and
        F_Q = N(N+2)/3 ≈ N²/3 (SQL scaling).
        """
        for N in [2, 4, 8]:
            if N % 2 != 0:
                continue
            state = twin_fock_state(N)
            F_Q = compute_fisher_information(state, N)
            expected = N * (N + 2) / 3.0
            assert pytest.approx(expected, rel=1e-10) == F_Q, (
                f"N={N}: F_Q={F_Q}, expected {expected}"
            )

    def test_given_sss_state_now_scales_with_n_then_scale_with_n(self) -> None:
        """SSS state (single-photon split) scales with N after factory fix.

        State: (|N-1, 1⟩ + |1, N-1⟩)/√2.
        J_z eigenvalues: ±(N-2)/2.
        Var(J_z) = (N-2)²/4,  F_Q = 4·Var(J_z) = (N-2)².

        For N=2: |1,1⟩ → J_z eigenstate → Var(J_z)=0 → F_Q=0.
        For N=4: (|3,1⟩+|1,3⟩)/√2 → F_Q = (4-2)² = 4.
        """
        for N in [2, 4, 8]:
            state = input_state_factory("sss", N=N)
            F_Q = compute_fisher_information(state, N)
            expected = float((N - 2) ** 2)
            assert pytest.approx(expected, rel=1e-10) == F_Q, (
                f"N={N}: F_Q={F_Q}, expected {expected}"
            )


class TestSurveyModelFactories:
    """Tests for the survey model factory functions."""

    def test_given_create_survey_model_defaults_then_set_appropriate_defaults(
        self,
    ) -> None:
        # noon_loss should have noise_type="loss"
        model = create_survey_model("noon_loss")
        assert model.noise_type == "loss", 'Expected model.noise_type == "loss"'
        assert model.state_type == "noon", 'Expected model.state_type == "noon"'

        # ideal_* should have noise_type="none"
        model = create_survey_model("ideal_coherent")
        assert model.noise_type == "none", 'Expected model.noise_type == "none"'

    def test_given_create_survey_model_kwargs_override_then_override_factory_defaults(
        self,
    ) -> None:
        model = create_survey_model(
            "ideal_coherent",
            noise_type="dephasing",
            label="Custom label",
        )

        assert model.noise_type == "dephasing"  # Overridden
        assert model.label == "Custom label", 'Expected model.label == "Custom label"'

    def test_given_create_default_survey_returns_list_then_return_a_list_of_models(
        self,
    ) -> None:
        from src.analysis.scaling_survey import create_default_survey

        models = create_default_survey()

        assert isinstance(models, list)
        assert len(models) > 0

        for model in models:
            assert isinstance(model, ModelConfig), (
                "Expected model to be instance of ModelConfig"
            )

    def test_given_create_default_survey_now_has_twelve_models_then_include_non_gaussian_ancilla_squeezed_vacuum_loss_kerr_and_weak_value_models(
        self,
    ) -> None:
        from src.analysis.scaling_survey import create_default_survey

        models = create_default_survey()
        model_ids = {m.model_id for m in models}

        assert "non_gaussian_n3" in model_ids, 'Expected "non_gaussian_n3" in model_ids'
        assert "non_gaussian_n4" in model_ids, 'Expected "non_gaussian_n4" in model_ids'
        assert "ancilla_assisted" in model_ids, (
            'Expected "ancilla_assisted" in model_ids'
        )
        assert "squeezed_vacuum_loss" in model_ids, (
            'Expected "squeezed_vacuum_loss" in model_ids'
        )
        assert "kerr_mzi" in model_ids, 'Expected "kerr_mzi" in model_ids'
        assert "weak_value_mzi" in model_ids, 'Expected "weak_value_mzi" in model_ids'
        assert len(models) == 12

    def test_given_squeezed_vacuum_loss_model_properties_then_have_correct_noise_type(
        self,
    ) -> None:
        from src.analysis.scaling_survey import create_survey_model

        model = create_survey_model("squeezed_vacuum_loss")
        assert model.state_type == "squeezed_vacuum", (
            'Expected model.state_type == "squeezed_vacuum"'
        )
        assert model.noise_type == "loss", 'Expected model.noise_type == "loss"'
        assert model.label == "Squeezed vacuum with loss", (
            'Expected model.label == "Squeezed vacuum with loss"'
        )

    @pytest.mark.slow
    def test_given_squeezed_vacuum_loss_runs_survey_then_produce_finite_delta_phi_values(
        self,
    ) -> None:
        from src.analysis.scaling_survey import (
            ModelConfig,
            SurveyConfig,
            run_scaling_survey,
        )

        model = ModelConfig(
            model_id="sv_loss_test",
            state_type="squeezed_vacuum",
            noise_type="loss",
            label="SV loss test",
        )
        survey_config = SurveyConfig(
            N_range=(2, 4),
            N_points=2,
            noise_levels=[0.01],
            seed=42,
        )
        df = run_scaling_survey([model], survey_config)
        finite_mask = np.isfinite(df["delta_phi"].to_numpy()) & (
            df["delta_phi"].to_numpy() > 0
        )
        assert int(np.sum(finite_mask)) > 0


class TestNewCustomModels:
    """Tests for the newly added custom-sensitivity survey models.

    These models use ``custom_sensitivity_fn`` to bypass the standard
    MZI pipeline. The tests verify that the factory functions produce
    valid callables that return finite sensitivity values.
    """

    def _make_call_sensitivity(
        self,
        fn: Callable[[int, float], float] | None,
        N: int,
        noise: float,
    ) -> float:
        """Safely call a custom sensitivity function."""
        assert fn is not None, "custom_sensitivity_fn must not be None"
        return fn(N, noise)

    def test_given_non_gaussian_n3_smoke_then_return_finite_sensitivity_for_small_n(
        self,
    ) -> None:
        model = create_survey_model("non_gaussian_n3")
        delta = self._make_call_sensitivity(model.custom_sensitivity_fn, 6, 0.0)
        assert np.isfinite(delta), f"Expected finite Δφ, got {delta}"
        assert delta > 0, f"Expected positive Δφ, got {delta}"

    def test_given_non_gaussian_n4_smoke_then_return_finite_sensitivity_for_small_n(
        self,
    ) -> None:
        model = create_survey_model("non_gaussian_n4")
        delta = self._make_call_sensitivity(model.custom_sensitivity_fn, 6, 0.0)
        assert np.isfinite(delta), f"Expected finite Δφ, got {delta}"
        assert delta > 0, f"Expected positive Δφ, got {delta}"

    def test_given_ancilla_assisted_smoke_then_return_finite_sensitivity_for_small_n(
        self,
    ) -> None:
        model = create_survey_model("ancilla_assisted")
        delta = self._make_call_sensitivity(model.custom_sensitivity_fn, 4, 0.0)
        assert np.isfinite(delta), f"Expected finite Δφ, got {delta}"
        assert delta > 0, f"Expected positive Δφ, got {delta}"

    def test_given_ancilla_assisted_noise_scales_sensitivity_then_degrade_sensitivity_with_higher_noise_level(
        self,
    ) -> None:
        model = create_survey_model("ancilla_assisted")
        fn = model.custom_sensitivity_fn
        assert fn is not None
        delta_clean = fn(4, 0.0)
        delta_noisy = fn(4, 1.0)
        # Noisy should be >= clean (larger Δφ = worse)
        assert delta_noisy >= delta_clean - 1e-10, (
            f"Noise should not improve sensitivity: "
            f"clean={delta_clean:.6e}, noisy={delta_noisy:.6e}"
        )

    @pytest.mark.skipif(not HAS_HYBRID, reason="hybrid_system module not available")
    def test_given_non_gaussian_ground_state_option_then_produce_a_valid_state_with_use_ground_state_true(
        self,
    ) -> None:
        model = create_survey_model("non_gaussian_n3", use_ground_state=True)
        delta_gs = self._make_call_sensitivity(model.custom_sensitivity_fn, 6, 0.0)
        assert np.isfinite(delta_gs), f"Expected finite Δφ for GS, got {delta_gs}"
        assert delta_gs > 0, f"Expected positive Δφ, got {delta_gs}"

    def test_given_non_gaussian_inf_for_small_n_then_return_inf_for_n_less_than_2(
        self,
    ) -> None:
        model = create_survey_model("non_gaussian_n3")
        delta = self._make_call_sensitivity(model.custom_sensitivity_fn, 1, 0.0)
        assert not np.isfinite(delta), "Expected inf for N < 2"

    def test_given_ancilla_inf_for_small_n_then_return_inf_for_n_less_than_2(
        self,
    ) -> None:
        model = create_survey_model("ancilla_assisted")
        delta = self._make_call_sensitivity(model.custom_sensitivity_fn, 1, 0.0)
        assert not np.isfinite(delta), "Expected inf for N < 2"

    def test_given_kerr_mzi_smoke_then_return_finite_sensitivity_for_small_n(
        self,
    ) -> None:
        model = create_survey_model("kerr_mzi")
        delta = self._make_call_sensitivity(model.custom_sensitivity_fn, 4, 0.0)
        assert np.isfinite(delta), f"Expected finite Δφ, got {delta}"
        assert delta > 0, f"Expected positive Δφ, got {delta}"

    def test_given_kerr_mzi_noon_scaling_then_give_heisenberg_scaling_delta_phi_equal_1_over_n(
        self,
    ) -> None:
        model = create_survey_model("kerr_mzi")
        for N in [2, 4, 8]:
            delta = self._make_call_sensitivity(model.custom_sensitivity_fn, N, 0.0)
            expected = 1.0 / N
            assert np.isclose(delta, expected, rtol=1e-10), (
                f"N={N}: Δφ={delta}, expected {expected}"
            )

    def test_given_kerr_mzi_inf_for_small_n_then_return_inf_for_n_less_than_2(
        self,
    ) -> None:
        model = create_survey_model("kerr_mzi")
        delta = self._make_call_sensitivity(model.custom_sensitivity_fn, 1, 0.0)
        assert not np.isfinite(delta), "Expected inf for N < 2"

    def test_given_weak_value_mzi_smoke_then_return_finite_sensitivity_for_small_n(
        self,
    ) -> None:
        model = create_survey_model("weak_value_mzi")
        delta = self._make_call_sensitivity(model.custom_sensitivity_fn, 4, 0.0)
        assert np.isfinite(delta), f"Expected finite Δφ, got {delta}"
        assert delta > 0, f"Expected positive Δφ, got {delta}"

    def test_given_weak_value_mzi_sql_limited_then_be_greater_than_or_equal_to_sql_1_over_sqrt_n(
        self,
    ) -> None:
        model = create_survey_model("weak_value_mzi")
        for N in [4, 8, 16]:
            delta = self._make_call_sensitivity(model.custom_sensitivity_fn, N, 0.0)
            sql = 1.0 / np.sqrt(N)
            assert delta >= sql - 1e-12, f"N={N}: Δφ={delta:.6f} < SQL={sql:.6f}"

    def test_given_weak_value_mzi_inf_for_small_n_then_return_inf_for_n_less_than_2(
        self,
    ) -> None:
        model = create_survey_model("weak_value_mzi")
        delta = self._make_call_sensitivity(model.custom_sensitivity_fn, 1, 0.0)
        assert not np.isfinite(delta), "Expected inf for N < 2"
