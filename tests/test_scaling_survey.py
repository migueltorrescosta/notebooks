"""Tests for the extended scaling survey orchestrator.

Tests verify:
1. Mini-survey integration test (2 models × 2 noise levels × 3 N values)
2. Noise channel tests (one-body loss produces finite Δφ and valid exponent)
"""

from __future__ import annotations

from typing import Callable

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

# Availability check for hybrid-system modules (used in conditional tests)
try:
    import src.physics.hybrid_system  # noqa: F401
    import src.physics.hybrid_mzi  # noqa: F401

    HAS_HYBRID = True
except ImportError:
    HAS_HYBRID = False


class TestScalingSurveyMiniIntegration:
    """Mini-survey integration tests."""

    def test_mini_survey_produces_expected_columns_and_rows(self) -> None:
        """Test that a mini-survey produces the expected DataFrame structure.

        2 models × 2 noise levels × 3 N values = 12 rows expected
        (assuming all computations succeed).
        """
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
        assert isinstance(df, pd.DataFrame)

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
        assert pd.api.types.is_numeric_dtype(df["N"])
        assert pd.api.types.is_numeric_dtype(df["noise_level"])
        assert pd.api.types.is_numeric_dtype(df["delta_phi"])

    def test_fit_all_exponents_produces_valid_output(self) -> None:
        """Test that fit_all_exponents produces fitted exponents."""
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
        assert isinstance(fit_df, pd.DataFrame)
        assert len(fit_df) > 0

        # Check for exponent columns
        assert "alpha" in fit_df.columns
        assert "alpha_err" in fit_df.columns
        assert "C" in fit_df.columns
        assert "R_squared" in fit_df.columns
        assert "valid" in fit_df.columns

        # Ideal coherent state should give SQL scaling (α ≈ -0.5)
        alpha_coherent = fit_df.loc[
            fit_df["model_id"] == "ideal_coherent", "alpha"
        ].iloc[0]
        assert -0.7 < alpha_coherent < -0.3


class TestScalingSurveyNoiseChannels:
    """Tests for different noise channels in the survey."""

    def test_model_config_validates_noise_type(self) -> None:
        """Test that ModelConfig rejects unknown noise types."""
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

    def test_loss_noise_produces_finite_delta_phi(self) -> None:
        """Test that one-body loss channel produces finite sensitivity values."""
        # Create model with loss noise
        model = ModelConfig(
            model_id="coherent_loss",
            state_type="coherent",
            noise_type="loss",
            label="Coherent with loss",
        )

        survey_config = SurveyConfig(
            N_range=(2, 8),
            N_points=3,
            noise_levels=[0.01],  # Small but non-zero loss
            seed=42,
        )

        df = run_scaling_survey([model], survey_config)

        # We should get finite values (not inf)
        # Note: some values might be inf if the simulation fails,
        # but at least some should be finite for small N
        finite_mask = np.isfinite(df["delta_phi"].values) & (df["delta_phi"].values > 0)

        # Count finite values
        n_finite = int(np.sum(finite_mask))
        assert n_finite > 0, "Expected at least some finite Δφ values"

    def test_dephasing_noise_produces_valid_survey(self) -> None:
        """Test that dephasing noise works in the survey."""
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
        assert len(df["noise_level"].unique()) == 3

    def test_survey_with_detection_noise(self) -> None:
        """Test detection noise (efficiency) channel."""
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
        finite_mask = np.isfinite(df["delta_phi"].values) & (df["delta_phi"].values > 0)
        assert np.sum(finite_mask) > 0

    def test_two_body_loss_configuration(self) -> None:
        """Test that two-body loss can be configured."""
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


class TestSurveyModelFactories:
    """Tests for the survey model factory functions."""

    def test_create_survey_model_defaults(self) -> None:
        """Test that create_survey_model sets appropriate defaults."""
        # noon_loss should have noise_type="loss"
        model = create_survey_model("noon_loss")
        assert model.noise_type == "loss"
        assert model.state_type == "noon"

        # ideal_* should have noise_type="none"
        model = create_survey_model("ideal_coherent")
        assert model.noise_type == "none"

    def test_create_survey_model_kwargs_override(self) -> None:
        """Test that kwargs override factory defaults."""
        model = create_survey_model(
            "ideal_coherent",
            noise_type="dephasing",
            label="Custom label",
        )

        assert model.noise_type == "dephasing"  # Overridden
        assert model.label == "Custom label"

    def test_create_default_survey_returns_list(self) -> None:
        """Test that create_default_survey returns a list of models."""
        from src.analysis.scaling_survey import create_default_survey

        models = create_default_survey()

        assert isinstance(models, list)
        assert len(models) > 0

        for model in models:
            assert isinstance(model, ModelConfig)

    def test_create_default_survey_now_has_ten_models(self) -> None:
        """Default survey should include non-Gaussian, ancilla, and squeezed-vacuum-loss models."""
        from src.analysis.scaling_survey import create_default_survey

        models = create_default_survey()
        model_ids = {m.model_id for m in models}

        assert "non_gaussian_n3" in model_ids
        assert "non_gaussian_n4" in model_ids
        assert "ancilla_assisted" in model_ids
        assert "squeezed_vacuum_loss" in model_ids
        assert len(models) == 10

    def test_squeezed_vacuum_loss_model_properties(self) -> None:
        """Squeezed-vacuum-loss model should have correct noise_type."""
        from src.analysis.scaling_survey import create_survey_model

        model = create_survey_model("squeezed_vacuum_loss")
        assert model.state_type == "squeezed_vacuum"
        assert model.noise_type == "loss"
        assert model.label == "Squeezed vacuum with loss"

    def test_squeezed_vacuum_loss_runs_survey(self) -> None:
        """Squeezed vacuum with loss should produce finite Δφ values in a survey."""
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
            N_range=(2, 6),
            N_points=3,
            noise_levels=[0.01],
            seed=42,
        )
        df = run_scaling_survey([model], survey_config)
        finite_mask = np.isfinite(df["delta_phi"].values) & (df["delta_phi"].values > 0)
        assert int(np.sum(finite_mask)) > 0


class TestNewCustomModels:
    """Tests for the newly added custom-sensitivity survey models.

    These models use ``custom_sensitivity_fn`` to bypass the standard
    MZI pipeline. The tests verify that the factory functions produce
    valid callables that return finite sensitivity values.
    """

    def _call_sensitivity(
        self, fn: Callable[[int, float], float] | None, N: int, noise: float
    ) -> float:
        """Safely call a custom sensitivity function."""
        assert fn is not None, "custom_sensitivity_fn must not be None"
        return fn(N, noise)

    def test_non_gaussian_n3_smoke(self) -> None:
        """Non-Gaussian n=3 model should return finite sensitivity for small N."""
        model = create_survey_model("non_gaussian_n3")
        delta = self._call_sensitivity(model.custom_sensitivity_fn, 6, 0.0)
        assert np.isfinite(delta), f"Expected finite Δφ, got {delta}"
        assert delta > 0, f"Expected positive Δφ, got {delta}"

    def test_non_gaussian_n4_smoke(self) -> None:
        """Non-Gaussian n=4 model should return finite sensitivity for small N."""
        model = create_survey_model("non_gaussian_n4")
        delta = self._call_sensitivity(model.custom_sensitivity_fn, 6, 0.0)
        assert np.isfinite(delta), f"Expected finite Δφ, got {delta}"
        assert delta > 0, f"Expected positive Δφ, got {delta}"

    def test_ancilla_assisted_smoke(self) -> None:
        """Ancilla-assisted model should return finite sensitivity for small N."""
        model = create_survey_model("ancilla_assisted")
        delta = self._call_sensitivity(model.custom_sensitivity_fn, 4, 0.0)
        assert np.isfinite(delta), f"Expected finite Δφ, got {delta}"
        assert delta > 0, f"Expected positive Δφ, got {delta}"

    def test_ancilla_assisted_noise_scales_sensitivity(self) -> None:
        """Higher noise_level should degrade (increase) sensitivity for ancilla model."""
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
    def test_non_gaussian_ground_state_option(self) -> None:
        """Using use_ground_state=True should produce a valid state."""
        model = create_survey_model("non_gaussian_n3", use_ground_state=True)
        delta_gs = self._call_sensitivity(model.custom_sensitivity_fn, 6, 0.0)
        assert np.isfinite(delta_gs), f"Expected finite Δφ for GS, got {delta_gs}"
        assert delta_gs > 0, f"Expected positive Δφ, got {delta_gs}"

    def test_non_gaussian_inf_for_small_N(self) -> None:
        """Non-Gaussian sensitivity should return inf for N < 2."""
        model = create_survey_model("non_gaussian_n3")
        delta = self._call_sensitivity(model.custom_sensitivity_fn, 1, 0.0)
        assert not np.isfinite(delta), "Expected inf for N < 2"

    def test_ancilla_inf_for_small_N(self) -> None:
        """Ancilla sensitivity should return inf for N < 2."""
        model = create_survey_model("ancilla_assisted")
        delta = self._call_sensitivity(model.custom_sensitivity_fn, 1, 0.0)
        assert not np.isfinite(delta), "Expected inf for N < 2"
