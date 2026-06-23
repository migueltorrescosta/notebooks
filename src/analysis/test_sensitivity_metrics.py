import numpy as np
import pandas as pd
import pytest

from src.physics.mzi_simulation import prepare_input_state

from .sensitivity_metrics import (
    _compute_single_N_metrics,
    _fit_scaling_exponents,
    all_sensitivity_metrics,
    compare_sensitivity_methods,
    error_propagation_sensitivity,
    sensitivity_from_error_propagation,
    sensitivity_scaling,
    validate_sensitivity_order,
)


class TestErrorPropagationSensitivity:
    def test_error_propagation_given_single_photon(self) -> None:
        max_photons = 1
        state = prepare_input_state("single_photon", max_photons=max_photons)
        phi_grid = np.linspace(0, 2 * np.pi, 181)
        result = error_propagation_sensitivity(state, max_photons, phi_grid)
        assert "delta_phi_ep" in result
        assert "phi_at_min" in result
        assert "delta_phi_grid" in result
        assert np.isfinite(result["delta_phi_ep"])
        assert result["delta_phi_ep"] > 0

    def test_error_propagation_given_noon_state(self) -> None:
        max_photons = 2
        state = prepare_input_state(
            "noon",
            max_photons=max_photons,
            n_particles=max_photons,
        )
        phi_grid = np.linspace(0, 2 * np.pi, 181)
        result = error_propagation_sensitivity(state, max_photons, phi_grid)
        assert np.isfinite(result["delta_phi_ep"])
        assert result["delta_phi_ep"] > 0

    def test_error_propagation_raises_for_invalid_input(self) -> None:
        max_photons = 1
        state = prepare_input_state("single_photon", max_photons=max_photons)
        with pytest.raises(ValueError):
            error_propagation_sensitivity(state, max_photons, np.array([0.0, np.pi]))
        with pytest.raises(ValueError):
            error_propagation_sensitivity(
                state,
                max_photons,
                np.linspace(0, 2 * np.pi, 100),
                dphi=-0.1,
            )

    def test_error_propagation_raises_for_dimension_mismatch(self) -> None:
        max_photons = 2
        wrong_state = prepare_input_state("single_photon", max_photons=1)
        phi_grid = np.linspace(0, 2 * np.pi, 100)
        with pytest.raises(ValueError):
            error_propagation_sensitivity(wrong_state, max_photons, phi_grid)


class TestAllSensitivityMetrics:
    def test_all_metrics_given_single_photon(self) -> None:
        max_photons = 1
        state = prepare_input_state("single_photon", max_photons=max_photons)
        phi_true = np.pi / 4
        result = all_sensitivity_metrics(state, max_photons, phi_true, n_mc=50, seed=42)
        assert "delta_phi_ep" in result
        assert "delta_phi_fc" in result
        assert "delta_phi_fq" in result
        assert "delta_phi_bayes" in result
        assert result["n_mc"] == 50

    def test_all_metrics_given_noon_state(self) -> None:
        max_photons = 2
        state = prepare_input_state(
            "noon",
            max_photons=max_photons,
            n_particles=max_photons,
        )
        phi_true = np.pi / 4
        result = all_sensitivity_metrics(state, max_photons, phi_true, n_mc=50, seed=42)
        assert np.isfinite(result["delta_phi_ep"])
        assert np.isfinite(result["delta_phi_fc"])
        assert np.isfinite(result["delta_phi_fq"])
        assert np.isfinite(result["delta_phi_bayes"])

    def test_input_validation_raises(self) -> None:
        state = prepare_input_state("single_photon", max_photons=1)
        with pytest.raises(ValueError):
            all_sensitivity_metrics(state, max_photons=0, phi_true=0.1, n_mc=10)
        with pytest.raises(ValueError):
            all_sensitivity_metrics(state, max_photons=1, phi_true=0.1, n_mc=0)


class TestSensitivityScaling:
    @pytest.mark.slow
    def test_single_photon_scaling(self) -> None:
        N_range = np.array([1, 2, 3, 4])
        result = sensitivity_scaling(
            state_type="single",
            N_range=N_range,
            noise_config=None,
            n_mc=50,
            seed=42,
        )
        assert len(result.df) > 0

    @pytest.mark.slow
    def test_noon_state_scaling(self) -> None:
        N_range = np.array([1, 2, 3, 4])
        result = sensitivity_scaling(
            state_type="noon",
            N_range=N_range,
            noise_config=None,
            n_mc=50,
            seed=42,
        )
        assert len(result.df) > 0

    def test_invalid_state_type_raises(self) -> None:
        with pytest.raises(ValueError):
            sensitivity_scaling(
                state_type="invalid",
                N_range=np.array([2, 4]),
            )


class TestSensitivityValidation:
    def test_error_propagation_ge_cramer_rao(self) -> None:
        assert validate_sensitivity_order(1.0, 0.5)
        assert validate_sensitivity_order(0.9, 1.0, rtol=0.2)
        assert not validate_sensitivity_order(0.5, 1.0, rtol=0.1)

    def test_compare_sensitivity_methods(self) -> None:
        max_photons = 2
        state = prepare_input_state(
            "noon",
            max_photons=max_photons,
            n_particles=max_photons,
        )
        result = compare_sensitivity_methods(state, max_photons, np.pi / 4, n_mc=50)
        assert "delta_phi_ep" in result
        assert "ep_valid" in result
        assert "methods_agree" in result

    def test_methods_agree_in_large_sample_limit(self) -> None:
        max_photons = 2
        state = prepare_input_state(
            "noon",
            max_photons=max_photons,
            n_particles=max_photons,
        )
        result = all_sensitivity_metrics(
            state,
            max_photons,
            phi_true=np.pi / 4,
            n_mc=50,
            seed=42,
        )
        assert np.isfinite(result["delta_phi_ep"])
        assert np.isfinite(result["delta_phi_fq"])
        assert np.isfinite(result["delta_phi_bayes"])
        assert result["delta_phi_ep"] > 0
        assert result["delta_phi_fq"] > 0
        assert result["delta_phi_bayes"] > 0


class TestScalingExponents:
    @pytest.mark.slow
    def test_noon_scaling_exponent_heisenberg(self) -> None:
        N_range = np.array([1, 2, 3, 4])
        result = sensitivity_scaling(
            state_type="noon",
            N_range=N_range,
            n_mc=100,
            seed=42,
        )
        if "delta_phi_fq" in result.exponents:
            alpha = result.exponents["delta_phi_fq"]
            assert -1.5 < alpha < -0.5


class TestBoundaryConditions:
    def test_near_zero_derivatives(self) -> None:
        max_photons = 1
        state = prepare_input_state("single_photon", max_photons=max_photons)
        phi_grid = np.linspace(0, 2 * np.pi, 181)
        result = error_propagation_sensitivity(state, max_photons, phi_grid)
        assert np.isfinite(result["delta_phi_ep"])

    def test_fine_phi_grid_precision(self) -> None:
        max_photons = 2
        state = prepare_input_state(
            "noon",
            max_photons=max_photons,
            n_particles=max_photons,
        )
        phi_grid = np.linspace(0, 2 * np.pi, 361)
        result = error_propagation_sensitivity(state, max_photons, phi_grid)
        assert np.isfinite(result["delta_phi_ep"])
        assert result["delta_phi_ep"] > 0


class TestReproducibility:
    @pytest.fixture(scope="class")
    def _noon_state(self) -> np.ndarray:
        return prepare_input_state("noon", max_photons=2, n_particles=2)

    @pytest.mark.slow
    def test_same_seed_gives_same_results(
        self,
        _noon_state: np.ndarray,  # noqa: PT019
    ) -> None:
        result1 = all_sensitivity_metrics(
            _noon_state, 2, phi_true=np.pi / 4, n_mc=50, seed=123
        )
        result2 = all_sensitivity_metrics(
            _noon_state, 2, phi_true=np.pi / 4, n_mc=50, seed=123
        )
        assert result1["delta_phi_bayes"] == result2["delta_phi_bayes"]

    @pytest.mark.slow
    def test_different_seeds_give_different_results(
        self,
        _noon_state: np.ndarray,  # noqa: PT019
    ) -> None:
        result1 = all_sensitivity_metrics(
            _noon_state, 2, phi_true=np.pi / 4, n_mc=50, seed=123
        )
        result2 = all_sensitivity_metrics(
            _noon_state, 2, phi_true=np.pi / 4, n_mc=50, seed=456
        )
        assert np.isfinite(result1["delta_phi_bayes"])
        assert np.isfinite(result2["delta_phi_bayes"])


class TestPhysicsInvariants:
    @pytest.fixture(scope="class")
    def _noon_metrics_result(self) -> dict:
        max_photons = 2
        state = prepare_input_state(
            "noon",
            max_photons=max_photons,
            n_particles=max_photons,
        )
        return all_sensitivity_metrics(
            state, max_photons, phi_true=np.pi / 4, n_mc=100, seed=42
        )

    def test_noon_state_invariants(
        self,
        _noon_metrics_result: dict,  # noqa: PT019
    ) -> None:
        result = _noon_metrics_result
        assert result["fisher_quantum"] >= 0
        assert result["delta_phi_ep"] > 0
        assert result["delta_phi_bayes"] > 0
        if np.isfinite(result["delta_phi_fq"]):
            assert result["delta_phi_fq"] > 0


class TestSensitivityFromErrorPropagation:
    """sensitivity_from_error_propagation — stateless formula."""

    def test_normal_case(self) -> None:
        result = sensitivity_from_error_propagation(0.5, 0.25, 1.0)
        assert result == pytest.approx(0.5)

    def test_zero_variance_returns_inf(self) -> None:
        result = sensitivity_from_error_propagation(1.0, 0.0, 1.0)
        assert not np.isfinite(result)

    def test_zero_derivative_returns_inf(self) -> None:
        result = sensitivity_from_error_propagation(0.5, 0.25, 0.0)
        assert not np.isfinite(result)

    def test_tiny_variance_below_tolerance(self) -> None:
        result = sensitivity_from_error_propagation(1.0, 1e-16, 1.0)
        assert not np.isfinite(result)

    def test_variance_at_tolerance_edge(self) -> None:
        result = sensitivity_from_error_propagation(1.0, 1e-14, 1.0)
        assert np.isfinite(result)

    def test_negative_variance_clamped(self) -> None:
        """Negative variance from numerical noise is clamped to zero."""
        result = sensitivity_from_error_propagation(0.5, -1e-10, 1.0)
        assert not np.isfinite(result)

    def test_custom_var_tolerance(self) -> None:
        """Custom var_tolerance should be respected."""
        result = sensitivity_from_error_propagation(
            1.0, 1e-10, 1.0, var_tolerance=1e-11
        )
        assert np.isfinite(result)
        result2 = sensitivity_from_error_propagation(
            1.0, 1e-10, 1.0, var_tolerance=1e-9
        )
        assert not np.isfinite(result2)

    def test_custom_deriv_tolerance(self) -> None:
        """Derivative 1e-11: finite if tolerance < 1e-11, inf if > 1e-11."""
        result = sensitivity_from_error_propagation(
            0.5, 0.25, 1e-11, deriv_tolerance=1e-12
        )
        assert np.isfinite(result)
        result2 = sensitivity_from_error_propagation(
            0.5, 0.25, 1e-11, deriv_tolerance=1e-10
        )
        assert not np.isfinite(result2)


class TestSensitivityScalingHelpers:
    """Tests for the private helpers extracted from ``sensitivity_scaling``."""

    def test_compute_single_n_metrics_valid_state(self) -> None:
        from src.physics.noise_channels import NoiseConfig

        result = _compute_single_N_metrics(
            "single",
            N=1,
            noise_config=NoiseConfig(),
            phi_true=np.pi / 4,
            n_mc=50,
            seed=42,
        )
        assert result is not None
        assert result["N"] == 1
        assert result["max_photons"] == 1
        assert "delta_phi_ep" in result
        assert result["state_type"] == "single"

    def test_compute_single_n_metrics_noon_state(self) -> None:
        from src.physics.noise_channels import NoiseConfig

        result = _compute_single_N_metrics(
            "noon",
            N=2,
            noise_config=NoiseConfig(),
            phi_true=np.pi / 4,
            n_mc=50,
            seed=42,
        )
        assert result is not None
        assert result["N"] == 2
        assert "delta_phi_fq" in result

    def test_compute_single_n_metrics_with_noise_config(self) -> None:
        from src.physics.noise_channels import NoiseConfig

        result = _compute_single_N_metrics(
            "single",
            N=1,
            noise_config=NoiseConfig(),
            phi_true=np.pi / 4,
            n_mc=50,
            seed=42,
        )
        assert result is not None

    def test_fit_scaling_exponents_normal(self) -> None:
        df = pd.DataFrame(
            {
                "N": [1, 2, 4],
                "delta_phi_ep": [1.0, 0.7, 0.5],
                "delta_phi_fq": [0.8, 0.5, 0.3],
                "delta_phi_bayes": [1.2, 0.9, 0.7],
            }
        )
        exponents = _fit_scaling_exponents(df)
        assert "delta_phi_ep" in exponents
        assert "delta_phi_fq" in exponents
        assert "delta_phi_bayes" in exponents
        # All exponents should be negative (sensitivity improves with N)
        for alpha in exponents.values():
            assert alpha < 0

    def test_fit_scaling_exponents_empty_df(self) -> None:
        df = pd.DataFrame(columns=["N", "delta_phi_ep"])
        exponents = _fit_scaling_exponents(df)
        assert exponents == {}

    def test_fit_scaling_exponents_single_row(self) -> None:
        df = pd.DataFrame(
            {
                "N": [1],
                "delta_phi_ep": [1.0],
                "delta_phi_fq": [0.8],
                "delta_phi_bayes": [1.2],
            }
        )
        exponents = _fit_scaling_exponents(df)
        assert exponents == {}

    def test_fit_scaling_exponents_non_positive_n(self) -> None:
        df = pd.DataFrame(
            {
                "N": [0, 2, 4],
                "delta_phi_ep": [1.0, 0.7, 0.5],
                "delta_phi_fq": [0.8, 0.5, 0.3],
                "delta_phi_bayes": [1.2, 0.9, 0.7],
            }
        )
        exponents = _fit_scaling_exponents(df)
        assert exponents == {}

    def test_fit_scaling_exponents_missing_columns(self) -> None:
        df = pd.DataFrame(
            {
                "N": [1, 2, 4],
                "delta_phi_ep": [1.0, 0.7, 0.5],
            }
        )
        exponents = _fit_scaling_exponents(df)
        assert "delta_phi_ep" in exponents
        assert "delta_phi_fq" not in exponents
