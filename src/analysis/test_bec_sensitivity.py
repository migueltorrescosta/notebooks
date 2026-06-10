"""Tests for BEC sensitivity analysis (src.analysis.bec_sensitivity)."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.bec_sensitivity import (
    compute_phase_uncertainty_lindblad,
    compute_sensitivity_vs_n,
)
from src.physics.noise_channels import NoiseConfig


class TestComputePhaseUncertaintyLindblad:
    """Lightweight smoke tests for Lindblad phase uncertainty.

    These tests verify that the function runs and returns a finite,
    positive value for small N. Full numerical validation requires
    dedicated scaling analysis (see reports/).
    """

    def test_returns_positive_float_for_css(self) -> None:
        from src.algorithms.spin_squeezing import coherent_spin_state

        noise = NoiseConfig()
        state = coherent_spin_state(4)
        dphi = compute_phase_uncertainty_lindblad(
            N=4,
            state=state,
            chi=1.0,
            T_decay=0.1,
            noise_config=noise,
        )
        assert np.isfinite(dphi)
        assert dphi > 0

    def test_returns_positive_float_for_twin_fock(self) -> None:
        from src.physics.states import generate_twin_fock_state

        noise = NoiseConfig()
        state = generate_twin_fock_state(4)
        dphi = compute_phase_uncertainty_lindblad(
            N=4,
            state=state,
            chi=1.0,
            T_decay=0.1,
            noise_config=noise,
        )
        assert np.isfinite(dphi)
        assert dphi > 0

    def test_raises_on_dimension_mismatch(self) -> None:
        noise = NoiseConfig()
        wrong_state = np.array([1.0, 0.0])  # dim=2 instead of N+1=3
        with pytest.raises(ValueError, match="dimension"):
            compute_phase_uncertainty_lindblad(
                N=2,
                state=wrong_state,
                chi=1.0,
                T_decay=0.1,
                noise_config=noise,
            )

    def test_raises_on_negative_chi(self) -> None:
        from src.algorithms.spin_squeezing import coherent_spin_state

        noise = NoiseConfig()
        state = coherent_spin_state(4)
        with pytest.raises(ValueError, match="non-negative"):
            compute_phase_uncertainty_lindblad(
                N=4,
                state=state,
                chi=-1.0,
                T_decay=0.1,
                noise_config=noise,
            )

    def test_raises_on_negative_time(self) -> None:
        from src.algorithms.spin_squeezing import coherent_spin_state

        noise = NoiseConfig()
        state = coherent_spin_state(4)
        with pytest.raises(ValueError, match="non-negative"):
            compute_phase_uncertainty_lindblad(
                N=4,
                state=state,
                chi=1.0,
                T_decay=-0.1,
                noise_config=noise,
            )

    def test_sql_fallback_at_zero_evolution(self) -> None:
        """At T=0 and chi=0, variance comes from initial state only."""
        from src.algorithms.spin_squeezing import coherent_spin_state

        noise = NoiseConfig()
        state = coherent_spin_state(4)
        dphi = compute_phase_uncertainty_lindblad(
            N=4,
            state=state,
            chi=0.0,
            T_decay=0.0,
            noise_config=noise,
        )
        assert np.isfinite(dphi)
        assert dphi > 0


class TestComputeSensitivityVsN:
    """Smoke tests for the sensitivity-vs-N scan.

    Only tests that the function produces a DataFrame with the expected
    structure for a small N sweep with the Lindblad method.
    """

    def test_returns_dataframe_with_expected_columns(self) -> None:
        noise = NoiseConfig()
        df = compute_sensitivity_vs_n(
            state_type="CSS",
            N_range=(4, 8),
            n_points=2,
            chi=1.0,
            noise_config=noise,
            method="Lindblad",
        )
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["N", "delta_phi"]
        assert len(df) > 0

    def test_delta_phi_values_are_finite_and_positive(self) -> None:
        noise = NoiseConfig()
        df = compute_sensitivity_vs_n(
            state_type="NOON",
            N_range=(4, 6),
            n_points=2,
            chi=1.0,
            noise_config=noise,
            method="Lindblad",
        )
        assert np.all(np.isfinite(df["delta_phi"]))
        assert np.all(df["delta_phi"] > 0)

    def test_with_noise_does_not_crash(self) -> None:
        noise = NoiseConfig(gamma_1=0.01, gamma_phi=0.01)
        df = compute_sensitivity_vs_n(
            state_type="Twin-Fock",
            N_range=(4, 8),
            n_points=2,
            chi=1.0,
            noise_config=noise,
            method="Lindblad",
        )
        assert len(df) > 0
