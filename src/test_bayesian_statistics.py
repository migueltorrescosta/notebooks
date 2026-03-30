"""Smoke tests for bayesian_statistics module."""

import numpy as np
import pandas as pd
import pytest

from src.bayesian_statistics import BayesUpdate


class TestBayesUpdate:
    def test_instantiation_with_valid_columns(self) -> None:
        """Test that BayesUpdate can be instantiated with correct columns."""
        df = pd.DataFrame(
            {"prior": [0.5, 0.5], "likelihood": [0.8, 0.2]}, index=["A", "B"]
        )
        # Should not raise
        bayes = BayesUpdate(df)
        assert bayes is not None

    def test_raises_on_missing_columns(self) -> None:
        """Test that BayesUpdate raises on missing columns."""
        df = pd.DataFrame({"prior": [0.5, 0.5]})
        with pytest.raises(AssertionError):
            BayesUpdate(df)

    def test_raises_on_extra_columns(self) -> None:
        """Test that BayesUpdate raises on extra columns."""
        df = pd.DataFrame(
            {"prior": [0.5, 0.5], "likelihood": [0.8, 0.2], "extra": [1, 2]},
            index=["A", "B"],
        )
        with pytest.raises(AssertionError):
            BayesUpdate(df)

    def test_posterior_is_calculated(self) -> None:
        """Test that posterior column is created and normalized."""
        df = pd.DataFrame(
            {"prior": [0.5, 0.5], "likelihood": [0.8, 0.2]}, index=["A", "B"]
        )
        bayes = BayesUpdate(df)
        assert "posterior" in bayes.df.columns

    def test_posterior_sums_to_one(self) -> None:
        """Test that posterior is normalized."""
        df = pd.DataFrame(
            {"prior": [0.5, 0.5], "likelihood": [0.8, 0.2]}, index=["A", "B"]
        )
        bayes = BayesUpdate(df)
        assert np.isclose(bayes.df["posterior"].sum(), 1.0)

    def test_normalize_columns_works(self) -> None:
        """Test that normalize_columns divides by row sums."""
        df = pd.DataFrame(
            {"prior": [1.0, 2.0], "likelihood": [4.0, 2.0]}, index=["A", "B"]
        )
        bayes = BayesUpdate(df)
        # After normalization, each column should sum to 1
        assert np.isclose(bayes.df["prior"].sum(), 1.0)
        assert np.isclose(bayes.df["likelihood"].sum(), 1.0)

    def test_calculate_posterior_basic(self) -> None:
        """Test posterior calculation: prior * likelihood."""
        df = pd.DataFrame(
            {"prior": [0.4, 0.6], "likelihood": [0.5, 0.5]}, index=["A", "B"]
        )
        bayes = BayesUpdate(df)
        # posterior = prior * likelihood, then normalized
        expected_A = (0.4 * 0.5) / ((0.4 * 0.5) + (0.6 * 0.5))  # = 0.4
        expected_B = (0.6 * 0.5) / ((0.4 * 0.5) + (0.6 * 0.5))  # = 0.6
        assert np.isclose(float(bayes.df.loc["A", "posterior"]), expected_A)  # type: ignore[arg-type]
        assert np.isclose(float(bayes.df.loc["B", "posterior"]), expected_B)  # type: ignore[arg-type]

    def test_plot_returns_figure(self) -> None:
        """Test that plot method returns a figure."""
        df = pd.DataFrame(
            {"prior": [0.5, 0.5], "likelihood": [0.8, 0.2]}, index=["A", "B"]
        )
        bayes = BayesUpdate(df)

        # Test that plot returns a figure object without error
        fig = bayes.plot()
        assert fig is not None
        # Close the figure to avoid display issues
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_categorical_index_detection(self) -> None:
        """Test that categorical index is detected."""
        df = pd.DataFrame(
            {"prior": [0.5, 0.5], "likelihood": [0.8, 0.2]},
            index=["A", "B"],  # String index = categorical
        )
        bayes = BayesUpdate(df)
        assert bayes.is_categorical is True

    def test_numeric_index_detection(self) -> None:
        """Test that numeric index is detected."""
        df = pd.DataFrame(
            {"prior": [0.5, 0.5], "likelihood": [0.8, 0.2]},
            index=[0, 1],  # Integer index = numeric
        )
        bayes = BayesUpdate(df)
        assert bayes.is_categorical is False
