"""Smoke tests for visualization module."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


class TestFourierTransform:
    def test_fourier_transform_runs_without_error(self):
        """Test that fourier_transform executes without raising an exception."""
        # Mock matplotlib to avoid display errors
        with patch("src.visualization.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_axs = [MagicMock(), MagicMock(), MagicMock()]
            mock_subplots.return_value = (mock_fig, mock_axs)

            # Simple Gaussian function
            def f(t):
                return np.exp(-(t**2))

            from src.visualization import fourier_transform

            # Should not raise
            fourier_transform(f, a=-10, b=10, time_domain_n=100)

    def test_fourier_transform_with_default_parameters(self):
        with patch("src.visualization.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_axs = [MagicMock(), MagicMock(), MagicMock()]
            mock_subplots.return_value = (mock_fig, mock_axs)

            def f(t):
                return np.sin(t)

            from src.visualization import fourier_transform

            # Should not raise with defaults
            fourier_transform(f)


class TestFiniteDimensionalPopulationsOverTime:
    def test_runs_with_valid_inputs(self):
        """Test with valid Hamiltonian and density matrix."""
        # 2x2 Hamiltonian (simple case)
        hamiltonian = np.array([[1, 0], [0, -1]])
        rho0 = np.array([[1, 0], [0, 0]])  # Pure state |0><0|

        with (
            patch("src.visualization.plt.stackplot"),
            patch("src.visualization.plt.legend"),
        ):
            from src.visualization import finite_dimensional_populations_over_time

            # Should not raise
            finite_dimensional_populations_over_time(
                hamiltonian, rho0, time_window_upper_bound=1
            )

    def test_raises_on_non_square_hamiltonian(self):
        hamiltonian = np.array([[1, 0, 0], [0, -1, 0]])  # Not square
        rho0 = np.array([[1, 0], [0, 0]])

        from src.visualization import finite_dimensional_populations_over_time

        with pytest.raises(AssertionError, match="not square"):
            finite_dimensional_populations_over_time(hamiltonian, rho0)

    def test_raises_on_mismatched_dimensions(self):
        hamiltonian = np.array([[1, 0], [0, -1]])
        rho0 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])  # 3x3, wrong size

        from src.visualization import finite_dimensional_populations_over_time

        with pytest.raises(AssertionError, match="does not match"):
            finite_dimensional_populations_over_time(hamiltonian, rho0)

    def test_raises_on_negative_time(self):
        hamiltonian = np.array([[1, 0], [0, -1]])
        rho0 = np.array([[1, 0], [0, 0]])

        from src.visualization import finite_dimensional_populations_over_time

        with pytest.raises(AssertionError, match="negative"):
            finite_dimensional_populations_over_time(
                hamiltonian, rho0, time_window_upper_bound=-1
            )

    def test_raises_on_non_traceless_rho0(self):
        hamiltonian = np.array([[1, 0], [0, -1]])
        rho0 = np.array([[0.5, 0], [0, 0.3]])  # Trace = 0.8, not 1

        from src.visualization import finite_dimensional_populations_over_time

        with pytest.raises(AssertionError, match="add up to 1"):
            finite_dimensional_populations_over_time(hamiltonian, rho0)

    def test_accepts_custom_labels(self):
        hamiltonian = np.array([[1, 0], [0, -1]])
        rho0 = np.array([[1, 0], [0, 0]])

        with (
            patch("src.visualization.plt.stackplot"),
            patch("src.visualization.plt.legend"),
        ):
            from src.visualization import finite_dimensional_populations_over_time

            finite_dimensional_populations_over_time(
                hamiltonian, rho0, labels=["ground", "excited"]
            )


class TestQuantumStateHeatmap:
    def test_runs_with_valid_inputs(self):
        """Test with valid Hamiltonian and density matrix."""
        hamiltonian = np.array([[1, 0], [0, -1]])
        rho0 = np.array([[1, 0], [0, 0]])

        with (
            patch("src.visualization.plt.subplots") as mock_subplots,
            patch("src.visualization.plt.draw"),
        ):
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            from src.visualization import quantum_state_heatmap

            # Should not raise
            quantum_state_heatmap(hamiltonian, rho0, t=0)

    def test_accepts_time_parameter(self):
        hamiltonian = np.array([[1, 0], [0, -1]])
        rho0 = np.array([[1, 0], [0, 0]])

        with (
            patch("src.visualization.plt.subplots") as mock_subplots,
            patch("src.visualization.plt.draw"),
        ):
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            from src.visualization import quantum_state_heatmap

            quantum_state_heatmap(hamiltonian, rho0, t=1.5)
