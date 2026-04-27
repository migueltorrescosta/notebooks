"""Unit tests for visualization module.

These tests verify that visualization functions:
1. Execute without errors
2. Call matplotlib functions with correct arguments
3. Properly validate input parameters
4. Handle edge cases correctly
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestFourierTransform:
    def test_creates_three_subplots(self) -> None:
        """Fourier transform should create a figure with exactly 3 subplots."""
        with patch("src.visualization.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_axs = [MagicMock(), MagicMock(), MagicMock()]
            mock_subplots.return_value = (mock_fig, mock_axs)

            from src.visualization.visualization import fourier_transform

            def f(t: float) -> float:
                return np.exp(-(t**2))

            fourier_transform(f, a=-10, b=10, time_domain_n=100)

            mock_subplots.assert_called_once_with(3)
            assert len(mock_axs) == 3

    def test_sets_correct_titles(self) -> None:
        """Each subplot should have the expected title."""
        with patch("src.visualization.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_axs = [MagicMock(), MagicMock(), MagicMock()]
            mock_subplots.return_value = (mock_fig, mock_axs)

            from src.visualization.visualization import fourier_transform

            fourier_transform(lambda t: t, a=0, b=1, time_domain_n=10)

            mock_axs[0].set_title.assert_called_once_with("Time Domain")
            mock_axs[1].set_title.assert_called_once_with(
                "Frequency Domain, absolute value"
            )
            mock_axs[2].set_title.assert_called_once_with("Frequency Domain, angle")

    def test_calls_plot_for_each_subplot(self) -> None:
        """Each subplot should have plot called on it."""
        with patch("src.visualization.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_axs = [MagicMock(), MagicMock(), MagicMock()]
            mock_subplots.return_value = (mock_fig, mock_axs)

            from src.visualization.visualization import fourier_transform

            fourier_transform(lambda t: t, a=0, b=1, time_domain_n=10)

            for ax in mock_axs:
                ax.plot.assert_called()

    def test_uses_fig_tight_layout(self) -> None:
        """Figure should use tight_layout for proper spacing."""
        with patch("src.visualization.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_axs = [MagicMock(), MagicMock(), MagicMock()]
            mock_subplots.return_value = (mock_fig, mock_axs)

            from src.visualization.visualization import fourier_transform

            fourier_transform(lambda t: 1.0, a=0, b=1, time_domain_n=10)

            mock_fig.tight_layout.assert_called_once_with(pad=1)

    def test_default_parameters_work(self) -> None:
        """Default parameters should allow the function to run."""
        with patch("src.visualization.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_axs = [MagicMock(), MagicMock(), MagicMock()]
            mock_subplots.return_value = (mock_fig, mock_axs)

            from src.visualization.visualization import fourier_transform

            # Should not raise
            fourier_transform(lambda t: np.sin(t))

    @pytest.mark.parametrize("n_samples", [10, 100, 1000])
    def test_different_sample_counts(self, n_samples: int) -> None:
        """Function should work with various sample counts."""
        with patch("src.visualization.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_axs = [MagicMock(), MagicMock(), MagicMock()]
            mock_subplots.return_value = (mock_fig, mock_axs)

            from src.visualization.visualization import fourier_transform

            fourier_transform(lambda t: 1.0, a=-1, b=1, time_domain_n=n_samples)

            # Verify plot was called (function executed successfully)
            for ax in mock_axs:
                ax.plot.assert_called()


class TestFiniteDimensionalPopulationsOverTime:
    """Tests for finite_dimensional_populations_over_time function."""

    @pytest.fixture
    def simple_hamiltonian(self) -> np.ndarray:
        """Simple 2x2 diagonal Hamiltonian."""
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    @pytest.fixture
    def pure_state(self) -> np.ndarray:
        """Pure density matrix for |0⟩⟨0|."""
        return np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)

    def test_calls_stackplot(
        self, simple_hamiltonian: np.ndarray, pure_state: np.ndarray
    ) -> None:
        """Function should call plt.stackplot to create the population plot."""
        with patch("src.visualization.plt.stackplot") as mock_stackplot:
            with patch("src.visualization.plt.legend"):
                from src.visualization.visualization import (
                    finite_dimensional_populations_over_time,
                )

                finite_dimensional_populations_over_time(
                    simple_hamiltonian, pure_state, time_window_upper_bound=10
                )

                mock_stackplot.assert_called()

    def test_stackplot_called_with_time_axis(
        self, simple_hamiltonian: np.ndarray, pure_state: np.ndarray
    ) -> None:
        """Stackplot should receive the time axis as first argument."""
        with patch("src.visualization.plt.stackplot") as mock_stackplot:
            with patch("src.visualization.plt.legend"):
                from src.visualization.visualization import (
                    finite_dimensional_populations_over_time,
                )

                finite_dimensional_populations_over_time(
                    simple_hamiltonian, pure_state, time_window_upper_bound=10
                )

                # First argument should be time axis (array of floats)
                call_args = mock_stackplot.call_args
                time_axis = call_args[0][0]
                assert isinstance(time_axis, np.ndarray)
                assert time_axis[0] == 0  # Should start at 0

    def test_stackplot_called_with_populations(
        self, simple_hamiltonian: np.ndarray, pure_state: np.ndarray
    ) -> None:
        """Stackplot should receive population arrays (array-like, time-invariant)."""
        with patch("src.visualization.plt.stackplot") as mock_stackplot:
            with patch("src.visualization.plt.legend"):
                from src.visualization.visualization import (
                    finite_dimensional_populations_over_time,
                )

                finite_dimensional_populations_over_time(
                    simple_hamiltonian, pure_state, time_window_upper_bound=10
                )

                # Check that stackplot was called
                mock_stackplot.assert_called()
                # First arg should be time axis, second should be array-like populations
                call_args = mock_stackplot.call_args
                assert isinstance(call_args[0][0], np.ndarray)  # time axis
                assert isinstance(
                    call_args[0][1], np.ndarray
                )  # populations (array-like)

    def test_legend_is_called(
        self, simple_hamiltonian: np.ndarray, pure_state: np.ndarray
    ) -> None:
        """Function should call plt.legend when labels are provided."""
        with patch("src.visualization.plt.stackplot"):
            with patch("src.visualization.plt.legend") as mock_legend:
                from src.visualization.visualization import (
                    finite_dimensional_populations_over_time,
                )

                finite_dimensional_populations_over_time(
                    simple_hamiltonian,
                    pure_state,
                    time_window_upper_bound=10,
                    labels=["ground", "excited"],
                )

                mock_legend.assert_called()

    def test_raises_on_non_square_hamiltonian(self) -> None:
        """Should raise AssertionError for non-square Hamiltonian."""
        hamiltonian = np.array([[1.0, 0.0], [0.0, -1.0], [0.0, 0.0]], dtype=complex)
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)

        from src.visualization.visualization import (
            finite_dimensional_populations_over_time,
        )

        with pytest.raises(AssertionError, match="not square"):
            finite_dimensional_populations_over_time(hamiltonian, rho0)

    def test_raises_on_dimension_mismatch(self) -> None:
        """Should raise AssertionError when rho0 dimensions don't match Hamiltonian."""
        hamiltonian = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        rho0 = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=complex
        )

        from src.visualization.visualization import (
            finite_dimensional_populations_over_time,
        )

        with pytest.raises(AssertionError, match="does not match"):
            finite_dimensional_populations_over_time(hamiltonian, rho0)

    def test_raises_on_negative_time(
        self, simple_hamiltonian: np.ndarray, pure_state: np.ndarray
    ) -> None:
        """Should raise AssertionError for negative time window."""
        from src.visualization.visualization import (
            finite_dimensional_populations_over_time,
        )

        with pytest.raises(AssertionError, match="negative"):
            finite_dimensional_populations_over_time(
                simple_hamiltonian, pure_state, time_window_upper_bound=-1
            )

    def test_raises_on_zero_time(
        self, simple_hamiltonian: np.ndarray, pure_state: np.ndarray
    ) -> None:
        """Should raise AssertionError for zero time window."""
        from src.visualization.visualization import (
            finite_dimensional_populations_over_time,
        )

        with pytest.raises(AssertionError, match="negative"):
            finite_dimensional_populations_over_time(
                simple_hamiltonian, pure_state, time_window_upper_bound=0
            )

    def test_raises_on_non_traceless_rho0(self, simple_hamiltonian: np.ndarray) -> None:
        """Should raise AssertionError when rho0 trace != 1."""
        rho0 = np.array([[0.8, 0.0], [0.0, 0.1]], dtype=complex)  # Trace = 0.9

        from src.visualization.visualization import (
            finite_dimensional_populations_over_time,
        )

        with pytest.raises(AssertionError, match="add up to 1"):
            finite_dimensional_populations_over_time(simple_hamiltonian, rho0)

    def test_raises_on_wrong_number_of_labels(
        self, simple_hamiltonian: np.ndarray, pure_state: np.ndarray
    ) -> None:
        """Should raise AssertionError when label count doesn't match state count."""
        from src.visualization.visualization import (
            finite_dimensional_populations_over_time,
        )

        with pytest.raises(AssertionError, match="labels"):
            finite_dimensional_populations_over_time(
                simple_hamiltonian, pure_state, labels=["only_one"]
            )

    def test_accepts_valid_labels(
        self, simple_hamiltonian: np.ndarray, pure_state: np.ndarray
    ) -> None:
        """Should accept labels array with correct count."""
        with patch("src.visualization.plt.stackplot"):
            with patch("src.visualization.plt.legend"):
                from src.visualization.visualization import (
                    finite_dimensional_populations_over_time,
                )

                # Should not raise
                finite_dimensional_populations_over_time(
                    simple_hamiltonian, pure_state, labels=["state_0", "state_1"]
                )


class TestQuantumStateHeatmap:
    """Tests for quantum_state_heatmap function."""

    @pytest.fixture
    def mixed_state(self) -> np.ndarray:
        """Mixed density matrix with both diagonal and off-diagonal elements."""
        return np.array([[0.5, 0.3], [0.3, 0.5]], dtype=complex)

    def test_creates_two_subplots(self) -> None:
        """Heatmap should create a figure with exactly 2 subplots (real and imag)."""
        hamiltonian = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)

        with patch("src.visualization.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            from src.visualization.visualization import quantum_state_heatmap

            quantum_state_heatmap(hamiltonian, rho0, t=0)

            mock_subplots.assert_called_once_with(1, 2)

    def test_both_subplots_show_imshow(self, mixed_state: np.ndarray) -> None:
        """Both subplots should call imshow to display the density matrix."""
        hamiltonian = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

        with patch("src.visualization.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            from src.visualization.visualization import quantum_state_heatmap

            quantum_state_heatmap(hamiltonian, mixed_state, t=0)

            mock_ax1.imshow.assert_called()
            mock_ax2.imshow.assert_called()

    def test_imshow_uses_correct_vmin_vmax(self, mixed_state: np.ndarray) -> None:
        """Imshow should use vmin=-1 and vmax=1 for proper color scaling."""
        hamiltonian = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

        with patch("src.visualization.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            from src.visualization.visualization import quantum_state_heatmap

            quantum_state_heatmap(hamiltonian, mixed_state, t=0)

            # Check both subplots use vmin=-1, vmax=1
            for ax_mock in [mock_ax1, mock_ax2]:
                call_kwargs = ax_mock.imshow.call_args[1]
                assert call_kwargs["vmin"] == -1
                assert call_kwargs["vmax"] == 1

    def test_sets_tick_labels(self, mixed_state: np.ndarray) -> None:
        """Both subplots should have x and y tick labels set."""
        hamiltonian = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

        with patch("src.visualization.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            from src.visualization.visualization import quantum_state_heatmap

            quantum_state_heatmap(hamiltonian, mixed_state, t=0)

            for ax_mock in [mock_ax1, mock_ax2]:
                ax_mock.set_xticks.assert_called()
                ax_mock.set_yticks.assert_called()

    def test_time_evolution_affects_result(self) -> None:
        """Different times should produce different density matrices."""
        hamiltonian = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)  # Pauli X
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)

        # Track what imshow was called with
        call_args_list = []

        with patch("src.visualization.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            from src.visualization.visualization import quantum_state_heatmap

            # Call at t=0
            quantum_state_heatmap(hamiltonian, rho0, t=0)
            call_args_list.append(mock_ax1.imshow.call_args[0][0].copy())

            # Call at t=π/2 (should give different result)
            quantum_state_heatmap(hamiltonian, rho0, t=np.pi / 2)
            call_args_list.append(mock_ax1.imshow.call_args[0][0].copy())

            # The two results should be different
            assert not np.allclose(call_args_list[0], call_args_list[1])


class TestLargerDimensions:
    """Test visualization functions with larger Hilbert spaces."""

    def test_populations_work_with_larger_dimensions(self) -> None:
        """Functions should handle 4x4 systems correctly."""
        hamiltonian = np.diag([1.0, 0.5, -0.5, -1.0])
        rho0 = np.diag([1.0, 0.0, 0.0, 0.0])

        with patch("src.visualization.plt.stackplot"):
            with patch("src.visualization.plt.legend"):
                from src.visualization.visualization import (
                    finite_dimensional_populations_over_time,
                )

                # Should not raise
                finite_dimensional_populations_over_time(
                    hamiltonian, rho0, time_window_upper_bound=10
                )

    def test_heatmap_works_with_3x3_system(self) -> None:
        """Heatmap should handle 3x3 density matrices."""
        hamiltonian = np.diag([1.0, 0.0, -1.0])
        rho0 = np.array(
            [[0.5, 0.2, 0.1], [0.2, 0.3, 0.1], [0.1, 0.1, 0.2]], dtype=complex
        )

        with patch("src.visualization.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            from src.visualization.visualization import quantum_state_heatmap

            quantum_state_heatmap(hamiltonian, rho0, t=0)

            # Verify imshow was called with 3x3 arrays
            for ax_mock in [mock_ax1, mock_ax2]:
                call_arg = ax_mock.imshow.call_args[0][0]
                assert call_arg.shape == (3, 3)
