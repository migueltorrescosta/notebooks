"""Unit tests for visualization module.

These tests verify that visualization functions:
1. Execute without errors
2. Call matplotlib functions with correct arguments
3. Properly validate input parameters
4. Handle edge cases correctly
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestFourierTransform:
    def test_given_fourier_transform_then_create_figure_with_exactly_3_subplots(
        self,
    ) -> None:
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_axs = [MagicMock(), MagicMock(), MagicMock()]
            mock_subplots.return_value = (mock_fig, mock_axs)

            from .visualization import fourier_transform

            def f(t: float) -> float:
                return np.exp(-(t**2))

            fourier_transform(f, a=-10, b=10, time_domain_n=100)

            mock_subplots.assert_called_once_with(3)
            assert len(mock_axs) == 3

    def test_given_each_subplot_then_have_expected_title(self) -> None:
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_axs = [MagicMock(), MagicMock(), MagicMock()]
            mock_subplots.return_value = (mock_fig, mock_axs)

            from .visualization import fourier_transform

            fourier_transform(lambda t: t, a=0, b=1, time_domain_n=10)

            mock_axs[0].set_title.assert_called_once_with("Time Domain")
            mock_axs[1].set_title.assert_called_once_with(
                "Frequency Domain, absolute value",
            )
            mock_axs[2].set_title.assert_called_once_with("Frequency Domain, angle")

    def test_given_each_subplot_then_have_plot_called_on_it(self) -> None:
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_axs = [MagicMock(), MagicMock(), MagicMock()]
            mock_subplots.return_value = (mock_fig, mock_axs)

            from .visualization import fourier_transform

            fourier_transform(lambda t: t, a=0, b=1, time_domain_n=10)

            for ax in mock_axs:
                ax.plot.assert_called()

    def test_given_figure_then_use_tight_layout_for_proper_spacing(self) -> None:
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_axs = [MagicMock(), MagicMock(), MagicMock()]
            mock_subplots.return_value = (mock_fig, mock_axs)

            from .visualization import fourier_transform

            fourier_transform(lambda t: 1.0, a=0, b=1, time_domain_n=10)

            mock_fig.tight_layout.assert_called_once_with(pad=1)

    def test_given_default_parameters_then_allow_function_to_run(self) -> None:
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_axs = [MagicMock(), MagicMock(), MagicMock()]
            mock_subplots.return_value = (mock_fig, mock_axs)

            from .visualization import fourier_transform

            # Should not raise
            fourier_transform(np.sin)

    @pytest.mark.parametrize("n_samples", [10, 100, 1000])
    def test_given_various_sample_counts_then_fourier_transform_works(
        self, n_samples: int
    ) -> None:
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_axs = [MagicMock(), MagicMock(), MagicMock()]
            mock_subplots.return_value = (mock_fig, mock_axs)

            from .visualization import fourier_transform

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

    def test_given_population_plot_then_call_plt_stackplot(
        self,
        simple_hamiltonian: np.ndarray,
        pure_state: np.ndarray,
    ) -> None:
        with patch("matplotlib.pyplot.stackplot") as mock_stackplot:
            with patch("matplotlib.pyplot.legend"):
                from .visualization import (
                    finite_dimensional_populations_over_time,
                )

                finite_dimensional_populations_over_time(
                    simple_hamiltonian,
                    pure_state,
                    time_window_upper_bound=10,
                )

                mock_stackplot.assert_called()

    def test_given_stackplot_then_receive_time_axis_as_first_argument(
        self,
        simple_hamiltonian: np.ndarray,
        pure_state: np.ndarray,
    ) -> None:
        with patch("matplotlib.pyplot.stackplot") as mock_stackplot:
            with patch("matplotlib.pyplot.legend"):
                from .visualization import (
                    finite_dimensional_populations_over_time,
                )

                finite_dimensional_populations_over_time(
                    simple_hamiltonian,
                    pure_state,
                    time_window_upper_bound=10,
                )

                # First argument should be time axis (array of floats)
                call_args = mock_stackplot.call_args
                time_axis = call_args[0][0]
                assert isinstance(time_axis, np.ndarray), (
                    "Expected time_axis to be instance of np.ndarray"
                )
                assert time_axis[0] == 0  # Should start at 0

    def test_given_stackplot_then_receive_population_arrays(
        self,
        simple_hamiltonian: np.ndarray,
        pure_state: np.ndarray,
    ) -> None:
        with patch("matplotlib.pyplot.stackplot") as mock_stackplot:
            with patch("matplotlib.pyplot.legend"):
                from .visualization import (
                    finite_dimensional_populations_over_time,
                )

                finite_dimensional_populations_over_time(
                    simple_hamiltonian,
                    pure_state,
                    time_window_upper_bound=10,
                )

                # Check that stackplot was called
                mock_stackplot.assert_called()
                # First arg should be time axis, second should be array-like populations
                call_args = mock_stackplot.call_args
                assert isinstance(call_args[0][0], np.ndarray)  # time axis
                assert isinstance(
                    call_args[0][1],
                    np.ndarray,
                )  # populations (array-like)

    def test_given_labels_are_provided_then_call_plt_legend(
        self,
        simple_hamiltonian: np.ndarray,
        pure_state: np.ndarray,
    ) -> None:
        with patch("matplotlib.pyplot.stackplot"):
            with patch("matplotlib.pyplot.legend") as mock_legend:
                from .visualization import (
                    finite_dimensional_populations_over_time,
                )

                finite_dimensional_populations_over_time(
                    simple_hamiltonian,
                    pure_state,
                    time_window_upper_bound=10,
                    labels=["ground", "excited"],
                )

                mock_legend.assert_called()

    def test_given_non_square_hamiltonian_then_raise_assertion_error(self) -> None:
        hamiltonian = np.array([[1.0, 0.0], [0.0, -1.0], [0.0, 0.0]], dtype=complex)
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)

        from .visualization import (
            finite_dimensional_populations_over_time,
        )

        with pytest.raises(AssertionError, match="not square"):
            finite_dimensional_populations_over_time(hamiltonian, rho0)

    def test_given_rho0_dims_dont_match_hamiltonian_then_raise_assertion_error(
        self,
    ) -> None:
        hamiltonian = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        rho0 = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=complex,
        )

        from .visualization import (
            finite_dimensional_populations_over_time,
        )

        with pytest.raises(AssertionError, match="does not match"):
            finite_dimensional_populations_over_time(hamiltonian, rho0)

    def test_given_negative_time_window_then_raise_assertion_error(
        self,
        simple_hamiltonian: np.ndarray,
        pure_state: np.ndarray,
    ) -> None:
        from .visualization import (
            finite_dimensional_populations_over_time,
        )

        with pytest.raises(AssertionError, match="negative"):
            finite_dimensional_populations_over_time(
                simple_hamiltonian,
                pure_state,
                time_window_upper_bound=-1,
            )

    def test_given_zero_time_window_then_raise_assertion_error(
        self,
        simple_hamiltonian: np.ndarray,
        pure_state: np.ndarray,
    ) -> None:
        from .visualization import (
            finite_dimensional_populations_over_time,
        )

        with pytest.raises(AssertionError, match="negative"):
            finite_dimensional_populations_over_time(
                simple_hamiltonian,
                pure_state,
                time_window_upper_bound=0,
            )

    def test_given_rho0_trace_not_1_then_raise_assertion_error(
        self, simple_hamiltonian: np.ndarray
    ) -> None:
        rho0 = np.array([[0.8, 0.0], [0.0, 0.1]], dtype=complex)  # Trace = 0.9

        from .visualization import (
            finite_dimensional_populations_over_time,
        )

        with pytest.raises(AssertionError, match="add up to 1"):
            finite_dimensional_populations_over_time(simple_hamiltonian, rho0)

    def test_given_label_count_mismatch_then_raise_assertion_error(
        self,
        simple_hamiltonian: np.ndarray,
        pure_state: np.ndarray,
    ) -> None:
        from .visualization import (
            finite_dimensional_populations_over_time,
        )

        with pytest.raises(AssertionError, match="labels"):
            finite_dimensional_populations_over_time(
                simple_hamiltonian,
                pure_state,
                labels=["only_one"],
            )

    def test_given_labels_with_correct_count_then_accept(
        self,
        simple_hamiltonian: np.ndarray,
        pure_state: np.ndarray,
    ) -> None:
        with patch("matplotlib.pyplot.stackplot"):
            with patch("matplotlib.pyplot.legend"):
                from .visualization import (
                    finite_dimensional_populations_over_time,
                )

                # Should not raise
                finite_dimensional_populations_over_time(
                    simple_hamiltonian,
                    pure_state,
                    labels=["state_0", "state_1"],
                )


class TestQuantumStateHeatmap:
    """Tests for quantum_state_heatmap function."""

    @pytest.fixture
    def mixed_state(self) -> np.ndarray:
        """Mixed density matrix with both diagonal and off-diagonal elements."""
        return np.array([[0.5, 0.3], [0.3, 0.5]], dtype=complex)

    def test_given_heatmap_then_create_figure_with_exactly_2_subplots(self) -> None:
        hamiltonian = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)

        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            from .visualization import quantum_state_heatmap

            quantum_state_heatmap(hamiltonian, rho0, t=0)

            mock_subplots.assert_called_once_with(1, 2)

    def test_given_both_subplots_then_call_imshow_to_display_density_matrix(
        self, mixed_state: np.ndarray
    ) -> None:
        hamiltonian = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            from .visualization import quantum_state_heatmap

            quantum_state_heatmap(hamiltonian, mixed_state, t=0)

            mock_ax1.imshow.assert_called()
            mock_ax2.imshow.assert_called()

    def test_given_imshow_then_use_vmin_minus_1_and_vmax_1_for_proper_color_scaling(
        self, mixed_state: np.ndarray
    ) -> None:
        hamiltonian = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            from .visualization import quantum_state_heatmap

            quantum_state_heatmap(hamiltonian, mixed_state, t=0)

            # Check both subplots use vmin=-1, vmax=1
            for ax_mock in [mock_ax1, mock_ax2]:
                call_kwargs = ax_mock.imshow.call_args[1]
                assert call_kwargs["vmin"] == -1, 'Expected call_kwargs["vmin"] == -1'
                assert call_kwargs["vmax"] == 1, 'Expected call_kwargs["vmax"] == 1'

    def test_given_both_subplots_then_have_x_and_y_tick_labels_set(
        self, mixed_state: np.ndarray
    ) -> None:
        hamiltonian = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            from .visualization import quantum_state_heatmap

            quantum_state_heatmap(hamiltonian, mixed_state, t=0)

            for ax_mock in [mock_ax1, mock_ax2]:
                ax_mock.set_xticks.assert_called()
                ax_mock.set_yticks.assert_called()

    def test_given_different_times_then_produce_different_density_matrices(
        self,
    ) -> None:
        hamiltonian = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)  # Pauli X
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)

        # Track what imshow was called with
        call_args_list = []

        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            from .visualization import quantum_state_heatmap

            # Call at t=0
            quantum_state_heatmap(hamiltonian, rho0, t=0)
            call_args_list.append(mock_ax1.imshow.call_args[0][0].copy())

            # Call at t=π/2 (should give different result)
            quantum_state_heatmap(hamiltonian, rho0, t=np.pi / 2)
            call_args_list.append(mock_ax1.imshow.call_args[0][0].copy())

            # The two results should be different
            assert call_args_list[0] != pytest.approx(call_args_list[1]), (
                "Expected call_args_list[0] != pytest.approx(call_args_list[1])"
            )


class TestLargerDimensions:
    """Test visualization functions with larger Hilbert spaces."""

    def test_given_functions_then_handle_4x4_systems_correctly(self) -> None:
        hamiltonian = np.diag([1.0, 0.5, -0.5, -1.0])
        rho0 = np.diag([1.0, 0.0, 0.0, 0.0])

        with patch("matplotlib.pyplot.stackplot"):
            with patch("matplotlib.pyplot.legend"):
                from .visualization import (
                    finite_dimensional_populations_over_time,
                )

                # Should not raise
                finite_dimensional_populations_over_time(
                    hamiltonian,
                    rho0,
                    time_window_upper_bound=10,
                )

    def test_given_heatmap_then_handle_3x3_density_matrices(self) -> None:
        hamiltonian = np.diag([1.0, 0.0, -1.0])
        rho0 = np.array(
            [[0.5, 0.2, 0.1], [0.2, 0.3, 0.1], [0.1, 0.1, 0.2]],
            dtype=complex,
        )

        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            from .visualization import quantum_state_heatmap

            quantum_state_heatmap(hamiltonian, rho0, t=0)

            # Verify imshow was called with 3x3 arrays
            for ax_mock in [mock_ax1, mock_ax2]:
                call_arg = ax_mock.imshow.call_args[0][0]
                assert call_arg.shape == (3, 3)
