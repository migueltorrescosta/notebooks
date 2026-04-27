"""Smoke tests for plotting module."""

from unittest.mock import MagicMock, patch

import numpy as np


class TestPlotArray:
    def test_plot_array_runs_without_error(self) -> None:
        """Test that plot_array executes without raising an exception."""
        # Mock streamlit components to avoid display errors
        with (
            patch("src.plotting.st.plotly_chart"),
            patch("src.plotting.px.imshow") as mock_imshow,
        ):
            mock_fig = MagicMock()
            mock_imshow.return_value = mock_fig

            # Create a simple test array
            test_array = np.array([[1, 2], [3, 4]])

            # Import and call the function
            from src.visualization.plotting import plot_array

            # Should not raise
            plot_array(test_array)

            # Verify imshow was called with correct parameters
            mock_imshow.assert_called_once()
            call_kwargs = mock_imshow.call_args.kwargs
            assert "text_auto" in call_kwargs
            assert "color_continuous_midpoint" in call_kwargs

    def test_plot_array_with_custom_midpoint(self) -> None:
        with (
            patch("src.plotting.st.plotly_chart"),
            patch("src.plotting.px.imshow") as mock_imshow,
        ):
            mock_fig = MagicMock()
            mock_imshow.return_value = mock_fig

            test_array = np.array([[-1, 0, 1], [2, 3, 4]])
            from src.visualization.plotting import plot_array

            plot_array(test_array, midpoint=0.5)

            call_kwargs = mock_imshow.call_args.kwargs
            assert call_kwargs["color_continuous_midpoint"] == 0.5

    def test_plot_array_with_text_auto_false(self) -> None:
        with (
            patch("src.plotting.st.plotly_chart"),
            patch("src.plotting.px.imshow") as mock_imshow,
        ):
            mock_fig = MagicMock()
            mock_imshow.return_value = mock_fig

            test_array = np.eye(3)
            from src.visualization.plotting import plot_array

            plot_array(test_array, text_auto=False)

            call_kwargs = mock_imshow.call_args.kwargs
            assert call_kwargs["text_auto"] is False

    def test_plot_array_with_custom_key(self) -> None:
        with (
            patch("src.plotting.st.plotly_chart") as mock_plotly,
            patch("src.plotting.px.imshow") as mock_imshow,
        ):
            mock_fig = MagicMock()
            mock_imshow.return_value = mock_fig

            test_array = np.array([[1, 2], [3, 4]])
            from src.visualization.plotting import plot_array

            plot_array(test_array, key="my_unique_key")

            mock_plotly.assert_called_once()
            assert mock_plotly.call_args.kwargs.get("key") == "my_unique_key"
