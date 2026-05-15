"""Smoke tests for plotting module."""

from unittest.mock import MagicMock, patch

import numpy as np


class TestPlotArray:
    def test_plot_array_should_execute_without_raising_exception(self) -> None:
        # Mock streamlit components to avoid display errors
        with (
            patch("src.visualization.plotting.st") as mock_st,
            patch("src.visualization.plotting.px.imshow") as mock_imshow,
        ):
            mock_fig = MagicMock()
            mock_imshow.return_value = mock_fig

            # Create a simple test array
            test_array = np.array([[1, 2], [3, 4]])

            # Import and call the function
            from .plotting import plot_array

            # Should not raise
            plot_array(test_array)

            # Verify imshow was called with correct parameters
            mock_imshow.assert_called_once()
            call_kwargs = mock_imshow.call_args.kwargs
            assert "text_auto" in call_kwargs, 'Expected "text_auto" in call_kwargs'
            assert "color_continuous_midpoint" in call_kwargs, (
                'Expected "color_continuous_midpoint" in call_kwargs'
            )
            # Verify streamlit plotly_chart was called
            mock_st.plotly_chart.assert_called_once()

    def test_plot_array_with_custom_midpoint(self) -> None:
        with (
            patch("src.visualization.plotting.st") as mock_st,
            patch("src.visualization.plotting.px.imshow") as mock_imshow,
        ):
            mock_fig = MagicMock()
            mock_imshow.return_value = mock_fig

            test_array = np.array([[-1, 0, 1], [2, 3, 4]])
            from .plotting import plot_array

            plot_array(test_array, midpoint=0.5)

            call_kwargs = mock_imshow.call_args.kwargs
            assert call_kwargs["color_continuous_midpoint"] == 0.5, (
                'Expected call_kwargs["color_continuous_midpoint"] == 0.5'
            )
            mock_st.plotly_chart.assert_called_once()

    def test_plot_array_with_text_auto_false(self) -> None:
        with (
            patch("src.visualization.plotting.st") as mock_st,
            patch("src.visualization.plotting.px.imshow") as mock_imshow,
        ):
            mock_fig = MagicMock()
            mock_imshow.return_value = mock_fig

            test_array = np.eye(3)
            from .plotting import plot_array

            plot_array(test_array, text_auto=False)

            call_kwargs = mock_imshow.call_args.kwargs
            assert call_kwargs["text_auto"] is False, (
                'Expected call_kwargs["text_auto"] to be False'
            )
            mock_st.plotly_chart.assert_called_once()

    def test_plot_array_with_custom_key(self) -> None:
        with (
            patch("src.visualization.plotting.st") as mock_st,
            patch("src.visualization.plotting.px.imshow") as mock_imshow,
        ):
            mock_fig = MagicMock()
            mock_imshow.return_value = mock_fig

            test_array = np.array([[1, 2], [3, 4]])
            from .plotting import plot_array

            plot_array(test_array, key="my_unique_key")

            mock_st.plotly_chart.assert_called_once()
            assert (
                mock_st.plotly_chart.call_args.kwargs.get("key") == "my_unique_key"
            ), (
                'Expected mock_st.plotly_chart.call_args.kwargs.get("key") == "my_unique_key"'
            )
