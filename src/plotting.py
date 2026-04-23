"""
Plotting utilities for visualization of quantum simulation results.

This module provides helper functions for creating interactive and
static visualizations using Plotly and Streamlit. Designed for quick
exploratory visualization of arrays and matrices.
"""

from typing import Optional

import numpy as np
import plotly.express as px
import streamlit as st


# A diverging color map, prng, is chosen to clearly distinguish positives from negatives
# https://matplotlib.org/stable/users/explain/colors/colormaps.html#diverging
def plot_array(
    my_array: np.ndarray,
    midpoint: Optional[float] = 0,
    text_auto: bool = True,
    key: Optional[str] = None,
) -> None:
    """Display a 2D array as an interactive heatmap in Streamlit.

    Uses a diverging colormap (purple-green-negative) centered at the
    specified midpoint, making it suitable for visualizing signed data
    like probability amplitudes or wavefunctions.

    Args:
        my_array: 2D numpy array to visualize. Can contain real or
            complex values (will display magnitude for complex).
        midpoint: Value in the array that maps to the neutral color
            (white/transparent). Default is 0.
        text_auto: If True, display numerical values in each cell.
        key: Optional Streamlit key for the plot element.

    Example:
        >>> import numpy as np
        >>> arr = np.random.randn(5, 5)
        >>> plot_array(arr, midpoint=0)  # Display in Streamlit
    """
    # Handle complex arrays by taking magnitude
    if np.iscomplexobj(my_array):
        my_array = np.abs(my_array)

    fig = px.imshow(
        my_array,
        text_auto=text_auto,
        aspect="auto",
        color_continuous_midpoint=midpoint,
        color_continuous_scale="prgn",
    )
    st.plotly_chart(fig, key=key)
