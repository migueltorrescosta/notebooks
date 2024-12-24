from typing import Optional
import numpy as np
import plotly.express as px
import streamlit as st

# A diverging color map, prng, is chosen to clearly distinguish positives from negatives
# https://matplotlib.org/stable/users/explain/colors/colormaps.html#diverging
def plot_array(
    my_array: np.array,
    midpoint: Optional[float] = 0,
    text_auto: bool=True,
    key: Optional[str] = None
) -> None:
    fig = px.imshow(
        my_array,
        text_auto=text_auto,
        aspect="auto",
        color_continuous_midpoint=midpoint,
        color_continuous_scale="prgn"
    )
    st.plotly_chart(fig, key=key)

