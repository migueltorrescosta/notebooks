"""Minimize Heatmap UI page - imports physics from src.optimization."""

import itertools
import numpy as np
import pandas as pd
import streamlit as st

from src.optimization import TEST_FUNCTIONS, MINIMIZERS

st.set_page_config(page_title="Minimize Heatmap", page_icon="📐️", layout="wide")

st.header("Minimize Heatmap", divider="blue")

with st.expander("📖 Methodology", expanded=False):
    st.markdown(r"""
    **Optimization Benchmarking** compares different numerical optimization algorithms on standard test functions.
    
    **Test Functions:** Standard benchmark functions include:
    - **Sphere**: $f(x,y) = x^2 + y^2$ (simple convex quadratic)
    - **Rastrigin**: Non-convex with many local minima
    - **Ackley**: Global minimum surrounded by local minima
    - **Rosenbrock**: Banana-shaped valley
    - **Himmelblau**: Four global minima
    
    **Methodology:**
    1. **Function Evaluation**: Compute $f(x,y)$ over a grid
    2. **Algorithm Selection**: Choose from gradient-based and derivative-free methods
    3. **Optimization**: Run each algorithm from starting point $(0, 0)$
    4. **Performance Comparison**: Compute normalized minimum values
    """)


with st.sidebar:
    st.header("Comparing minimizers", divider="blue")
    c1, c2 = st.columns(2)
    with c1:
        selected_optimizer = st.selectbox("Minimizer", list(MINIMIZERS.keys()))
        optimizer = MINIMIZERS[selected_optimizer]
    with c2:
        selected_function = st.selectbox("Test function", list(TEST_FUNCTIONS.keys()))
        test_function = TEST_FUNCTIONS[selected_function]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        x_min = st.number_input("$x_{min}$", value=-3.0)
    with c2:
        x_max = st.number_input("$x_{max}$", min_value=x_min, value=3.0)
    with c3:
        y_min = st.number_input("$y_{min}$", value=-3.0)
    with c4:
        y_max = st.number_input("$y_{max}$", min_value=y_min, value=3.0)


@st.cache_data
def gen_surface_df(f: str) -> pd.DataFrame:
    """Cached surface dataframe generation."""
    return pd.DataFrame(
        [
            {"x": round(x, 3), "y": round(y, 3), "f": TEST_FUNCTIONS[f](x, y)}
            for x, y in itertools.product(np.linspace(-3, 3, 101), repeat=2)
        ]
    )


@st.cache_data
def gen_performance_df() -> pd.DataFrame:
    """Cached performance dataframe generation."""
    # Limit to subset for faster computation
    subset_functions = ["Rastrigin", "Ackley", "Sphere", "Rosenbrock", "Himmelblau"]
    subset_minimizers = ["Nelder-Mead", "Powell", "CG", "BFGS"]

    results = []
    for function, minimizer in itertools.product(subset_functions, subset_minimizers):
        opt = MINIMIZERS[minimizer]
        func = TEST_FUNCTIONS[function]
        result = opt(func, [0, 0])
        results.append(
            {
                "minimizer": minimizer,
                "test_function": function,
                "argmin": result["x"],
            }
        )

    df = pd.DataFrame(results)

    def calc_min(row: pd.Series) -> float:
        func = TEST_FUNCTIONS[row["test_function"]]
        argmin = row["argmin"]
        return func(argmin[0], argmin[1])

    df["min"] = df.apply(calc_min, axis=1)
    df["normalized_min"] = df.groupby("test_function")["min"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
    )
    return df


st.header("Comparing minimizers", divider="blue")

c1, c2 = st.columns(2)
with c1:
    import matplotlib.pyplot as plt
    import seaborn as sns

    argmin_df = gen_performance_df()
    fig, ax = plt.subplots()
    sns.heatmap(
        argmin_df.pivot(
            index="test_function", columns="minimizer", values="normalized_min"
        ),
        ax=ax,
        cmap="viridis",
    )
    st.pyplot(fig)
    st.caption("Yellow is BAD")

with c2:
    surface_df = gen_surface_df(f=selected_function)
    fig, ax = plt.subplots()
    sns.heatmap(surface_df.pivot(index="x", columns="y", values="f"), cmap="viridis")
    st.pyplot(fig)
