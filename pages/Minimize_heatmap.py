from collections.abc import Callable
from typing import Any
from matplotlib import pyplot as plt
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Minimize Heatmap", page_icon="📐️", layout="wide")

st.header("Minimize Heatmap", divider="blue")

with st.expander("📖 Methodology", expanded=False):
    st.markdown("""
    **Optimization Benchmarking** compares different numerical optimization algorithms on standard test functions.
    
    **Test Functions:** Standard benchmark functions from optimization literature include:
    - **Sphere**: $f(x,y) = x^2 + y^2$ (simple convex quadratic)
    - **Rastrigin**: Non-convex with many local minima
    - **Ackley**: Global minimum surrounded by local minima
    - **Rosenbrock**: Banana-shaped valley
    - **Himmelblau**: Four global minima
    - **Eggholder**: Highly irregular with many local optima
    
    **Methodology:**
    1. **Function Evaluation**: Compute $f(x,y)$ over a grid to visualize the landscape
    2. **Algorithm Selection**: Choose from gradient-based (CG, BFGS, L-BFGS-B, TNC, SLSQP) and derivative-free (Nelder-Mead, Powell, COBYLA, COBYQA, trust-constr) methods
    3. **Optimization**: Run each algorithm from starting point $(0, 0)$ to find local/global minimum
    4. **Performance Comparison**: Compute normalized minimum values across all algorithms/functions
    5. **Visualization**: Heatmap shows algorithm performance (yellow = poor, purple = good)
    
    **Key Insight:** No single algorithm performs best on all functions—derivative-free methods
    work well for non-smooth problems, while gradient methods excel on smooth convex functions.
    
    **Reference:** [Test functions for optimization](https://en.wikipedia.org/wiki/Test_functions_for_optimization)
    """)

# 1. Use the functions in https://en.wikipedia.org/wiki/Test_functions_for_optimization
# 2. Add Nelder-Mead, Simulated Annealing, Q-learning?
# 3. Plot the evolution of estimated optimal parameters over time


def rastrigin(x: float, y: float) -> float:
    return (
        10 * 2
        + (x**2 - 10 * np.cos(2 * np.pi * x))
        + (y**2 - 10 * np.cos(2 * np.pi * y))
    )


def ackley(x: float, y: float) -> float:
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))


def sphere(x: float, y: float) -> float:
    return x**2 + y**2


def rosenbrock(x: float, y: float) -> float:
    return np.log(1 + (1 - x) ** 2 + 100 * (y - x**2) ** 2)


def beale(x: float, y: float) -> float:
    return np.log(
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y**2) ** 2
        + (2.625 - x + x * y**3) ** 2
    )


def goldstein_price(x: float, y: float) -> float:
    return (
        1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
    ) * (
        30
        + (2 * x - 3 * y) ** 2
        * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    )


def booth(x: float, y: float) -> float:
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


def bukin(x: float, y: float) -> float:
    return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)


def matyas(x: float, y: float) -> float:
    return 0.26 * (x**2 + y**2) - 0.48 * x * y


def levi(x: float, y: float) -> float:
    return (
        np.sin(3 * np.pi * x) ** 2
        + (x - 1) ** 2 * (1 + np.sin(3 * np.pi * y) ** 2)
        + (y - 1) ** 2 * (1 + np.sin(2 * np.pi + y) ** 2)
    )


def himmelblau(x: float, y: float) -> float:
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def three_hump_camel(x: float, y: float) -> float:
    return 2 * x**2 - 1.05 * x**4 + np.divide(x**6, 6) + x * y + y**2


def eggholder(x: float, y: float) -> float:
    return (-1000 * y + 47) * np.sin(
        np.sqrt(np.abs(500 * x + 1000 * y + 47))
    ) - 1000 * x * np.sin(np.sqrt(np.abs(1000 * (x - y) + 47)))


def holder_table(x: float, y: float) -> float:
    return -1 * np.abs(
        np.sin(x)
        * np.cos(y)
        * np.exp(np.abs(1 - np.divide(np.sqrt(x**2 + y**2), np.pi)))
    )


def schaffer(x: float, y: float) -> float:
    return 0.5 + np.divide(
        np.cos(np.sin(np.abs(x**2 - y**2))) ** 2 - 0.5,
        (1 + 0.001 * (x**2 + y**2)) ** 2,
    )


def shekel(x: float, y: float) -> float:
    return -1 * sum(
        [
            np.divide(1, c + (x - a) ** 2 + (y - b) ** 2)
            for c, b, a in [(4, 1, 1), (5, 0, 1), (6, 0, 1)]
        ]
    )


test_functions: dict[str, Callable[[float, float], float]] = {
    "Rastrigin": rastrigin,
    "Ackley": ackley,
    "Sphere": sphere,
    "Rosenbrock": rosenbrock,
    "Beale": beale,
    "Goldstein-Price": goldstein_price,
    "Booth": booth,
    "Bukin": bukin,
    "Matyas": matyas,
    "Levi": levi,
    "Himmelblau": himmelblau,
    "Three-hump camel": three_hump_camel,
    "Eggholder": eggholder,
    "Holder table": holder_table,
    "Schaffer": schaffer,
    "Shekel": shekel,
}  # https://en.wikipedia.org/wiki/Test_functions_for_optimization


def gen_minimizer_from_scipy(method: str) -> Callable[..., Any]:
    def minimizer(temp_f: Callable[..., float], x0: float) -> Any:
        def wrapped(v: np.ndarray) -> float:
            return temp_f(v[0], v[1])

        # Use getattr to bypass type checking on scipy.optimize.minimize
        return getattr(__import__("scipy.optimize", fromlist=["minimize"]), "minimize")(
            wrapped, x0=x0, method=method
        )

    return minimizer


minimizers = {
    "Nelder-Mead": gen_minimizer_from_scipy(method="Nelder-Mead"),
    "Powell": gen_minimizer_from_scipy(method="Powell"),
    "CG": gen_minimizer_from_scipy(method="CG"),
    "BFGS": gen_minimizer_from_scipy(method="BFGS"),
    "L-BFGS-B": gen_minimizer_from_scipy(method="L-BFGS-B"),
    "TNC": gen_minimizer_from_scipy(method="TNC"),
    "COBYLA": gen_minimizer_from_scipy(method="COBYLA"),
    "COBYQA": gen_minimizer_from_scipy(method="COBYQA"),
    "SLSQP": gen_minimizer_from_scipy(method="SLSQP"),
    "trust-constr": gen_minimizer_from_scipy(method="trust-constr"),
    # "Newton-CG": gen_minimizer_from_scipy(method="Newton-CG"), # These require the Jacobian
    # "dogleg": gen_minimizer_from_scipy(method="dogleg"),
    # "trust-ncg": gen_minimizer_from_scipy(method="trust-ncg"),
    # "trust-exact": gen_minimizer_from_scipy(method="trust-exact"),
    # "trust-krylov": gen_minimizer_from_scipy(method="trust-krylov"),
}  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize


with st.sidebar:
    st.header("Comparing minimizers", divider="blue")
    c1, c2 = st.columns(2)
    with c1:
        selected_optimizer = st.selectbox("Minimizer", list(minimizers.keys()))
        optimizer = minimizers[selected_optimizer]
    with c2:
        selected_function = st.selectbox("Test function", list(test_functions.keys()))
        test_function = test_functions[selected_function]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        x_min = st.number_input("$x_{min}$", value=-3.0)
    with c2:
        x_max = st.number_input("$x_{max}$", min_value=x_min, value=3.0)
    with c3:
        y_min = st.number_input("$y_{min}$", value=-3.0)
    with c4:
        y_max = st.number_input("$y_{max}$", min_value=y_min, value=3.0)


def gen_surface_df(f: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"x": round(x, 3), "y": round(y, 3), "f": test_functions[f](x, y)}
            for x, y in itertools.product(np.linspace(-3, 3, 501), repeat=2)
        ]
    )


def gen_performance_df() -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "minimizer": minimizer,
                "test_function": function,
                "argmin": minimizers[minimizer](test_functions[function], x0=[0, 0])[
                    "x"
                ],
            }
            for function, minimizer in itertools.product(
                test_functions.keys(), minimizers.keys()
            )
        ]
    )

    def calc_min(row: pd.Series) -> float:
        test_func = test_functions[row["test_function"]]
        argmin = row["argmin"]
        return test_func(argmin[0], argmin[1])

    df["min"] = df.apply(calc_min, axis=1)
    df["normalized_min"] = df.groupby("test_function")[["min"]].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )

    return df


st.header("Comparing minimizers", divider="blue")

c1, c2 = st.columns(2)
with c1:
    argmin_df = gen_performance_df()
    fig, ax = plt.subplots()
    sns.heatmap(
        argmin_df.pivot(
            index="test_function", columns="minimizer", values="normalized_min"
        ),
        ax=ax,
        cmap="viridis",
    )
    st.write(fig)
    st.caption("Yellow is BAD")

with c2:
    surface_df = gen_surface_df(f=selected_function)
    fig, ax = plt.subplots()
    sns.heatmap(surface_df.pivot(index="x", columns="y", values="f"), cmap="viridis")
    st.write(fig)
