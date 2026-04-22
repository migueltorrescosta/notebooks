from enum import Enum
from functools import partial

from matplotlib import pyplot as plt
from src.plotting import plot_array
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import streamlit as st

# LAYOUT
st.set_page_config(page_title="Fisher information", page_icon="🛗️", layout="wide")


class Distributions(Enum):
    Binomial = "Binomial"


# INPUTS
def binomial_pdf(x: int, p: float, n: int) -> float:
    return float(
        scipy.stats.binom.cdf(x, n=n, p=p) - scipy.stats.binom.cdf(x - 1, n=n, p=p)
    )


with st.sidebar:
    st.header("Fisher Information", divider="blue")
    distribution = st.selectbox("Function", [dist.value for dist in Distributions])

    match distribution:
        case Distributions.Binomial.value:
            c1, c2 = st.columns(2)
            with c1:
                n = st.number_input("$N$ Trials", min_value=1, value=3)
            with c2:
                theta_sample_size = st.number_input(
                    "$\\theta$'s granularity", min_value=5, value=100
                )
            if n * theta_sample_size > 1000:
                st.error(
                    f"There are {n * theta_sample_size} $(x,\\theta)$ tuples to calculate. This will be slow ⚠️"
                )
            valid_x = range(n + 1)
            valid_theta = np.linspace(0, 1, theta_sample_size + 1)
            pdf = partial(binomial_pdf, n=n)

    st.header("Cramer Rao", divider="orange")
    fisher_clip = st.number_input(
        "Max absolute fisher information ( plot )", min_value=0.001, value=0.1
    )

st.header("Fisher Information", divider="blue")

with st.expander("📖 Methodology", expanded=False):
    st.markdown("""
    **Fisher Information** measures how much information a random variable $X$ carries about an unknown parameter $\\theta$.
    The Fisher Information for a parameter $\\theta$ is defined as
    $\\mathcal{I}(\\theta) = \\mathbb{E}\\left[\\left(\\frac{\\partial}{\\partial \\theta}\\log f_\\theta(X)\\right)^2\\,\\bigg|\\,\\theta\\right]$,
    where $f_\\theta(x)$ is the probability density (or mass) function of $X$ parameterized by $\\theta$.
    
    **Methodology:**
    1. **PDF computation**: For each combination of outcome $x$ and parameter value $\\theta$, compute the probability $f_\\theta(x)$
    2. **Log-likelihood gradient**: Approximate $\\frac{\\partial}{\\partial \\theta}\\log f_\\theta(x)$ using finite differences along the $\\theta$ axis
    3. **Expectation**: Weight the squared gradient by the PDF values and average over all outcomes $x$
    
    **Cramer-Rao Bound:** The variance of any unbiased estimator $\\hat{\\theta}$ of $\\theta$ satisfies 
    $\\mathrm{var}[\\hat{\\theta}] \\geq \\frac{1}{\\mathcal{I}(\\theta)}$.    
    This bound provides a fundamental limit on the precision of parameter estimation.
    """)

match distribution:
    case Distributions.Binomial.value:
        st.latex(f"""
            f_\\theta(x) = \\mathbb{{P}}[X=x] = {{{n} \\choose x}} x^{{\\theta}}({n}-x)^{{1 - \\theta}}, x \\in \\{{ 0 , 1 , \\dots , {n} \\}} 
        """)

# Vectorized PDF computation using broadcasting
x_arr = np.arange(n + 1)[:, None]  # (n+1, 1)
theta_arr = np.linspace(0, 1, theta_sample_size + 1)[None, :]  # (1, m)
pdf_values = scipy.stats.binom.pmf(x_arr, n=n, p=theta_arr)  # (n+1, m)

# Create DataFrame directly from array - no Python loops
df_pdf = pd.DataFrame(
    pdf_values.T,  # Transpose to (m, n+1) with theta as rows, x as columns
    index=pd.Index(theta_arr.flatten(), name="theta"),
    columns=pd.Index(x_arr.flatten(), name="x"),
)

plot_array(np.array(df_pdf), midpoint=None, text_auto=False)


def quick_and_dirty(my_array: np.ndarray) -> None:
    fig, ax = plt.subplots()
    sns.heatmap(my_array, cmap="viridis")
    st.write(fig)


c1, c2, c3, c4 = st.columns(4)
with c1:
    st.latex(r"f_\theta(x)")
    quick_and_dirty(np.array(df_pdf))
with c2:
    st.latex(r"\log f_\theta(x)")
    quick_and_dirty(np.log(np.array(df_pdf)))
with c3:
    st.latex(r"\left ( \frac{\partial}{\partial \theta} \log f_\theta(x) \right )")
    quick_and_dirty(np.gradient(np.log(np.array(df_pdf)), axis=0))
with c4:
    st.latex(r"\left ( \frac{\partial}{\partial \theta} \log f_\theta(x) \right )^2")
    quick_and_dirty(np.gradient(np.log(np.array(df_pdf)), axis=0) ** 2)

# axis 0 is x, axis 1 is p
fisher_information = np.average(
    np.gradient(np.log(np.array(df_pdf)), axis=0) ** 2,
    weights=np.array(df_pdf),
    axis=1,
)
fisher_information_df = pd.DataFrame(
    {
        "fisher_information": np.clip(
            fisher_information, -1 * fisher_clip, fisher_clip
        ),
        "cramer_rao_bound": np.divide(1, fisher_information),
        "theta": valid_theta,
    }
)
st.header("Cramer Rao", divider="orange")
st.dataframe(
    fisher_information_df.T
)  # TODO: Understand why increasing the size of N impacts the number of Nones in this pd.DataFrame
c1, c2 = st.columns(2)
with c1:
    st.latex(r"""
        \mathcal{I}(\theta) \coloneqq
        \mathbb{E}_X\left [ \left (
            \frac{\partial}{\partial \theta} \log f_\theta(x)
            \right )^2 | \theta \right ]
    """)
    st.line_chart(data=fisher_information_df, x="theta", y="fisher_information")

with c2:
    st.latex(r"\mathrm{var}[\hat \theta] \geq \frac{1}{\mathcal{I}(\theta)}")

    st.line_chart(data=fisher_information_df, x="theta", y="cramer_rao_bound")
