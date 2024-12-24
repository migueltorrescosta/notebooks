from enum import Enum

from matplotlib import pyplot as plt
from src.plotting import plot_array
from tqdm import tqdm
import itertools
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
with st.sidebar:
    st.header("Fisher Information", divider="blue")
    distribution = st.selectbox("Function", [dist.value for dist in Distributions])

    match distribution:
        case Distributions.Binomial.value:
            c1, c2 = st.columns(2)
            with c1:
                n = st.number_input("$N$ Trials", min_value=1, value=3)
            with c2:
                theta_sample_size = st.number_input("$\\theta$'s granularity", min_value=5, value=100)
            if n * theta_sample_size > 1000:
                st.error(f"There are {n * theta_sample_size} $(x,\\theta)$ tuples to calculate. This will be slow ⚠️")
            pdf = lambda x, p: scipy.stats.binom.cdf(x, n=n, p=p) - scipy.stats.binom.cdf(x-1, n=n, p=p)
            valid_x = range(n+1)
            valid_theta = np.linspace(0, 1, theta_sample_size + 1)

    st.header("Cramer Rao", divider="orange")
    fisher_clip = st.number_input("Max absolute fisher information ( plot )", min_value=.001, value=.1)

st.header("Fisher Information", divider="blue")

match distribution:
    case Distributions.Binomial.value:
        st.latex(f"""
            f_\\theta(x) = \\mathbb{{P}}[X=x] = {{{n} \\choose x}} x^{{\\theta}}({n}-x)^{{1 - \\theta}}, x \\in \\{{ 0 , 1 , \\dots , {n} \\}} 
        """)

df_pdf = pd.DataFrame(
    data=[(x, p, pdf(x,p)) for x,p in tqdm(itertools.product(valid_x, valid_theta), total=len(valid_x) * len(valid_theta))],
    columns=["x", "theta", "pdf"]
).pivot(columns="x", index="theta", values="pdf")

plot_array(df_pdf, midpoint=None, text_auto=False)

def quick_and_dirty(my_array: np.array):
    fig, ax = plt.subplots()
    sns.heatmap(
        my_array,
        cmap="viridis"
    )
    st.write(fig)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.latex("f_\\theta(x)")
    quick_and_dirty(np.array(df_pdf))
with c2:
    st.latex("\log_\\theta f(x)")
    quick_and_dirty(np.log(np.array(df_pdf)))
with c3:
    st.latex("\\left ( \\frac{\partial}{\partial \\theta} \log_\\theta f(x) \\right )")
    quick_and_dirty(np.gradient(np.log(np.array(df_pdf)), axis=0))
with c4:
    st.latex("\\left ( \\frac{\partial}{\partial \\theta} \log f_\\theta(x) \\right )^2")
    quick_and_dirty(np.gradient(np.log(np.array(df_pdf)), axis=0)**2)

# axis 0 is x, axis 1 is p
fisher_information = np.average(
    np.gradient(np.log(np.array(df_pdf)), axis=0)**2,
    weights=np.array(df_pdf),
    axis=1,
)
fisher_information_df = pd.DataFrame({
    "fisher_information": np.clip(fisher_information, -1 * fisher_clip, fisher_clip),
    "cramer_rao_bound": np.divide(1, fisher_information),
    "theta": valid_theta
})
st.header("Cramer Rao", divider="orange")
st.dataframe(fisher_information_df.T) # TODO: Understand why increasing the size of N impacts the number of Nones in this pd.DataFrame
c1, c2 = st.columns(2)
with c1:
    st.latex("""
        \\mathcal{I}(\\theta) \\coloneqq
        \\mathbb{E}_X\\left [ \\left (
            \\frac{\partial}{\partial \\theta} \log f_\\theta(x)
            \\right )^2 | \\theta \\right ]
    """)
    st.line_chart(
        data = fisher_information_df,
        x="theta",
        y="fisher_information"
    )

with c2:
    st.latex("\\mathrm{var}[\hat \\theta] \geq \\frac{1}{\mathcal{I}(\\theta)}")

    st.line_chart(
        data = fisher_information_df,
        x="theta",
        y="cramer_rao_bound"
    )
