from enum import Enum
from src.bayesian_statistics import BayesUpdate
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bayesian Updates", page_icon="👶️", layout="wide")


class Distributions(str, Enum):
    Geometric = "Geometric"
    Linear = "Linear"
    Gaussian = "Gaussian"

# START
with st.sidebar:

    st.header("Setup", divider="blue")
    c1, c2 = st.columns(2)
    with c1:
        prior_distribution: Distributions = st.selectbox("Prior", [dist.value for dist in Distributions])
        match prior_distribution:
            case Distributions.Geometric.value:
                st.latex("a(x-b)^c")
                a = st.number_input("$a$", value=1.)
                b = st.number_input("$b$", value=0.)
                c = st.number_input("$c$", value=1.)
                prior = lambda x:a*(x-b)**c
                st.latex(f"{a}(x-{b})^{{ {c} }}")
            case Distributions.Linear.value:
                st.latex("ax+b")
                a = st.number_input("$a$", value=1.)
                b = st.number_input("$b$", value=0.)
                prior = lambda x:a*x+b
            case Distributions.Gaussian.value:
                st.latex("ae^{-(x-\\mu)^2}")
                a = st.number_input("$a$", value=1.)
                mu = st.number_input("$\\mu$", value=0.)
                prior = lambda x:a*np.exp(x-mu)**2

    with c2:
        likelihood_distribution: Distributions = st.selectbox("Likelihood", [dist.value for dist in Distributions])
        match likelihood_distribution:
            case Distributions.Geometric.value:
                st.latex("A(x-B)^C")
                a = st.number_input("$A$", value=1.)
                b = st.number_input("$B$", value=0.)
                c = st.number_input("$C$", value=1.)
                likelihood = lambda x:a*(x-b)**c
            case Distributions.Linear.value:
                st.latex("Ax+B")
                a = st.number_input("$A$", value=1.)
                b = st.number_input("$B$", value=0.)
                likelihood = lambda x:a*x+b
            case Distributions.Gaussian.value:
                st.latex("Ae^{-(x-\\mu)^2}")
                a = st.number_input("$A$", value=1.)
                mu = st.number_input("$\\mu$", value=0.)
                likelihood = lambda x:a*np.exp(x-mu)**2

domain = np.linspace(-1,1,201)
bayes_context_object = BayesUpdate(pd.DataFrame({
    "prior": [prior(x) for x in domain],
    "likelihood": [likelihood(x) for x in domain]
}, index=domain))
c1, c2, c3 = st.columns(3)
with c1:
    st.line_chart(bayes_context_object.df["prior"], x_label="prior")
with c2:
    st.line_chart(bayes_context_object.df["likelihood"], x_label="likelihood")
with c3:
    st.line_chart(bayes_context_object.df["posterior"], x_label="posterior")

st.header("Raw data", divider="orange")
st.dataframe(bayes_context_object.df.T)