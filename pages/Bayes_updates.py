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


domain = np.linspace(-1,1,201)


with st.sidebar:

    st.header("Setup", divider="blue")
    c1, c2 = st.columns(2)
    with c1:
        prior_distribution: Distributions = st.selectbox("Prior", [dist.value for dist in Distributions])
        match prior_distribution:
            case Distributions.Geometric.value:
                st.latex("a_1(x-b_1)^{c_1}")
                a = st.number_input("$a_1$", value=1.)
                b = st.number_input("$b_1$", value=0.)
                c = st.number_input("$c_1$", value=1.)
                prior = lambda x:a*(x-b)**c
                st.latex(f"{a}(x{-1 * b:+})^{{ {c} }}")
            case Distributions.Linear.value:
                st.latex("a_1x+b_1")
                a = st.number_input("$a_1$", value=1.)
                b = st.number_input("$b_1$", value=0.)
                prior = lambda x:a*x+b
            case Distributions.Gaussian.value:
                st.latex("e^{-a_1(x-\\mu_1)^2}")
                a = st.number_input("$a_1$", value=1.)
                mu = st.number_input("$\\mu_1$", value=0.)
                prior = lambda x:np.exp(-1 * a * (x-mu)**2)

        prior_vector = [max(prior(x), 0) for x in domain]
        prior_vector /= sum(prior_vector)

    with c2:
        likelihood_distribution: Distributions = st.selectbox("Likelihood", [dist.value for dist in Distributions])
        match likelihood_distribution:
            case Distributions.Geometric.value:
                st.latex("a_2(x-b_2)^{c_2}")
                a = st.number_input("$a_2$", value=1.)
                b = st.number_input("$b_2$", value=0.)
                c = st.number_input("$c_2$", value=1.)
                likelihood = lambda x:a*(x-b)**c
                st.latex(f"{a}(x{-1 * b:+})^{{ {c} }}")
            case Distributions.Linear.value:
                st.latex("a_2x+b_2")
                a = st.number_input("$a_2$", value=1.)
                b = st.number_input("$b_2$", value=0.)
                likelihood = lambda x:a*x+b
            case Distributions.Gaussian.value:
                st.latex("e^{-a_2(x-\\mu_2)^2}")
                a = st.number_input("$a_2$", value=1.)
                mu = st.number_input("$\\mu_2$", value=0.)
                likelihood = lambda x:np.exp(-1 * a * (x-mu)**2)

        likelihood_vector = [max(likelihood(x), 0) for x in domain]
        likelihood_vector /= sum(likelihood_vector)

posterior_vector = prior_vector * likelihood_vector
posterior_vector /= sum(posterior_vector)
c1, c2, c3 = st.columns(3)
with c1:
    st.line_chart(prior_vector, x_label="prior")
with c2:
    st.line_chart(likelihood_vector, x_label="likelihood")
with c3:
    st.line_chart(posterior_vector, x_label="posterior")

st.header("Raw data", divider="orange")
st.dataframe(pd.DataFrame({
    "prior": prior_vector,
    "likelihood": likelihood_vector,
    "posterior": posterior_vector,
}).T)