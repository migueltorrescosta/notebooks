from enum import Enum
from functools import partial

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bayesian Updates", page_icon="👶️", layout="wide")


class Distributions(str, Enum):
    Geometric = "Geometric"
    Linear = "Linear"
    Gaussian = "Gaussian"


def prior_polynomial(x: float, a: float, b: float, c: float) -> float:
    return a * (x - b) ** c


def prior_gaussian(x: float, a: float, mu: float) -> float:
    return np.exp(-1 * a * (x - mu) ** 2)


def likelihood_polynomial(x: float, a: float, b: float, c: float) -> float:
    return a * (x - b) ** c


def likelihood_gaussian(x: float, a: float, mu: float) -> float:
    return np.exp(-1 * a * (x - mu) ** 2)


domain = np.linspace(-1, 1, 201)
prior_vector = np.array([0.0])
likelihood_vector = np.array([0.0])


with st.sidebar:
    st.header("Setup", divider="blue")
    c1, c2 = st.columns(2)
    with c1:
        prior_distribution = st.selectbox(
            "Prior", [dist.value for dist in Distributions]
        )
        match prior_distribution:
            case Distributions.Geometric.value:
                st.latex("a_1(x-b_1)^{c_1}")
                a = st.number_input("$a_1$", value=1.0)
                b = st.number_input("$b_1$", value=0.0)
                c = st.number_input("$c_1$", value=1.0)
                st.latex(f"{a}(x{-1 * b:+})^{{ {c} }}")
                prior_fn = partial(prior_polynomial, a=a, b=b, c=c)
            case Distributions.Linear.value:
                st.latex("a_1x+b_1")
                a = st.number_input("$a_1$", value=1.0)
                b = st.number_input("$b_1$", value=0.0)
                c = 1.0  # Linear is polynomial with c=1
                prior_fn = partial(prior_polynomial, a=a, b=b, c=c)
            case Distributions.Gaussian.value:
                st.latex("e^{-a_1(x-\\mu_1)^2}")
                a = st.number_input("$a_1$", value=1.0)
                mu = st.number_input("$\\mu_1$", value=0.0)
                prior_fn = partial(prior_gaussian, a=a, mu=mu)

        prior_vector = np.array([max(prior_fn(x), 0) for x in domain])
        prior_vector = prior_vector / np.sum(prior_vector)

    with c2:
        likelihood_distribution = st.selectbox(
            "Likelihood", [dist.value for dist in Distributions]
        )
        match likelihood_distribution:
            case Distributions.Geometric.value:
                st.latex("a_2(x-b_2)^{c_2}")
                a = st.number_input("$a_2$", value=1.0)
                b = st.number_input("$b_2$", value=0.0)
                c = st.number_input("$c_2$", value=1.0)
                st.latex(f"{a}(x{-1 * b:+})^{{ {c} }}")
                likelihood_fn = partial(likelihood_polynomial, a=a, b=b, c=c)
            case Distributions.Linear.value:
                st.latex("a_2x+b_2")
                a = st.number_input("$a_2$", value=1.0)
                b = st.number_input("$b_2$", value=0.0)
                c = 1.0  # Linear is polynomial with c=1
                likelihood_fn = partial(likelihood_polynomial, a=a, b=b, c=c)
            case Distributions.Gaussian.value:
                st.latex("e^{-a_2(x-\\mu_2)^2}")
                a = st.number_input("$a_2$", value=1.0)
                mu = st.number_input("$\\mu_2$", value=0.0)
                likelihood_fn = partial(likelihood_gaussian, a=a, mu=mu)

        likelihood_vector = np.array([max(likelihood_fn(x), 0) for x in domain])
        likelihood_vector = likelihood_vector / np.sum(likelihood_vector)

    posterior_vector = prior_vector * likelihood_vector
    posterior_vector = posterior_vector / np.sum(posterior_vector)
c1, c2, c3 = st.columns(3)
with c1:
    st.line_chart(prior_vector, x_label="prior")
with c2:
    st.line_chart(likelihood_vector, x_label="likelihood")
with c3:
    st.line_chart(posterior_vector, x_label="posterior")

st.header("Raw data", divider="orange")
st.dataframe(
    pd.DataFrame(
        {
            "prior": prior_vector,
            "likelihood": likelihood_vector,
            "posterior": posterior_vector,
        }
    ).T
)
