from enum import Enum
from functools import partial

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bayesian Updates", page_icon="👶️", layout="wide")

st.header("Bayesian Updates", divider="blue")

with st.expander("📖 Methodology", expanded=False):
    st.markdown(r"""
    **Bayesian Updates** is a method for updating probability estimates based on new evidence.
    
    **Core Concept:** We start with a **prior** belief about a parameter, incorporate new data through a **likelihood** function,
    and obtain an updated **posterior** distribution.
    
    **Methodology:**
    1. **Prior Distribution**: Define an initial probability distribution over the parameter space
       - Polynomial prior: $a(x-b)^c$ for geometric/linear cases
       - Gaussian prior: $e^{-a(x-\mu)^2}$ for smooth peaked distributions
    2. **Likelihood Function**: Model how likely observed data is given different parameter values
       - Uses the same functional forms as the prior
    3. **Posterior Calculation**: Apply Bayes' theorem: $P(\theta|data) \propto P(data|\theta) \cdot P(\theta)$
       - Normalize the product of prior and likelihood
    4. **Visualization**: Compare prior, likelihood, and posterior side-by-side
    
    **Physical Context:** In quantum metrology, this approach is used to update estimates of unknown phase shifts
    based on measurement outcomes, combining prior knowledge with experimental data.
    """)


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
                st.latex(r"a_1(x-b_1)^{c_1}")
                a = st.number_input("$a_1$", value=1.0)
                b = st.number_input("$b_1$", value=0.0)
                c = st.number_input("$c_1$", value=1.0)
                st.latex(f"{a}(x{-1 * b:+})^{{ {c} }}")
                prior_fn = partial(prior_polynomial, a=a, b=b, c=c)
            case Distributions.Linear.value:
                st.latex(r"a_1x+b_1")
                a = st.number_input("$a_1$", value=1.0)
                b = st.number_input("$b_1$", value=0.0)
                c = 1.0  # Linear is polynomial with c=1
                prior_fn = partial(prior_polynomial, a=a, b=b, c=c)
            case Distributions.Gaussian.value:
                st.latex(r"e^{-a_1(x-\\mu_1)^2}")
                a = st.number_input("$a_1$", value=1.0)
                mu = st.number_input("$\\mu_1$", value=0.0)
                prior_fn = partial(prior_gaussian, a=a, mu=mu)

        prior_vector = np.maximum(prior_fn(domain), 0)
        prior_vector = prior_vector / np.sum(prior_vector)

    with c2:
        likelihood_distribution = st.selectbox(
            "Likelihood", [dist.value for dist in Distributions]
        )
        match likelihood_distribution:
            case Distributions.Geometric.value:
                st.latex(r"a_2(x-b_2)^{c_2}")
                a = st.number_input("$a_2$", value=1.0)
                b = st.number_input("$b_2$", value=0.0)
                c = st.number_input("$c_2$", value=1.0)
                st.latex(f"{a}(x{-1 * b:+})^{{ {c} }}")
                likelihood_fn = partial(likelihood_polynomial, a=a, b=b, c=c)
            case Distributions.Linear.value:
                st.latex(r"a_2x+b_2")
                a = st.number_input("$a_2$", value=1.0)
                b = st.number_input("$b_2$", value=0.0)
                c = 1.0  # Linear is polynomial with c=1
                likelihood_fn = partial(likelihood_polynomial, a=a, b=b, c=c)
            case Distributions.Gaussian.value:
                st.latex(r"e^{-a_2(x-\\mu_2)^2}")
                a = st.number_input("$a_2$", value=1.0)
                mu = st.number_input("$\\mu_2$", value=0.0)
                likelihood_fn = partial(likelihood_gaussian, a=a, mu=mu)

        likelihood_vector = np.maximum(likelihood_fn(domain), 0)
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
