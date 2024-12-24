from altair import value
from dataclasses import dataclass
from plotly import express as px
from scipy.special import comb
from src.enums import ProbabilityDistribution
from src.plotting import plot_array
from tqdm import tqdm
from typing import Any, Dict, Tuple
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from multiprocessing import Pool

st.set_page_config(page_title="Numerical Quantum Time Evolution", page_icon="🦖️", layout="wide")

st.cache_data.clear()
st.cache_resource.clear()

# START
with st.sidebar:


    st.header("Setup", divider="blue")
    initial_wave_packet: ProbabilityDistribution = st.selectbox("$X$", [prob_dist.value for prob_dist in ProbabilityDistribution])

    match initial_wave_packet:

        case ProbabilityDistribution.ParticleDecay.value:
            c1, c2, c3 = st.columns(3)
            with c1:
                n_particles = st.number_input("$N$ particles", min_value=1, value=50, max_value=5000)
            with c2:
                decay_lambda = st.number_input("$\lambda$ ( decay )", min_value=0.001, value=1.)
            with c3:
                t_max = st.number_input("$t_{max}$", min_value=1, value=2)

            st.markdown(f"""
                The corresponding halflife is
                
                $\\frac{{\\log(2)}}{{\\lambda}} \\approx \\frac{{{np.log(2):.5f}}}{{{decay_lambda:.5f}}} \\approx {np.divide(np.log(2), decay_lambda):.5f}$.
            """)

st.header("Setup", divider="blue")

match initial_wave_packet:

    case ProbabilityDistribution.ParticleDecay.value:

        def f(args: Tuple[float, float])-> Dict[str, Any]:
            k,t = args
            return {
                "k": k,
                "t": t,
                "prob": comb(n_particles, k) * (1-np.exp(-1 * decay_lambda * t))**k * (np.exp(-1 * decay_lambda * t))**(n_particles - k)
            }


        granularity = 500
        iterator = itertools.product(
            range(n_particles+1),
            [round(x,7) for x in np.linspace(0,t_max,granularity + 1)]
        )

        # pool = Pool()
        # for result in pool.imap(f, iterator):
        #     print(result)
        # df = pd.DataFrame(tqdm(pool.imap(f, iterator), total = (n_particles + 1)*(granularity + 1)))
        df = pd.DataFrame([f(args) for args in tqdm(iterator, total=(n_particles + 1) * (granularity + 1))])
        plot_array(df.pivot(columns="t", index="k", values="prob"), midpoint=None, text_auto=False)
        st.header("Raw data", divider="green")
        st.dataframe(df.pivot(columns="k", index="t", values="prob"))