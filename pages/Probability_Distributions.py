from scipy.special import comb
from src.utils.enums import ProbabilityDistribution
from src.visualization.plotting import plot_array
from tqdm import tqdm
from typing import Any, Dict, Tuple
import itertools
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Numerical quantum Time Evolution", page_icon="🦖️", layout="wide"
)

st.header("Probability Distributions", divider="blue")

with st.expander("📖 Methodology", expanded=False):
    st.markdown("""
    **Particle Decay Model** calculates the probability distribution for the number of particles remaining
    after time $t$ in a system with exponential decay.
    
    **Physical Model:**
    - $N$: Initial number of particles
    - $\\lambda$: Decay rate (probability per unit time that a particle decays)
    - $t$: Observation time
    
    **The Decay Process:**
    - Each particle decays independently with probability $p_{decay}(t) = 1 - e^{-\\lambda t}$
    - The number of particles that have decayed by time $t$ follows a binomial distribution
    
    **Methodology:**
    1. **Decay Probability**: Compute $p = 1 - e^{-\\lambda t}$ for each time step
    2. **Binomial Distribution**: For $k$ particles remaining:
       $$P(K = k) = \\binom{N}{k} p^k (1-p)^{N-k}$$
    3. **Heatmap Visualization**: Display $P(K = k, t)$ over all $k \\in [0, N]$ and $t \\in [0, t_{max}]$
    
    **Key Properties:**
    - **Halflife**: $t_{1/2} = \\frac{\\ln 2}{\\lambda}$ — time for half the particles to decay
    - **Expected value**: $\\mathbb{E}[K] = N e^{-\\lambda t}$
    - **Variance**: $\\mathrm{Var}(K) = N e^{-\\lambda t}(1 - e^{-\\lambda t})$
    
    **Physical Context:** This model describes first-order chemical reactions, radioactive decay,
    and other Poisson processes in the binomial approximation.
    """)

# START
with st.sidebar:
    st.header("Setup", divider="blue")
    initial_wave_packet_str = st.selectbox(
        "$X$", [prob_dist.value for prob_dist in ProbabilityDistribution]
    )
    initial_wave_packet: ProbabilityDistribution = ProbabilityDistribution(
        initial_wave_packet_str
    )

    match initial_wave_packet:
        case ProbabilityDistribution.ParticleDecay.value:
            c1, c2, c3 = st.columns(3)
            with c1:
                n_particles: int = st.number_input(
                    "$N$ particles", min_value=1, value=50, max_value=5000
                )
            with c2:
                decay_lambda = st.number_input(
                    r"$\lambda$ ( decay )", min_value=0.001, value=1.0
                )
            with c3:
                t_max = st.number_input("$t_{max}$", min_value=1, value=2)

            st.markdown(f"""
                The corresponding halflife is
                
                $\\frac{{\\log(2)}}{{\\lambda}} \\approx \\frac{{{np.log(2):.5f}}}{{{decay_lambda:.5f}}} \\approx {np.divide(np.log(2), decay_lambda):.5f}$.
            """)

st.header("Setup", divider="blue")

match initial_wave_packet:
    case ProbabilityDistribution.ParticleDecay.value:

        def f(args: Tuple[float, float]) -> Dict[str, Any]:
            k, t = args
            return {
                "k": k,
                "t": t,
                "prob": comb(n_particles, k)
                * (1 - np.exp(-1 * decay_lambda * t)) ** k
                * (np.exp(-1 * decay_lambda * t)) ** (n_particles - k),
            }

        granularity = 500
        iterator = itertools.product(
            range(n_particles + 1),
            [round(x, 7) for x in np.linspace(0, t_max, granularity + 1)],
        )

        # pool = Pool()
        # for result in pool.imap(f, iterator):
        #     print(result)
        # df = pd.DataFrame(tqdm(pool.imap(f, iterator), total = (n_particles + 1)*(granularity + 1)))
        df = pd.DataFrame(
            [
                f(args)
                for args in tqdm(iterator, total=(n_particles + 1) * (granularity + 1))
            ]
        )
        plot_array(
            np.array(df.pivot(columns="t", index="k", values="prob")),
            midpoint=None,
            text_auto=False,
        )
        st.header("Raw data", divider="green")
        st.dataframe(df.pivot(columns="k", index="t", values="prob"))
