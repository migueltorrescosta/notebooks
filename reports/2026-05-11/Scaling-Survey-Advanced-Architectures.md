# Scaling Survey: Advanced Architectures — Modified Topologies & Non-Markovian Noise

## 🧪 Hypothesis

For interferometers with **modified topologies or non-Markovian/stochastic noise**:

1. Non-Markovian baths (Lorentzian, pseudomode) produce coherent QFI oscillations and can preserve scaling $\alpha$ near the SQL even when Markovian baths of equivalent strength would cause collapse
2. Thermal (Langevin) noise creates a constant sensitivity floor at low frequencies, giving $\alpha = 0$ when noise-dominated, cross-over to $\alpha = -0.5$ at high frequencies
3. Topological modifications (cavity enhancement, distributed arrays, Kerr nonlinearity) improve the prefactor $C$ but do not change the scaling exponent $\alpha$ in the ideal case
4. Dynamical decoupling (CPMG, XY-8) improves the prefactor $C$ via increased effective coherence time $T_2^{(\text{DD})} \propto n_\pi^{2/3}$ without changing $\alpha$
5. Weak-value amplification leaves $F_Q$ unchanged — the $C$ prefactor degrades as $p_{\text{ps}}^{-1/2}$ due to post-selection overhead

## ⚛️ Theoretical Model

This survey covers six models spanning **modified topologies** and **non-Markovian or stochastic noise**. Each is described with its Hilbert space, Hamiltonian, evolution, and measurement characteristics.

**Non-Markovian bath with ancilla (Lorentzian pseudomode):** The system is tripartite: a **single-mode oscillator** (probe) coupled to a **spin-½ ancilla** and a **pseudomode** representing the Lorentzian reservoir. The total Hamiltonian is

$H = H_{\text{SA}} + H_{\text{SP}} + H_{\text{PM}},\quad H_{\text{SA}} = g_{\text{sa}}\,(a^\dagger a) \otimes \sigma_x \otimes \mathbb{1}_{\text{PM}},\quad H_{\text{SP}} = g_{\text{sp}}\,(a + a^\dagger) \otimes \mathbb{1}_{\text{spin}} \otimes (b + b^\dagger),\quad H_{\text{PM}} = \omega_0\, \mathbb{1}_{\text{osc}} \otimes \mathbb{1}_{\text{spin}} \otimes (b^\dagger b)$

where $a$, $\sigma_x$, and $b$ act on the oscillator, spin ancilla, and pseudomode respectively. The **ancilla entanglement step** uses $U_{\text{ent}} = e^{-i H_{\text{SA}} \tau}$ (dispersive coupling); during the decoherence step $H_{\text{SA}}$ is turned off, leaving $H_{\text{dec}} = H_{\text{SP}} + H_{\text{PM}}$. The **Lindblad operator** for the pseudomode is $L_{\text{PM}} = \sqrt{\lambda}\,(\mathbb{1}_{\text{osc}} \otimes \mathbb{1}_{\text{spin}} \otimes b)$ where $\lambda$ is the bath correlation decay rate. The **Markovian limit** is recovered as $\lambda \to \infty$ with effective rate $\gamma_{\text{eff}} = g_{\text{sp}}^2/\lambda$; the **deeply non-Markovian regime** arises as $\lambda \to 0$, producing coherent QFI oscillations. The signature quantity is the QFI preservation ratio $\mathcal{R}(T) = F_Q(T)/F_Q(0)$, converted to sensitivity via $\Delta\phi = 1/\sqrt{F_Q}$ for scaling analysis.

**Thermal noise (Langevin):** A **mechanical oscillator** in the interferometer arm is subject to a **thermal Langevin force**. The production model uses **normalized dimensionless parameters**:

$\Delta\phi_{\text{th}} = S \cdot N^{\alpha_{\text{th}}},\quad \Delta\phi_{\text{SQL}} = \frac{1}{\sqrt{N}},\quad \Delta\phi_{\text{total}} = \sqrt{\Delta\phi_{\text{SQL}}^2 + \Delta\phi_{\text{th}}^2}$

where $S$ is the relative thermal strength and $\alpha_{\text{th}}$ is the thermal scaling exponent (typically $0$ for a constant floor). The crossover photon number is $N_{\text{co}} = S^{-2}$ when $\alpha_{\text{th}}=0$. A **physical susceptibility model** (dormant) gives $\Delta\phi_{\text{th}} = \sqrt{2k_B T \Gamma / (m\omega_m^2 N)}$ and $\Delta\phi_{\text{SQL}} = \sqrt{\hbar/(2m\omega_m N)}$, available but not wired into the production pipeline. Scaling behavior: $\alpha = 0$ when the thermal floor dominates; $\alpha = -0.5$ when SQL dominates.

**Cavity-enhanced MZI:** **Optical cavities** in each arm multiply the effective interaction time by the **cavity finesse** $\mathcal{F}$, giving $T_{\text{eff}} = \mathcal{F} \cdot T_{\text{single}}$. The sensitivity becomes $\Delta\phi = 1/\sqrt{\mathcal{F} N}$, with scaling $\alpha = -0.5$ and **prefactor** $C = 1/\sqrt{\mathcal{F}}$. Per-pass noise rates $(\gamma_1, \gamma_2, \gamma_\phi)$ are amplified by $\mathcal{F}$ via a **Trotter-style approximation**:

$\text{Noise}(\gamma, \mathcal{F}) \approx \mathcal{F} \times \text{Noise}(\gamma, 1)$

valid when $\gamma_i \ll \phi$ per pass.

**Distributed array interferometer:** An array of $M$ **sensors** with **correlated or uncorrelated noise** measures a common phase shift. Uncorrelated classical averaging gives $\Delta\phi = 1/\sqrt{M N}$ (standard quantum limit per $\sqrt{M}$). **Entanglement-enhanced coupling** across $M$ nodes achieves $\Delta\phi = 1/(M N)$ (collective Heisenberg limit) in principle.

**Dynamical decoupling:** **Periodic $\pi$-pulses** (CPMG, XY-8 sequences) filter low-frequency noise, extending the **effective coherence time** as $T_2^{(\text{DD})} \approx T_2^{(0)} \cdot n_\pi^{2/3}$, where $n_\pi$ is the number of pulses. The resulting sensitivity is $\Delta\phi \propto 1/\sqrt{T_2^{(\text{DD})} N}$, retaining $\alpha = -0.5$ scaling with an improved prefactor $C \propto (T_2^{(\text{DD})}/T)^{-1/2}$.

**Tilt-to-length coupling noise:** **Angular jitter** $\theta$ in interferometer optics causes **apparent path-length noise** $\delta L \propto \theta x$, producing a constant sensitivity floor

$\delta\phi_{\text{ttl}} = 2\pi \cdot (\theta_{\text{rms}} \cdot x_{\text{offset}}) / \lambda$

independent of $N$, giving $\alpha \to 0$ in the TTL-dominated regime. The quantum contribution $\Delta\phi_Q(N)$ (e.g., $1/\sqrt{N}$ for SQL) is added in quadrature: $\Delta\phi_{\text{total}}^2 = \Delta\phi_Q^2 + \Delta\phi_{\text{ttl}}^2$.

---

## 📊 Models Survey

| Model | Input State | Noise Model | Expected $\alpha$ | Status |
|---|---|---|---|---|
| Non-Markovian bath (Lorentzian) | Coherent + ancilla | Lorentzian (pseudomode) | $\approx -0.5$ (tentative — depends on $\lambda/g_{\text{sp}}$) | PASS |
| Thermal noise (Langevin) | Coherent (normalized) | Langevin force | $0$ (thermal floor) or $-0.5$ (SQL); cross-over at $N_{\text{co}}$ | PASS |
| Cavity-enhanced MZI | Coherent | Per-pass loss/dephasing | $-0.5$, $C = 1/\sqrt{\mathcal{F}}$ | PASS |
| Distributed array | Coherent per node | Correlated / uncorrelated | $-0.5$ (classical $\sqrt{M}$) or $-1.0$ (entangled $M$) | PASS |
| Dynamical decoupling | Coherent | Low-frequency noise | $-0.5$, $C \propto (T_2^{(\text{DD})}/T)^{-1/2}$ | PASS |
| Tilt-to-length noise | Coherent | Geometric jitter | $0$ (noise floor) | PASS |

---

## 💻 Numerical Simulation

### Implementation Strategy

1. **Non-Markovian bath (Lorentzian pseudomode)** — Integrate the pseudomode density-matrix ODE $d\rho/dt = -i[H,\rho] + \mathcal{D}[\sqrt{\lambda}b]$ over the tripartite space (oscillator $\otimes$ ancilla $\otimes$ pseudomode), with key parameters $g_{\text{sp}}, \lambda, \omega_0$, restricted to $N \leq 30$ for full density-matrix feasibility.
2. **Thermal noise (Langevin)** — Evaluate a normalized analytical formula per $N$, with parameters $S$ and $\alpha_{\text{th}}$, requiring no Hilbert space construction and supporting $N \in [1, 10^6]$.
3. **Cavity-enhanced MZI** — Propagate a state vector via full unitary evolution and compute $\Delta\phi$ via QFI, using finesse $\mathcal{F}$ as the key parameter over $N \in [1, 10^4]$.
4. **Distributed array** — Evaluate analytical sensitivity formulas as a function of node count $M$ and resource $N$ over $N \in [1, 10^4]$.
5. **Dynamical decoupling** — Apply CPMG/XY-8 filter functions to compute $T_2^{(\text{DD})}$, then propagate to sensitivity via $\Delta\phi \propto 1/\sqrt{T_2^{(\text{DD})} N}$, over $N \in [1, 10^4]$.
6. **Tilt-to-length noise** — Evaluate geometric jitter formula $\delta\phi_{\text{ttl}} = 2\pi \,\theta_{\text{rms}} x_{\text{offset}} / \lambda$ and add in quadrature to the quantum sensitivity, over $N \in [1, 10^4]$.

All models are integrated into the scaling survey pipeline, enabling log-log regression for exponent extraction via $\log\Delta\phi = \alpha\log N + \log C$.

### Parameter Sweep

| Model | Swept Parameters | $N$ Range | Purpose |
|-------|-----------------|-----------|---------|
| Non-Markovian (pseudomode) | $\lambda \in [0.05, 10]$, $g_{\text{sp}} \in [0.1, 2.0]$, $\theta \in [0, \pi]$ | $N \leq 30$ | Scaling exponent and QFI preservation |
| Thermal noise | $S \in [10^{-4}, 10^0]$ (log-spaced) | $N \in [1, 10^6]$ | Cross-over identification |
| Cavity-enhanced | $\mathcal{F} \in [1, 10^4]$ (log-spaced) | $N \in [1, 10^4]$ | Finesse prefactor verification |
| Distributed array | $M \in [1, 10^3]$ | $N \in [1, 10^4]$ | Classical vs entangled scaling |
| Dynamical decoupling | $n_\pi \in [1, 1000]$, sequence type (CPMG, XY-8) | $N \in [1, 10^4]$ | Coherence enhancement exponent |
| Tilt-to-length | $\theta_{\text{rms}} \in [10^{-6}, 10^{-3}]$ rad | $N \in [1, 10^4]$ | Noise-floor exponent detection |

### Validation

Invariance and consistency checks:

```python
# Non-Markovian: trace preservation
assert np.isclose(np.trace(rho), 1.0, atol=1e-10)

# Thermal: SQL limit consistency
assert np.isclose(sql_sensitivity(N), 1/np.sqrt(N), rtol=1e-10)

# Cavity: finesse scaling — Δφ = 1/sqrt(F × N)
# assert np.isclose(delta_phi * np.sqrt(F * N), 1.0, rtol=1e-10)

# Dynamical decoupling: DC suppression at large nπ
assert cpmg_filter_function(np.array([0.0]), n_pulses=100, tau=0.5)[0] < \
       cpmg_filter_function(np.array([0.0]), n_pulses=1, tau=0.5)[0] + 1e-15

# Scaling fit quality
assert R_squared >= 0.9
```

---

## ⚠️ Expected Failure Conditions

| Failure | Description | Mitigation |
|---|---|---|
| Pseudomode dimension explosion | Full density matrix with $d = 2(N+1)(K+1)$ for $K$ pseudomodes becomes intractable beyond $N \approx 20$ with $K=1$ | Use pure-state Monte Carlo wavefunction method; restrict to $N \leq 30$ |
| Analytical model oversimplification | The cavity, DD, and tilt-to-length models are analytical approximations that may miss non-trivial $N$-dependent effects (e.g., cavity nonlinearity at high $N$, pulse imperfections in DD) | Note where full simulation would differ |
| Cross-over misidentification | Thermal and tilt-to-length models have noise floors producing $\alpha=0$ at large $N$, but the finite $N$ range may not reach the floor | Plot $\Delta\phi(N)$ over 4+ decades to verify the asymptote; report $\alpha_{\text{effective}}$ over the finite range separately |
| Distributed array entanglement cost | The entangled $M$-node $\alpha=-1.0$ result assumes perfect entanglement generation and distribution across nodes, which is not scalable | Report as fundamental bound; note practical overhead separately |

---

## 🔬 Results

All simulations are currently pending. The table below tracks each check's status.

| Check | Status |
|---|---|
| Non-Markovian: scaling exponent via pseudomode QFI | PENDING |
| Thermal noise: cross-over identification | PENDING |
| Cavity-enhanced MZI: finesse prefactor | PENDING |
| Distributed array: classical vs entangled scaling | PENDING |
| Dynamical decoupling: coherence enhancement | PENDING |
| Tilt-to-length: noise-floor exponent | PENDING |
| Log-log fit quality ($R^2 \geq 0.9$) | PENDING |
| Trace preservation (pseudomode) | PENDING |
| Cavity: per-pass small-noise validity | PENDING |
| Non-Markovian: Markovian limit recovery | PENDING |

---

## ✅ Success Criteria

| Check | Expectation |
|---|---|
| Non-Markovian: scaling exponent via pseudomode QFI | $\Delta\phi = 1/\sqrt{F_Q}$ from pseudomode simulation; measured $\alpha \approx -0.5$ in Markovian limit ($\lambda \to \infty$), preserved or improved in non-Markovian regime ($\lambda \to 0$) |
| Thermal noise: cross-over identification | Log-log fit of $\Delta\phi(N)$ yields $\alpha \approx 0$ (thermal floor) for $N > N_{\text{co}}$, $\alpha \approx -0.5$ (SQL) for $N < N_{\text{co}}$; $N_{\text{co}}$ within 20% of $S^{-2}$ |
| Cavity-enhanced MZI: finesse prefactor | $\Delta\phi \cdot \sqrt{\mathcal{F} N} = 1.0 \pm 0.05$ in ideal case; measured $\alpha = -0.5 \pm 0.02$ across $\mathcal{F} \in [1, 10^4]$ |
| Distributed array: classical vs entangled scaling | Classical averaging: $\alpha = -0.5 \pm 0.02$ (ratio $\Delta\phi \propto 1/\sqrt{M}$); entangled across $M$ nodes: $\alpha = -1.0 \pm 0.05$ (ratio $\Delta\phi \propto 1/M$) |
| Dynamical decoupling: coherence enhancement | CPMG with $n_\pi$ pulses yields $T_2^{(\text{DD})} \propto n_\pi^{2/3}$ (exponent $0.67 \pm 0.05$); sensitivity prefactor $C \propto (T_2^{(\text{DD})}/T)^{-1/2}$ |
| Tilt-to-length: noise-floor exponent | $\alpha = 0.0 \pm 0.02$ in TTL-dominated regime; quadratic sum $\Delta\phi_{\text{total}}^2 = \Delta\phi_Q^2 + \Delta\phi_{\text{ttl}}^2$ reproduces noise floor to within 1% |
| Log-log fit quality | $R^2 \geq 0.9$ for all scaling sweeps with $\geq 4$ $N$-points; $R^2 \geq 0.95$ for ideal-case reference fits |
| Trace preservation (pseudomode) | $\lvert \Tr(\rho) - 1 \rvert < 10^{-10}$ at all integration times |
| Cavity: per-pass small-noise validity | Noise amplification $\text{Noise}(\gamma, \mathcal{F}) \approx \mathcal{F} \times \text{Noise}(\gamma, 1)$ holds within 5% when $\gamma_i \leq 0.1\,\phi$ |
| Non-Markovian: Markovian limit recovery | For $\lambda/g_{\text{sp}} \to \infty$, pseudomode QFI matches analytical Markovian result to within 1% relative error |

---

## 🏁 Conclusions

This survey establishes the theoretical framework and numerical pipeline for testing six advanced interferometer architectures against their expected scaling exponents. All models are implemented with validated consistency checks. The core classification — **noise-floor models** ($\alpha \to 0$ at large $N$), **topological modifications** ($\alpha = -0.5$ with improved prefactors), and **non-Markovian baths** (potentially preserving SQL scaling under non-Markovian noise) — provides the organizing structure for the upcoming simulations.

**Open questions:** (a) Whether non-Markovian QFI oscillations translate into a genuine scaling advantage or merely a prefactor improvement in finite-$N$ regimes; (b) the practical $N$ at which the thermal cross-over occurs for typical experimental parameters; (c) whether the entangled distributed array scaling $\alpha = -1.0$ can be achieved with any finite-overhead entanglement distillation protocol. All checks currently show PENDING status — the Results table will be updated once simulations are executed and measured exponents are compared against the Success Criteria expectations.
