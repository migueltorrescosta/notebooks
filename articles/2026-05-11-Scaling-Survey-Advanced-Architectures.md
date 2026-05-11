# Scaling Survey: Advanced Architectures — Modified Topologies & Non-Markovian Noise

## Hypothesis

For interferometers with **modified topologies or non-Markovian/stochastic noise**:

1. Non-Markovian baths (Lorentzian, pseudomode) produce coherent QFI oscillations and can preserve scaling $\alpha$ near the SQL even when Markovian baths of equivalent strength would cause collapse
2. Thermal (Langevin) noise creates a constant sensitivity floor at low frequencies, giving $\alpha = 0$ when noise-dominated, cross-over to $\alpha = -0.5$ at high frequencies
3. Topological modifications (cavity enhancement, distributed arrays, Kerr nonlinearity) improve the prefactor $C$ but do not change the scaling exponent $\alpha$ in the ideal case
4. Dynamical decoupling (CPMG, XY-8) improves the prefactor $C$ via increased effective coherence time $T_2^{(\text{DD})} \propto n_\pi^{2/3}$ without changing $\alpha$
5. Weak-value amplification leaves $F_Q$ unchanged — the $C$ prefactor degrades as $p_{\text{ps}}^{-1/2}$ due to post-selection overhead

---

## Literature Review

| Concept & Motivation | Article | Year |
|---|---|---|
| Pseudomode method for non-Markovian open quantum systems: maps Lorentzian-structured reservoir to a single damped pseudomode + Markovian bath | Garraway, *Phys. Rev. A* 55, 2290 | 1997 |
| Gravitational phase shifts in fiber interferometers: phase noise accumulation over long baselines; path-imbalance sensitivity scaling | *High-Sensitivity Fiber Interferometer for Gravitational Phase Shift Measurement* ([arXiv:2506.09770](https://arxiv.org/abs/2506.09770)) | 2025 |
| Variance-based dark-matter detection with fluctuations: super-binomial variance metrics as sensitivity signal; relevant for scaling under stochastic noise | *Fluctuations in Atom Interferometers as a New Tool for Dark Matter* ([arXiv:2602.23427](https://arxiv.org/abs/2602.23427)) | 2026 |
| Multiparameter noise budgeting for space interferometers: high-dimensional sensitivity analysis, parameter-space reduction; tilt-to-length noise methodology | *Multiparameter Hierarchical Sensitivity Analysis of Tilt-to-Length Coupling Noise in Taiji Interferometer* | 2025/2026 |
| Ancilla-assisted metrology in non-Markovian environments: pseudomode-based QFI; preservation ratio $\mathcal{R}(T)$ scaling | `articles/2026-05-09-Ancilla-Assisted-Metrology-Non-Markovian.md` (prior repo) | 2026 |
| Quantum limits for interferometry (review): QCRB, SQL, Heisenberg limit | *Sensitivity of Quantum-Enhanced Interferometers* (review) | — |

---

## Theoretical Model

This survey covers models with **different Hilbert spaces and solvers**. Each is described independently.

### Model 1: Non-Markovian Bath with Ancilla (Lorentzian, Pseudomode)

**System**: Tripartite: single-mode oscillator (probe) × spin-½ (ancilla) × pseudomode (Lorentzian bath).
**Method**: Pseudomode — replace the reservoir with a single damped harmonic oscillator.

$$H = H_{\text{SA}} + H_{\text{SP}} + H_{\text{PM}},\quad
H_{\text{SA}} = g_{\text{sa}}\,(a^\dagger a) \otimes \sigma_x \otimes I_{\text{PM}},\quad
H_{\text{SP}} = g_{\text{sp}}\,(a + a^\dagger) \otimes I_{\text{spin}} \otimes (b + b^\dagger),\quad
H_{\text{PM}} = \omega_0\, I_{\text{osc}} \otimes I_{\text{spin}} \otimes (b^\dagger b)$$

Here $a$, $\sigma_x$, $b$ act on the oscillator, spin ancilla, and pseudomode respectively.
The ancilla entanglement step uses $U_{\text{ent}} = e^{-i H_{\text{SA}} \tau}$ (dispersive coupling).
During the decoherence step $H_{\text{SA}}$ is turned off, leaving $H_{\text{dec}} = H_{\text{SP}} + H_{\text{PM}}$.

Lindblad for pseudomode: $L_{\text{PM}} = \sqrt{\lambda}\,(I_{\text{osc}} \otimes I_{\text{spin}} \otimes b)$ where $\lambda$ is the bath correlation decay rate.

**Regimes**: $\lambda \to \infty$ → Markovian (rate $\gamma_{\text{eff}} = g_{\text{sp}}^2/\lambda$); $\lambda \to 0$ → deeply non-Markovian (coherent oscillations in QFI).

**Implementation**: `pseudomode_system.py`. This module computes QFI preservation ratios $\mathcal{R}(T) = F_Q(T)/F_Q(0)$ with and without ancilla. Scaling sweeps (Δφ vs N) are not directly implemented in the physics module itself, but the survey pipeline in `scaling_survey.py` bridges this by converting $F_Q \to \Delta\phi = 1/\sqrt{F_Q}$ via `_ancilla_sensitivity_fn`, enabling exponent fitting.

### Model 2: Thermal Noise (Langevin)

**System**: Mechanical oscillator in the MZI arm, subject to thermal Langevin force.

**Sensitivity** (analytical model — dimensionless normalized implementation):

$$\Delta\phi_{\text{th}} = S \cdot N^{\alpha_{\text{th}}},\quad
\Delta\phi_{\text{SQL}} = \frac{1}{\sqrt{N}},\quad
\Delta\phi_{\text{total}} = \sqrt{\Delta\phi_{\text{SQL}}^2 + \Delta\phi_{\text{th}}^2}$$

where $S$ is the relative thermal strength and $\alpha_{\text{th}}$ is the thermal scaling exponent (typically $0$ for a constant floor). The crossover $N_{\text{co}} = S^{-2}$ when $\alpha_{\text{th}}=0$.

**Physical model (dormant — `mechanical_susceptibility`, `force_psd_thermal` defined but unused in production)**:
$$\Delta\phi_{\text{th}} = \sqrt{\frac{2k_B T \Gamma}{m\omega_m^2 N}},\quad
\Delta\phi_{\text{SQL}} = \sqrt{\frac{\hbar}{2m\omega_m N}}$$

**Scaling**: $\alpha = 0$ when thermal floor dominates; $\alpha = -0.5$ when SQL dominates.

**Implementation**: `thermal_langevin.py`. The production path uses normalized dimensionless parameters (`thermal_strength`, `thermal_exponent`) for flexible scaling surveys. Full physical susceptibility functions are available but not wired into the sensitivity pipeline.

### Model 3: Cavity-Enhanced MZI

**System**: MZI with optical cavities in each arm. Effective interaction time $T_{\text{eff}} = \mathcal{F} \cdot T_{\text{single}}$ where $\mathcal{F}$ is cavity finesse.

**Sensitivity**: $\Delta\phi = 1/\sqrt{\mathcal{F} N}$. Scaling $\alpha = -0.5$, prefactor $C = 1/\sqrt{\mathcal{F}}$.

**Noise model**: Per-pass noise rates $(\gamma_1, \gamma_2, \gamma_\phi)$ are scaled by $\mathcal{F}$ using a Trotter-style approximation:
$$\text{Noise}(\gamma, \mathcal{F}) \approx \mathcal{F} \times \text{Noise}(\gamma, 1)$$
This is valid when $\gamma_i \ll \phi$ per pass (small noise relative to phase per pass).

**Implementation**: `cavity_mzi.py` provides `cavity_enhanced_mzi` for state-vector evolution and `cavity_enhanced_sensitivity(N, $\phi$, config)` for direct sensitivity computation via QFI on the output state.

### Model 4: Distributed Array Interferometer

**System**: $M$ sensors with correlated or uncorrelated noise. Collective phase sensitivity.

**Sensitivity**:
- Uncorrelated classical averaging: $\Delta\phi = 1/\sqrt{M N}$ (SQL per $\sqrt{M}$)
- Entanglement-enhanced across $M$ nodes: $\Delta\phi = 1/(M N)$ (collective HL)

**Implementation**: `distributed_mzi.py::distributed_mzi_sensitivity`.

### Model 5: Dynamical Decoupling

**System**: MZI with periodic $\pi$-pulses (CPMG, XY-8) to filter low-frequency noise.

**Effective coherence time**: $T_2^{(\text{DD})} \approx T_2^{(0)} \cdot n_\pi^{2/3}$ for CPMG.

**Sensitivity**: $\Delta\phi \propto 1/\sqrt{T_2^{(\text{DD})} N}$. Scaling $\alpha = -0.5$; prefactor $C \propto (T_2^{(\text{DD})}/T)^{-1/2}$.

**Implementation**: `dynamical_decoupling.py` (filter function + analytical model).

### Model 6: Tilt-to-Length Coupling Noise

**System**: Angular jitter $\theta$ in interferometer optics causes apparent path-length noise $\delta L \propto \theta x$.

**Sensitivity floor**: $\delta\phi_{\text{ttl}} = 2\pi \cdot (\theta_{\text{rms}} \cdot x_{\text{offset}}) / \lambda$ (constant noise floor, independent of $N$, giving $\alpha \to 0$). The quantum contribution $\Delta\phi_Q(N)$ (e.g., $1/\sqrt{N}$ for SQL) is added in quadrature externally via the `quantum_sensitivity` parameter; the TTL model itself is $N$-agnostic.

**Implementation**: `tilt_to_length_noise.py` (geometric noise model). The `$F_Q$`-based sensitivity floor equals the TTL phase noise; the total sensitivity is the quadratic sum of quantum and TTL contributions.

---

## Models Survey

| Model | Solver Type | Expected $\alpha$ | Scaling Sweep Integration | Implementation |
|---|---|---|---|---|---|
| Non-Markovian bath (Lorentzian) | Pseudomode ODE | Tentative — module reports $F_Q$ ratios; `scaling_survey.py` converts to $\Delta\phi = 1/\sqrt{F_Q}$ | ✅ Integrated via `_ancilla_sensitivity_fn` in `scaling_survey.py`; included in `create_default_survey` | `pseudomode_system.py` ✅ |
| Thermal noise (Langevin) | Normalized analytical | $0$ (thermal floor) or $-0.5$ (SQL regime); cross-over at $N_{\text{co}}$ | ✅ Standard `custom_sensitivity_fn` via `run_scaling_survey` | `thermal_langevin.py` (normalized model — physical susceptibility functions are defined but unused in production) |
| Cavity-enhanced MZI | Unitary (state vector + QFI) | $-0.5$, $C = 1/\sqrt{\mathcal{F}}$ | ✅ `cavity_enhanced_sensitivity(N, ...)` computes $\Delta\phi(N)$ directly; integrated via `custom_sensitivity_fn` | `cavity_mzi.py` ✅ |
| Distributed array | Analytical (multi-MZI) | $-0.5$ (classical, $\sqrt{M}$) or $-1.0$ (entangled, $M$) | ✅ Integrated via `custom_sensitivity_fn` in page pipeline (both classical and entangled variants) | `distributed_mzi.py` ✅ |
| Dynamical decoupling | Filter function | $-0.5$, $C \propto (T_2^{\text{(DD)}}/T)^{-1/2}$ | ✅ Integrated via `custom_sensitivity_fn` in page pipeline | `dynamical_decoupling.py` ✅ |
| Tilt-to-length noise | Geometric model | $0$ (noise floor); $C = 2\pi \cdot \theta_{\text{rms}} x_{\text{offset}} / \lambda$ | ✅ Integrated via `custom_sensitivity_fn` in page pipeline | `tilt_to_length_noise.py` ✅ |

---

## Numerical Simulation

### Implementation Strategy

Each model uses a different numerical approach:

| Model | Method | Key Parameters | $N$ Range | Scaling Sweep Available? |
|---|---|---|---|---|---|
| Non-Markovian | Pseudomode ODE: $d\rho/dt = -i[H,\rho] + \mathcal{D}[\sqrt{\lambda}b]$ | $g_{\text{sp}}, \lambda, \omega_0$ | $N \leq 30$ (full density matrix) | ✅ $\Delta\phi = 1/\sqrt{F_Q}$ via survey pipeline; module reports $F_Q$ ratios |
| Thermal noise | Evaluate normalized analytical formula per $N$ | $S$ (strength), $\alpha_{\text{th}}$ (exponent) | $N \in [1, 10^6]$ (no Hilbert space) | ✅ Standard `custom_sensitivity_fn` via `run_scaling_survey` |
| Cavity-enhanced | Full unitary (state vector); $\Delta\phi$ via QFI | $\mathcal{F}$ | $N \in [1, 10^4]$ | ✅ `cavity_enhanced_sensitivity(N, ...)` computes $\Delta\phi$; integrated via `custom_sensitivity_fn` |
| Distributed array | Analytical sensitivity formula | $M$ | $N \in [1, 10^4]$ | ✅ `distributed_mzi_sensitivity` returns $\Delta\phi$; wired via `custom_sensitivity_fn` |
| Dynamical decoupling | Filter-function → $T_2^{(\text{DD})}$ → sensitivity | $n_\pi$, pulse sequence | $N \in [1, 10^4]$ | ✅ `dd_sensitivity_scaling` computes $\Delta\phi$; wired via `custom_sensitivity_fn` |
| Tilt-to-length | $2\pi \cdot \theta_{\text{rms}} x_{\text{offset}} / \lambda$ (constant floor) | $\theta_{\text{rms}}, x_{\text{offset}}, \lambda$ | $N \in [1, 10^4]$ | ✅ `ttl_scaling_sweep` computes $\Delta\phi$; wired via `custom_sensitivity_fn` |

### Parameter Sweep

- Resource sweep per model capabilities (see $N$ range above)
- Noise parameters swept across ranges characteristic of each model
- Scaling exponent via log-log regression: $\log\Delta\phi = \alpha\log N + \log C$
- For non-Markovian: report full $F_Q(t)$ trajectory envelope, not just endpoint

### Validation

```python
# Non-Markovian: trace preservation (validate_pseudomode_density in pseudomode_system.py)
assert np.isclose(np.trace(rho), 1.0, atol=1e-10)

# Thermal: SQL limit consistency (sql_sensitivity in thermal_langevin.py)
assert np.isclose(sql_sensitivity(N), 1/np.sqrt(N), rtol=1e-10)

# Cavity: finesse scaling (via cavity_enhanced_sensitivity — returns Δφ directly)
# Δφ = 1/sqrt(F * N) is the analytical expectation
# assert np.isclose(delta_phi * np.sqrt(F * N), 1.0, rtol=1e-10)

# Dynamical decoupling: dc suppression at large nπ (cpmg_filter_function)
from src.physics.dynamical_decoupling import cpmg_filter_function
assert cpmg_filter_function(np.array([0.0]), n_pulses=100, tau=0.5)[0] < \
       cpmg_filter_function(np.array([0.0]), n_pulses=1, tau=0.5)[0] + 1e-15

# Scaling fit quality (from scaling_fit.py -> ScalingFitResult.R_squared)
assert R_squared >= 0.9
```

---

## Likely Failure Conditions

1. **Pseudomode dimension explosion**: Full density matrix with $d = 2(N+1)(K+1)$ for $K$ pseudomodes becomes intractable beyond $N \approx 20$ with $K=1$. Mitigation: pure-state Monte Carlo wavefunction method; restrict to $N \leq 30$.

2. **Analytical model oversimplification**: The cavity, DD, and tilt-to-length models are analytical approximations. They may miss non-trivial $N$-dependent effects (e.g., cavity nonlinearity at high $N$, pulse imperfections in DD). Mitigation: note where full simulation would differ.

3. **Cross-over misidentification**: Thermal and tilt-to-length models have noise floors that produce $\alpha=0$ at large $N$, but the finite $N$ range may not reach the floor. Mitigation: plot $\Delta\phi(N)$ over 4+ decades to verify the asymptote; report $\alpha_{\text{effective}}$ over the finite range separately.

4. **Distributed array entanglement cost**: The entangled $M$-node $\alpha=-1.0$ result assumes perfect entanglement generation and distribution across nodes, which is not scalable. Mitigation: report as fundamental bound; note practical overhead separately.
