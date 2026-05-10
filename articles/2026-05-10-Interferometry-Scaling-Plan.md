# Interferometry Sensitivity Scaling: A Unified Numerical Survey

## Hypothesis

The **scaling exponent** $\alpha$ in $\Delta\phi \propto N^{\alpha}$ (where $N$ is a resource count such as photon number or atom number) is not universal — it depends sensitively on:

1. **Input state type** (classical, squeezed, entangled, non-Gaussian)
2. **Noise model** (loss, dephasing, thermal, non-Markovian)
3. **Measurement strategy** (parity, homodyne, Bayesian)
4. **Interferometer topology** (single-pass, cavity-enhanced, distributed)

This article proposes a **unified numerical survey** that sweeps across a grid of physical models — from the ideal MZI to noisy open-system metrology — and documents the observed scaling exponent $\alpha$ for each combination. The goal is to produce a reference phase diagram of the form $(\text{state}, \text{noise}) \to \alpha$, identifying:

- Which combinations achieve Heisenberg scaling ($\alpha = -1$)
- Which are limited to SQL ($\alpha = -0.5$)
- Where intermediate or fractional exponents arise (e.g., $\alpha = -2/3$ for spin-squeezed states)
- Where decoherence causes a **scaling collapse** ($\alpha \to 0$ for large $N$)

---

## Literature Review

| Concept & Motivation | Article | Year |
|---|---|---|
| Noise resilience in high-bandwidth atom interferometers: Kalman filtering + $k$-reversal modulation for separating signal from noise; informs practical sensitivity pipelines with time-varying noise | *Noise Resilience in a High-Bandwidth Atom Interferometer* ([arXiv:2504.07236](https://arxiv.org/abs/2504.07236)) | 2025 |
| Gravitational phase shifts in fiber interferometers: Phase noise accumulation over long baselines; path-imbalance sensitivity scaling | *High-Sensitivity Fiber Interferometer for Gravitational Phase Shift Measurement* ([arXiv:2506.09770](https://arxiv.org/abs/2506.09770)) | 2025 |
| Variance-based dark-matter detection with fluctuations: Super-binomial variance metrics as sensitivity signal; relevant for scaling under stochastic noise | *Fluctuations in Atom Interferometers as a New Tool for Dark Matter* ([arXiv:2602.23427](https://arxiv.org/abs/2602.23427)) | 2026 |
| Multiparameter noise budgeting for space interferometers: High-dimensional sensitivity analysis, parameter-space reduction; system-level noise-budgeting methodology | *Multiparameter Hierarchical Sensitivity Analysis of Tilt-to-Length Coupling Noise in Taiji Interferometer* (journal link) | 2025/2026 |
| Noise subtraction in compact heterodyne interferometers: Laser frequency noise, thermal noise, optical path noise; ~10× sensitivity improvement via subtraction | *Investigation and Mitigation of Noise Contributions in a Compact Heterodyne Interferometer* | 2022 |
| Quantum limits for interferometry (review): QCRB, SQL, Heisenberg limit as foundations for all scaling calculations | *Sensitivity of Quantum-Enhanced Interferometers* (review) | — |
| Pseudomode method for non-Markovian open quantum systems: Maps Lorentzian-structured reservoir to a single damped pseudomode + Markovian bath | Garraway, *Phys. Rev. A* 55, 2290 | 1997 |

### Prior Work in This Repo

| Concept & Motivation | Article | Year |
|---|---|---|
| High-order non-Gaussian squeezing under Markovian decoherence: Existing hybrid Lindblad framework; QFI decoherence sweeps at fixed $\langle n \rangle$ for n=2,3,4 | `articles/2026-05-07-High-Order-Squeezing-Plan.md` | 2026 |
| Ancilla-assisted metrology in non-Markovian environments: Pseudomode-based non-Markovian QFI; demonstrates $\mathcal{R}(T)$ preservation ratio scaling | `articles/2026-05-09-Ancilla-Assisted-Metrology-Non-Markovian.md` | 2026 |

---

## Theoretical Model

### General Framework

The survey adopts a **unified metrology pipeline**. Each stage can be independently swapped, producing a factorial design of experiments:

| Stage | Options | Key Parameter |
|---|---|---|
| **Input state** | Coherent, Fock, NOON, CSS, SSS, Twin-Fock, Squeezed vacuum, Non-Gaussian (n=3,4) | $\langle N \rangle$ |
| **Entanglement** | None, OAT ($\chi J_z^2$), TNT ($\chi J_z^3$), Two-axis countertwisting | Squeezing parameter $r$, time $t$ |
| **Phase imprint** | Linear ($e^{i\phi J_z}$), Displacement ($e^{i\phi (a+a^\dagger)}$) | $\phi$ |
| **Decoherence** | Markovian loss ($\gamma_1$), Dephasing ($\gamma_\phi$), Two-body loss ($\gamma_2$), Non-Markovian (Lorentzian, $\lambda$), Thermal noise | Rates $\gamma$, bath correlation $\lambda$ |
| **Readout/Estimator** | Error propagation, Classical Fisher, Quantum Fisher (SLD), Bayesian | Measurement basis |

### Sensitivity Metrics

The primary output is the **scaling exponent** $\alpha$, extracted via log-log linear regression:

$$
\log \Delta\phi = \alpha \log N + \log C
$$

where $\Delta\phi$ is computed using one of four methods (see `src/analysis/sensitivity_metrics.py`, `src/analysis/fisher_information.py`):

| Method | Formula |
|---|---|
| Error propagation | $\Delta\phi_{\text{EP}} = \sigma_{J_z} / \lvert \partial\langle J_z\rangle/\partial\phi \rvert$ |
| Classical Fisher | $\Delta\phi_C = 1/\sqrt{F_C}$ |
| Quantum Fisher (pure) | $\Delta\phi_Q = 1/\sqrt{F_Q},\ F_Q = 4\,\text{Var}(G)$ |
| Quantum Fisher (mixed) | $\Delta\phi_Q = 1/\sqrt{F_Q}$ (SLD eigen-decomposition) |
| Bayesian | $\Delta\phi_B = \text{Std}[\phi \vert m_0]$ (circular) |

> **Note on Bayesian estimation**: The Bayesian estimator's finite-sample uncertainty creates a floor that masks true scaling at large $N$. Mitigation: scale $n_{\text{MC}}$ with $N$ and verify the Bayesian sensitivity converges to the Fisher bound.

### Controlled Variables

To ensure fair comparison across models, for each model we fix:

- **Resource normalization**: Same $\langle N \rangle$ (mean photon/atom number) at the interferometer input
- **Phase bias**: $\phi = \pi/4$ (maximum slope for most states)
- **Noise strength**: Expressed in dimensionless rates $\gamma \cdot T$ (unitless)
- **Measurement strategy**: Parity detection on output modes (unless otherwise noted)

---

## Models Survey

| Model | Implementation Considerations | Expected Scaling $\Delta\phi(N) = C \cdot N^{\alpha}$ |
|---|---|---|
| **Ideal MZI with coherent state** | Two-mode Fock basis, $U_{\text{BS}}(\pi/4,0)$, $U_\phi = e^{i\phi n_1}$. Already fully implemented in `mzi_simulation.py`. | $\alpha = -0.5$, $C = 1$ (SQL, $\Delta\phi = 1/\sqrt{N}$) |
| **Ideal MZI with NOON state** | NOON state $(\vert N,0\rangle + \vert 0,N\rangle)/\sqrt{2}$ via `mzi_states.py::noon_state`. Fragile to loss. | $\alpha = -1.0$, $C = 1$ (HL, $\Delta\phi = 1/N$) |
| **MZI + one-body loss ($\gamma_1$)** | Lindblad $L = \sqrt{\gamma_1} a$. Use `lindblad_solver.py` or `noise_channels.py::apply_noise_channel`. Scaling collapse expected for NOON states. | $\alpha \to -0.5$, $C \to (1 + \gamma_1 T)^{1/2}$ under strong loss (NOON degrades to SQL); $\alpha = -0.5$, $C = e^{\gamma_1 T/2}$ for coherent (robust) |
| **MZI + phase diffusion ($\gamma_\phi$)** | Lindblad $L = \sqrt{\gamma_\phi} J_z$. Implemented in `noise_channels.py`. Dephasing rate $\gamma_\phi$ sweeps. | $\alpha$ degrades from $-0.5$ toward $0$ as $\gamma_\phi$ increases; $C$ grows as $\exp(\gamma_\phi T)$ for squeezed states; entangled states ($\alpha=-1$) lose advantage faster, $\alpha \to -0.5$ then $0$ |
| **MZI + two-body loss ($\gamma_2$)** | Lindblad $L = \sqrt{\gamma_2} a^2$. Rate $\propto N(N-1)$. Implemented in `noise_channels.py`. | $\alpha \to 0$, $C \to (2\gamma_2 T)^{-1/2}$ at large $N$ for all states (loss scales as $N^2$, sensitivity saturates) |
| **MZI + detection noise ($\eta$)** | Binomial $P(k\vert n,\eta)$ detection channel. Implemented in `noise_channels.py::detection_channel_pmf`. | $\alpha \to -0.5$, $C \to \eta^{-1/2}$; Heisenberg limit unreachable for $\eta < 1$; $C$ diverges as $\eta \to 0$ |
| **Spin-squeezed state (OAT, $\chi J_z^2$)** | OAT Hamiltonian $H = \chi J_z^2$ in Dicke basis. CSS $\to$ SSS via `spin_squeezing.py::generate_squeezed_state`. Use `optimal_squeezing_time()` for each $N$ — OAT squeezing peaks at $t_{\text{opt}} \propto N^{-2/3}$, beyond which the squeezed quadrature re-widens. | $\alpha = -2/3$, $C \approx (2^{1/3}/3^{1/6})\,\chi^{-1/3}$ (ideal OAT at optimal $t \propto N^{-2/3}$); degrades under $\gamma_\phi$, $C \to \infty$ at $\gamma_\phi \gg \chi$ |
| **Twin-Fock state** | $\vert N/2,0\rangle$ in Dicke basis. Already implemented in `pages/BEC_Sensitivity_Scaling.py`. | $\alpha = -1.0$, $C \approx 1$ (ideal, $\Delta\phi \approx 1/N$ via number-difference readout); $\alpha \to -0.5$, $C \to \sqrt{2}$ with strong loss |
| **Squeezed-vacuum injection** | Single-mode squeezed vacuum $S(\xi)\vert0\rangle$ injected into dark port of MZI. Needs `squeezed_vacuum_state()` extension to `mzi_states.py`. | $\alpha = -0.5$, $C = e^{-r}$ (improvement factor $e^{-r}$ over SQL, independent of $N$; $r$ is squeezing parameter) |
| **Cavity-enhanced MZI** | Increased effective interaction time $T_{\text{eff}} = \mathcal{F} \cdot T_{\text{single}}$ where $\mathcal{F}$ is finesse. Models as repeated phase interrogation via `cavity_enhanced_mzi()`. | $\alpha = -0.5$, $C = 1/\sqrt{\mathcal{F}}$ (SQL with finesse-controlled prefactor; $\Delta\phi = 1/\sqrt{\mathcal{F} N}$) |
| **Non-Gaussian states (n=3,4 squeezing)** | Hybrid oscillator-spin system from `hybrid_system.py`. Spin-dependent forces at 3rd/4th order. Existing code in `pseudomode_system.py` and `articles/2026-05-07-*-Plan.md`. | n=3: $\alpha \approx -0.75$, $C \approx 1.5$; n=4: $\alpha \approx -0.85$, $C \approx 2.0$ at fixed $\langle n\rangle$ (fractional between SQL and HL) |
| **Ancilla-assisted MZI** | Spin ancilla coupled via $H_{\text{sa}} = g \cdot n \otimes \sigma_x$. State $\vert\alpha\rangle \otimes \vert\downarrow\rangle$. Existing `hybrid_mzi.py`. | $\alpha$ same as underlying state; $C$ reduced by factor $\mathcal{R}(T)^{-1/2}$ where $\mathcal{R}(T) = F_Q(T)/F_Q(0)$ is QFI preservation ratio; benefit largest under non-Markovian noise |
| **Non-Markovian bath (Lorentzian)** | Pseudomode method: $H_{\text{sp}} = g_{\text{sp}}(a + a^\dagger)(b + b^\dagger)$, $L_{\text{pm}} = \sqrt{\lambda}\, b$. Existing `pseudomode_system.py`. Deeply non-Markovian baths ($\lambda \ll 1$) produce coherent QFI oscillations — report full $F_Q(t)$ trajectories and extract envelope rather than endpoint. | $\alpha$ ranges from $-0.5$ (Markovian $\lambda \to \infty$, $C = (g_{\text{sp}}^2/\lambda)^{-1/2}$) to $-0.3$ to $-0.5$ (non-Markovian $\lambda \to 0$, effective $\gamma_{\text{eff}} = g_{\text{sp}}^2/\lambda \to 0$); $C$ depends on $\lambda/g_{\text{sp}}^2$ |
| **MZI + thermal noise (Langevin)** | Thermal Langevin force $F_{\text{th}}$ with PSD $S_F(\omega) = 2m\Gamma k_B T$. Not yet implemented — requires stochastic differential equation solver (`thermal_langevin_solver()`). | $\alpha = 0$, $C = \sqrt{2 k_B T \Gamma / (m \omega_m^2)}$ at low frequencies (thermal floor dominates); $\alpha = -0.5$, $C = \sqrt{\hbar/(2m\omega_m)}$ at high frequencies (SQL) |
| **Distributed array interferometer** | $M$ sensors with correlated noise $H_{\text{corr}}$. Entanglement-enhanced network. Extends `mzi_simulation.py` to multiple interferometers (`distributed_interferometer()`). | $\alpha = -1.0$, $C = 1$ with entanglement across $M$ nodes (collective HL); $\alpha = -0.5$, $C = 1$ with uncorrelated classical averaging over $M$ sensors (SQL per $\sqrt{M}$ improvement) |
| **Kerr-nonlinear MZI** | $H = \chi (a^\dagger a)^2$ in each arm. Phase magnification via nonlinearity. Not yet implemented (`kerr_mzi_simulation()`). The Kerr nonlinearity also introduces intensity-dependent phase diffusion — compare full Lindblad simulation to approximate analytic models and verify $\chi$ is small enough that diffusion does not dominate. | $\alpha \approx -0.75$ to $-1.0$ depending on $\chi T$, $C \approx (1/\chi T)^{1/2}$ ideal; phase diffusion from Kerr term adds $C \propto \exp(\chi^2 T)$ at large $\chi$ |
| **Squeezed-light + lossy MZI** | Squeezed vacuum injection + optical loss. Combine `squeezed_vacuum_state()` with `noise_channels.py`. | $\alpha = -0.5$, $C = \sqrt{1 - \eta + \eta e^{-2r}}$ (effective squeezing degrades with loss $\eta$; $C \to 1$ as $\eta \to 0$) |
| **Dynamical decoupling sequence** | Periodic $\pi$-pulses (CPMG, XY-8) to filter noise. Requires time-dependent Hamiltonian simulation (`dynamical_decoupling_pulse()`). Not yet implemented. | $\alpha \approx -0.5$, $C \propto (T_2^{\text{(DD)}}/T)^{-1/2}$ where $T_2^{\text{(DD)}} \approx T_2^{(0)} \cdot n_{\pi}^{2/3}$ for CPMG; filter-function engineering improves $C$ but does not change $\alpha$ |
| **Tilt-to-length coupling noise** | $\delta L \propto \theta x$ where $\theta$ is angular jitter. Systematic noise rather than fundamental — affects $\alpha$ via noise-floor scaling. Not yet implemented. | $\alpha \to 0$ when tilt noise dominates; $C = \theta_{\text{rms}} \cdot L / \lambda$ (constant noise floor, no entanglement advantage; $\alpha$ and $C$ both depend on jitter PSD) |
| **Weak-value amplification MZI** | Post-selected amplification $A_w = \langle f\vert A \vert i\rangle / \langle f \vert i \rangle$. Fisher information unchanged despite signal amplification. Not yet implemented. | $\alpha = -0.5$, $C = 1$ (no advantage over conventional; same FI despite larger signal; $C$ degrades with post-selection probability $p_{\text{ps}}$ as $C \propto 1/\sqrt{p_{\text{ps}}}$) |

---

## Numerical Simulation

### Implementation Strategy

The survey follows a **composable pipeline** architecture:

```
input_state(N, state_type)
    → entangle(state, ent_type, params)
    → phase_imprint(state, phi)
    → decohere(state, noise_config, T)
    → readout(state, measurement_basis)
    → sensitivity_estimator(state, method)
    → extract_exponent(N_array, delta_phi_array)
```

Each stage is a pure function acting on a state vector or density matrix, matching the existing patterns in `src/physics/` and `src/analysis/`.

### Parameter Sweep Design

For each model, the survey performs:

1. **Resource sweep**: $N \in \{2, 4, 8, 16, 32, 64, 128, 256\}$ (truncated to computational limits)
2. **Noise sweep**: $\gamma \in \{0, 10^{-3}, 10^{-2}, 10^{-1}, 1\}$ (dimensionless)
3. **Measurement**: Parity, $J_z$, number-difference for each $(N, \gamma)$ pair
4. **Output**: $\alpha$ per $(\text{model}, \text{method}, \text{noise level})$ triple

### Numerical Validation

Every simulation step includes:

```python
# Trace preservation (Lindblad/RK4)
assert np.isclose(np.trace(rho), 1.0, atol=1e-10)

# Hermiticity
assert np.allclose(rho, rho.conj().T, atol=1e-10)

# Positivity (for density matrices)
assert np.min(np.linalg.eigvalsh(rho)) >= -1e-8

# QFI bounds
assert 0 <= F_Q <= F_Q_max, "QFI must respect physical bounds"

# Scaling fit quality (minimum number of points, R² threshold)
assert len(N_array) >= 4, "Need at least 4 N values for reliable exponent fit"
assert R_squared >= 0.9, "Log-log fit must be linear"
```

### Data Products

| Output | Format | Description |
|---|---|---|
| `scaling_map.csv` | CSV | $(\text{model}, \text{noise}, \alpha, \alpha_{\text{err}}, R^2)$ for all sweeps |
| `scaling_map.json` | JSON | Same data in structured format for downstream analysis |
| `sensitivity_curves/` | Figures | Log-log $\Delta\phi$ vs $N$ curves for each model |
| `phase_diagram/` | Figures | Heatmap of $\alpha$ across (state, noise) grid |

---

## Likely Failure Conditions

1. **Hilbert space explosion for large $N$**: Two-mode Fock basis with $(N_{\max}+1)^2$ dimension becomes prohibitive for $N > 50$. **Mitigation**: Use Dicke basis (dimension $N+1$) for symmetric states; use TWA (`truncated_wigner.py`) for $N > 1000$; use covariance-matrix methods for Gaussian states.

2. **Lindblad solver performance**: Full density-matrix evolution for mixed states requires $d^2 \times d^2$ Liouvillian. For $d = 2(N+1)(K+1)$ with pseudomode, this becomes intractable beyond $N \approx 20$. **Mitigation**: Use pure-state Monte Carlo wavefunction method (`quantum_trajectories`) instead of master equation; restrict pseudomode studies to $N \leq 30$.

3. **Scaling exponent degeneracy**: Different combinations of (state, noise) can yield the same $\alpha$, making attribution ambiguous (e.g., thermal-noise-limited SQL $\alpha=-0.5$ vs. coherent-state SQL $\alpha=-0.5$). **Mitigation**: Report additional metrics (prefactor $C$, R², Bayesian model evidence) to disambiguate.

4. **Finite-size effects**: For small $N$ ($N < 8$), the discrete nature of Fock space causes quantization artifacts in $\Delta\phi$ that distort the scaling fit. **Mitigation**: Exclude $N < 4$ from log-log regression; verify power-law behavior holds for $N \geq 8$.

5. **Phase bias dependence**: $\Delta\phi(\phi)$ is not flat; choosing $\phi = \pi/4$ works for most states but some (e.g., NOON) have maximal sensitivity at different operating points. **Mitigation**: Sweep $\phi$ and report sensitivity at each state's optimal operating point.
