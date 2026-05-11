# Scaling Survey: Two-Mode Fock Basis — Markovian MZI Models

## Hypothesis

For MZIs using **two-mode Fock-basis states** under Markovian decoherence:

1. Coherent states always achieve SQL scaling ($\alpha = -0.5$) regardless of noise
2. NOON states achieve Heisenberg scaling ($\alpha = -1.0$) only in the absence of loss; any loss collapses them to SQL ($\alpha \to -0.5$)
3. Squeezed-vacuum injection improves the prefactor $C$ by $e^{-r}$ but does not change $\alpha$; loss degrades this advantage
4. Two-body loss is predicted to cause complete scaling collapse ($\alpha \to 0$) for all states at large $N$, to be verified by the survey

The survey quantifies these transitions as a function of noise strength $\gamma$, reporting $(\text{state}, \text{noise}) \to \alpha$ and identifying critical noise thresholds where $\alpha$ changes.

---

## Literature Review

| Concept & Motivation | Article | Year |
|---|---|---|
| Quantum limits for interferometry: QCRB, SQL, Heisenberg limit; foundational for all scaling calculations | *Sensitivity of Quantum-Enhanced Interferometers* (review) | — |
| Noise resilience in high-bandwidth atom interferometers: Kalman filtering for separating signal from noise in lossy environments | *Noise Resilience in a High-Bandwidth Atom Interferometer* ([arXiv:2504.07236](https://arxiv.org/abs/2504.07236)) | 2025 |
| Noise subtraction in compact heterodyne interferometers: laser frequency noise, thermal noise, optical path noise; ~10× sensitivity improvement via subtraction | *Investigation and Mitigation of Noise Contributions in a Compact Heterodyne Interferometer* | 2022 |
| Squeezed-vacuum injection in MZIs: theory and experimental demonstrations of sub-SQL sensitivity via dark-port squeezing | *Squeezed-light-enhanced MZI* (various) | — |

---

## Theoretical Model

### Hilbert Space

Two-mode bosonic Fock space $\mathcal{H} = \text{span}\{\vert n_1, n_2\rangle\}$ truncated at maximum photon number $N_{\max}$ per mode, giving dimension $(N_{\max}+1)^2$.

### MZI Unitary

The Mach–Zehnder interferometer operation in the Fock basis:

$$U_{\text{MZI}} = U_{\text{BS}}(\pi/4,0)\, U_\phi\, U_{\text{BS}}(\pi/4,0)$$

where the beam-splitter unitary is $U_{\text{BS}}(\theta,\phi) = \exp\big(i\theta(e^{i\phi}a^\dagger b + e^{-i\phi}a b^\dagger)\big)$ and the phase shift is $U_\phi = e^{i\phi n_2}$. The basis conventions match `mzi_simulation.py`.

### Input States

| State | Construction | Code Location |
|---|---|---|
| Coherent | $\vert\alpha\rangle\otimes\vert0\rangle$, $\langle N\rangle = \vert\alpha\vert^2$ | `mzi_states.py` |
| NOON | $(\vert N,0\rangle + \vert0,N\rangle)/\sqrt{2}$ | `mzi_states.py::noon_state` |
| Squeezed vacuum (single-mode) | $S(\xi)\vert0\rangle$ in mode 0, vacuum in mode 1; $S(\xi)=e^{(\xi^* a^2 - \xi a^{\dagger 2})/2}$ | `mzi_states.py::squeezed_vacuum_state` |

### Noise Channels (Lindblad Form)

Density matrix evolution: $\dot\rho = -i[H,\rho] + \sum_k \big(L_k\rho L_k^\dagger - \frac12\{L_k^\dagger L_k,\rho\}\big)$.

| Noise Type | Lindblad Operators $L_k$ | Code |
|---|---|---|
| One-body loss | $\sqrt{\gamma_1}\,a_1$ (mode 1, phase-shifted arm) | `mzi_lindblad.py::build_mzi_lindblad_operators` |
| Phase diffusion | $\sqrt{\gamma_\phi}\,J_z = \sqrt{\gamma_\phi}\,(n_1 - n_2)/2$ | `mzi_lindblad.py::build_mzi_lindblad_operators` |
| Two-body loss | $\sqrt{\gamma_2}\,a_1^2$ (pair loss from mode 1) | `mzi_lindblad.py::build_mzi_lindblad_operators` |
| Detection noise | $P(k\vert n,\eta) = \binom{n}{k}\eta^k(1-\eta)^{n-k}$ (binomial detection) | `noise_channels.py::apply_detection_noise` (Monte Carlo) or `detection_channel_pmf` (exact) |

### Modified MZI Variants in this Framework

| Variant | Modification | Unitary | Code |
|---|---|---|---|
| Kerr-nonlinear MZI | Add $\chi(n_1^2 + n_2^2)$ to phase shift | $e^{i(\phi n_2 + \chi T (n_1^2 + n_2^2))}$ | `kerr_mzi.py::kerr_mzi` |
| Weak-value MZI | Post-select on near-orthogonal final state | Standard MZI + post-selection $\vert f\rangle\langle f\vert$ | `weak_value_mzi.py::weak_value_mzi` |

For the standard linear phase $\phi$ coupled via $n_2$, the Kerr nonlinearity does not change the scaling exponent ($\alpha = -1.0$ for NOON probes) because the generator $n_2$ commutes with the diagonal Kerr term $\chi(n_1^2 + n_2^2)$ — the QFI is invariant. Enhanced scaling (super-Heisenberg) would require a nonlinear generator (e.g., $\phi \cdot n_2^2$), which is a separate model not covered here. Weak-value amplification amplifies the signal but preserves Fisher information ($\alpha = -0.5$); the prefactor degrades as $C \propto 1/\sqrt{p_{\text{ps}}}$.

### Measurement Strategies

| Measurement | Operator | Method | Implementation Status |
|---|---|---|---|
| Quantum Fisher (SLD) | $F_Q = \text{Tr}[\rho L^2]$ (symmetric logarithmic derivative) | SLD eigen-decomposition; pure-state shortcut $4\,\text{Var}(J_z)$ | `fisher_information.py` ✅ |
| Number-difference | $J_z = (n_1 - n_2)/2$ | Error propagation $\Delta\phi = \sigma_{J_z}/|\partial\langle J_z\rangle/\partial\phi|$ | `sensitivity_metrics.py::error_propagation_sensitivity` ✅ |
| Classical Fisher (CFI) | $F_C = \sum (\partial P/\partial\phi)^2 / P$ | Central-difference derivative of $P(m|\phi)$ | `fisher_information.py::classical_fisher_information` ✅ |
| Parity | $\Pi = e^{i\pi n_2}$ | Not implemented as standalone measurement in the survey pipeline; parity-specific studies are deferred | ⏳ Planned |

### Sensitivity Metrics

$$\Delta\phi_{\text{EP}} = \frac{\Delta J_z}{\vert\partial\langle J_z\rangle/\partial\phi\vert},\quad
\Delta\phi_{\text{QFI}} = \frac{1}{\sqrt{F_Q}}\quad(\text{default}),\quad
\Delta\phi_{\text{CFI}} = \frac{1}{\sqrt{F_C}}$$

### Controlled Variables

- Resource normalization: same $\langle N\rangle$ across compared states
- Phase bias: $\phi = \pi/4$ (maximum slope for most states)
- Noise strength: dimensionless $\gamma\cdot T$; default sweep $\{0, 10^{-3}, 10^{-2}, 10^{-1}\}$ (see Parameter Sweep)
- Sensitivity method: QFI-based (default), error propagation for comparison

---

## Models Survey

| Model | Input State | Noise | Expected $\alpha$ | Implementation Status |
|---|---|---|---|---|
| Ideal MZI coherent | $\vert\alpha\rangle\otimes\vert0\rangle$ | None | $-0.5$ ($\Delta\phi = 1/\sqrt{N}$) | `mzi_simulation.py` ✅ |
| Ideal MZI NOON | $(\vert N,0\rangle + \vert0,N\rangle)/\sqrt{2}$ | None | $-1.0$ ($\Delta\phi = 1/N$) | `mzi_states.py` ✅ |
| MZI + one-body loss | Any | $L=\sqrt{\gamma_1}\,a_1$ (mode 1) | Coherent: $-0.5$; NOON: $-0.5$ (collapse) | `mzi_lindblad.py::build_mzi_lindblad_operators` ✅ |
| MZI + phase diffusion | Any | $L=\sqrt{\gamma_\phi}\,J_z$ | Coherent: $-0.5$; SSS: degrades toward $0$ | `mzi_lindblad.py::build_mzi_lindblad_operators` ✅ |
| MZI + two-body loss | Any | $L=\sqrt{\gamma_2}\,a_1^2$ (mode 1) | All states: $\alpha \to 0$ at large $N$ (prediction, to be verified by survey) | `mzi_lindblad.py::build_mzi_lindblad_operators` ✅ |
| MZI + detection noise | Any | $\eta$ binomial channel | $\alpha \to -0.5$, $C \to \eta^{-1/2}$ | ⚠️ Approximate (bound $F_Q \to \eta F_Q$ only; full binomial convolution in `noise_channels.py` not yet wired into survey) |
| Squeezed-vacuum injection | $S(\xi)\vert0\rangle\otimes\vert0\rangle$ (mode 0 squeezed, mode 1 vacuum) | None | $-0.5$, $C = e^{-r}$ (prefactor only) | `mzi_states.py::squeezed_vacuum_state` ✅ |
| Squeezed-vacuum + lossy MZI | Squeezed vacuum + one-body loss | $\gamma_1$ | $-0.5$, $C = \sqrt{1-\eta+\eta e^{-2r}}$ | `create_survey_model("squeezed_vacuum_loss")` ✅ |
| Kerr-nonlinear MZI | Coherent or vacuum | Kerr $\chi(n_1^2+n_2^2)$ | $-1.0$ (Heisenberg, linear generator $n_2$); $F_Q$ invariant under Kerr since $[n_2,\chi(n_1^2+n_2^2)]=0$ | `kerr_mzi.py` ✅ |
| Weak-value MZI | Coherent | Post-selection | $-0.5$, $C \propto 1/\sqrt{p_{\text{ps}}}$; same FI as conventional | `weak_value_mzi.py` ✅ |

---

## Numerical Simulation

### Implementation Strategy

Composable pipeline, each stage a pure function:

```
input_state(N, type) → add_noise(state, config, T) → phase_imprint(state, φ)
  → readout(state) → sensitivity(method) → extract_exponent(N, Δφ)
```

### Parameter Sweep

- Resource sweep: $N \in \{2, 4, 8, 16, 32, 64\}$ (Fock basis limits $N \leq 64$)
- Noise sweep: $\gamma \in \{0, 10^{-3}, 10^{-2}, 10^{-1}\}$ (dimensionless; `SurveyConfig` default)
- Measurement: QFI (default), error propagation for validation
- Scaling exponent via log-log linear regression: $\log\Delta\phi = \alpha\log N + \log C$
- Minimum 4 $N$ points, $R^2 > 0.9$ for valid fit

### Validation

```python
assert np.isclose(np.trace(rho), 1.0, atol=1e-8)       # trace preservation (mzi_lindblad.py default)
assert np.allclose(rho, rho.conj().T, atol=1e-8)       # hermiticity (mzi_lindblad.py default)
assert np.min(np.linalg.eigvalsh(rho)) >= -1e-8         # positivity
assert R_squared >= 0.9                                  # fit quality (scaling_fit.py default)
```

---

## Likely Failure Conditions

1. **Hilbert space explosion**: $(N_{\max}+1)^2$ dimension is prohibitive for $N > 64$. Per-state-type truncation (`_max_photons_for_state` in `scaling_survey.py`): definite Fock-like states (NOON, twin-Fock) use $N_{\max}=N$; coherent-like (coherent, squeezed vacuum, CSS) use $N_{\max} = \max(2N, N+20)$ to capture Poisson tails. For $N=64$, the latter gives $N_{\max}=148$ → dimension $22,\!201$. Mitigation: Dicke basis for symmetric states, covariance-matrix methods for Gaussian states.

2. **Degenerate exponents**: Different (state, noise) combos can yield identical $\alpha$ (e.g., coherent SQL vs. NOON collapsed to SQL). Mitigation: report prefactor $C$, $R^2$, and model evidence to disambiguate.

3. **Finite-size artifacts**: $N < 4$ produces quantization artifacts in $\Delta\phi$. Mitigation: exclude $N < 4$ from log-log regression; require $N \geq 8$ for final fit.

4. **Phase-bias dependence**: $\Delta\phi(\phi)$ is not flat; $\phi=\pi/4$ works for most states but NOON has maximal sensitivity at different $\phi$. Mitigation: report sensitivity at each state's optimal operating point.
