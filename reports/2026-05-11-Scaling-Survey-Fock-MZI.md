# Scaling Survey: Two-Mode Fock Basis — Markovian MZI Models

## 🧪 Hypothesis

For MZIs using **two-mode Fock-basis states** under Markovian decoherence:

1. Coherent states always achieve SQL scaling ($\alpha = -0.5$) regardless of noise
2. NOON states achieve Heisenberg scaling ($\alpha = -1.0$) only in the absence of loss; any loss collapses them to SQL ($\alpha \to -0.5$)
3. Squeezed-vacuum injection ($S(\xi)\vert0\rangle$ in mode 0, vacuum in mode 1) achieves Heisenberg-like scaling ($\alpha \to -1.0$) at large $N$ because $F_Q = 2\langle N\rangle(\langle N\rangle + 1)$, exceeding the NOON-state Heisenberg limit by a factor of 2 in the prefactor; loss degrades this advantage
4. Two-body loss is predicted to cause complete scaling collapse ($\alpha \to 0$) for all states at large $N$, to be verified by the survey

The survey quantifies these transitions as a function of noise strength $\gamma$, reporting $(\text{state}, \text{noise}) \to \alpha$ and identifying critical noise thresholds where $\alpha$ changes.

## ⚛️ Theoretical Model

The simulation operates in a **two-mode bosonic Fock space** $\mathcal{H} = \text{span}\{\vert n_1, n_2\rangle\}$ truncated at maximum photon number $N_{\max}$ per mode, giving dimension $(N_{\max}+1)^2$. The **Mach–Zehnder interferometer** is implemented by the unitary $U_{\text{MZI}} = U_{\text{BS}}(\pi/4,0)\, U_\phi\, U_{\text{BS}}(\pi/4,0)$ where the **beam-splitter unitary** is $U_{\text{BS}}(\theta,\phi) = \exp\big(i\theta(e^{i\phi}a^\dagger b + e^{-i\phi}a b^\dagger)\big)$ and the **phase shift** is $U_\phi = e^{i\phi n_2}$.

The following **input states** are considered in the survey:

| State | Construction |
|---|---|
| Coherent | $\vert\alpha\rangle\otimes\vert0\rangle$, $\langle N\rangle = \vert\alpha\vert^2$ |
| NOON | $(\vert N,0\rangle + \vert0,N\rangle)/\sqrt{2}$ |
| Twin-Fock | Uniform superposition $\sum_{n=0}^N \vert n, N-n\rangle/\sqrt{N+1}$ (SQL-limited under $J_z$; **not** the standard $\vert N/2, N/2\rangle$ twin-Fock) |
| Squeezed vacuum (single-mode) | $S(\xi)\vert0\rangle$ in mode 0, vacuum in mode 1; $S(\xi)=e^{(\xi^* a^2 - \xi a^{\dagger 2})/2}$ |
| Single-photon split ("SSS") | $(\vert N-1,1\rangle + \vert 1,N-1\rangle)/\sqrt{2}$; $F_Q = (N-2)^2$ |

Noise is incorporated through the **Lindblad master equation** $\dot\rho = -i[H,\rho] + \sum_k \big(L_k\rho L_k^\dagger - \frac12\{L_k^\dagger L_k,\rho\}\big)$, using the following noise channels:

| Noise Type | Lindblad Operator $L_k$ |
|---|---|
| One-body loss | $\sqrt{\gamma_1}\,a_1$ (mode 1, phase-shifted arm) |
| Phase diffusion | $\sqrt{\gamma_\phi}\,J_z = \sqrt{\gamma_\phi}\,(n_1 - n_2)/2$ |
| Two-body loss | $\sqrt{\gamma_2}\,a_1^2$ (pair loss from mode 1) |
| Detection noise | $P(k\vert n,\eta) = \binom{n}{k}\eta^k(1-\eta)^{n-k}$ (binomial detection) |

Two **modified MZI variants** extend the basic model: a **Kerr-nonlinear MZI** adds $\chi(n_1^2 + n_2^2)$ to the phase shift ($e^{i(\phi n_2 + \chi T (n_1^2 + n_2^2))}$), and a **weak-value MZI** applies post-selection on a near-orthogonal final state. For the standard linear phase generator $n_2$, the Kerr nonlinearity commutes with the generator, leaving the QFI invariant; enhanced scaling (super-Heisenberg) would require a nonlinear generator (e.g., $\phi \cdot n_2^2$), a separate model not covered here. Weak-value amplification preserves Fisher information ($\alpha = -0.5$) with prefactor $C = 1/\cos\delta = 1/\sqrt{1-p_{\text{ps}}}$ where $\delta = \pi/2 - \theta_{\text{ps}}$ is the deviation from orthogonality and $p_{\text{ps}} = \sin^2\delta$ is the post-selection probability.

The survey uses **$J_z = (n_1 - n_2)/2$ as the phase generator** for all QFI computations, consistent with the number-difference measurement used in error-propagation sensitivity. For definite-$N$ states (NOON, Fock, single-photon-split), $\text{Var}(J_z) = \text{Var}(n_2)$, so the $J_z$ and $n_2$ conventions coincide. For indefinite-$N$ states (coherent, squeezed vacuum), $\text{Var}(J_z) \neq \text{Var}(n_2)$, introducing a constant-factor offset in absolute $F_Q$ values, but **scaling exponents $\alpha$ are unaffected** by this choice. Reference QFI values under the $J_z$ convention:

| State | $F_Q$ | $\Delta\phi$ |
|---|---|---|
| NOON | $N^2$ | $1/N$ |
| Twin-Fock (uniform superposition) | $N(N+2)/3 \approx N^2/3$ | $\sqrt{3/N(N+2)} \approx \sqrt{3}/N$ |
| Coherent ($\vert\alpha\rangle\otimes\vert0\rangle$) | $\lvert\alpha\rvert^2 = N$ | $1/\sqrt{N}$ (SQL) |
| Squeezed vacuum ($\vert\xi\rangle\otimes\vert0\rangle$) | $2\langle N\rangle(\langle N\rangle + 1)$ | $1/\sqrt{2\langle N\rangle(\langle N\rangle + 1)}$ |

Three **measurement strategies** provide sensitivity estimates: **quantum Fisher information** ($F_Q = \text{Tr}[\rho L^2]$ via SLD eigen-decomposition with $J_z$ generator; pure-state shortcut $4\,\text{Var}(J_z)$), **number-difference** ($J_z$ error propagation $\Delta\phi = \sigma_{J_z}/\vert\partial\langle J_z\rangle/\partial\phi\vert$), and **classical Fisher information** ($F_C = \sum (\partial P/\partial\phi)^2 / P$ via central-difference derivative). **Parity measurement** ($\Pi = e^{i\pi n_2}$) is noted but deferred to dedicated studies. The default sensitivity metric is QFI-based ($\Delta\phi_{\text{QFI}} = 1/\sqrt{F_Q}$), with error propagation used for validation.

Controlled variables across the survey include: **resource normalization** (same $\langle N\rangle$ across compared states), **phase bias** ($\phi = \pi/4$, maximum slope for most states), **noise strength** (dimensionless $\gamma\cdot T$; default sweep $\{0, 10^{-3}, 10^{-2}, 10^{-1}\}$), and **sensitivity method** (QFI-based by default).

## 📊 Models Survey

| Model | Input State | Noise | Expected $\alpha$ | Implementation Status |
|---|---|---|---|---|
| Ideal MZI coherent | $\vert\alpha\rangle\otimes\vert0\rangle$ | None | $-0.5$ ($\Delta\phi = 1/\sqrt{N}$) | ✅ |
| Ideal MZI NOON | $(\vert N,0\rangle + \vert0,N\rangle)/\sqrt{2}$ | None | $-1.0$ ($\Delta\phi = 1/N$) | ✅ |
| Ideal MZI twin-Fock | Uniform $\sum\vert n,N-n\rangle/\sqrt{N+1}$ | None | $-0.5$ ($\Delta\phi \approx \sqrt{3}/N$, SQL scaling under $J_z$) | ✅ |
| MZI + one-body loss | Any | $L=\sqrt{\gamma_1}\,a_1$ (mode 1) | Coherent: $-0.5$; NOON: $-0.5$ (collapse) | ✅ |
| MZI + phase diffusion | Any | $L=\sqrt{\gamma_\phi}\,J_z$ | Coherent: $-0.5$; NOON: $-0.5$ (collapse); SSS: degrades toward $0$ | ✅ |
| MZI + two-body loss | Any | $L=\sqrt{\gamma_2}\,a_1^2$ (mode 1) | All states: $\alpha \to 0$ at large $N$ (prediction; analysis pending) | ✅ |
| MZI + detection noise | Any | $\eta$ binomial channel | $\alpha \to -0.5$, $C \to \eta^{-1/2}$ | 🔄 (full binomial convolution not yet wired into survey) |
| Squeezed-vacuum injection | $S(\xi)\vert0\rangle\otimes\vert0\rangle$ (mode 0 squeezed, mode 1 vacuum) | None | $-1.0$ at large $N$, $F_Q = 2\langle N\rangle(\langle N\rangle+1)$ | ✅ |
| Squeezed-vacuum + lossy MZI | Squeezed vacuum + one-body loss | $\gamma_1$ | $\alpha$ degrades from $-1.0$ toward $-0.5$ with increasing loss | ✅ |
| Kerr-nonlinear MZI | Coherent or vacuum | Kerr $\chi(n_1^2+n_2^2)$ | $-1.0$ (Heisenberg, linear generator $n_2$); $F_Q$ invariant | ✅ |
| Weak-value MZI | Coherent | Post-selection | $-0.5$, $C = 1/\cos\delta = 1/\sqrt{1-p_{\text{ps}}}$ | ✅ |

## 💻 Numerical Simulation

### Implementation Strategy

1. **Pipeline construction** — Chain pure functions in a composable pipeline: `input_state(N, type) → add_noise(state, config, T) → phase_imprint(state, φ) → readout(state) → sensitivity(method) → extract_exponent(N, Δφ)`.
2. **Dimension management** — Truncate per state type: definite Fock-like states (NOON, twin-Fock) at $N_{\max}=N$; coherent-like states (coherent, squeezed vacuum) at $N_{\max} = \max(2N, N+20)$ to capture Poisson tails.
3. **State preparation** — Factory functions produce each input state; the SSS state is constructed as $(\vert N-1,1\rangle + \vert 1,N-1\rangle)/\sqrt{2}$ with $F_Q = (N-2)^2$, enabling meaningful $N$-scaling analysis.

### Parameter Sweep

| Parameter | Values / Range | Purpose |
|-----------|----------------|---------|
| Resource $N$ | $\{2, 3, 5, 8, 13, 22, 37, 64\}$ (log-spaced, 8 points) | Scaling exponent extraction |
| Noise strength $\gamma$ | $\{0, 10^{-3}, 10^{-2}, 10^{-1}\}$ (dimensionless $\gamma\cdot T$) | Study noise-induced collapse |
| Measurement method | QFI (default), error propagation | Sensitivity quantification |
| Scaling fit | $\log\Delta\phi = \alpha\log N + \log C$ via linear regression | Exponent extraction |
| Fit quality threshold | $R^2 > 0.9$, min 4 $N$ points | Valid fit guarantee |

### Validation

```python
assert np.isclose(np.trace(rho), 1.0, atol=1e-8)       # trace preservation
assert np.allclose(rho, rho.conj().T, atol=1e-8)       # hermiticity
assert np.min(np.linalg.eigvalsh(rho)) >= -1e-8         # positivity
assert R_squared >= 0.9                                  # fit quality
```

## ⚠️ Expected Failure Conditions

| Failure | Description | Mitigation |
|---|---|---|
| Hilbert space explosion | $(N_{\max}+1)^2$ dimension is prohibitive for $N > 64$. Definite Fock-like states use $N_{\max}=N$; coherent-like use $N_{\max} = \max(2N, N+20)$ to capture Poisson tails. For $N=64$, the latter gives $N_{\max}=128$ → dimension $16,\!641$. | Use Dicke basis for symmetric states, covariance-matrix methods for Gaussian states. |
| Degenerate exponents | Different (state, noise) combos can yield identical $\alpha$ (e.g., coherent SQL vs. NOON collapsed to SQL). | Report prefactor $C$, $R^2$, and model evidence to disambiguate. |
| Finite-size artifacts | $N < 4$ produces quantization artifacts in $\Delta\phi$. | Exclude $N < 4$ from log-log regression; require $N \geq 8$ for final fit. |
| Phase-bias dependence | $\Delta\phi(\phi)$ is not flat; $\phi=\pi/4$ works for most states but NOON has maximal sensitivity at different $\phi$. | Report sensitivity at each state's optimal operating point. |

## 🔬 Results

| Check | Status |
|---|---|
| **Coherent-state SQL scaling (ideal)** | ⏳ |
| **NOON Heisenberg scaling (ideal)** | ⏳ |
| **Squeezed-vacuum scaling (ideal)** | ⏳ |
| **NOON collapse under one-body loss** | ⏳ |
| **Two-body loss: scaling collapse** | ⏳ |
| **Phase-diffusion: coherent resilience** | ⏳ |
| **Kerr-nonlinear MZI: invariant exponent** | ⏳ |
| **Weak-value MZI: Fisher invariance** | ⏳ |
| **Quantum state invariants** | ⏳ |
| **Log-log fit quality** | ⏳ |
| **Minimum $N$ threshold enforced** | ⏳ |
| **Phase-bias optimality** | ⏳ |

All twelve checks are pending. No numerical experiments have been run yet; this report currently serves as the survey design document.

## ✅ Success Criteria

| Check | Expectation |
|---|---|
| **Coherent-state SQL scaling (ideal)** | $\alpha = -0.5 \pm 0.02$ for noise-free MZI; $\Delta\phi = 1/\sqrt{N}$ to within 2% for $N \geq 4$ |
| **NOON Heisenberg scaling (ideal)** | $\alpha = -1.0 \pm 0.02$; $\Delta\phi = 1/N$ to within 2% for $N \geq 2$ under $J_z$ generator |
| **Squeezed-vacuum scaling (ideal)** | $F_Q = 2\langle N\rangle(\langle N\rangle + 1)$ verified to within 1%; $\alpha \to -1.0$ for $\langle N \rangle \gg 1$ |
| **NOON collapse under one-body loss** | $\alpha$ transitions from $-1.0$ to $-0.5 \pm 0.05$ for $\gamma_1 T \geq 10^{-2}$; $R^2 \geq 0.9$ for the collapsed fit |
| **Two-body loss: scaling collapse** | All input states show $\alpha \to 0 \pm 0.05$ at large $N$ for $\gamma_2 T > 0$; verified over $N \in [4, 64]$ |
| **Phase-diffusion: coherent resilience** | Coherent states maintain $\alpha = -0.5 \pm 0.02$ for all $\gamma_\phi \in [0, 1]$ |
| **Kerr-nonlinear MZI: invariant exponent** | $F_Q$ changes by $< 10^{-8}$ when Kerr term $\chi(n_1^2 + n_2^2)$ is added (commuting generator); $\alpha$ unchanged to machine precision |
| **Weak-value MZI: Fisher invariance** | $F_Q$ unchanged vs conventional MZI (same state, same $\phi$); prefactor $C_{\text{wv}} = 1/\sqrt{1-p_{\text{ps}}}$ verified within 1% |
| **Quantum state invariants** | $\Tr(\rho) = 1.0 \pm 10^{-8}$; $\rho = \rho^\dagger$ to $10^{-8}$; $\min \text{eigvals}(\rho) \geq -10^{-8}$ at all evolution times |
| **Log-log fit quality** | $R^2 \geq 0.9$ for all scaling fits with $\geq 4$ $N$-points; $R^2 \geq 0.95$ for ideal-case references |
| **Minimum $N$ threshold enforced** | $N < 4$ excluded from scaling regressions to avoid finite-size quantization artifacts |
| **Phase-bias optimality** | Each state's $\Delta\phi(\phi)$ sampled over $\phi \in [0, \pi/2]$ to confirm operating point; reported $\alpha$ uses the state's optimal $\phi$ |

## 🏁 Conclusions

This report establishes the design for a comprehensive survey of scaling exponents ($\alpha$) in Mach–Zehnder interferometry across a suite of two-mode Fock-basis states under Markovian decoherence. The theoretical framework — Hilbert space, interferometer unitary, input states, noise channels, generator convention, and measurement strategies — is fully specified, and the numerical pipeline is ready. The Models Survey table provides the definitive reference mapping each (state, noise) combination to its expected scaling exponent $\alpha$, including predicted transitions (NOON collapse under loss, two-body scaling collapse, squeezed-vacuum Heisenberg scaling). All success criteria are defined; numerical results are pending execution.

🔍 **Open items**: (a) Run the full survey sweep to populate the Results table with $\alpha$ measurements. (b) Verify predicted thresholds for NOON collapse under one-body loss and two-body scaling collapse. (c) Validate analytical bounds for detection noise ($F_Q \to \eta F_Q$) with the full binomial convolution. (d) Extend parity-measurement analysis and determine whether parity CFI recovers QFI scaling for relevant states.
