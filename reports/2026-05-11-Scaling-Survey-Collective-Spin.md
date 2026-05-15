# Scaling Survey: Dicke Basis — Collective Spin & Entanglement-Enhanced Interferometry

## 🧪 Hypothesis

For interferometers using **collective spin states in the symmetric Dicke subspace**:

1. Spin-squeezed states (OAT) achieve $\alpha = -2/3$ at optimal squeezing time $t_{\text{opt}} \propto N^{-1/3}$, interpolating between SQL and HL
2. Twin-Fock states achieve $\alpha = -1.0$ (Heisenberg scaling) via number-difference readout
3. Non-Gaussian states (n=3,4 hybrid squeezing) achieve fractional exponents $\alpha \approx -0.75$ to $-0.85$ at fixed $\langle n\rangle$
4. Ancilla-assisted metrology preserves the state's scaling exponent $\alpha$ but improves the prefactor $C$ by $\mathcal{R}(T)^{-1/2}$ under non-Markovian noise
5. All entanglement-enhanced models are more fragile than coherent states under phase diffusion ($\gamma_\phi$): the advantage degrades monotonically with $\gamma_\phi$

## ⚛️ Theoretical Model

The **Hilbert space** is the symmetric Dicke subspace $\mathcal{H}_J = \text{span}\{\vert J, m\rangle\}$ with total spin $J = N/2$ and $m \in \{-J, -J+1, \dots, J\}$, of dimension $N+1$. The **collective angular momentum operators** are $J_x = \frac12(a_1^\dagger a_2 + a_2^\dagger a_1)$, $J_y = \frac{i}{2}(a_2^\dagger a_1 - a_1^\dagger a_2)$, and $J_z = \frac12(n_1 - n_2)$, which satisfy the su(2) algebra $[J_i, J_j] = i\epsilon_{ijk} J_k$.

The **entanglement Hamiltonians** for spin squeezing are **one-axis twisting (OAT)** $H = \chi J_z^2$, with unitary $e^{-i\chi t J_z^2}$, and **two-axis countertwisting (TNT)** $H = \chi(J_+^2 + J_-^2)/2$, with unitary $e^{-i\chi t (J_+^2+J_-^2)/2}$. The OAT optimal squeezing time scales as $t_{\text{opt}} \propto N^{-1/3}$, yielding $\xi^2_{\min} \propto N^{-2/3}$ and sensitivity exponent $\alpha = -2/3$. The TNT Hamiltonian achieves Heisenberg-limited squeezing with $\xi^2_{\min} \propto N^{-1}$.

**Extended models** incorporate larger Hilbert spaces beyond pure spin squeezing. **Non-Gaussian states** are prepared via a hybrid oscillator-spin Hamiltonian $H_n = \frac{\Omega_n}{2}\, \sigma_{\text{op}} \otimes (a^n e^{-i\theta_n} + a^{\dagger n} e^{i\theta_n})$ on the extended space $\mathcal{H}_J \otimes \mathcal{H}_{\text{osc}}$. **Ancilla-assisted metrology** couples the probe to a spin ancilla via $H_{\text{int}} = g\, n_{\text{photon}} \otimes J_z$ on $\mathcal{H}_{\text{Fock}} \otimes \mathbb{C}^{d_A}$. For non-Markovian noise, the system extends to $\mathcal{H}_{\text{osc}} \otimes \mathbb{C}^2 \otimes \mathcal{H}_{\text{pm}}$ with probe-ancilla coupling $H_{\text{sa}} = g_{\text{sa}}\, a^\dagger a \otimes \sigma_x \otimes \mathbb{1}_{\text{pm}}$.

The **input states** considered are: the **coherent spin state (CSS)** $\vert J, -J\rangle$ rotated to the $x$-axis; the **spin-squeezed state (SSS)** obtained by OAT evolution at $t_{\text{opt}}$; the **Twin-Fock state** $\vert N/2, 0\rangle$ in the Dicke basis (requires even $N$); **non-Gaussian states** from hybrid oscillator-spin evolution under $H_n$; and **ancilla-assisted states** coupling the probe to an ancillary register.

The interferometric **circuit protocol** follows a standard Mach-Zehnder sequence: a $\pi/2$ beam splitter (rotation about $J_y$), a phase imprint $e^{i\phi n_2}$ (equivalent to $e^{-i\phi J_z}$ up to a global phase), and a final $\pi/2$ beam splitter. The **sensitivity metrics** are:

| Method | Formula |
|---|---|
| Error propagation | $\Delta\phi_{\text{EP}} = \Delta J_z / \lvert \partial\langle J_z\rangle/\partial\phi\rvert$ |
| Quantum Fisher (pure) | $\Delta\phi_{\text{QFI}} = 1/\sqrt{F_Q}$, $F_Q = 4\,\text{Var}(J_z)$ |
| Quantum Fisher (mixed) | $\Delta\phi_{\text{QFI}} = 1/\sqrt{F_Q}$ via SLD eigen-decomposition |
| Classical Fisher | $\Delta\phi_{\text{CFI}} = 1/\sqrt{\sum_k (\partial_\phi p_k)^2 / p_k}$ |

---

## 📊 Models Survey

| Model | Input State | Noise | Expected $\alpha$ | Implementation Status |
|---|---|---|---|---|
| OAT spin-squeezed ($\chi J_z^2$ at $t_{\text{opt}}$) | CSS | Ideal | $-2/3$, $C \approx (2^{1/3}/3^{1/6})\chi^{-1/3}$ | ✅ |
| Twin-Fock | $\vert N/2,0\rangle$ (even $N$) | Ideal; one-body loss $\gamma_1$ | $-1.0$ (ideal), $-0.5$ (with loss) | ✅ |
| Two-axis CT (TNT) | CSS | Ideal | $-1.0$ (Heisenberg-limited) | ✅ |
| Non-Gaussian $n=3$ | Vacuum $\otimes \vert\downarrow\rangle$ under $H_3$ | Ideal | $\approx -0.75$, $C \approx 1.5$ | ✅ |
| Non-Gaussian $n=4$ | Vacuum $\otimes \vert\downarrow\rangle$ under $H_4$ | Ideal | $\approx -0.85$, $C \approx 2.0$ | ✅ |
| Ancilla-assisted | Probe $\otimes$ ancilla | Non-Markovian $\mathcal{R}(T)$ | Same $\alpha$ as probe; $C$ improved by $\mathcal{R}(T)^{-1/2}$ | ✅ |

---

## 💻 Numerical Simulation

### Implementation Strategy

1. **Dicke-basis propagation** (OAT/TNT): ODE for $e^{-iHt}\vert\psi_0\rangle$ in $N+1$ dimensions (unitary) or Lindblad master equation (mixed)
2. **Optimal time search**: For OAT, sweep $t$, compute $\xi^2_R(t)$, pick $t_{\text{opt}}$ per $N$, then compute sensitivity at $t_{\text{opt}}$
3. **Hybrid systems**: Extended Hilbert space dimension $2(N+1)$ for non-Gaussian (oscillator $\otimes$ spin), $2(N+1)(K+1)$ for ancilla + pseudomode
4. **Phase imprint**: $e^{i\phi n_2}$ in the two-mode Fock basis (diagonal: $e^{i\phi n_2}$), applied after the first beam splitter

### Parameter Sweep

| Parameter | Values / Range | Purpose |
|-----------|----------------|---------|
| Resource $N$ | $\{2, 4, 8, 16, 32, 64, 128\}$ (Dicke dimension $N+1$) | Scaling exponent extraction over larger range |
| Phase diffusion $\gamma_\phi$ | $\{0, 10^{-3}, 10^{-2}, 10^{-1}, 1\}$ | Primary decoherence channel |
| OAT squeezing time $t/\chi^{-1}$ | Swept to find $t_{\text{opt}}$ per $N$ | Verify $\xi^2_{\min} \propto N^{-2/3}$ |
| Scaling fit | $\log\Delta\phi = \alpha\log N + \log C$ via linear regression | Exponent extraction |

### Validation

```python
# Squeezing parameter bounds
assert np.min(eigvalsh(rho)) >= -1e-8, "Positivity"
assert np.isclose(np.trace(rho), 1.0, atol=1e-10), "Trace preservation"

# OAT optimal time consistency
assert np.isclose(t_opt, (6.0/N)**(1/3)/chi, rtol=0.1), "t_opt ∝ N^{-1/3}"

# QFI bounds for pure symmetric states
assert F_Q <= 4 * N**2, "QFI ≤ 4N² for J_z generator"
```

---

## ⚠️ Expected Failure Conditions

| Failure | Description | Mitigation |
|---|---|---|
| Dicke basis limitation for hybrid models | Adding oscillator or ancilla increases dimension to $(N+1) \times d_{\text{ext}}$. For $N > 100$ and $d_{\text{ext}} > 10$, this becomes expensive. | Restrict hybrid models to $N \leq 30$. |
| Optimal time degeneracy | The sensitivity $\Delta\phi(t)$ can have multiple local minima at different $t$ for the same $N$. | Sweep $t$ with fine grid and report global minimum. |
| Non-Gaussian state preparation | The time-evolved state may not be the optimal metrological state at finite $N$. | Report QFI of multiple state preparations (vacuum-evolved, ground state) and select the best. |
| Ancilla readout overhead | The ancilla-assisted protocol requires projective measurements on the ancilla, which adds statistical overhead not captured by QFI alone. | Report Bayesian sensitivity $n_{\text{shots}} \to \Delta\phi$ alongside QFI bound. |

---

## 🔬 Results

All checks are pending numerical experiments. The table below tracks completion status for each Success Criterion.

| # | Check | Status |
|---|---|---|
| 1 | OAT squeezing scaling ($\xi^2_{\min} \propto N^{-2/3}$) | ⏳ |
| 2 | OAT sensitivity exponent ($\alpha = -2/3$) | ⏳ |
| 3 | Twin-Fock Heisenberg scaling ($\alpha = -1.0$) | ⏳ |
| 4 | Two-axis countertwisting (TNT) | ⏳ |
| 5 | Non-Gaussian $n=3$ scaling ($\alpha \approx -0.75$) | ⏳ |
| 6 | Non-Gaussian $n=4$ scaling ($\alpha \approx -0.85$) | ⏳ |
| 7 | Ancilla-assisted prefactor preservation | ⏳ |
| 8 | Phase-diffusion degradation with $\gamma_\phi$ | ⏳ |
| 9 | Quantum state invariants throughout evolution | ⏳ |
| 10 | QFI upper bound ($F_Q \leq 4N^2$) | ⏳ |
| 11 | Log-log fit quality ($R^2 \geq 0.9$) | ⏳ |
| 12 | NOON vs Twin-Fock comparison under one-body loss | ⏳ |

---

## ✅ Success Criteria

| # | Check | Expectation |
|---|---|---|
| 1 | **OAT squeezing scaling** | $\xi^2_{\min} \propto N^{-2/3}$; fit exponent $-0.67 \pm 0.05$; $t_{\text{opt}} \propto N^{-1/3}$ within 10% of $(6/N)^{1/3}/\chi$ |
| 2 | **OAT sensitivity exponent** | $\alpha = -2/3 \pm 0.03$ from log-log fit of $\Delta\phi(N)$ at $t_{\text{opt}}$; $R^2 \geq 0.95$ |
| 3 | **Twin-Fock Heisenberg scaling** | $\Delta\phi = 1/\sqrt{F_Q}$ with $F_Q = N(N+2)/3$; measured $\alpha = -1.0 \pm 0.03$ in ideal case; $\alpha \to -0.5$ under one-body loss $\gamma_1 > 0$ |
| 4 | **Two-axis countertwisting (TNT)** | $\xi^2_{\min} \propto N^{-1}$ (Heisenberg-limited); $\alpha = -1.0 \pm 0.05$ via QFI |
| 5 | **Non-Gaussian $n=3$ scaling** | Measured $\alpha = -0.75 \pm 0.05$ at fixed $\langle n \rangle$; $C \approx 1.5$; $R^2 \geq 0.90$ |
| 6 | **Non-Gaussian $n=4$ scaling** | Measured $\alpha = -0.85 \pm 0.05$ at fixed $\langle n \rangle$; $C \approx 2.0$; $R^2 \geq 0.90$ |
| 7 | **Ancilla-assisted prefactor preservation** | $\alpha_{\text{ancilla}} = \alpha_{\text{probe}} \pm 0.02$ (scaling exponent unchanged); prefactor $C_{\text{ancilla}} = C_{\text{probe}} / \sqrt{\mathcal{R}(T)}$ within 5% |
| 8 | **Phase-diffusion degradation** | Entanglement-enhanced states (OAT, Twin-Fock) show $\alpha$ degrading monotonically with $\gamma_\phi$; coherent-state $\alpha$ stays at $-0.5 \pm 0.02$ for all $\gamma_\phi$ |
| 9 | **Quantum state invariants** | $\Tr(\rho) = 1.0 \pm 10^{-10}$; $\rho = \rho^\dagger$ (Hermitian to $10^{-10}$); $\min \text{eigvals}(\rho) \geq -10^{-8}$ throughout evolution |
| 10 | **QFI upper bound** | $F_Q \leq 4N^2$ for pure symmetric states under $J_z$ generator; verified within floating-point tolerance |
| 11 | **Log-log fit quality (all models)** | $R^2 \geq 0.9$ for all scaling sweeps with $\geq 5$ $N$-points; outliers flagged when fit residuals exceed $3\sigma$ |
| 12 | **NOON vs Twin-Fock comparison under loss** | Both states collapse to $\alpha = -0.5$ under $\gamma_1 > 0$; prefactor $C$ differs by at most a constant factor set by the initial $F_Q$ ratio |

All checks are pending execution of the numerical sweeps described in Numerical Simulation. Once completed, each check will be assigned a status (✅/❌/🔄) and this section will summarize which of the five hypotheses were supported.

#### 📐 Analytical Bounds

For pure states in the symmetric Dicke subspace with generator $J_z$, the **quantum Fisher information** satisfies $F_Q \leq 4N^2$, corresponding to the Heisenberg limit $\Delta\phi_{\text{HL}} = 1/(2N)$. The **standard quantum limit** for coherent states is $\Delta\phi_{\text{SQL}} = 1/\sqrt{N}$. The **spin-squeezing parameter** $\xi^2_R = N\,\text{Var}(J_z)/\langle J_x\rangle^2$ is bounded below by $1/F_Q$ for the optimal readout direction, providing a computable witness for metrological advantage. For Twin-Fock states, the exact QFI is $F_Q = N(N+2)/3$ in the ideal case.

---

## 🏁 Conclusions

This survey establishes a unified framework for evaluating scaling exponents $\alpha$ across five collective-spin models in the symmetric Dicke subspace. The **theoretical model** covers OAT ($\alpha = -2/3$), TNT and Twin-Fock ($\alpha = -1.0$), non-Gaussian states ($\alpha \approx -0.75$ to $-0.85$), and ancilla-assisted protocols ($\alpha$ preserved, prefactor improved). All implementations are conceptually complete (✅ in the Models Survey), but numerical experiments sweeping $N$ and noise parameters have yet to be executed.

Once the full parameter sweeps are run, the Success Criteria table will reveal which of the five hypotheses are supported and whether the predicted scaling exponents hold within the stated tolerances.

#### 🔍 Open Items

🔍 **Numerical verification**: The predicted exponents $\alpha = -2/3$ for OAT and $\alpha = -1.0$ for Twin-Fock/TNT are analytically motivated and should be confirmed via log-log regression over $N \in \{2, \dots, 128\}$.

🔍 **Noise robustness**: To what extent does phase diffusion $\gamma_\phi$ degrade the scaling exponent for each model? Hypothesis 5 predicts monotonic degradation for entanglement-enhanced states but robustness for coherent states.

🔍 **Ancilla overhead**: The QFI bound for ancilla-assisted protocols does not account for the additional measurement cost of projective ancilla readout. Bayesian estimation with finite $n_{\text{shots}}$ should be compared to the ideal QFI bound to assess practical advantage.
