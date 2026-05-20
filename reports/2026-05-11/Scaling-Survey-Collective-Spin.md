# Scaling Survey: Dicke Basis — Collective Spin & Entanglement-Enhanced Interferometry

## 🧪 Hypothesis

For interferometers using **collective spin states in the symmetric Dicke subspace**:

1. Spin-squeezed states (OAT) achieve $\alpha = -2/3$ at optimal squeezing time $t_{\text{opt}} \propto N^{-1/3}$, interpolating between SQL and HL
2. Balanced Dicke superposition (uniform superposition $\sum_{n=0}^N \vert n,N-n\rangle/\sqrt{N+1}$, often called Twin-Fock in the Fock-basis literature) achieves $\alpha = -1.0$ (Heisenberg scaling) via number-difference readout, with $F_Q = N(N+2)/3$
3. Non-Gaussian states (n=3,4 hybrid squeezing) achieve fractional exponents $\alpha \approx -0.75$ to $-0.85$ at fixed $\langle n\rangle$
4. Ancilla-assisted metrology preserves the state's scaling exponent $\alpha$ but improves the prefactor $C$; the non-Markovian improvement $\mathcal{R}(T)^{-1/2}$ is studied in the pseudomode module (`pseudomode_system.py`), while the scaling survey implements a Markovian dispersive ancilla protocol
5. All entanglement-enhanced models are more fragile than coherent states under phase diffusion ($\gamma_\phi$): the advantage degrades monotonically with $\gamma_\phi$

## ⚛️ Theoretical Model

The **Hilbert space** is the symmetric Dicke subspace $\mathcal{H}_J = \text{span}\{\vert J, m\rangle\}$ with total spin $J = N/2$ and $m \in \{-J, -J+1, \dots, J\}$, of dimension $N+1$. The **collective angular momentum operators** are $J_x = \frac12(a_1^\dagger a_2 + a_2^\dagger a_1)$, $J_y = \frac{i}{2}(a_2^\dagger a_1 - a_1^\dagger a_2)$, and $J_z = \frac12(n_1 - n_2)$, which satisfy the su(2) algebra $[J_i, J_j] = i\epsilon_{ijk} J_k$.

The **entanglement Hamiltonians** for spin squeezing are **one-axis twisting (OAT)** $H = \chi J_z^2$, with unitary $e^{-i\chi t J_z^2}$, and **two-axis countertwisting (TNT)** $H = \chi(J_+^2 + J_-^2)/2$, with unitary $e^{-i\chi t (J_+^2+J_-^2)/2}$. The OAT optimal squeezing time scales as $t_{\text{opt}} \propto N^{-1/3}$, yielding $\xi^2_{\min} \propto N^{-2/3}$ and sensitivity exponent $\alpha = -2/3$. The TNT Hamiltonian achieves Heisenberg-limited squeezing with $\xi^2_{\min} \propto N^{-1}$.

**Extended models** incorporate larger Hilbert spaces beyond pure spin squeezing. **Non-Gaussian states** are prepared via a hybrid oscillator-spin Hamiltonian $H_n = \frac{\Omega_n}{2}\, \sigma_{\text{op}} \otimes (a^n e^{-i\theta_n} + a^{\dagger n} e^{i\theta_n})$ on the extended space $\mathcal{H}_J \otimes \mathcal{H}_{\text{osc}}$. **Ancilla-assisted metrology** couples the probe to a spin ancilla via $H_{\text{int}} = g\, n_{\text{photon}} \otimes J_z$ on $\mathcal{H}_{\text{Fock}} \otimes \mathbb{C}^{d_A}$. For non-Markovian noise, the system extends to $\mathcal{H}_{\text{osc}} \otimes \mathbb{C}^2 \otimes \mathcal{H}_{\text{pm}}$ with probe-ancilla coupling $H_{\text{sa}} = g_{\text{sa}}\, a^\dagger a \otimes \sigma_x \otimes \mathbb{1}_{\text{pm}}$.

The **input states** considered are: the **coherent spin state (CSS)** $\vert J, -J\rangle$ rotated to the $x$-axis (used for OAT/TNT entangling); the **spin-squeezed state (SSS)** obtained by OAT evolution at $t_{\text{opt}}$; the **balanced Dicke superposition** $\sum_{n=0}^N \vert n,N-n\rangle/\sqrt{N+1}$ (implemented in the code as `twin_fock_state`, distinct from the standard $\vert N/2,N/2\rangle$ Twin-Fock — see **Note on naming** below); **non-Gaussian states** from hybrid oscillator-spin evolution under $H_n$; and **ancilla-assisted states** coupling the probe to an ancillary register.

> **Note on naming**: The code's `twin_fock_state` returns the uniform superposition of all $\vert n,N-n\rangle$ Fock states (balanced Dicke superposition), *not* the single Fock state $\vert N/2,N/2\rangle$. This choice preserves $\langle J_z\rangle=0$ and achieves near-Heisenberg QFI $F_Q=N(N+2)/3$. The standard $\vert N/2,N/2\rangle$ Twin-Fock has $F_Q=N$ (SQL scaling) and is not used here.

> **Note on CSS conventions**: In the Dicke-basis context (OAT/TNT entanglers), CSS means $\vert J,-J\rangle_x$ — the eigenstate of $J_x$ with eigenvalue $-J$, which has definite particle number $N$. In the Fock-basis scaling survey pipeline (`input_state_factory("css", ...)`), it refers to a two-mode coherent state with $\alpha=\sqrt{N}$ on mode 0, which does not have definite $N$. The two conventions give the same SQL scaling $\alpha=-0.5$ but differ in finite-$N$ prefactors. The OAT/TNT models consistently use the Dicke-basis CSS.

The interferometric **circuit protocol** follows a standard **Mach-Zehnder** sequence: a $\pi/2$ beam splitter, a phase imprint $e^{i\phi n_2}$ (equivalent to $e^{-i\phi J_z}$ up to a global phase), and a final $\pi/2$ beam splitter. The scaling survey computes sensitivity via the **Quantum Fisher Information** $F_Q = 4\,\text{Var}(J_z)$ for pure states, which is independent of the beam splitter convention and depends only on the probe state. This means the survey's $\alpha$ predictions hold regardless of whether the Dicke-basis rotation (about $J_y$) or Fock-basis beam-splitter Hamiltonian ($a_1^\dagger a_2 + a_2^\dagger a_1$) is used. The **sensitivity metrics** are:

| Method | Formula |
|---|---|
| **Error propagation** | $\Delta\phi_{\text{EP}} = \Delta J_z / \lvert \partial\langle J_z\rangle/\partial\phi\rvert$ |
| Quantum Fisher (pure) | $\Delta\phi_{\text{QFI}} = 1/\sqrt{F_Q}$, $F_Q = 4\,\text{Var}(J_z)$ |
| Quantum Fisher (mixed) | $\Delta\phi_{\text{QFI}} = 1/\sqrt{F_Q}$ via SLD eigen-decomposition |
| Classical Fisher | $\Delta\phi_{\text{CFI}} = 1/\sqrt{\sum_k (\partial_\phi p_k)^2 / p_k}$ |

---

## 📊 Models Survey

| Model | Input State | Noise | Expected $\alpha$ | Implementation Status |
|---|---|---|---|---|---|
| OAT spin-squeezed ($\chi J_z^2$ at $t_{\text{opt}}$) | CSS ($\vert J,-J\rangle_x$) | Ideal; phase diffusion $\gamma_\phi$ | $-2/3$, $C \approx (2^{1/3}/3^{1/6})\chi^{-1/3}$ | PASS via `coherent_oat` entangler |
| Balanced Dicke superposition (uniform $\sum\vert n,N-n\rangle/\sqrt{N+1}$) | Even $N$ | Ideal; one-body loss $\gamma_1$ | $-1.0$ (ideal, $F_Q=N(N+2)/3$), $-0.5$ (with loss) | PASS as `ideal_twin_fock` |
| Two-axis CT (TNT) | CSS ($\vert J,-J\rangle_x$) | Ideal | $-1.0$ (Heisenberg-limited) | PARTIAL via `tnt` entangler (not in default survey; uses OAT $t_{\text{opt}}$ as approximation) |
| Non-Gaussian $n=3$ | Vacuum $\otimes \vert\downarrow\rangle$ under $H_3$ | Ideal | $\approx -0.75$, $C \approx 1.5$ | PASS as `non_gaussian_n3` |
| Non-Gaussian $n=4$ | Vacuum $\otimes \vert\downarrow\rangle$ under $H_4$ | Ideal | $\approx -0.85$, $C \approx 2.0$ | PASS as `non_gaussian_n4` |
| Ancilla-assisted (Markovian, survey pipeline) | Probe $\otimes$ ancilla | Markovian (dispersive coupling) | Same $\alpha$ as probe; $C$ improved | PASS as `ancilla_assisted` (g_sp=0, lam=0) |
| Ancilla-assisted (non-Markovian, pseudomode) | Probe $\otimes$ ancilla $\otimes$ bath | Non-Markovian $\mathcal{R}(T)$ | Same $\alpha$; $C$ improved by $\mathcal{R}(T)^{-1/2}$ | PASS in `pseudomode_system.py` (separate module; not in scaling survey) |

---

## 💻 Numerical Simulation

### Implementation Strategy

1. **Dicke-basis propagation** (OAT/TNT): ODE for $e^{-iHt}\vert\psi_0\rangle$ in $N+1$ dimensions (unitary) or Lindblad master equation (mixed)
2. **Optimal time search**: For OAT, sweep $t$, compute $\xi^2_R(t)$, pick $t_{\text{opt}}$ per $N$, then compute sensitivity at $t_{\text{opt}}$. For TNT, the code uses the OAT formula $t_{\text{opt}} = (6/N)^{1/3}$ as an approximation (known limitation — the true TNT optimal time may differ).
3. **Hybrid systems**: Extended Hilbert space dimension $2(N+1)$ for non-Gaussian (oscillator $\otimes$ spin), $2(N+1)(K+1)$ for ancilla + pseudomode
4. **Phase imprint**: $e^{i\phi n_2}$ in the two-mode Fock basis (diagonal: $e^{i\phi n_2}$), applied after the first beam splitter

### Parameter Sweep

| Parameter | Values / Range | Purpose |
|-----------|----------------|---------|
| Resource $N$ | $\{2, 4, 8, 16, 32, 64, 128\}$ (Dicke dimension $N+1$) | Scaling exponent extraction over larger range |
| Phase diffusion $\gamma_\phi$ | $\{0, 10^{-3}, 10^{-2}, 10^{-1}\}$ (code default); survey supports up to $1$ | Primary decoherence channel |
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
| TNT optimal time approximation | The code reuses the OAT formula $t_{\text{opt}} = (6/N)^{1/3}$ for TNT, which is not the true TNT optimal time. The QFI may still be correct but the squeezing dynamics are approximate. | For accurate TNT squeezing, implement a proper $t_{\text{opt}}$ search via $\xi^2$ minimization. |
| Ancilla survey is Markovian | The scaling survey's ancilla model (`_ancilla_sensitivity_fn`) uses $g_{\text{sp}}=0, \lambda=0$ (no bath coupling). The non-Markovian $R(T)$ claims are verified only in the separate `pseudomode_system.py` module, which is not part of the scaling survey pipeline. | The non-Markovian model can be adapted into a `custom_sensitivity_fn` for the survey if needed. |

---

## 🔬 Results

The scaling survey pipeline is fully implemented and can be run interactively via the `Interferometry_Scaling_Survey.py` page. Numerical sweeps for the specific collective-spin models described here have not yet been executed; the table below tracks completion status.

| # | Check | Status |
|---|---|---|
| 1 | OAT squeezing scaling ($\xi^2_{\min} \propto N^{-2/3}$) | PENDING |
| 2 | OAT sensitivity exponent ($\alpha = -2/3$) | PENDING |
| 3 | Balanced Dicke superposition (Twin-Fock) Heisenberg scaling ($\alpha = -1.0$) | PENDING |
| 4 | Two-axis countertwisting (TNT) — via `tnt` entangler (note: uses OAT $t_{\text{opt}}$ approximation) | PENDING |
| 5 | Non-Gaussian $n=3$ scaling ($\alpha \approx -0.75$) | PENDING |
| 6 | Non-Gaussian $n=4$ scaling ($\alpha \approx -0.85$) | PENDING |
| 7 | Ancilla-assisted prefactor preservation (Markovian survey model) | PENDING |
| 8 | Phase-diffusion degradation with $\gamma_\phi$ | PENDING |
| 9 | Quantum state invariants throughout evolution | PENDING |
| 10 | QFI upper bound ($F_Q \leq 4N^2$) | PENDING |
| 11 | Log-log fit quality ($R^2 \geq 0.9$) | PENDING |
| 12 | Balanced Dicke vs NOON comparison under one-body loss | PENDING |

---

## ✅ Success Criteria

| # | Check | Expectation |
|---|---|---|---|
| 1 | **OAT squeezing scaling** | $\xi^2_{\min} \propto N^{-2/3}$; fit exponent $-0.67 \pm 0.05$; $t_{\text{opt}} \propto N^{-1/3}$ within 10% of $(6/N)^{1/3}/\chi$ |
| 2 | **OAT sensitivity exponent** | $\alpha = -2/3 \pm 0.03$ from log-log fit of $\Delta\phi(N)$ at $t_{\text{opt}}$; $R^2 \geq 0.95$ |
| 3 | **Balanced Dicke superposition Heisenberg scaling** | $\Delta\phi = 1/\sqrt{F_Q}$ with $F_Q = N(N+2)/3$; measured $\alpha = -1.0 \pm 0.03$ in ideal case; $\alpha \to -0.5$ under one-body loss $\gamma_1 > 0$ |
| 4 | **Two-axis countertwisting (TNT)** | $\xi^2_{\min} \propto N^{-1}$ (Heisenberg-limited); $\alpha = -1.0 \pm 0.05$ via QFI. Note: code uses OAT $t_{\text{opt}}$ as approximation — squeezing scaling may differ from true TNT. |
| 5 | **Non-Gaussian $n=3$ scaling** | Measured $\alpha = -0.75 \pm 0.05$ at fixed $\langle n \rangle$; $C \approx 1.5$; $R^2 \geq 0.90$ |
| 6 | **Non-Gaussian $n=4$ scaling** | Measured $\alpha = -0.85 \pm 0.05$ at fixed $\langle n \rangle$; $C \approx 2.0$; $R^2 \geq 0.90$ |
| 7 | **Ancilla-assisted prefactor preservation (survey model)** | $\alpha_{\text{ancilla}} = \alpha_{\text{probe}} \pm 0.02$ (scaling exponent unchanged); $C$ improvement depends on coupling parameters. The non-Markovian $\mathcal{R}(T)^{-1/2}$ prefactor improvement applies to the pseudomode module only. |
| 8 | **Phase-diffusion degradation** | Entanglement-enhanced states (OAT, balanced Dicke superposition) show $\alpha$ degrading monotonically with $\gamma_\phi$; coherent-state $\alpha$ stays at $-0.5 \pm 0.02$ for all $\gamma_\phi$ |
| 9 | **Quantum state invariants** | $\Tr(\rho) = 1.0 \pm 10^{-10}$; $\rho = \rho^\dagger$ (Hermitian to $10^{-10}$); $\min \text{eigvals}(\rho) \geq -10^{-8}$ throughout evolution |
| 10 | **QFI upper bound** | $F_Q \leq 4N^2$ for pure symmetric states under $J_z$ generator; verified within floating-point tolerance |
| 11 | **Log-log fit quality (all models)** | $R^2 \geq 0.9$ for all scaling sweeps with $\geq 5$ $N$-points; outliers flagged when fit residuals exceed $3\sigma$ |
| 12 | **NOON vs Balanced Dicke comparison under loss** | Both states collapse to $\alpha = -0.5$ under $\gamma_1 > 0$; prefactor $C$ differs by at most a constant factor set by the initial $F_Q$ ratio |

All checks are pending execution of the numerical sweeps described in Numerical Simulation. Once completed, each check will be assigned a status (PASS/FAIL/PARTIAL) and this section will summarize which of the five hypotheses were supported.

#### ⚖️ Analytical Bounds

For pure states in the symmetric Dicke subspace with generator $J_z$, the **quantum Fisher information** satisfies $F_Q \leq 4N^2$, corresponding to the Heisenberg limit $\Delta\phi_{\text{HL}} = 1/(2N)$. The **standard quantum limit** for coherent states is $\Delta\phi_{\text{SQL}} = 1/\sqrt{N}$. The **spin-squeezing parameter** $\xi^2_R = N\,\text{Var}(J_z)/\langle J_x\rangle^2$ is bounded below by $1/F_Q$ for the optimal readout direction, providing a computable witness for metrological advantage. For Twin-Fock states, the exact QFI is $F_Q = N(N+2)/3$ in the ideal case.

---

## 🏁 Conclusions

This survey establishes a unified framework for evaluating scaling exponents $\alpha$ across five collective-spin models in the symmetric Dicke subspace. The **theoretical model** covers OAT ($\alpha = -2/3$), TNT and balanced Dicke superposition ($\alpha = -1.0$), non-Gaussian states ($\alpha \approx -0.75$ to $-0.85$), and ancilla-assisted protocols ($\alpha$ preserved, prefactor improved). The scaling survey pipeline is implemented and functional (accessible via the `Interferometry_Scaling_Survey.py` page), but numerical experiments specific to the collective-spin models have yet to be executed.

**Key clarifications from code validation**: (a) The "Twin-Fock" state in the code is actually the balanced Dicke superposition $\sum\vert n,N-n\rangle/\sqrt{N+1}$ (uniform in Fock space, not the $\vert N/2,N/2\rangle$ Fock state) — this does not affect the $\alpha=-1.0$ prediction, which holds. (b) The scaling survey's ancilla model is Markovian (dispersive coupling); the non-Markovian $\mathcal{R}(T)$ claims are verified in the separate `pseudomode_system.py` module. (c) The TNT implementation uses the OAT $t_{\text{opt}}$ as an approximation — acceptable for QFI-based scaling but not for squeezing-dynamics studies.

Once the full parameter sweeps are run, the Success Criteria table will reveal which of the five hypotheses are supported and whether the predicted scaling exponents hold within the stated tolerances.

#### 🔍 Open Items

**Numerical verification**: The predicted exponents $\alpha = -2/3$ for OAT and $\alpha = -1.0$ for balanced Dicke superposition/TNT are analytically motivated and should be confirmed via log-log regression over $N \in \{2, \dots, 128\}$.

**Noise robustness**: To what extent does phase diffusion $\gamma_\phi$ degrade the scaling exponent for each model? Hypothesis 5 predicts monotonic degradation for entanglement-enhanced states but robustness for coherent states.

**Ancilla overhead**: The QFI bound for ancilla-assisted protocols does not account for the additional measurement cost of projective ancilla readout. Bayesian estimation with finite $n_{\text{shots}}$ should be compared to the ideal QFI bound to assess practical advantage.

**TNT optimal time**: The code reuses the OAT formula $t_{\text{opt}} = (6/N)^{1/3}$ for TNT. A dedicated $t_{\text{opt}}$ search for TNT dynamics would verify whether the squeezing scaling matches the analytical prediction $\xi^2_{\min} \propto N^{-1}$.
