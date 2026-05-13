# Scaling Survey: Dicke Basis — Collective Spin & Entanglement-Enhanced Interferometry

## 🧪 Hypothesis

For interferometers using **collective spin states in the symmetric Dicke subspace**:

1. Spin-squeezed states (OAT) achieve $\alpha = -2/3$ at optimal squeezing time $t_{\text{opt}} \propto N^{-1/3}$, interpolating between SQL and HL
2. Twin-Fock states achieve $\alpha = -1.0$ (Heisenberg scaling) via number-difference readout
3. Non-Gaussian states (n=3,4 hybrid squeezing) achieve fractional exponents $\alpha \approx -0.75$ to $-0.85$ at fixed $\langle n\rangle$
4. Ancilla-assisted metrology preserves the state's scaling exponent $\alpha$ but improves the prefactor $C$ by $\mathcal{R}(T)^{-1/2}$ under non-Markovian noise
5. All entanglement-enhanced models are more fragile than coherent states under phase diffusion ($\gamma_\phi$): the advantage degrades monotonically with $\gamma_\phi$



## ⚛️ Theoretical Model

### Hilbert Space

Symmetric Dicke subspace $\mathcal{H}_J = \text{span}\{\vert J, m\rangle\}$ with total spin $J = N/2$ and $m \in \{-J, -J+1, \dots, J\}$. Dimension $N+1$. Collective operators:

$$J_x = \frac12(a_1^\dagger a_2 + a_2^\dagger a_1),\quad
J_y = \frac{i}{2}(a_2^\dagger a_1 - a_1^\dagger a_2),\quad
J_z = \frac12(n_1 - n_2)$$

### Entanglement Hamiltonians

| Name | Hamiltonian | Unitary | Effect |
|---|---|---|---|
| One-axis twisting (OAT) | $H = \chi J_z^2$ | $e^{-i\chi t J_z^2}$ | CSS → SSS, $\xi^2_{\min} \propto N^{-2/3}$ at $t_{\text{opt}} = (6/N)^{1/3}/\chi$ |
| Two-axis countertwisting (TNT) | $H = \chi(J_+^2 + J_-^2)/2$ | $e^{-i\chi t (J_+^2+J_-^2)/2}$ | CSS → Heisenberg-limited squeezing |

The OAT optimal squeezing time scales as $t_{\text{opt}} \propto N^{-1/3}$ (not $N^{-2/3}$), yielding minimum squeezing parameter $\xi^2_{\min} \propto N^{-2/3}$ and sensitivity exponent $\alpha = -2/3$. Implementation: `spin_squeezing.py::generate_squeezed_state`, `optimal_squeezing_time`.

### Extended Models

| Model | Extended Hilbert Space | Coupling | Code |
|---|---|---|---|---|
| Non-Gaussian (n=3,4) | $\mathcal{H}_J \otimes \mathcal{H}_{\text{osc}}$ (hybrid oscillator-spin) | $H_n = \frac{\Omega_n}{2}\, \sigma_{\text{op}} \otimes (a^n e^{-i\theta_n} + a^{\dagger n} e^{i\theta_n})$ | `hybrid_system.py::hybrid_hamiltonian_n` |
| Ancilla-assisted (two-mode + ancilla MZI) | $\mathcal{H}_{\text{Fock}} \otimes \mathbb{C}^{d_A}$ (two-mode system + spin-J ancilla) | $H_{\text{int}} = g\, n_{\text{photon}} \otimes J_z$ (phase coupling) or $g\,(a+a^\dagger) \otimes J_x$ (flip-flop) | `mzi_simulation.py::system_ancilla_interaction_unitary` |
| Ancilla-assisted (non-Markovian) | $\mathcal{H}_{\text{osc}} \otimes \mathbb{C}^2 \otimes \mathcal{H}_{\text{pm}}$ (probe + spin ancilla + pseudomode bath) | $H_{\text{sa}} = g_{\text{sa}}\, a^\dagger a \otimes \sigma_x \otimes \mathbb{1}_{\text{pm}}$ | `pseudomode_system.py::build_pseudomode_hamiltonian` |

### Input States

| State | Description | Code |
|---|---|---|
| CSS (coherent spin) | $\vert J, -J\rangle$ rotated to $x$-axis | `mzi_states.py::input_state_factory` with `state_type="css"` |
| SSS (spin-squeezed) | CSS after OAT evolution at $t_{\text{opt}}$ | `spin_squeezing.py::generate_squeezed_state` |
| Twin-Fock | $\vert N/2, 0\rangle$ in Dicke basis (must have even $N$) | `mzi_states.py::twin_fock_state` |
| Non-Gaussian | Hybrid oscillator-spin state time-evolved from vacuum under $H_n$ | `hybrid_system.py::hybrid_hamiltonian_n` + unitary evolution |
| Ancilla + probe (MZI) | $\vert\text{system}\rangle \otimes \vert 0 \ldots 0\rangle_{\text{ancilla}}$ | `mzi_simulation.py::evolve_mzi` |
| Ancilla + probe (non-Markovian) | $\vert\alpha\rangle_{\text{osc}} \otimes \vert\downarrow\rangle_{\text{spin}} \otimes \vert0\rangle_{\text{pm}}$ | `pseudomode_system.py::pseudomode_initial_state` |

### Sensitivity Metrics

| Method | Formula |
|---|---|
| Error propagation | $\Delta\phi_{\text{EP}} = \Delta J_z / \lvert \partial\langle J_z\rangle/\partial\phi\rvert$ |
| Quantum Fisher (pure) | $\Delta\phi_{\text{QFI}} = 1/\sqrt{F_Q}$, $F_Q = 4\,\text{Var}(G)$ where $G = J_z$ |
| Quantum Fisher (mixed) | $\Delta\phi_{\text{QFI}} = 1/\sqrt{F_Q}$ via SLD eigen-decomposition |
| Classical Fisher | $\Delta\phi_{\text{CFI}} = 1/\sqrt{\sum_k (\partial_\phi p_k)^2 / p_k}$ |

---

## 📊 Models Survey

| Model | Input | Entanglement | Expected $\alpha$ | Implementation |
|---|---|---|---|---|
| OAT spin-squeezed | CSS | $\chi J_z^2$ at $t_{\text{opt}}$ | $-2/3$, $C \approx (2^{1/3}/3^{1/6})\chi^{-1/3}$ | `spin_squeezing.py` ✅ |
| Twin-Fock | $\vert N/2,0\rangle$ (even $N$) | None (inherent) | $-1.0$; degrades to $-0.5$ with loss | `mzi_states.py::twin_fock_state` ✅ |
| Two-axis CT (TNT) | CSS | $\chi(J_+^2+J_-^2)/2$ | $-1.0$ (Heisenberg-limited squeezing) | `src/analysis/scaling_survey.py::_apply_entanglement` ✅ |
| Non-Gaussian n=3 | Vacuum $\otimes \vert\downarrow\rangle$ evolved under $H_{n=3}$ | 3rd-order coupling | $\approx -0.75$, $C \approx 1.5$ | `hybrid_system.py::hybrid_hamiltonian_n` + `hybrid_mzi.py::qfi_hybrid_mzi` ✅ (via `src/analysis/scaling_survey.py::_non_gaussian_sensitivity_fn`) |
| Non-Gaussian n=4 | Vacuum $\otimes \vert\downarrow\rangle$ evolved under $H_{n=4}$ | 4th-order coupling | $\approx -0.85$, $C \approx 2.0$ | `hybrid_system.py::hybrid_hamiltonian_n` + `hybrid_mzi.py::qfi_hybrid_mzi` ✅ (via `src/analysis/scaling_survey.py::_non_gaussian_sensitivity_fn`) |
| Ancilla-assisted | Probe $\otimes\vert\downarrow\rangle$ or $\vert\alpha\rangle\otimes\vert\downarrow\rangle\otimes\vert0\rangle_{\text{pm}}$ | $g_{\text{sa}}\, n \otimes \sigma_x$ | Same $\alpha$ as probe; $C$ improved by $\mathcal{R}(T)^{-1/2}$ | `pseudomode_system.py` or `mzi_simulation.py` ✅ (via `src/analysis/scaling_survey.py::_ancilla_sensitivity_fn`) |

---

## 💻 Numerical Simulation

### Implementation Strategy

1. **Dicke-basis propagation** (OAT/TNT): ODE for $e^{-iHt}\vert\psi_0\rangle$ in $N+1$ dimensions (unitary) or Lindblad master equation (mixed)
2. **Optimal time search**: For OAT, sweep $t$, compute $\xi^2_R(t)$, pick $t_{\text{opt}}$ per $N$, then compute sensitivity at $t_{\text{opt}}$
3. **Hybrid systems**: Extended Hilbert space dimension $2(N+1)$ for non-Gaussian (oscillator $\otimes$ spin), $2(N+1)(K+1)$ for ancilla + pseudomode
4. **Phase imprint**: $e^{i\phi n_2}$ in the two-mode Fock basis (diagonal: $e^{i\phi n_2}$), applied after the first beam splitter

### Parameter Sweep

- Resource sweep: $N \in \{2, 4, 8, 16, 32, 64, 128\}$ (Dicke dimension $N+1$ allows larger $N$ than Fock)
- Noise sweep: $\gamma_\phi \in \{0, 10^{-3}, 10^{-2}, 10^{-1}, 1\}$ (phase diffusion primary decoherence)
- For OAT: sweep $t/\chi^{-1}$ to find $t_{\text{opt}}$ per $N$, verify $\xi^2_{\min} \propto N^{-2/3}$
- Scaling exponent via log-log regression: $\log\Delta\phi = \alpha\log N + \log C$

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

## ⚠️ Likely Failure Conditions

| Failure | Description | Mitigation |
|---------|-------------|------------|
| Dicke basis limitation for hybrid models | Adding oscillator or ancilla increases dimension to $(N+1) \times d_{\text{ext}}$. For $N > 100$ and $d_{\text{ext}} > 10$, this becomes expensive. | Restrict hybrid models to $N \leq 30$. |
| Optimal time degeneracy | The sensitivity $\Delta\phi(t)$ can have multiple local minima at different $t$ for the same $N$. | Sweep $t$ with fine grid and report global minimum. |
| Non-Gaussian state preparation | The time-evolved state (or true ground state via `hybrid_system.py::hybrid_ground_state_n`) may not be the optimal metrological state at finite $N$. | Report QFI of multiple state preparations (vacuum-evolved, ground state) and select the best. |
| Ancilla readout overhead | The ancilla-assisted protocol requires projective measurements on the ancilla, which adds statistical overhead not captured by QFI alone. | Report Bayesian sensitivity $n_{\text{shots}} \to \Delta\phi$ alongside QFI bound. |

---

## ✅ Success Criteria

| # | Check | Expectation |
|---|---|---|
| 1 | **OAT squeezing scaling** | $\xi^2_{\min} \propto N^{-2/3}$; fit exponent $-0.67 \pm 0.05$; $t_{\text{opt}} \propto N^{-1/3}$ within 10% of $(6/N)^{1/3}/\chi$ |
| 2 | **OAT sensitivity exponent** | $\alpha = -2/3 \pm 0.03$ from log-log fit of $\Delta\phi(N)$ at $t_{\text{opt}}$; $R^2 \geq 0.95$ |
| 3 | **Twin-Fock Heisenberg scaling** | $\Delta\phi = 1/\sqrt{F_Q}$ with $F_Q = N(N+2)/3$; measured $\alpha = -1.0 \pm 0.03$ in ideal case; $\alpha \to -0.5$ under one-body loss $\gamma_1 > 0$ |
| 4 | **Two-axis countertwisting (TNT)** | $\xi^2_{\min} \propto N^{-1}$ (Heisenberg-limited); $\alpha = -1.0 \pm 0.05$ via QFI |
| 5 | **Non-Gaussian n=3 scaling** | Measured $\alpha = -0.75 \pm 0.05$ at fixed $\langle n \rangle$; $C \approx 1.5$; $R^2 \geq 0.90$ |
| 6 | **Non-Gaussian n=4 scaling** | Measured $\alpha = -0.85 \pm 0.05$ at fixed $\langle n \rangle$; $C \approx 2.0$; $R^2 \geq 0.90$ |
| 7 | **Ancilla-assisted: prefactor preservation** | $\alpha_{\text{ancilla}} = \alpha_{\text{probe}} \pm 0.02$ (scaling exponent unchanged); prefactor $C_{\text{ancilla}} = C_{\text{probe}} / \sqrt{\mathcal{R}(T)}$ within 5% |
| 8 | **Phase-diffusion degradation** | Entanglement-enhanced states (OAT, Twin-Fock) show $\alpha$ degrading monotonically with $\gamma_\phi$; coherent-state $\alpha$ stays at $-0.5 \pm 0.02$ for all $\gamma_\phi$ |
| 9 | **Quantum state invariants** | $\Tr(\rho) = 1.0 \pm 10^{-10}$; $\rho = \rho^\dagger$ (Hermitian to $10^{-10}$); $\min \text{eigvals}(\rho) \geq -10^{-8}$ throughout evolution |
| 10 | **QFI upper bound** | $F_Q \leq 4N^2$ for pure symmetric states under $J_z$ generator; verified within floating-point tolerance |
| 11 | **Log-log fit quality (all models)** | $R^2 \geq 0.9$ for all scaling sweeps with $\geq 5$ $N$-points; outliers flagged when fit residuals exceed $3\sigma$ |
| 12 | **NOON vs Twin-Fock comparison under loss** | Both states collapse to $\alpha = -0.5$ under $\gamma_1 > 0$; prefactor $C$ differs by at most a constant factor set by the initial $F_Q$ ratio |

---

## 🔬 Results and Next Steps

| # | Check | Status |
|---|---|---|
| 1 | **OAT squeezing scaling** | ⏳ |
| 2 | **OAT sensitivity exponent** | ⏳ |
| 3 | **Twin-Fock Heisenberg scaling** | ⏳ |
| 4 | **Two-axis countertwisting (TNT)** | ⏳ |
| 5 | **Non-Gaussian n=3 scaling** | ⏳ |
| 6 | **Non-Gaussian n=4 scaling** | ⏳ |
| 7 | **Ancilla-assisted: prefactor preservation** | ⏳ |
| 8 | **Phase-diffusion degradation** | ⏳ |
| 9 | **Quantum state invariants** | ⏳ |
| 10 | **QFI upper bound** | ⏳ |
| 11 | **Log-log fit quality (all models)** | ⏳ |
| 12 | **NOON vs Twin-Fock comparison under loss** | ⏳ |
