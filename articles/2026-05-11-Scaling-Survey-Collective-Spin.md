# Scaling Survey: Dicke Basis — Collective Spin & Entanglement-Enhanced Interferometry

## 🧪 Hypothesis

For interferometers using **collective spin states in the symmetric Dicke subspace**:

1. Spin-squeezed states (OAT) achieve $\alpha = -2/3$ at optimal squeezing time $t_{\text{opt}} \propto N^{-1/3}$, interpolating between SQL and HL
2. Twin-Fock states achieve $\alpha = -1.0$ (Heisenberg scaling) via number-difference readout
3. Non-Gaussian states (n=3,4 hybrid squeezing) achieve fractional exponents $\alpha \approx -0.75$ to $-0.85$ at fixed $\langle n\rangle$
4. Ancilla-assisted metrology preserves the state's scaling exponent $\alpha$ but improves the prefactor $C$ by $\mathcal{R}(T)^{-1/2}$ under non-Markovian noise
5. All entanglement-enhanced models are more fragile than coherent states under phase diffusion ($\gamma_\phi$): the advantage degrades monotonically with $\gamma_\phi$

---

## 📖 Literature Review

| Concept & Motivation | Article | Year |
|---|---|---|
| Quantum limits for interferometry: QCRB, SQL, Heisenberg limit; foundations for all scaling calculations | *Sensitivity of Quantum-Enhanced Interferometers* (review) | — |
| High-order non-Gaussian squeezing under Markovian decoherence: hybrid Lindblad framework; QFI decoherence sweeps at fixed $\langle n \rangle$ for n=2,3,4 | `articles/2026-05-07-High-Order-Squeezing-Plan.md` (prior repo) | 2026 |
| Ancilla-assisted metrology in non-Markovian environments: pseudomode-based non-Markovian QFI; $\mathcal{R}(T)$ preservation ratio scaling | `articles/2026-05-09-Ancilla-Assisted-Metrology-Non-Markovian.md` (prior repo) | 2026 |
| OAT squeezing theory: Kitagawa & Ueda (1993) — optimal squeezing time $t_{\text{opt}} \propto N^{-1/3}$, minimum squeezing parameter $\xi^2_{\min} \propto N^{-2/3}$ | Kitagawa & Ueda, *Phys. Rev. A* 47, 5138 | 1993 |
| Two-axis countertwisting: $H \propto J_+^2 + J_-^2$ achieves $\xi^2 \propto N^{-1}$ (Heisenberg-limited squeezing) | Kitagawa & Ueda, *Phys. Rev. A* 47, 5138 | 1993 |

---

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
| Ancilla-assisted (non-Markovian) | $\mathcal{H}_{\text{osc}} \otimes \mathbb{C}^2 \otimes \mathcal{H}_{\text{pm}}$ (probe + spin ancilla + pseudomode bath) | $H_{\text{sa}} = g_{\text{sa}}\, a^\dagger a \otimes \sigma_x \otimes I_{\text{pm}}$ | `pseudomode_system.py::build_pseudomode_hamiltonian` |

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

1. **Dicke basis limitation for hybrid models**: Adding oscillator or ancilla increases dimension to $(N+1) \times d_{\text{ext}}$. For $N > 100$ and $d_{\text{ext}} > 10$, this becomes expensive. Mitigation: restrict hybrid models to $N \leq 30$.

2. **Optimal time degeneracy**: The sensitivity $\Delta\phi(t)$ can have multiple local minima at different $t$ for the same $N$. Mitigation: sweep $t$ with fine grid and report global minimum.

3. **Non-Gaussian state preparation**: The time-evolved state (or true ground state via `hybrid_system.py::hybrid_ground_state_n`) may not be the optimal metrological state at finite $N$. Mitigation: report QFI of multiple state preparations (vacuum-evolved, ground state) and select the best.

4. **Ancilla readout overhead**: The ancilla-assisted protocol requires projective measurements on the ancilla, which adds statistical overhead not captured by QFI alone. Mitigation: report Bayesian sensitivity $n_{\text{shots}} \to \Delta\phi$ alongside QFI bound.
