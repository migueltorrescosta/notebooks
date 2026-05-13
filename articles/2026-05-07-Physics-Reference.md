# PHYSICS.md — Quantum Metrology Simulation Suite

Comprehensive reference for the physical models, operators, and sensitivity algorithms implemented in the MZI simulation codebase.

---

## ⚛️ 1. Hilbert Spaces & Basis Conventions

### 1.1 Two-Mode Fock Space (Interferometer System)

| Property           | Specification                                                          |
| ------------------ | ---------------------------------------------------------------------- |
| **Modes**          | $a$ (mode 0), $b$ (mode 1)                                             |
| **Basis**          | $\vert n_1, n_2\rangle$ with $n_1, n_2 \in \{0, 1, \ldots, N_{\max}\}$ |
| **Dimension**      | $(N_{\max}+1)^2$                                                       |
| **State ordering** | Index = $n_1 \times (N_{\max}+1) + n_2$                                |
| **Truncation**     | Maximum photons per mode = $N_{\max}$                                  |

### 1.2 Dicke Basis (Ancilla / Collective Spins)

| Property                   | Specification                                            |
| -------------------------- | -------------------------------------------------------- |
| **Total angular momentum** | $J = N/2$ for $N$ particles                              |
| **Basis**                  | $\vert J, m\rangle$ with $m \in \{-J, -J+1, \ldots, J\}$ |
| **Dimension**              | $N + 1 = 2J + 1$                                         |
| **Eigenvalues of $J_z$**   | $m = +J, +J-1, \ldots, -J$ (descending)                  |

### 1.3 Combined System-Ancilla Space

$$
\mathcal{H}_{\text{total}} = \mathcal{H}_{\text{sys}} \otimes \mathcal{H}_{\text{anc}}
$$

Dimension: $(N_{\max}+1)^2 \times (2J+1)$

---

## 2. Input States

### 2.1 Available State Types

| State             | Equation                                                                                                                               | Scaling                         | Code Function               |
| ----------------- |----------------------------------------------------------------------------------------------------------------------------------------| ------------------------------- | --------------------------- |
| **Vacuum**        | $\vert 0,0\rangle$                                                                                                                     | —                               | `vacuum_state()`            |
| **Fock**          | $\vert n,0\rangle$ or $\vert 0,n\rangle$                                                                                               | —                               | `fock_state(n1, n2)`        |
| **Single photon** | $\vert 1,0\rangle$ or $\vert 0,1\rangle$                                                                                               | —                               | `single_photon_state(mode)` |
| **NOON**          | $\frac{\vert N,0\rangle + \vert 0,N\rangle}{\sqrt{2}}$                                                                                 | $\Delta\phi \propto 1/N$        | `noon_state(N)`             |
| **Coherent**      | $\vert \alpha\rangle = e^{-                                 \vert \alpha\vert ^2/2} \sum_n \frac{\alpha^n}{\sqrt{n!}}\vert n,0\rangle$ | $\Delta\phi \propto 1/\sqrt{N}$ | `coherent_state(alpha)`     |
| **CSS/GHZ**       | $\frac{\vert 0\ldots0\rangle + \vert 1\ldots1\rangle}{\sqrt{2}}$                                                                       | $\Delta\phi \propto 1/\sqrt{N}$ | `generate_css_state(N)`     |

### 2.2 State Preparation

```python
prepare_input_state(
    state_type: "vacuum" | "single_photon" | "coherent" | "fock" | "noon",
    max_photons: int,
    n_particles: int = 1,
    alpha: complex = 1.0,
    mode: int = 0
) -> np.ndarray
```

---

## ⚛️ 3. Mach-Zehnder Interferometer Evolution

### 3.1 Circuit Pipeline

```
|ψ_in⟩ → BS₁ → Phase Shift → [Ancilla Coupling] → BS₂ → |ψ_out⟩
```

### 3.2 Unitary Transformations

#### Beam Splitter (BS)

Transformation on mode operators:
$$
a \rightarrow \cos\theta \cdot a + i e^{i\phi_{\text{bs}}} \sin\theta \cdot b
$$
$$
b \rightarrow \cos\theta \cdot b - i e^{-i\phi_{\text{bs}}} \sin\theta \cdot a
$$

Unitary in Fock basis:
$$
U_{\text{BS}}(\theta, \phi_{\text{bs}}) = \exp\left[i\left(\theta(a^\dagger b e^{-i\phi_{\text{bs}}} + b^\dagger a e^{i\phi_{\text{bs}}}) + \text{h.c.}\right)\right]
$$

Matrix elements computed via binomial expansion:
$$
\langle n_1', n_2'\vert{}U_{\text{BS}}\vert{}n_1, n_2\rangle = \sum_{m=0}^{n_1} \sum_{k=0}^{n_2} \binom{n_1}{m}\binom{n_2}{k} \cos^{n_1+n_2-m-k}(\theta) \sin^{m+k}(\theta) i^{m+k} e^{ik\phi_{\text{bs}}}
$$

**50/50 beam splitter**: $\theta = \pi/4$

#### Phase Shift

Applied to mode 1 (second arm):
$$
U_{\phi} = \exp(i\phi \cdot n_1) = \text{diag}\left(e^{i\phi n_2}\right)
$$

#### System-Ancilla Coupling

Two coupling types available:

**Phase coupling** (parameter estimation):
$$
H_{\text{int}} = g \cdot n_{\text{photon}} \otimes J_z^{\text{(anc)}}
$$
$$
U_{\text{int}} = \exp(-i H_{\text{int}} t) = \exp\left(-ig t \cdot (a^\dagger a) \otimes J_z\right)
$$

**Flip-flop coupling** (entanglement generation):
$$
H_{\text{int}} = g \cdot (a + a^\dagger) \otimes J_x^{\text{(anc)}}
$$
$$
U_{\text{int}} = \exp\left(-ig t \cdot (a + a^\dagger) \otimes J_x\right)
$$

### 3.3 Evolution Function

```python
evolve_mzi(
    initial_system_state, theta, phi_bs, phi_phase, g, interaction_time, coupling_type, max_photons, ancilla_dim
) -> np.ndarray  # Shape: (sys_dim * anc_dim,)
```

---

## 4. Operators

### 4.1 Two-Mode Fock Operators

**Annihilation operators:**
$$
a_0\vert{}n_1, n_2\rangle = \sqrt{n_1}\vert{}n_1-1, n_2\rangle
$$
$$
a_1\vert{}n_1, n_2\rangle = \sqrt{n_2}\vert{}n_1, n_2-1\rangle
$$

**Number operators:**
$$
n_0 = a_0^\dagger a_0, \quad n_1 = a_1^\dagger a_1
$$

**Population imbalance (J_z in two-mode mapping):**
$$
J_z = \frac{n_0 - n_1}{2}
$$

### 4.2 Angular Momentum Operators (Dicke Basis)

**J_z (diagonal):**
$$
J_z\vert{}J,m\rangle = m\vert{}J,m\rangle, \quad m \in \{-J, -J+1, \ldots, J\}
$$

**J_x (off-diagonal):**
$$
J_+\vert{}J,m\rangle = \sqrt{J(J+1) - m(m+1)}\vert{}J,m+1\rangle
$$
$$
J_-\vert{}J,m\rangle = \sqrt{J(J+1) - m(m-1)}\vert{}J,m-1\rangle
$$
$$
J_x = \frac{J_+ + J_-}{2}
$$

Matrix elements:
$$
\langle J,m'\vert{}J_x\vert{}J,m\rangle = \frac{1}{2}\sqrt{J(J+1) - m(m+1)}\delta_{m',m+1} + \frac{1}{2}\sqrt{J(J+1) - m(m-1)}\delta_{m',m-1}
$$

---

## 📊 5. Sensitivity Metrics

### 5.1 Error Propagation

Based on input-output relation and output variance:

$$
\Delta\phi_{\text{EP}} = \frac{\sigma_{J_z}}{\left\vert{}\frac{\partial \langle J_z\rangle}{\partial \phi}\right\vert{}}
$$

where:
- $\sigma_{J_z} = \sqrt{\langle J_z^2\rangle - \langle J_z\rangle^2}$ is the output variance
- $\frac{\partial\langle J_z\rangle}{\partial\phi}$ is computed via central differences

**Implementation:** `error_propagation_sensitivity()` in `sensitivity_metrics.py`

### 5.2 Classical Fisher Information (CFI)

For measurement outcomes $m$ with probabilities $P(m\vert{}\phi)$:

$$
F_C(\phi) = \sum_m \frac{\left(\frac{\partial P(m\vert{}\phi)}{\partial \phi}\right)^2}{P(m\vert{}\phi)}
$$

Cramér-Rao bound:
$$
\Delta\phi_C = \frac{1}{\sqrt{F_C(\phi)}}
$$

**Numerical derivative (central difference):**
$$
\frac{\partial P}{\partial\phi} \approx \frac{P(\phi+\delta\phi/2) - P(\phi-\delta\phi/2)}{\delta\phi}
$$

**Implementation:** `classical_fisher_information()` in `fisher_information.py`

### 5.3 Quantum Fisher Information (QFI)

For **pure states** $\vert{}\psi(\phi)\rangle$ with generator $G$:

$$
F_Q = 4 \cdot \text{Var}(G) = 4\left(\langle G^2\rangle - \langle G\rangle^2\right)
$$

where $G = J_z$ (phase generator for MZI).

For **mixed states** $\rho(\phi)$:

$$
F_Q = 2\sum_{i<j} \frac{(\lambda_i - \lambda_j)^2}{\lambda_i + \lambda_j}\vert{}\langle i\vert{}G\vert{}j\rangle\vert{}^2 + \sum_i 4\lambda_i \vert{}\Delta G_{ii}\vert{}^2
$$

where $\rho = \sum_i \lambda_i \vert{}i\rangle\langle i\rangle$ and $\Delta G_{ii} = \langle i\vert{}G\vert{}i\rangle - \text{Tr}(\rho G)$.

**Ultimate bound:**
$$
\Delta\phi_Q = \frac{1}{\sqrt{F_Q}}
$$

**Implementation:** `quantum_fisher_information()` and `quantum_fisher_information_dm()` in `fisher_information.py`

### 5.4 Bayesian Sensitivity

Posterior distribution via Bayes' rule:

$$
P(\phi\vert{}m_0) = \frac{P(m_0\vert{}\phi) \cdot \pi(\phi)}{P(m_0)}
$$

with uniform prior $\pi(\phi) = 1/(2\pi)$ on $[0, 2\pi)$.

**Sensitivity (posterior standard deviation):**

Linear approximation:
$$
\Delta\phi_B = \sqrt{\langle\phi^2\rangle - \langle\phi\rangle^2}
$$

Circular statistics (for phase wrap-around):
$$
\Delta\phi_{\text{circ}} = \sqrt{-2\ln\vert{}\langle e^{i\phi}\rangle\vert{}}
$$

where $\langle e^{i\phi}\rangle = \sum_\phi P(\phi\vert{}m_0) e^{i\phi}$.

**Implementation:** `bayesian_estimator()` in `bayesian_phase_estimation.py`

### 5.5 Comparison of Methods

| Method | Formula | Best For | Computational Cost |
|--------|----------|----------|---------------------|
| Error Propagation | $\Delta\phi_{\text{EP}} = \sigma/\vert\partial\langle O\rangle/\partial\phi\vert$ | Quick estimates | Low |
| Classical Fisher | $\Delta\phi_C = 1/\sqrt{F_C}$ | Optimized measurements | Medium |
| Quantum Fisher | $\Delta\phi_Q = 1/\sqrt{F_Q}$ | Theoretical bounds | Medium (pure) / High (mixed) |
| Bayesian | $\Delta\phi_B = \text{Std}[\phi\vert m_0]$ | Finite samples, prior info | High |

**Inequality chain:**
$$
\Delta\phi_Q \leq \Delta\phi_C \leq \Delta\phi_{\text{EP}}
$$

---

## ⚠️ 6. Noise Channels (Open Quantum Systems)

### 6.1 Lindblad Master Equation

$$
\frac{d\rho}{dt} = -i[H,\rho] + \sum_k \left(L_k\rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\}\right)
$$

### 6.2 Available Noise Channels

| Channel | Lindblad Operator $L_k$ | Physical Rate | Effect |
|---------|--------------------------|--------------|--------|
| **One-body loss** | $L = \sqrt{\gamma_1} \cdot a$ | $\gamma_1$ (s⁻¹) | Single particle decay |
| **Two-body loss** | $L = \sqrt{\gamma_2} \cdot a^2$ | $\gamma_2$ (s⁻¹ per pair) | Pairwise loss $\propto N(N-1)$ |
| **Phase diffusion** | $L = \sqrt{\gamma_\phi} \cdot J_z$ | $\gamma_\phi$ (s⁻¹) | Dephasing between $m$ states |
| **Detection noise** | Binomial$(k; n, \eta)$ | $\eta \in [0,1]$ | Imperfect detection efficiency |

### 6.3 Noise Configuration

```python
@dataclass
class NoiseConfig:
    gamma_1: float = 0.0    # One-body loss rate
    gamma_2: float = 0.0    # Two-body loss rate  
    gamma_phi: float = 0.0  # Phase diffusion rate
    eta: float = 1.0        # Detection efficiency
```

### 6.4 Detection Noise Model

For $n$ actual particles, detected count $k$ follows binomial distribution:

$$
P(k\vert{}n, \eta) = \binom{n}{k} \eta^k (1-\eta)^{n-k}
$$

**Special cases:**
- $\eta = 1$: Perfect detection, $P(k\vert{}n) = \delta_{k,n}$
- $\eta = 0$: No detection, uniform over possible outcomes

**Implementation:** `apply_detection_noise()` and `detection_channel_pmf()` in `noise_channels.py`

---

## 7. System-Ancilla Rabi Model

### 7.1 Hamiltonian

For ancilla level $k$ with dimension $N$ (ancilla dimension):

$$
H = (-J_S \sigma_x + \delta_S \sigma_z) + \alpha_x \sigma_x J_z + \alpha_z \sigma_z J_z
$$

where:
- $J_S$: System tunneling strength
- $\delta_S$: System energy shift
- $\alpha_x, \alpha_z$: Coupling coefficients
- $J_z$ acts on state $\vert{}J,m\rangle$ with $m = (N-2k)/2$

### 7.2 Rabi Frequency

For ancilla level $k$:

$$
\omega_k = \sqrt{\left(\alpha_z \frac{N-2k}{2} + \delta_S\right)^2 + \left(\alpha_x \frac{N-2k}{2} - J_S\right)^2}
$$

### 7.3 Observable: $\langle\sigma_z\rangle(t)$

$$
\langle\sigma_z\rangle(t) \approx \cos^2(\omega_k t) + \sin^2(\omega_k t) \cdot \left(\frac{z_{\text{coeff}}}{\omega_k}\right)^2 - \sin^2(\omega_k t) \cdot \left(\frac{x_{\text{coeff}}}{\omega_k}\right)
$$

where:
- $x_{\text{coeff}} = \alpha_x(N-2k)/2 - J_S$
- $z_{\text{coeff}} = \alpha_z(N-2k)/2 + \delta_S$

### 7.4 Sensitivities

$$
\frac{\partial\langle\sigma_z\rangle}{\partial J_S} = \sin^2(\omega_k t) \cdot \frac{\alpha_x \cdot x_{\text{coeff}}}{\omega_k^2}
$$

$$
\frac{\partial\langle\sigma_z\rangle}{\partial \delta_S} = \sin^2(\omega_k t) \cdot \frac{\alpha_z \cdot z_{\text{coeff}}}{\omega_k^2}
$$

**Implementation:** `compute_rabi_frequency()` and `sensitivity()` in `sensitivity_analysis.py`

---

## 8. Scaling Laws

### 8.1 Standard Quantum Limit (SQL)

For classical states (coherent, CSS, twin-Fock):

$$
\Delta\phi_{\text{SQL}} \propto \frac{1}{\sqrt{N}}
$$

Quantum Fisher Information: $F_Q = N$

### 8.2 Heisenberg Limit (HL)

For maximally entangled states (NOON):

$$
\Delta\phi_{\text{HL}} \propto \frac{1}{N}
$$

Quantum Fisher Information: $F_Q = N^2$

### 8.3 Scaling Summary

| State Type | $F_Q$ | $\Delta\phi$ | Scaling Exponent $\alpha$ in $\Delta\phi \propto N^\alpha$ |
|------------|---------|---------------|----------------------------------------------------------|
| Coherent / CSS | $N$ | $1/\sqrt{N}$ | $\alpha = -0.5$ |
| Twin-Fock | $N$ | $1/\sqrt{N}$ | $\alpha = -0.5$ |
| NOON | $N^2$ | $1/N$ | $\alpha = -1.0$ |

### 8.4 Scaling Analysis Function

```python
sensitivity_scaling(
    state_type: "css" | "noon" | "twin_fock" | "single",
    N_range: np.ndarray,
    noise_config: NoiseConfig | None = None,
    phi_true: float = π/4,
    n_mc: int = 200,
    rng_seed: int = 42
) -> SensitivityScalingResult
```

**Log-log fit:**
$$
\log(\Delta\phi) = \alpha \cdot \log(N) + \log(C)
$$
$$
\alpha = \text{np.polyfit}(\log(N), \log(\Delta\phi), 1)[0]
$$

---

## 💻 9. Numerical Methods

### 9.1 Unitary Evolution

**Method:** `scipy.linalg.expm()` for matrix exponentiation

$$
U = \exp(-iHt)
$$

**Avoid:** Direct matrix inversion (`np.linalg.inv`) — use `np.linalg.solve` for linear systems.

### 9.2 Finite Differences

**Central difference (interior points):**
$$
f'(x_i) \approx \frac{f(x_{i+1}) - f(x_{i-1})}{2\Delta x}
$$

**Forward/backward difference (boundaries):**
$$
f'(x_0) \approx \frac{f(x_1) - f(x_0)}{\Delta x}
$$

### 9.3 Random Number Generation

All stochastic processes use `numpy.random.default_rng(seed)` for reproducibility.

**Default behavior:** Deterministic (fixed fallback seed).

### 9.4 Performance Constraints

| Metric | Constraint |
|--------|-------------|
| Individual simulation | $< 100$ ms |
| State dimension growth | Polynomial in $N_{\max}$ |
| Memory | Tensor methods for large Hilbert spaces |

---

## 📐 10. Physical Invariants & Validation

### 10.1 Conservation Laws

**Probability conservation:**
```python
assert np.isclose(np.sum(probabilities), 1.0), "Probability must be conserved"
```

**Unitarity:**
```python
assert np.allclose(U @ U.conj().T, np.eye(n)), "Operator must be unitary"
```

**Density matrix trace:**
$$
\text{Tr}(\rho) = 1
$$

### 10.2 Inequality Constraints

**Quantum vs Classical Fisher:**
$$
F_Q \geq F_C
$$

**Sensitivity bounds:**
$$
\Delta\phi_Q \leq \Delta\phi_C \leq \Delta\phi_{\text{EP}}
$$

**Lindblad completeness:**
$$
\sum_k L_k^\dagger L_k \leq \mathbb{1}
$$

---

## 📝 11. Quick Reference: Key Equations

| Concept | Equation |
|---------|----------|
| **NOON state** | $\vert\text{NOON}\rangle = \frac{\vert N,0\rangle + \vert0,N\rangle}{\sqrt{2}}$ |
| **QFI (pure)** | $F_Q = 4\text{Var}(G) = 4(\langle G^2\rangle - \langle G\rangle^2)$ |
| **QFI (mixed)** | $F_Q = 2\sum_{i<j} \frac{(\lambda_i-\lambda_j)^2}{\lambda_i+\lambda_j}\vert\langle i\vert G\vert j\rangle\vert^2 + \sum_i 4\lambda_i\vert\Delta G_{ii}\vert^2$ |
| **Cramér-Rao** | $\Delta\phi \geq \frac{1}{\sqrt{F}}$ |
| **Error propagation** | $\Delta\phi_{\text{EP}} = \frac{\sigma_O}{\vert d\langle O\rangle/d\phi\vert}$ |
| **Bayesian (circular)** | $\Delta\phi_B = \sqrt{-2\ln\vert\langle e^{i\phi}\rangle\vert}$ |
| **Heisenberg limit** | $\Delta\phi_{\text{HL}} = \frac{1}{N}$ |
| **SQL** | $\Delta\phi_{\text{SQL}} = \frac{1}{\sqrt{N}}$ |
| **Lindblad eq.** | $\dot{\rho} = -i[H,\rho] + \sum_k (L_k\rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k,\rho\})$ |

---

## 12. File Organization

| Module | Purpose |
|--------|---------|
| `src/physics/mzi_simulation.py` | MZI evolution, state preparation, unitaries |
| `src/physics/noise_channels.py` | Noise models, Lindblad operators |
| `src/physics/angular_momentum.py` | Spin operators $J_x, J_z$ |
| `src/analysis/sensitivity_metrics.py` | All sensitivity methods, scaling analysis |
| `src/analysis/fisher_information.py` | Classical & quantum Fisher information |
| `src/analysis/bayesian_phase_estimation.py` | Bayesian inference, posterior computation |
| `src/analysis/sensitivity_analysis.py` | System-ancilla Rabi model, observables |
| `src/algorithms/spin_squeezing.py` | Spin squeezing for enhanced sensitivity |
| `src/evolution/lindblad_solver.py` | Master equation integration |

---

*Last updated: 2026-04-30*
