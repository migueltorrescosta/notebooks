# Weighted Joint Measurement in Ancilla-Assisted Metrology: Generalization to N System and M Ancilla Particles

## 🧪 Hypothesis

For a system S of N particles and an ancilla A of M particles (each in two-mode bosonic Fock spaces), where only S couples to the unknown phase rate $\theta$ via $H_{\text{enc}} = \theta J_z^S$, and S and A interact via $H_{\text{int}} = \alpha_{xx} J_x^S \otimes J_x^A + \alpha_{xz} J_x^S \otimes J_z^A + \alpha_{zx} J_z^S \otimes J_x^A + \alpha_{zz} J_z^S \otimes J_z^A$ during a holding time $T_H$, the **weighted joint measurement** $M(a,b) = a\,J_z^S + b\,J_z^A$ with $a^2 + b^2 = 1$ (L$_2$-normalised) can exploit S-A correlations to improve sensitivity beyond the S-only measurement.

The central hypotheses are:

1. **Optimal weights unlock enhanced sensitivity.** For any $N \ge 1$, $M \ge 1$, and non-zero interaction, there exists an optimal pair $(a^*, b^*)$ with $a^{*2} + b^{*2} = 1$ that minimises $\Delta\theta$. The equal-weight case $(a,b) = (1/\sqrt{2}, 1/\sqrt{2})$ from previous work is generally suboptimal. The optimal weight ratio $a^*/b^*$ is determined by the relative magnitudes of $\text{Var}(J_z^S)$, $\text{Var}(J_z^A)$, and $\text{Cov}(J_z^S, J_z^A)$, and varies with $\alpha_{ij}$, $N$, $M$, and $T_H$.

2. **N-scaling beyond SQL is accessible via the joint measurement.** With optimised weights and appropriate interaction, the sensitivity scales as $\Delta\theta \propto 1/(N^\nu T_H)$ with $\nu > 1/2$ (beating the SQL scaling $\nu = 1/2$) and approaching $\nu \to 1$ (Heisenberg scaling) in the limit of large $N$ with ideal interaction. This is possible because the weighted joint measurement can extract variance from both S and A while the signal derivative $\partial\langle M\rangle/\partial\theta$ scales with $N$ through $J_z^S$.

3. **Ancilla size M contributes diminishing returns.** For fixed $N$, increasing $M$ beyond $M \sim N$ provides negligible improvement, because $\theta$ couples only to $J_z^S$ (operator norm $N/2$) and the ancilla serves only as a readout-enhancement resource. The interaction strength $\alpha_{ij}$ must be large enough to distribute $\theta$-dependent phase information from S to A; a larger $M$ helps only if the interaction effectively couples S to all $M$ ancilla particles.

4. **The QFI bound remains $\Delta\theta \ge 1/(N T_H)$**, set by the maximal eigenvalue $J_S = N/2$ of the generator $J_z^S$. The ancilla and the weighted measurement cannot surpass this Heisenberg limit, but can saturate it in the ideal interaction regime.

## ⚛️ Theoretical Model

The total Hilbert space is $\mathcal{H}_{\text{tot}} = \mathcal{H}_S \otimes \mathcal{H}_A$, where each subsystem is a **two-mode bosonic Fock space** with fixed particle number. The **system** has $N$ particles, occupying the symmetric Dicke subspace of dimension $d_S = N+1$ with total spin $J_S = N/2$, spanned by $\{\,\vert J_S, m_S\rangle \mid m_S = -J_S, -J_S+1, \dots, J_S\,\}$. The **ancilla** has $M$ particles, with dimension $d_A = M+1$ and total spin $J_A = M/2$, spanned by $\{\,\vert J_A, m_A\rangle \mid m_A = -J_A, \dots, J_A\,\}$. The full space has dimension $d = (N+1)(M+1)$ with ordered tensor-product basis $\{\vert J_S, m_S\rangle \otimes \vert J_A, m_A\rangle\}$.

The **collective angular momentum operators** for each subsystem are obtained from standard SU(2) algebra. For a spin-$J$ system, $J_z$ is diagonal with eigenvalues $m \in \{-J, -J+1, \dots, J\}$, $J_x$ has matrix elements $\langle J, m' \vert J_x \vert J, m \rangle = \tfrac12\sqrt{J(J+1) - m(m\pm 1)}\,\delta_{m', m\pm 1}$, and $J_y$ is defined analogously. These operators satisfy $[J_x, J_z] = i J_y$ and are efficiently constructed via `dicke_basis.jz_operator(N)` and `dicke_basis.jx_operator(N)` from the existing codebase. They are embedded into the full space via Kronecker products: $J_z^S = J_z(N) \otimes \mathbb{1}_{M+1}$, $J_x^S = J_x(N) \otimes \mathbb{1}_{M+1}$, $J_z^A = \mathbb{1}_{N+1} \otimes J_z(M)$, $J_x^A = \mathbb{1}_{N+1} \otimes J_x(M)$.

The **initial state** is a pure product state $\vert\Psi_0\rangle = \vert\psi_S\rangle \otimes \vert\psi_A\rangle$, where each subsystem is restricted to the family of **coherent spin states (CSS)** for the baseline investigation. A CSS is parameterised by two angles $(\Theta, \Phi)$:
$\vert\Theta, \Phi\rangle = \exp(-i\Phi J_z)\,\exp(-i\Theta J_y)\,\vert J, -J\rangle,$
where $\Theta \in [0, \pi]$ and $\Phi \in [0, 2\pi)$. The CSS family has the structure of a generalised Bloch sphere for spin $J$, and reduces to the standard single-qubit parameterisation for $J=1/2$ (i.e., $N=1$). **A gauge freedom** — global phase invariance around $J_z$ before the first beam splitter, combined with the beam splitter's ability to generate any in-plane rotation — allows us to fix $\Phi_S = 0$ and $\Phi_A = 0$ without loss of generality. This reduction from 4 to 2 initial-state degrees of freedom is conservative: it assumes the first beam-splitter rotation $e^{-i T_{\text{BS}1} J_x}$ explores the same reachable set as the full $(\Theta, \Phi)$ parameterisation with a second $J_x$-rotation. If this symmetry turns out to be only approximate (e.g., for Hamiltonians that break the $J_z$-rotation symmetry of the initial state), the full $(\Theta, \Phi)$ parameterisation can be restored.

The **circuit** proceeds in three steps. First, a **beam-splitter unitary** generated by $J_x$ acts independently on S and A: $U_{\text{BS}}(T) = \exp(-i T J_x)$, applied as $U_{\text{BS}}^{(2)}(T) = U_{\text{BS}}(T; N) \otimes U_{\text{BS}}(T; M)$ with independent times $T_{\text{BS}1}$ (before the hold) and $T_{\text{BS}2}$ (after the hold). The $U_{\text{BS}}(T)$ for each subsystem is computed via `scipy.linalg.expm(-1j * T * J_x)`.

Second, during the **holding time** $T_H$, the system evolves under
$H_{\text{hold}} = \theta\,(J_z^S) + H_{\text{int}},$
where $H_{\text{int}} = \alpha_{xx} J_x^S \otimes J_x^A + \alpha_{xz} J_x^S \otimes J_z^A + \alpha_{zx} J_z^S \otimes J_x^A + \alpha_{zz} J_z^S \otimes J_z^A.$ Crucially, **only the system S accumulates phase from $\theta$**; the ancilla experiences no $\theta$-dependent evolution. The hold unitary is $U_{\text{hold}}(T_H) = \exp(-i T_H H_{\text{hold}})$, computed via `torch.linalg.matrix_exp` acting on the $d$-dimensional space (the PyTorch implementation enables automatic differentiation through this operation). Third, the second beam splitter is applied. The complete evolution is
$\vert\Psi_{\text{final}}\rangle = U_{\text{BS}}^{(2)}(T_{\text{BS}2})\, U_{\text{hold}}(T_H)\, U_{\text{BS}}^{(2)}(T_{\text{BS}1})\, (\vert\psi_S\rangle \otimes \vert\psi_A\rangle).$

The **measurement** observable is the weighted sum
$M(a,b) = a\,J_z^S + b\,J_z^A$, $a^2 + b^2 = 1.$
For a pure final state, the expectation value is $\langle M \rangle = a\langle J_z^S\rangle + b\langle J_z^A\rangle$ and the variance is
$\text{Var}(M) = a^2\,\text{Var}(J_z^S) + b^2\,\text{Var}(J_z^A) + 2ab\,\text{Cov}(J_z^S, J_z^A),$
where $\text{Cov}(J_z^S, J_z^A) = \langle J_z^S J_z^A\rangle - \langle J_z^S\rangle\langle J_z^A\rangle$. The **sensitivity** via error propagation is $\Delta\theta = \sqrt{\text{Var}(M)} / \vert\partial\langle M\rangle/\partial\theta\vert$. The weight angle $\phi$ is **not optimised by the outer gradient-based solver**; instead, at each circuit-parameter evaluation, $\phi^*$ is found by a 1D golden-section search over $\phi \in [0, 2\pi)$ that minimises $\Delta\theta(\phi)$ using the six pre-computed moments. This sub-optimisation is exact and fast (no matrix exponentials needed per $\phi$-evaluation). The outer optimiser sees the objective $f(\text{params}) = \min_\phi \Delta\theta(\phi, \text{params})$, and the gradient $\nabla f$ is obtained via automatic differentiation through the circuit at the optimal $\phi^*$, applying the envelope theorem: $\partial f / \partial p_i = \partial \Delta\theta / \partial p_i |_{\phi=\phi^*}$ because $\partial \Delta\theta / \partial \phi |_{\phi=\phi^*} = 0$.

The **QFI bound** follows from the same argument as the $N=M=1$ case. The generator of $\theta$ is $J_z^S$, with operator norm $\|J_z^S\| = J_S = N/2$. For a pure initial state $\vert\Psi_0\rangle$, the symmetric logarithmic derivative at $\theta=0$ is $h_0 = \int_0^{T_H} e^{i t H_{\text{int}}} J_z^S e^{-i t H_{\text{int}}} \, dt$. Since $\|e^{i t H_{\text{int}}} J_z^S e^{-i t H_{\text{int}}}\| = \|J_z^S\| = N/2$ for all $t$, the operator norm of $h_0$ is at most $T_H N/2$. Consequently $\text{Var}(h_0) \leq (T_H N/2)^2$, giving $F_Q \leq T_H^2 N^2$, and the Cramér–Rao bound implies
$\Delta\theta \geq \frac{1}{\sqrt{F_Q}} \geq \frac{1}{N T_H}.$
This is the **Heisenberg limit** for $N$ particles. The joint measurement with optimised weights respects this bound. The SQL ratio $r_{\text{SQL}} = \Delta\theta \sqrt{N} T_H$ is tracked as a diagnostic benchmark: $r_{\text{SQL}} < 1$ indicates beating the SQL. It is never asserted.

### Parameter Summary

| Symbol | Meaning | Range / Value |
|--------|---------|---------------|
| $N$ | System particle number | 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 (scaling) |
| $M$ | Ancilla particle number | 0, 1, 2, 3, 4, 6, 8, 12, 16 (scaling) |
| $d_S = N+1$ | System Hilbert space dimension | 2, 3, 4, ... |
| $d_A = M+1$ | Ancilla Hilbert space dimension | 1, 2, 3, ... |
| $\Theta_S$ | System CSS polar angle (azimuth fixed to 0) | $[0, \pi]$ |
| $\Theta_A$ | Ancilla CSS polar angle (azimuth fixed to 0) | $[0, \pi]$ |
| $T_{\text{BS}1}, T_{\text{BS}2}$ | Beam-splitter times | $[0, \pi]$ |
| $T_H$ | Holding time | $[0.1, 20]$ |
| $\alpha_{ij}$ | Interaction coefficients | $[-2, 2]$ |
| $\phi$ | Measurement weight angle (sub-optimised, not in outer loop) | $[0, 2\pi)$ |
| $\theta$ (true) | Phase rate | 1.0 (default) |

### Analytical Benchmarks

Two analytically solvable limits serve as **required validation steps** before any scaling sweeps:

1. **Zero interaction** ($\alpha_{ij} = 0$): The system and ancilla evolve independently. The weighted measurement sensitivity reduces to:
   $\Delta\theta = \frac{\sqrt{a^2\,\text{Var}(J_z^S) + b^2\,\text{Var}(J_z^A)}}{|a|\,|\partial\langle J_z^S\rangle/\partial\theta|},$
   where the ancilla contributes only noise (zero signal derivative because $\partial\langle J_z^A\rangle/\partial\theta = 0$). The optimal weight is $a^* = 1$, $b^* = 0$ (S-only measurement), and the sensitivity cannot beat SQL. The numerical pipeline must reproduce this exactly (relative error $< 10^{-10}$).

2. **$\alpha_{zz}$-only interaction** ($\alpha_{xx} = \alpha_{xz} = \alpha_{zx} = 0$, $\alpha_{zz} \neq 0$): The hold Hamiltonian is $H_{\text{hold}} = \theta J_z^S + \alpha_{zz} J_z^S \otimes J_z^A$. Since $[J_z^S \otimes J_z^A, J_z^S] = 0$, the evolution is **diagonal in the product Dicke basis** $|m_S, m_A\rangle$. The phase $\theta$ is imprinted as $e^{-i(\theta + \alpha_{zz} m_A) T_H m_S}$ — a simple phase per $m_S$ eigenvalue. All expectation values, variances, and covariances can be computed analytically as explicit functions of $N, M, \alpha_{zz}, T_H$. The numerical pipeline must reproduce these exact expressions (relative error $< 10^{-10}$).

These benchmarks catch implementation errors (wrong basis ordering, incorrect operator embedding, sign errors in Hamiltonian) before the optimiser encounters expensive, high-dimensional landscapes.

## 💻 Numerical Simulation

### Implementation Strategy

1. **Operator construction** — Build $J_z(N)$, $J_x(N)$, $J_y(N)$ for spin $J=N/2$ via `dicke_basis.jz_operator(N)` and `dicke_basis.jx_operator(N)` (which return $(N+1)\times(N+1)$ numpy arrays). Embed into the full $(N+1)(M+1)$-dimensional space via Kronecker products: $J_z^S = J_z(N) \otimes \mathbb{1}_{M+1}$, $J_x^S = J_x(N) \otimes \mathbb{1}_{M+1}$, $J_z^A = \mathbb{1}_{N+1} \otimes J_z(M)$, $J_x^A = \mathbb{1}_{N+1} \otimes J_x(M)$. Construct $H_{\text{int}}$ as a linear combination of the four tensor products, and the measurement operator $M(a,b) = a J_z^S + b J_z^A$ where $(a,b) = (\cos\phi, \sin\phi)$. Convert all operators to `torch.tensor(dtype=torch.complex128)` for the AD-enabled computation graph.

2. **State preparation** — Prepare each subsystem as a CSS with $\Phi_S = \Phi_A = 0$ (azimuth gauge fixing). Use the Dicke-basis rotation: $\vert\Theta, 0\rangle = \exp(-i\Theta J_y)\,\vert J, -J\rangle$. The ground state $\vert J, -J\rangle$ is the last basis vector in the Dicke ordering $m = J, J-1, \dots, -J$. The rotation exponential is computed via `torch.linalg.matrix_exp`. The full state is the Kronecker product $\vert\Psi_0\rangle = \vert\psi_S\rangle \otimes \vert\psi_A\rangle$.

3. **Beam-splitter unitaries** — Compute $U_{\text{BS}}(T; J) = \exp(-i T J_x)$ for each subsystem via `torch.linalg.matrix_exp`. The combined unitary is $U_{\text{BS}}^{(2)}(T) = U_{\text{BS}}(T; N) \otimes U_{\text{BS}}(T; M)$ for independent $T_{\text{BS}1}$ and $T_{\text{BS}2}$.

4. **Hold unitary** — Compute $U_{\text{hold}}(T_H) = \exp(-i T_H H_{\text{hold}})$ via `torch.linalg.matrix_exp` on the full $d \times d$ matrix. $H_{\text{hold}}$ includes the $\theta$-dependent term $\theta J_z^S$ and the interaction $H_{\text{int}}$.

5. **Forward simulation function** — Define a differentiable PyTorch function that takes the 9 circuit parameters ($\Theta_S, \Theta_A, T_{\text{BS}1}, T_{\text{BS}2}, T_H, \alpha_{xx}, \alpha_{xz}, \alpha_{zx}, \alpha_{zz}$), computes $|\Psi_{\text{final}}\rangle$ at $\theta$ and $|\Psi_{\text{final}}^{\pm}\rangle$ at $\theta \pm \delta$ (for the derivative), and returns the six moments ($\langle J_z^S\rangle, \langle J_z^A\rangle, \text{Var}(J_z^S), \text{Var}(J_z^A), \text{Cov}(J_z^S, J_z^A)$) and their $\theta$-derivatives computed via central finite differences with step $\delta = 10^{-6}$.

6. **Weight sub-optimisation** — Given the six moments and their $\theta$-derivatives, compute $\Delta\theta(\phi)$ analytically for any $\phi$. Perform a golden-section search over $\phi \in [0, 2\pi)$ to find $\phi^*$ that minimises $\Delta\theta$. This is fast (no matrix exponentials, pure arithmetic). The optimal objective value is $f_{\text{params}} = \Delta\theta(\phi^*)$.

7. **Gradient computation** — Compute gradients of $f_{\text{params}}$ with respect to all 9 circuit parameters via `torch.autograd.grad`. The gradients are evaluated at $\phi^*$ (envelope theorem: $\partial\Delta\theta(\phi^*)/\partial p = \partial\Delta\theta/\partial p|_{\phi=\phi^*}$ since $\partial\Delta\theta/\partial\phi|_{\phi^*} = 0$). The finite-difference step for $\theta$ is applied as a non-differentiable preprocessing step (the $\pm\delta$ evaluations are detached from the gradient graph), while all matrix exponentials and state evolution remain in the graph.

8. **Gradient-based optimisation** — Use `scipy.optimize.minimize(method='L-BFGS-B')` with bounds on all parameters. At each iteration, scipy provides a candidate parameter vector; convert to torch tensors, run the forward+sub-optimisation, extract $f$ and $\nabla f$ via AD, and return both to scipy. L-BFGS-B respects bound constraints (e.g., $\Theta_S \in [0, \pi]$, $T_H \in [0.1, 20]$, $\alpha_{ij} \in [-2, 2]$). Use multiple random starting points (10+ seeds per $(N,M)$ configuration) to mitigate local-minima risk; the gradient-based search within each basin handles the high-dimensional landscape efficiently.

9. **Analytical benchmark validation** — Before any scaling sweeps, validate the pipeline against the two analytical benchmarks (§ ⚛️, sub-section "Analytical Benchmarks"). Each benchmark must reproduce exact expressions to within $10^{-10}$ relative error across a grid of $N, M, \alpha, T_H$ values.

10. **Statistical scaling extraction** — After optimisation converges per $(N,M)$, record the best $\Delta\theta$. For each $N$, use $K \ge 20$ random seeds and bootstrap-resample (with replacement, $10^4$ resamples) the set of best $\Delta\theta$ values to obtain a distribution for the scaling exponent $\nu$. Fit $\log\Delta\theta = -\nu \log N + c$ with weights $w_N = 1/\sigma_N^2$, where $\sigma_N$ is the standard deviation across seeds. Report $\nu$ as **median with 95% confidence interval**. Also fit a quadratic model $\log\Delta\theta = -\nu \log N + \beta(\log N)^2 + c$ and report $\beta$ as a curvature diagnostic — a non-zero $\beta$ indicates finite-size corrections are significant.

### Parameter Sweep

| Sweep Type | Parameters Scanned | Fixed Parameters | Purpose |
|------------|-------------------|------------------|---------|
| Weight optimisation | $\phi$ (golden-section) | Any circuit params | Find $(a^*, b^*)$ at each circuit configuration |
| $N$-scaling, optimal weights | $N \in \{1,2,3,4,6,8,12,16,24,32\}$ | $M = N$, fully optimised | Extract $\nu$ with 95% CI; 10 N-points |
| $M$-scaling (ancilla size) | $M \in \{0,1,2,3,4,6,8,12,16\}$ | $N = 4$, fully optimised | Diminishing returns from ancilla |
| $\alpha$-scan with weight re-optimisation | $\alpha_{xx} \in [-2, 2]$ (21 pts) | $N$, $M$, circuit params re-optimised per point | Weighted measurement robustness |
| $(N,M)$ grid | $N \times M$ product grid | Fully optimised per $(N,M)$ | Complete sensitivity landscape |
| $T_H$ bound investigation | $T_H \in [0.1, 20]$ | $N$, $M$, fully optimised | Verify $\Delta\theta \propto 1/T_H$ with weights |
| Analytical benchmark: $\alpha=0$ | $N=\{1,2,4,8\}$, $M=N$ | $\alpha_{ij}=0$, random circuit params | Validate against closed-form expression |
| Analytical benchmark: $\alpha_{zz}$-only | $N=\{1,2,4,8\}$, $M=N$ | $\alpha_{zz}\in\{0.5,1.0,2.0\}$, rest zero | Validate against diagonal-form expression |

### Validation

The following invariants and benchmarks are asserted at every evaluation point. Violation of any invariant triggers an error, not silent correction.

- **Dimension consistency**: $\dim(J_z^S) = \dim(J_z^A) = d \times d$, and $\dim(\vert\psi\rangle) = d$, where $d = (N+1)(M+1)$.
- **Unitarity**: $U_{\text{BS}}^\dagger U_{\text{BS}} = \mathbb{1}_{N+1}$ (system), $= \mathbb{1}_{M+1}$ (ancilla), and $U_{\text{hold}}^\dagger U_{\text{hold}} = \mathbb{1}_d$ to machine precision.
- **State normalisation**: $\langle\Psi_0\vert\Psi_0\rangle = 1$ and $\langle\Psi_{\text{final}}\vert\Psi_{\text{final}}\rangle = 1$.
- **Variance non-negativity**: $\text{Var}(J_z^S) \ge 0$, $\text{Var}(J_z^A) \ge 0$, $\text{Var}(M) \ge 0$ (tolerance $10^{-12}$ for numerical rounding).
- **Sensitivity positivity**: $\Delta\theta > 0$ for all valid configurations.
- **Weight normalisation**: $a^2 + b^2 = 1$ holds by construction but is verified at each evaluation.
- **Heisenberg bound**: $\Delta\theta \cdot N \cdot T_H \ge 1$ (unconditional; tolerance $10^{-6}$).
- **CSS normalisation**: $\langle\Theta,0\vert\Theta,0\rangle = 1$ for all subsystem CSS states.
- **Analytical benchmark: zero interaction**: When $\alpha_{xx} = \alpha_{xz} = \alpha_{zx} = \alpha_{zz} = 0$, the optimal weight is $a^* = 1$, $b^* = 0$ (S-only measurement); the sensitivity obeys $\Delta\theta \ge 1/(\sqrt{N} T_H)$ (cannot beat SQL without interaction); and the numerical result matches the closed-form expression to within $10^{-10}$ relative error.
- **Analytical benchmark: $\alpha_{zz}$-only**: When only $\alpha_{zz} \neq 0$, the evolution is diagonal in the product Dicke basis and the sensitivity matches the exact expression to within $10^{-10}$ relative error.
- **$N=M=1$ regression**: At $a = b = 1/\sqrt{2}$, the sensitivity recovers $\Delta\theta = 1/T_H$ (previous report's equal-weight result) to within $0.1\%$ relative error.
- **SQL ratio** (diagnostic, not an assertion): $r_{\text{SQL}} = \Delta\theta \sqrt{N} T_H$ is tracked; $r_{\text{SQL}} < 1$ indicates SQL beating. Never asserted.

#### 🔧 Implementation Status

| Component | Description | Tests Planned |
|-----------|-------------|---------------|
| **Collective operators** | $J_z(N)$, $J_x(N)$, $J_y(N)$ for arbitrary $N$ via `dicke_basis` | Unit: eigenvalues, commutation, embedding, torch conversion |
| **Kronecker embedding** | $J_z^S = J_z(N) \otimes \mathbb{1}_{M+1}$, etc., and $H_{\text{int}}$ | Unit: dimension, hermiticity |
| **CSS parameterisation (torch)** | $\vert\Theta,0\rangle = e^{-i\Theta J_y} \vert J,-J\rangle$ with $\Phi=0$ | Unit: normalisation, angle bounds, AD-enabled |
| **Weighted measurement operator** | $M(a,b) = a J_z^S + b J_z^A$ with L$_2$ constraint | Unit: spectrum, weight normalisation |
| **Weight sub-optimisation** | Golden-section search over $\phi$ at fixed circuit parameters | Integration: verify analytic $\Delta\theta(\phi)$ matches brute-force sweep |
| **Beam-splitter unitaries (torch)** | $U_{\text{BS}}(T; J) = \exp(-i T J_x)$ per subsystem | Unit: unitarity, $J=1/2$ consistency, AD-enabled |
| **Hold unitary (torch)** | $\exp(-i T_H [\theta J_z^S + H_{\text{int}}])$ via `torch.linalg.matrix_exp` | Unit: unitarity, $\theta$-derivative stability, AD-enabled |
| **Full circuit (torch)** | BS1 $\to$ Hold $\to$ BS2 with CSS initial states | Integration: normalisation, $N=M=1$ regression, AD correctness |
| **Sensitivity** | $\Delta\theta = \sqrt{\text{Var}(M)} / \vert\partial\langle M\rangle/\partial\theta\vert$ with covariance | Integration: matches QFI for special cases |
| **Analytical benchmark: $\alpha=0$** | Closed-form sensitivity for zero interaction | Regression: reproduces exact expressions to $10^{-10}$ |
| **Analytical benchmark: $\alpha_{zz}$** | Closed-form sensitivity for commuting interaction | Regression: reproduces exact diagonal expressions to $10^{-10}$ |
| **Gradient computation** | `torch.autograd.grad` through full circuit at $\phi^*$ | Unit: finite-difference verification of AD gradients |
| **L-BFGS-B wrapper** | 9-parameter optimisation (8 circuit + $\phi$ sub-optimised) | Integration: SQL baseline recovery, benchmark-case convergence |
| **$N$-scaling fitter** | Weighted log-log regression with bootstrap CI | Unit: fit quality, coverage of confidence intervals |
| **Statistical pipeline** | $K \ge 20$ seeds, bootstrap resampling, curvature diagnostic | Integration: reproducibility, coverage of $\nu$ CI |
| **Validation suite** | All invariants (trace, unitarity, positivity, HL bound, analytical benchmarks) | E2E: complete pipeline |

## ⚠️ Expected Failure Conditions

| Failure | Description | Mitigation |
|---------|-------------|------------|
| **$T_H$ boundary saturation** | The optimiser drives $T_H$ to its upper bound because the sensitivity always improves with longer holding ($\Delta\theta \propto 1/T_H$). | Expand bound check; accept as expected SQL/HL behaviour. Do not interpret $T_H$ at boundary as optimal trade-off. |
| **Interaction coefficients driven to zero** | The optimiser converges to $\alpha_{ij} \approx 0$, especially for large $N$ where the system alone can achieve near-Heisenberg scaling. | Compare weighted joint measurement with S-only at the same $N$ and $\alpha$; the advantage may manifest at intermediate $\alpha$. |
| **Weight angle $\phi$ has trivial optimum at $\phi=0$ (S-only)** | For $M \ll N$ or weak interaction, the optimal weight is $a^* \approx 1$, $b^* \approx 0$ (S-only measurement). The ancilla contributes no useful information. | Verify this regime systematically; it confirms that the weighted measurement gracefully degrades to S-only when the ancilla is uninformative. |
| **Neeligible covariance** | The covariance term $2ab\,\text{Cov}(J_z^S, J_z^A)$ is small compared to $a^2\text{Var}(J_z^S) + b^2\text{Var}(J_z^A)$, making the joint measurement approximately equivalent to separate S and A measurements. | Compute and report the covariance fraction $\rho = \text{Cov}(J_z^S, J_z^A) / \sqrt{\text{Var}(J_z^S)\,\text{Var}(J_z^A)}$ as a diagnostic. |
| **CSS restriction is too limiting** | The optimal initial state for a given $N$, $M$, and $\alpha$ may require spin-squeezed or non-Gaussian states that are not accessible within the CSS family. | Extend to general states in a follow-up investigation; the CSS family provides a lower bound on achievable sensitivity. |
| **Dimension growth with $N$ and $M$** | The full space dimension $d = (N+1)(M+1)$ grows quadratically. At $N=M=32$, $d = 1089$, and matrix exponentials of size $1089 \times 1089$ plus optimisation become computationally expensive. | Cap at $N=M=24$ ($d=625$) for routine sweeps; use $N=32$ ($d=1089$) only for confirming asymptotic trend. PyTorch's GPU support can expedite large exponentials. |
| **Fringe extremum** | The derivative $\partial\langle M\rangle/\partial\theta$ may vanish for certain parameter combinations, causing $\Delta\theta \to \infty$ numerically. | Detect and discard configurations where $\vert\partial\langle M\rangle/\partial\theta\vert < 10^{-8}$, or exclude from fits. |
| **Gradient vanishing in L-BFGS-B** | The objective landscape may contain plateaus where $\nabla f \approx 0$ despite being far from a true optimum. | Monitor gradient norm in optimisation history; if $\|\nabla f\| < 10^{-10}$ for $>20$ iterations without progress, restart from a new random seed. |
| **Envelope theorem failure** | If $\phi^*$ lies at a boundary of $[0,2\pi)$ and $\partial\Delta\theta/\partial\phi \neq 0$ at that boundary, the envelope theorem does not hold and the gradient from AD at $\phi^*$ is incorrect. | Check $\phi^*$ is interior ($0 < \phi^* < 2\pi$) after each gradient call; if at boundary, fall back to finite-difference gradient over the circuit parameters. |
| **L-BFGS-B convergence to poor local minimum** | The 9D landscape is non-convex; L-BFGS-B is a local optimiser that may not find the global optimum from a single starting point. | Use $K \ge 20$ random seeds per $(N,M)$; report the distribution of final $\Delta\theta$ values (not just the best). |
| **Computational budget exceeded** | A full sweep with $K=20$ seeds, 10 N-points, and $d=625$ (N=24) matrix exponentials may require $>10^5$ objective evaluations. | Prioritise small-$N$ sweeps first; use result-tracking to detect convergence early; limit $N \le 16$ for comprehensive sweeps and $N \le 24$ for scaling confirmation. |

## 🔬 Results

| Experiment | Status | Description |
|------------|--------|-------------|
| Analytical benchmark: zero interaction | ⏳ | Validate against closed-form expression for all $N \le 8$ |
| Analytical benchmark: $\alpha_{zz}$-only | ⏳ | Validate against diagonal-form expression for all $N \le 8$ |
| Gradient reconstruction test | ⏳ | Verify AD gradients match finite-difference gradients (relative error $< 10^{-5}$) |
| Weight sweep at fixed $N=M=1$, $\alpha_{xx}=1$ | ⏳ | Verify optimal $(a^*, b^*)$ differs from $(1/\sqrt{2}, 1/\sqrt{2})$ |
| $N$-scaling with optimal weights, $M=N$ | ⏳ | Extract $\nu$ with 95% CI; compare to SQL ($\nu=1/2$) and HL ($\nu=1$) |
| $M$-scaling at fixed $N=4$ | ⏳ | Diminishing returns from ancilla size |
| $\alpha_{xx}$-scan with weight re-optimisation, $N=M=1$ | ⏳ | Compare with previous equal-weight results; confirm weight optimisation improves sensitivity |
| $(N,M)$ grid scan, full optimisation | ⏳ | Complete landscape of $\Delta\theta(N, M)$ |
| $N=M=1$ regression test | ⏳ | Verify $a=b=1/\sqrt{2}$ reproduces previous report's equal-weight sensitivity |

Key results to be reported once numerical experiments are completed. The central outputs will be:
- Optimal weight ratio $a^*/b^*$ as a function of $\alpha_{ij}$, $N$, $M$
- Scaling exponent $\nu$ with 95% bootstrap confidence interval for the weighted joint measurement vs S-only
- Comparison of best $\Delta\theta$ (weighted joint) with the Heisenberg limit $1/(N T_H)$ and SQL $1/(\sqrt{N} T_H)$
- Curvature diagnostic $\beta$ for finite-size corrections
- Distribution of $\Delta\theta$ across random seeds (evidence of optimisation quality)
- Gradient norm convergence history for L-BFGS-B traces

## ✅ Success Criteria

| #  | Check | Expectation                                                                                                                          |
|----|-------|--------------------------------------------------------------------------------------------------------------------------------------|
| 1  | Analytical benchmark: zero interaction | All $N \le 8$ configurations reproduce closed-form to $10^{-10}$ relative error                                                      |
| 2  | Analytical benchmark: $\alpha_{zz}$-only | All $N \le 8$ configurations reproduce diagonal-form expression to $10^{-10}$ relative error                                         |
| 3  | AD gradient accuracy | $\|\nabla f_{\text{AD}} - \nabla f_{\text{FD}}\| / \|\nabla f_{\text{AD}}\| < 10^{-5}$ for 10 random parameter configurations        |
| 4  | Weight sweep at $N=M=1$, $\alpha_{xx}=1$ finds $a^* \neq 1/\sqrt{2}$ | At least 5% improvement in $\Delta\theta$ over equal-weight case                                                                     |
| 5  | $N$-scaling exponent $\nu$ 95% CI lower bound $> 0.55$ | The weighted joint measurement beats SQL at a statistically significant level                                                        |
| 6  | $\nu \leq 1$ for all configurations (HL bound) | 95% CI upper bound $\le 1.05$ (allowing small numerical tolerance) for all valid fits                                                |
| 7  | $M$-scaling shows diminishing returns | $\Delta\theta$ improvement from $M \to 2M$ is $< 50\%$ of improvement from $M=0$ to $M=1$ at fixed $N=4$                             |
| 8  | HL bound $F_Q \leq T_H^2 N^2$ never violated | $\Delta\theta \geq 1/(N T_H)$ for all configurations                                                                                 |
| 9  | Weighted joint measurement achieves $\Delta\theta \sqrt{N} T_H < 1$ for at least one configuration | SQL-beating configuration exists                                                                                                     |
| 10 | Weighted joint beats S-only at same parameters | For at least 50% of probed $\alpha$ values, $\Delta\theta_{\text{weighted}} < \Delta\theta_{\text{S-only}}$ at same $(N, M, \alpha)$ |
| 11 | All unitarity, normalisation, positivity assertions pass | $> 99\%$ of evaluated configurations pass                                                                                            |
| 12 | Regression: $N=M=1$, $\phi=\pi/4$ reproduces previous equal-weight results | $\Delta\theta$ within $1\%$ of previous report at same parameters                                                                    |
| 13 | Scaling fit $R^2 > 0.92$ for $N$-scaling with $\ge 8$ N-points | Fits pass $R^2$ threshold; LOOCV $R^2 > 0.85$                                                                                        |
| 14 | Curvature diagnostic $\beta$ reported | Quadratic term coefficient $\vert \beta \vert < 0.1$ or explicitly flagged as significant                                            |
| 15 | All $\nu$ estimates include 95% bootstrap CI | CIs are non-degenerate (width $< 0.3$) for $N \ge 6$ points                                                                          |

## ⚖️ Physical Invariants

The following analytical bounds constrain all numerical results:

1. **Heisenberg Limit**: $\Delta\theta \geq 1/(N T_H)$. This bound is unconditional — it depends only on $N$ and $T_H$, and is enforced by the validation suite.

2. **Standard Quantum Limit** (reference benchmark): $\Delta\theta_{\text{SQL}} = 1/(\sqrt{N} T_H)$. This is the per-particle sensitivity achievable with product states. It is not a fundamental bound and is never asserted; the SQL ratio $r_{\text{SQL}} = \Delta\theta \sqrt{N} T_H$ is tracked as a diagnostic.

3. **Operator norm saturation**: $\|J_z^S\| = N/2$, giving the maximum variance $\text{Var}(J_z^S) \leq (N/2)^2$. For a $J_z^S$-only measurement, $\Delta\theta \geq 1/(N T_H)$ is the same HL; the S-only measurement can in principle saturate this with a NOON state and optimal readout. The joint measurement with $J_z^A$ adds ancilla variance but no direct $\theta$ signal, so optimal weighting must balance the ancilla's noise contribution against the information it carries through $\text{Cov}(J_z^S, J_z^A)$.

4. **Conservation of total spin**: For each subsystem independently, $[J_x, J_y] = i J_z$ and $J^2 = J_x^2 + J_y^2 + J_z^2 = J(J+1)\mathbb{1}$. The beam-splitter unitary preserves $J^2$ within each subsystem. The hold Hamiltonian also preserves $(J^S)^2$ and $(J^A)^2$ individually because $J_x^S$ and $J_z^S$ both commute with $(J^S)^2$, and their tensor products with ancilla operators preserve this property. This means the total spin per subsystem is fixed, and the evolution stays within the $d = (N+1)(M+1)$-dimensional Dicke subspace.

5. **Covariance bound**: $|\text{Cov}(J_z^S, J_z^A)| \leq \sqrt{\text{Var}(J_z^S)\,\text{Var}(J_z^A)} \leq (N/2)(M/2) = NM/4$, by the Cauchy–Schwarz inequality. The maximum possible correlation between S and A is bounded by the product of their individual variances.

6. **Envelope theorem (Danskin's theorem)**: For the objective $f(\text{params}) = \min_\phi \Delta\theta(\phi, \text{params})$, if $\phi^*$ is a unique interior minimiser, then $\nabla f = \partial \Delta\theta / \partial \text{params}|_{\phi=\phi^*}$. This justifies the AD approach that differentiates through the circuit at the optimal $\phi^*$ without differentiating through the $\phi$-subproblem.

## 🏁 Conclusions

This report outlines a plan to generalise the ancilla-assisted joint measurement framework from the $N=M=1$, equal-weight case $(J_z^S + J_z^A)/\sqrt{2}$ studied in `2026-05-15-Ancilla-Assisted-Metrology-Joint-Measurement.md` to arbitrary $N$ and $M$, incorporating three key methodological improvements:

- **Gradient-based optimisation (L-BFGS-B + PyTorch AD)**: Replaces Nelder–Mead in 12D with a 9-parameter (after conservative symmetry reduction) gradient-based solver that can efficiently navigate the non-convex landscape. The $\phi$ weight sub-optimisation is handled in closed form outside the outer loop, exploiting the envelope theorem for correct gradient computation.
- **Conservative symmetry reduction**: The gauge choice $\Phi_S = \Phi_A = 0$ and removal of $\phi$ from the outer optimiser reduces the search space from 12 to 9 parameters, improving convergence without risking loss of valid configurations.
- **Statistical rigor**: Scaling exponent $\nu$ is extracted via weighted regression with bootstrap confidence intervals (95% CI), $K \ge 20$ seeds per $(N,M)$, and a curvature diagnostic to detect finite-size corrections.

The theoretical framework is a direct generalisation of the $N=M=1$ case: the same circuit structure (BS1 → hold → BS2), the same interaction Hamiltonian $H_{\text{int}}$, and the same error-propagation sensitivity formula carry over, with the collective operators $J_z(N)$ and $J_x(N)$ replacing the $2\times2$ Pauli matrices.

Two analytical benchmarks — zero interaction and $\alpha_{zz}$-only interaction — provide exact closed-form expressions that validate the numerical implementation before any scaling sweeps begin. The computational budget (up to 1 day) allows for $N \le 24$ with $K \ge 20$ seeds and the full suite of analytical benchmarks and diagnostics.

The key questions are:
- **(a)** Does the optimal weight ratio $a^*/b^*$ deviate significantly from $1/\sqrt{2}$ for non-zero $\alpha_{ij}$? If so, the previous report's conclusion that $J_z^S + J_z^A$ is suboptimal is confirmed and quantified.
- **(b)** What scaling exponent $\nu$ does the weighted joint measurement achieve as $N$ grows? With 95% bootstrap confidence, does it beat the SQL ($\nu > 0.55$)? The QFI bound permits any $\nu \in [1/2, 1]$.
- **(c)** How many ancilla particles $M$ are needed to approach the optimal sensitivity? The diminishing-returns hypothesis predicts $M \sim N$ is sufficient.
- **(d)** Does the weighted joint measurement improve robustness to interaction-induced degradation, as the previous report found for the equal-weight case?
- **(e)** Are finite-size corrections significant? The curvature diagnostic $\beta$ quantifies how close the observed scaling is to asymptotic behaviour.

The numerical implementation builds on existing infrastructure (`dicke_basis.jz_operator`, `dicke_basis.jx_operator`, `scipy.optimize.minimize`, error-propagation sensitivity) and adds PyTorch-based automatic differentiation for the circuit simulation, CSS state preparation for arbitrary spin $J$ with $\Phi=0$ gauge fixing, Kronecker-product embedding for the S+A system, a golden-section sub-optimisation for the weight angle $\phi$, and bootstrap-enhanced statistical fitting for the scaling exponent.

#### 🔍 Open Items

**Optimal measurement beyond linear combinations**: The weighted sum $M = a J_z^S + b J_z^A$ is still only a linear combination of the two subsystem angular momenta. The most general informationally complete measurement on the $(N+1)(M+1)$-dimensional space could achieve the QFI bound $F_Q$ exactly. How much of the gap between the weighted linear measurement and the QFI bound can be closed by allowing arbitrary POVMs or by adapting the measurement to the interaction parameters?

**Non-CSS initial states**: The CSS family is a restricted subspace of each subsystem's Hilbert space (dimension 2 for each subsystem, regardless of $N$ or $M$). Spin-squeezed states (OAT), Dicke superpositions, and NOON states may offer better initial variance properties that the joint measurement could exploit. A full-state optimisation over the $(N+1)$-dimensional system space is computationally intensive but could reveal the true optimal sensitivity.

**Interaction-based readout**: An additional interaction period after the hold but before the second beam splitter (as proposed in `2026-05-15-Ancilla-Assisted-Metrology-Joint-Measurement.md`, open item d) could map ancilla correlations back onto the system, potentially enabling Heisenberg-limited sensitivity with an S-only measurement.

**Optimal $(a,b)$ for specific interaction families**: The four coefficients $\alpha_{xx}, \alpha_{xz}, \alpha_{zx}, \alpha_{zz}$ generate qualitatively different S-A correlations. Is there a universal optimal weight ratio $a^*/b^*$ that depends only on $N$ and $M$, or does it depend strongly on which $\alpha_{ij}$ dominates?

**Finite-$N$ prefactors and sub-leading corrections**: For small $N$, the CSS-based sensitivity may differ significantly from the asymptotic scaling. Characterising the pre-factor $C$ in $\Delta\theta = C/(N^\nu T_H)$ as a function of $N$ and $M$ would be valuable for experimental planning.

**Gradient-based optimisation vs global search**: L-BFGS-B is a local optimiser. If the 9D landscape has many competing basins, differential evolution or CMA-ES may be needed. The multi-seed strategy hedges against this, but a systematic comparison of optimiser performance would strengthen the conclusions.
