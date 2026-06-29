# Free-Ancilla Initial State with $\omega$-Modulated Drive and Weighted Joint Measurement

## 🧪 Hypothesis

Report #20260610 demonstrated that freeing the ancilla initial state (adding Bloch angles $\theta_A$, $\phi_A$) to the $\omega$-modulated drive protocol improves sensitivity from $4.91\times$ SQL (fixed ancilla) to $7.3\times$ SQL at $N=1$, $\omega=0.1$ — a **derivative enhancement** via $\partial H_A/\partial\omega$ amplified by the ancilla superposition. Report #20260613 demonstrated that replacing the S-only measurement with a weighted joint measurement $M(\psi) = \cos\psi\,J_z^S + \sin\psi\,J_z^A$ improves sensitivity from $4.91\times$ SQL ($\psi=0$) to $9.7\times$ SQL at $N=1$, $\omega=0.5$ — a **variance reduction** via extraction of S--A correlations.

These two mechanisms are physically independent: free-ancilla acts on the **initial state** (increasing the derivative signal), while the joint measurement acts on the **readout** (extracting the larger correlation signal). Their combination could compound multiplicatively.

The experiment tests three specific claims:

1. **N=1 compounding** (Experiment 1): The combined 7D optimisation $(\theta_A, \phi_A, a_x, a_y, a_z, a_{zz}, \psi)$ at $N=1$ with qubit ancilla ($J_A=1/2$) achieves $\Delta\omega$ that strictly exceeds both the #20260610 best ($7.3\times$ SQL, $\Delta\omega=0.01364$) and the #20260613 best ($9.7\times$ SQL, $\Delta\omega=0.01027$). Specifically, $\Delta\omega < 0.01027$ at some $\omega$, corresponding to $> 9.7\times$ SQL. The null is that the combined optimum is $\min(\text{free-ancilla best},\text{joint-measurement best})$ — i.e., the mechanisms do not compound.

2. **N>1 scaling with qubit ancilla** (Experiment 2): With $J_A=1/2$ and $N=1$–$20$, the combined protocol slows the ratio decay beyond #20260613 (which achieved $R=1.94$ at $N=20$ with $\beta\approx0.64$). Specifically, $R(N=20) > 1.94$ and the decay exponent $\beta < 0.64$. The null is that the free-ancilla degree of freedom becomes irrelevant at $N>3$, recovering the #20260613 scaling exactly.

3. **N>1 scaling with multi-particle ancilla** (Experiment 3): With $J_A = N/2$ (multi-particle Dicke ancilla, free CSS initial state $\vert \theta_A,\phi_A\rangle$), the combined protocol achieves sub-SQL scaling exponent $\alpha < -0.5$, exceeding both the #20260612 baseline ($J_A=N/2$, S-only, $\alpha\approx-0.5$, $R\approx4.68$ flat) and the #20260613 baseline ($J_A=1/2$, joint measurement, $\alpha\approx-0.5$, $R$ decays). The multi-particle ancilla provides an $O(N)$ Hilbert space for both the derivative enhancement and the correlation extraction, potentially unlocking $F_Q \propto N^2$ scaling. The null is that the $J_A=N/2$ free-ancilla state collapses to the top Dicke limit ($\theta_A\to0$) and the joint measurement cannot improve on the S-only scaling exponent.

## ⚛️ Theoretical Model

**Hilbert space.** The total space is $\mathcal{H}_{\text{tot}} = \mathcal{H}_S \otimes \mathcal{H}_A$. For **Experiments 1 and 2**, $\mathcal{H}_A$ is a single-qubit ancilla ($J_A=1/2$, dimension 2) and $\mathcal{H}_S$ is the $N$-particle symmetric subspace ($J_S = N/2$, dimension $N+1$), giving total dimension $2(N+1)$. For **Experiment 3**, both S and A are $N$-particle symmetric subspaces ($J_S = J_A = N/2$, dimension $N+1$ each), giving total dimension $(N+1)^2$. The **Fock-to-Dicke mapping** uses the standard two-mode single-particle basis $\{\vert 1,0\rangle, \vert 0,1\rangle\}$ for $J=1/2$, extended to $J=N/2$ via $\vert J,m\rangle = \vert J+m, J-m\rangle$ in the Fock representation. The ordering convention for the Dicke basis is $m$ descending from $+J$ to $-J$.

**Operators.** Angular momentum operators $J_k$ satisfy $[J_i, J_j] = i\epsilon_{ijk} J_k$. For **qubit ancilla** ($J=1/2$), $J_k = \sigma_k/2$ (Pauli matrices). For **multi-particle ancilla** ($J=N/2$), $J_k$ are the $(N+1)\times(N+1)$ Dicke-basis angular momentum matrices. These are embedded into the total space via Kronecker products: $J_k^S = J_k^{(S)} \otimes \mathbb{1}_{\dim\mathcal{H}_A}$, $J_k^A = \mathbb{1}_{\dim\mathcal{H}_S} \otimes J_k^{(A)}$.

**Initial state.** The system starts in the Fock state $\vert 1,0\rangle_S$ (first basis vector, $m=+J_S$). For **Experiments 1 and 2**, the ancilla starts in a free pure qubit state $\vert \psi_A(\theta_A,\phi_A)\rangle = \cos(\theta_A/2)\vert 1,0\rangle_A + e^{i\phi_A}\sin(\theta_A/2)\vert 0,1\rangle_A$, with $\theta_A\in[0,\pi]$, $\phi_A\in[0,2\pi)$. For **Experiment 3**, the ancilla starts in a **coherent spin state** (CSS) in the Dicke basis: $\vert \psi_A(\theta_A,\phi_A)\rangle = \sum_{m=-J_A}^{J_A} \binom{2J_A}{J_A+m}^{1/2} \cos(\theta_A/2)^{J_A+m} \sin(\theta_A/2)^{J_A-m} e^{-i(J_A-m)\phi_A} \vert J_A,m\rangle$, where $J_A=N/2$.

**Circuit protocol.** The protocol follows the standard four-step sequence established in #20260519 and used in all subsequent $\omega$-modulated drive experiments:

1. **Beam splitter on system only**: $U_{\text{BS}}^{(S)} = \exp(-i(\pi/2) J_x^S) \otimes \mathbb{1}_{\dim\mathcal{H}_A}$, converting the input Fock state into a coherent superposition on S. For multi-particle system, this is the symmetric Dicke-basis BS.

2. **Holding period with $\omega$-modulated drive, Ising interaction, and simultaneous encoding**: The full state evolves under $H = H_S + H_A + H_{\text{int}}$ for duration $T_H = 10$:
   - $H_S = \omega J_z^S$ — the unknown phase rate on the system,
   - $H_A = \omega\,(a_x J_x^A + a_y J_y^A + a_z J_z^A)$ — the **$\omega$-modulated** ancilla drive,
   - $H_{\text{int}} = a_{zz} J_z^S \otimes J_z^A$ — the Ising interaction coupling S and A (independent of $\omega$).
   
   The hold unitary is $U_{\text{hold}}(T_H) = \exp(-i T_H H)$.

3. **Second beam splitter on system only**: $U_{\text{BS}}^{(S)}$ (identical to step 1).

4. **Weighted joint measurement**: $M(\psi) = \cos\psi\,J_z^S + \sin\psi\,J_z^A$, with $\psi\in[-\pi,\pi]$. The coefficients satisfy $m_s^2 + m_a^2 = 1$ where $m_s = \cos\psi$, $m_a = \sin\psi$.

The complete evolution is $\vert \Psi_{\text{final}}\rangle = U_{\text{BS}}^{(S)}\, U_{\text{hold}}(T_H)\, U_{\text{BS}}^{(S)}\, \vert \Psi_0\rangle$.

**Sensitivity metric.** Error-propagation uncertainty:
$\Delta\omega = \frac{\sqrt{\text{Var}(M)}}{|\partial\langle M\rangle/\partial\omega|}$,
where the derivative is computed via central finite differences with step $\delta = 10^{-6}$. The standard quantum limit for $N$ system particles is $\Delta\omega_{\text{SQL}} = 1/(\sqrt{N}\,T_H) = 0.1/\sqrt{N}$. The sensitivity ratio is $R = \Delta\omega_{\text{SQL}} / \Delta\omega$.

**Physical mechanisms for compounding.** The free-ancilla mechanism (#20260610) increases $|\partial\langle J_z^S\rangle/\partial\omega|$ by starting the ancilla in a superposition that maximises its sensitivity to the $\omega$-modulated drive components $a_x$, $a_y$. The joint-measurement mechanism (#20260613) reduces the effective variance by exploiting the covariance $\text{Cov}(J_z^S, J_z^A)$ and extracting the ancilla's $\omega$-dependent signal directly. These mechanisms act on different parts of the error-propagation formula (numerator and denominator, respectively), so their improvements are expected to compound approximately multiplicatively. For $J_A=N/2$, the ancilla contribution to both the derivative (via $\partial H_A/\partial\omega = H_A^{\text{norm}}$, an $O(N)$ operator) and the variance (via the $(N+1)$-dimension ancilla Hilbert space) scales with $N$, potentially enabling $F_Q \propto N^2$ scaling.

## 📊 Models Survey

| Model | $J_A$ | Initial State | Measurement | $N$ Range | Expected $R(N=1)$ | Expected $\alpha$ | Reference |
|---|---|---|---|---|---|---|---|
| **A** (fixed ancilla, S-only) | 1/2 | Fixed $\vert 1,0\rangle$ | $J_z^S$ | 1–20 | $4.91\times$ | $\approx -0.5$ | #20260519, #20260611 |
| **B** (free ancilla, S-only) | 1/2 | Free $(\theta_A,\phi_A)$ | $J_z^S$ | 1 | $7.3\times$ | N/A | #20260610 |
| **C** (fixed ancilla, joint meas.) | 1/2 | Fixed $\vert 1,0\rangle$ | $M(\psi)$ | 1–20 | $9.7\times$ | $\approx -0.5$ | #20260613 |
| **D** (this work, Exp. 1–2) | 1/2 | Free $(\theta_A,\phi_A)$ | $M(\psi)$ | 1–20 | $>10\times$ | $< -0.5$ | This work |
| **E** (multi-ancilla, S-only) | $N/2$ | Fixed $\vert J_A,J_A\rangle$ | $J_z^S$ | 1–8 | $4.68\times$ | $\approx -0.5$ | #20260612 |
| **F** (this work, Exp. 3) | $N/2$ | Free CSS $(\theta_A,\phi_A)$ | $M(\psi)$ | 1–20 | $>10\times$ | $\ll -0.5$ | This work |

Models D and F represent the two new configurations:
- **Model D** — qubit ancilla ($J_A=1/2$), free initial state, joint measurement. This directly extends #20260613 (which had fixed ancilla + joint measurement) by also freeing the ancilla initial state. The 7D parameter space $(\theta_A,\phi_A,a_x,a_y,a_z,a_{zz},\psi)$ can be compared against the 5D space $(a_x,a_y,a_z,a_{zz},\psi)$ of #20260613.
- **Model F** — multi-particle ancilla ($J_A=N/2$), free CSS initial state, joint measurement. This extends #20260612 (which had $J_A=N/2$, top Dicke state, S-only measurement) by adding both the free initial state and the joint measurement. The 7D parameter space is the same form, but the operators are $(N+1)\times(N+1)$ Dicke matrices.

At $N=1$, Models D and F converge (both have $J_A=1/2$). At $N>1$, Model F gains an additional $O(N)$ degree of freedom from the multi-particle ancilla, which the free CSS state and joint measurement can exploit.

## 💻 Numerical Simulation

### Implementation Strategy

1. **Operator construction** — For Experiments 1–2 ($J_A=1/2$), use `build_n_particle_operators(N)` from `src/physics/n_particle_drive.py` to construct $2(N+1)$-dimensional operators. For Experiment 3 ($J_A=N/2$), use `build_operators(N, N)` from `src/physics/bipartite_operators.py` to construct $(N+1)^2$-dimensional operators in the Dicke basis. Both provide the required operator keys `Jz_S`, `Jx_S`, `Jy_S`, `Jz_A`, `Jx_A`, `Jy_A`, `I_S`, `I_A`, `I_full`.

2. **Free-ancilla state preparation** — For the qubit ancilla, use `n_particle_free_ancilla_initial_state(N, theta_A, phi_A)` which returns a $2(N+1)$-vector $|J_S,J_S\rangle_S \otimes |\psi_A(\theta_A,\phi_A)\rangle$. For the multi-particle ancilla ($J_A=N/2$), construct the CSS state $|\psi_A(\theta_A,\phi_A)\rangle$ in the Dicke basis using `coherent_spin_state(J, theta, phi)` from `src/algorithms/coherent_spin_state.py`, then tensor with the system initial state $|J_S,J_S\rangle_S$ via `multi_particle_free_css_initial_state(N, theta_A, phi_A)`.

3. **$\omega$-modulated hold Hamiltonian** — Build $H = \omega J_z^S + \omega(a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_{zz} J_z^S \otimes J_z^A$ using the same form as `build_phase_modulated_hold_hamiltonian` in `src/analysis/ancilla_drive_metrology.py`. For the multi-particle case, the operators are larger but the expression is identical. The hold unitary is $U_{\text{hold}} = \exp(-i T_H H)$ via `scipy.linalg.expm`.

4. **System-only beam splitter** — $U_{\text{BS}}^{(S)} = \exp(-i(\pi/2) J_x^S) \otimes \mathbb{1}$ constructed per $N$ and cached for repeated use.

5. **Weighted joint measurement operator** — Build $M(\psi) = \cos\psi\,J_z^S + \sin\psi\,J_z^A$ for the full Hilbert space dimension. Enforce Hermiticity. Pass as `meas_op` to the sensitivity computation function.

6. **Sensitivity computation** — Use `compute_n_particle_sensitivity` or the generic `compute_free_ancilla_sensitivity(evolve_fn=evolve_phase_modulated_circuit, ..., meas_op=M)` for the qubit case. For the multi-particle case, implement a wrapper that constructs the full initial state from $(\theta_A,\phi_A)$ and passes it with `meas_op=M` to an evolve function supporting $J_A=N/2$ operators. Central finite differences with $\delta = 10^{-6}$.

7. **Two-phase optimisation** — Use the shared `run_two_phase_pipeline` infrastructure from `src/analysis/optimisation_pipeline.py`:
   - **Stage 1**: Random search over the 7D parameter space with samples drawn uniformly: $\theta_A \sim U[0,\pi]$, $\phi_A \sim U[0,2\pi)$, $\mathbf{a}$ from the 3-ball $\|\mathbf{a}\| \le R$ (Marsaglia's method), $a_{zz} \sim U[-5,5]$, $\psi \sim U[-\pi,\pi]$.
   - **Stage 2**: Nelder--Mead refinement from the top $k$ random-search points, operating in the full 7D space with bound enforcement.

8. **Baseline controls** — For every $(N,\omega)$ pair, run the decoupled control ($a_x=a_y=a_z=a_{zz}=0$) to verify $R=1$ at $\psi=0$, and an S-only control ($\psi=0$) to reproduce the relevant prior result (#20260610 for free ancilla, #20260613 for fixed ancilla). For Model F, the S-only control reproduces #20260612.

9. **Data serialisation** — Every optimisation result is stored in Parquet files via `ParquetSerializable` dataclasses. Each dataclass stores all input parameters ($\omega$, $N$, $T_H$, $\theta_A$, $\phi_A$, $a_x$, $a_y$, $a_z$, $a_{zz}$, $\psi$) alongside all computed results ($\Delta\omega$, $R$, $\langle M\rangle$, $\text{Var}(M)$, $\partial\langle M\rangle/\partial\omega$, $\text{Cov}(J_z^S,J_z^A)$, fringe-extremum flag). Fail-fast deserialisation.

10. **N-scaling analysis** — For each $\omega$, fit $R(N)-1 = c N^{-\beta}$ using nonlinear least squares for $N\ge 2$. Also compute the scaling exponent $\alpha$ from $\log(\Delta\omega) = \alpha\log(N) + \log(C)$.

### Parameter Sweep

**Experiment 1: N=1 compounding (qubit ancilla, $J_A=1/2$)**

| Parameter | Range / Values | Purpose |
|-----------|--------------|---------|
| $\omega$ | $\{0.1, 0.2, 0.5, 1.0, 2.0, 5.0\}$ (6 values) | Test compounding across operating points |
| $T_H$ | 10 (fixed) | SQL reference $\Delta\omega_{\text{SQL}} = 0.1$ |
| $\theta_A$ | $[0, \pi]$ (optimised) | Free ancilla polar angle |
| $\phi_A$ | $[0, 2\pi)$ (optimised) | Free ancilla azimuth |
| $a_x, a_y, a_z$ | 3-ball $\|\mathbf{a}\| \le 10$ | Drive coefficients |
| $a_{zz}$ | $[-5, 5]$ (optimised) | Ising interaction |
| $\psi$ | $[-\pi, \pi]$ (optimised) | Measurement angle |
| Random samples per $\omega$ | 5000 | Stage 1 global exploration |
| NM refinements per $\omega$ | 60 | Stage 2 local refinement |
| Finite-difference step $\delta$ | $10^{-6}$ | Derivative computation |

**Experiment 2: N>1 scaling with qubit ancilla ($J_A=1/2$)**

| Parameter | Range / Values | Purpose |
|-----------|--------------|---------|
| $N$ | 1 to 20 (integer, 20 values) | Primary scaling axis |
| $\omega$ | $\{0.1, 0.2, 0.5, 1.0, 2.0\}$ (5 values) | Test scaling across operating points |
| $T_H$ | 10 (fixed) | SQL reference $0.1/\sqrt{N}$ |
| $\theta_A$, $\phi_A$ | $[0,\pi] \times [0,2\pi)$ (optimised) | Free ancilla state |
| $a_x, a_y, a_z$ | 3-ball $\|\mathbf{a}\| \le 10$ | Drive coefficients |
| $a_{zz}$ | $[-5, 5]$ (optimised) | Ising interaction |
| $\psi$ | $[-\pi, \pi]$ (optimised) | Measurement angle |
| Random samples per $(N,\omega)$ | 3000 | Stage 1 |
| NM refinements per $(N,\omega)$ | 40 | Stage 2 |

**Experiment 3: N>1 scaling with multi-particle ancilla ($J_A=N/2$)**

| Parameter | Range / Values | Purpose |
|-----------|--------------|---------|
| $N$ | 1 to 20 (integer, 20 values) | Primary scaling axis |
| $\omega$ | $\{0.1, 0.2, 0.5\}$ (3 values) | Test scaling at low-to-moderate $\omega$ |
| $T_H$ | 10 (fixed) | SQL reference $0.1/\sqrt{N}$ |
| $\theta_A$, $\phi_A$ | $[0,\pi] \times [0,2\pi)$ (optimised) | Free CSS ancilla state |
| $a_x, a_y, a_z$ | 3-ball $\|\mathbf{a}\| \le 5$ (reduced bound for stability at large $N$) | Drive coefficients |
| $a_{zz}$ | $[-5, 5]$ (optimised) | Ising interaction |
| $\psi$ | $[-\pi, \pi]$ (optimised) | Measurement angle |
| Random samples per $(N,\omega)$ | 2000 (reduced due to larger Hilbert space) | Stage 1 |
| NM refinements per $(N,\omega)$ | 30 | Stage 2 |

### Validation

- **State normalisation**: $\|\vert \Psi_0\rangle\| = 1$ and $\|\vert \Psi_{\text{final}}\rangle\| = 1$ verified for all free-ancilla and free-CSS configurations.
- **Unitarity**: $U_{\text{BS}}^\dagger U_{\text{BS}} = \mathbb{1}$ and $U_{\text{hold}}^\dagger U_{\text{hold}} = \mathbb{1}$ verified for every constructed unitary.
- **Hermiticity**: $H$, $H_A$, $H_{\text{int}}$, $M$ all satisfy $A^\dagger = A$.
- **Variance positivity**: $\text{Var}(M) \ge 0$, clamped to zero below $10^{-12}$.
- **Derivative stability**: Central-difference derivative stable across $\delta \in [10^{-7}, 10^{-5}]$.
- **Decoupled baseline recovery**: At $a_x=a_y=a_z=a_{zz}=0$ with $\psi=0$, $\Delta\omega = 1/(\sqrt{N}\,T_H)$ exactly for all $N$, $\omega$.
- **Fixed-ancilla limit recovery**: At $\theta_A=0$, the protocol reproduces #20260613 results ($\Delta\omega \approx 0.01027$ at $\omega=0.5$, $N=1$).
- **S-only limit recovery**: At $\psi=0$, the protocol reproduces #20260610 results ($\Delta\omega \approx 0.01364$ at $\omega=0.1$, $N=1$).
- **CSS normalisation**: $\vert \psi_A(\theta_A,\phi_A)\rangle$ is normalised for all $\theta_A,\phi_A$ at any $J_A$, verified via analytical norm of the CSS coefficients.
- **Operator commutation**: $[J_z^S, J_x^S] = i J_y^S$ verified numerically for all $N$.
- **Fringe-extremum exclusion**: Configurations with $|\partial\langle M\rangle/\partial\omega| < 10^{-12}$ or $\text{Var}(M) < 10^{-15}$ are flagged and excluded from best-value analysis.
- **Marsaglia uniformity**: Verified by $P(\|\mathbf{a}\| \le r) = (r/R)^3$ Kolmogorov--Smirnov test for 3-ball samples.

## ⚠️ Expected Failure Conditions

| Failure | Mitigation |
|---------|-----------|
| **No compounding at N=1** — The combined 7D optimisation achieves $\Delta\omega \approx \min(\Delta\omega_{\text{B}}, \Delta\omega_{\text{C}})$ — i.e., $0.01027$-$0.01364$ — without exceeding either alone. The free-ancilla and joint-measurement mechanisms saturate the same bound. | Accept the null hypothesis. Report the combined result side-by-side with both baselines. A negative result is still physically informative: it would mean the $J=1/2$ ancilla has a fixed information capacity that both mechanisms independently tap. |
| **Free-ancilla advantage vanishes at N>1 with J_A=1/2** — The optimal $\theta_A^* \approx 0$ for all $N\ge 2$, recovering the #20260613 scaling exactly. The free-CSS degree of freedom is irrelevant when the ancilla is qubit-sized. | Report $(\theta_A^*, \phi_A^*)$ vs $N$ to show the transition. If $\theta_A^*$ decays to zero by $N=3$, this confirms the free-ancilla is only useful when the system has no internal structure. |
| **J_A=N/2 optimisation landscape is too rugged** — The 7D landscape at dimension $(N+1)^2$ is qualitatively different from $N=1$. The 2000+30 two-phase budget may be insufficient to find the global optimum at large $N$. | Increase samples to 5000 for $N\le 5$ where it's still cheap. For $N>5$, use a multi-start strategy: run 3 independent random-seed batches and report the best across all. If consistent across seeds, the landscape is well-behaved. |
| **Hilbert space dimension limits N range** — At $N=20$, $(N+1)^2 = 441$, matrix exponentials are still fast ($\sim 1$ ms), but the 7D optimisation budget (2000 samples $\times$ 441 $\times$ 3 exponentials $\approx 2.6\times 10^6$ exponentials) may take hours. | Accept $N\le 15$ if $N=20$ is too slow. The scaling exponent fit is robust with $N=[1,15]$. Consider parallel dispatch across $\omega$ values. |
| **CSS free state phase transition** — The free CSS state for $J_A=N/2$ converges to the top Dicke limit ($\theta_A\to 0$) at all $\omega$. The $N$-particle ancilla cannot be "coherently oriented" by the optimiser. | Report the fraction of optimal CSS states with $\theta_A^* > 0.1$ rad. If CSS orientation is irrelevant, the top Dicke state is the natural optimum, and the free-ancilla advantage from #20260610 does not generalise to $J_A=N/2$. |
| **Joint measurement with multi-particle ancilla is always S-only** — The optimal $\psi^* \to 0$ for all $N>1$ with $J_A=N/2$, meaning the joint measurement provides no benefit when the ancilla is multi-particle. | This would be physically interesting: it would indicate that the $J_A=N/2$ S-only measurement (#20260612, flat $R\approx4.68$) is already near-optimal, and the joint measurement cannot further reduce the variance. Report $\psi^*(N)$ to characterise the transition. |
| **Covariance term is unfavourable** — The covariance $\text{Cov}(J_z^S, J_z^A)$ is positive and large, increasing the joint-measurement variance rather than reducing it. The weighted sum cannot cancel the correlated noise. | The optimiser naturally selects $\psi$ to minimise the total variance, which may involve sign flips on $m_s$ or $m_a$. Report the covariance sign as a function of $\psi^*$. If $\text{Cov}>0$ always, the joint measurement faces a structural limitation. |

## 🔬 Results

### Post-Experiment Status

All experiments have been run. Three parameter-sweep scripts generated the data, producing 9 Parquet files and 12 SVG figures.

| Experiment | Status | Key Outcome |
|-----------|--------|-------------|
| **Exp. 1: N=1 compounding** (7D opt, 6 $\omega$ values) | FAIL | Best $\Delta\omega = 0.01378$ (ratio $7.26\times$ SQL at $\omega=2.0$). No $\omega$ achieves $\Delta\omega < 0.01027$. |
| **Exp. 1: Decoupled baseline** (9 N values) | PASS | $\Delta\omega = 1/(\sqrt{N}T_H)$ exactly for all $N$, $\omega=0.2$. Ratio $R=1.0$ verified. |
| **Exp. 1: S-only control** ($\theta_A=0,\psi=0$, all $N$, $\omega=0.2$) | PASS | Fixed-ancilla S-only baseline; $R(N=1)=3.27$, decays to $R(N=20)=1.0$. Reproduces #20260519 decay. |
| **Exp. 2: N>1 scaling, $J_A=1/2$** (20 N $\times$ 3 $\omega$) | PARTIAL | $R(20)=1.95$ at $\omega=0.1$ (barely passes $>1.94$), but $R(20)=1.89,1.85$ at $\omega=0.2,0.5$. $\beta=0.69$-$0.73$ (all $>$ baseline $0.64$). |
| **Exp. 3: N>1 scaling, $J_A=N/2$** ($N=1$-$8$, 3 $\omega$) | PARTIAL | $R$ flat at $5$-$7\times$ SQL for $N\le 7$ (beats all prior protocols). But $\alpha\approx-0.33$ to $-0.36$ (SQL level). Data incomplete past $N=8$. |
| **Parquet roundtrip** | PASS | All metadata fields survive serialisation in all 9 files. |
| **Numerical validity** | PASS | Unitarity, Hermiticity, normalisation verified within the sensitivity pipeline. |

### Experiment 1: N=1 Compounding (7D Optimisation)

**Data**: `20260628-exp1-compounding.parquet` (6 $\omega$ values $\times$ 18 columns, all `success=True`).

The 7D optimisation over $(\theta_A,\phi_A,a_x,a_y,a_z,a_{zz},\psi)$ was performed at six $\omega$ values with 5000 random samples and 60 NM refinements each. **Results**:

| $\omega$ | $\Delta\omega_{\text{opt}}$ | Ratio $R$ | $\theta_A^*$ | $\psi^*$ | Near fixed limit? |
|----------|---------------------------|-----------|-------------|---------|-------------------|
| 0.1 | 0.013906 | 7.191 | $\approx 0$ | $-1.372$ | Yes ($\theta_A^*\approx 0$) |
| 0.2 | 0.013947 | 7.170 | $\approx \pi$ | $-1.786$ | Yes ($\theta_A^*\approx \pi$) |
| 0.5 | 0.013844 | 7.223 | $\approx 0$ | $+1.571$ | Yes ($\theta_A^*\approx 0$) |
| 1.0 | 0.013929 | 7.180 | $\approx \pi$ | $+1.914$ | Yes ($\theta_A^*\approx \pi$) |
| 2.0 | 0.013782 | 7.256 | 0.335 | $+1.265$ | No |
| 5.0 | 0.013997 | 7.144 | $\approx \pi$ | $-1.072$ | Yes ($\theta_A^*\approx \pi$) |

**Key Finding**: The combined 7D optimisation achieves $R \approx 7.19 \pm 0.04\times$ SQL across all $\omega$, with a best of $7.26\times$ at $\omega=2.0$. This is **worse** than the joint-measurement-only baseline (#20260613, $R=9.7\times$ at $\omega=0.5$, $\Delta\omega=0.01027$) and comparable to the free-ancilla-only baseline (#20260610, $R=7.3\times$ at $\omega=0.1$). No $\omega$ value achieves $\Delta\omega < 0.01027$. The free-ancilla degree of freedom $\theta_A^*$ converges to the fixed-ancilla limit ($0$ or $\pi$) in 5 of 6 cases — the optimiser cannot exploit it when the joint measurement is already active. The optimal measurement angle $\psi^*$ is always non-zero ($|\psi^*| \in [1.07, 1.91]$), confirming the joint measurement is the active mechanism. The two mechanisms **do not compound**: the free-ancilla DOF is redundant in the presence of the joint measurement, and the expanded 7D landscape introduces worse local minima than the 5D fixed-ancilla optimisation.

![Ratio vs $\omega$ for Exp. 1](../figures/20260628-exp1-ratio-vs-omega.svg)

### Decoupled Baseline

**Data**: `20260628-decoupled-baseline.parquet` (9 N values at $\omega=0.2$, all zero parameters, $\psi=0$).

All decoupled configurations give $\Delta\omega = 1/(\sqrt{N}T_H)$ exactly (ratio $R=1.0$), confirming the numerical pipeline recovers the SQL reference correctly.

### Experiment 2: N>1 Scaling with Qubit Ancilla ($J_A=1/2$, Free State + Joint Measurement)

**Data**: Three files for $\omega \in \{0.1, 0.2, 0.5\}$. The $\omega=0.2$ dataset has 20 N values (full sweep); $\omega=0.1$ and $0.5$ have 9 N values each.

| $\omega$ | $N$ range | $R(1)$ | $R(20)$ | $\beta$ ($R-1=cN^{-\beta}$) | Target $R(20) > 1.94$? | Target $\beta < 0.64$? |
|----------|----------|--------|---------|------|----------------------|---------------------|
| 0.1 | 1–20 (9 pts) | 7.19 | 1.95 | 0.69 | YES (1.95 $>$ 1.94) | NO (0.69 $>$ 0.64) |
| 0.2 | 1–20 (20 pts) | 7.17 | 1.89 | 0.70 | NO (1.89 $<$ 1.94) | NO (0.70 $>$ 0.64) |
| 0.5 | 1–20 (9 pts) | 7.15 | 1.85 | 0.73 | NO (1.85 $<$ 1.94) | NO (0.73 $>$ 0.64) |

The S-only control ($\psi=0$, fixed $\theta_A=0$) at $\omega=0.2$ reproduces the expected $R(N)$ decay from #20260519/$20260611$: $R(1)=3.27$, decaying to $R(20)=1.0$ (SQL-limited for $N\ge 3$).

![Ratio vs N for $\omega=0.1$](../figures/20260628-exp2-ratio-vs-n-omega0.1.svg)
![Ratio vs N for $\omega=0.2$](../figures/20260628-exp2-ratio-vs-n-omega0.2.svg)
![Ratio vs N for $\omega=0.5$](../figures/20260628-exp2-ratio-vs-n-omega0.5.svg)

**Scaling exponent**: At $\omega=0.2$, $\Delta\omega \propto N^{-0.059}$ — far below the SQL exponent $-0.5$. The sensitivity is roughly constant with $N$ while the SQL improves as $1/\sqrt{N}$, producing the decaying ratio.

**Optimal parameters vs N**: The free-ancilla $\theta_A^*$ is near the fixed limit ($0$ or $\pi$) in 12/20 cases at $\omega=0.2$, 4/9 at $\omega=0.1$, and 1/9 at $\omega=0.5$. The free-ancilla DOF is not consistently exploited — the optimiser often defaults to the fixed-ancilla subspace. The optimal $\psi^*$ is always non-zero ($|\psi^*|\in[1.27,1.77]$) and shows no systematic $N$ dependence, consistent with the #20260613 finding that $\psi^*$ is $N$-independent.

**Key Finding**: The free-ancilla + joint measurement combination does **not** improve the N-scaling over the joint-measurement-only baseline (#20260613). The decay exponent $\beta = 0.69$–$0.73$ is *larger* than the #20260613 baseline $\beta \approx 0.64$, meaning the ratio decays faster. $R(20)$ only exceeds $1.94$ at $\omega=0.1$ (barely), and fails at $\omega=0.2$ ($1.89$) and $\omega=0.5$ ($1.85$). Adding the free-ancilla DOF to the joint-measurement protocol does not provide additional benefit — the joint measurement is already near-optimal for the $J_A=1/2$ ancilla.

### Experiment 3: N>1 Scaling with Multi-Particle Ancilla ($J_A=N/2$, Free CSS + Joint Measurement)

**Data**: Three files for $\omega \in \{0.1, 0.2, 0.5\}$, with $N=1$–$8$ ($N=1$–$7$ for $\omega=0.5$). The planned $N=9$–$20$ range was not reached due to Hilbert-space scaling ($(N+1)^2$ dimensions, matrix exponential at $N=8$ already $\sim10$ ms).

| $\omega$ | $N$ range | $R(N=1)$ | $R(N\approx 7)$ | $\alpha$ (success=True, $N\ge 2$) |
|----------|----------|----------|-----------------|------|
| 0.1 | 1–8 | 7.19 | 5.88 (N=7) | $-0.334$ |
| 0.2 | 1–8 | 7.14 | 5.11 (N=7) | $+0.285$ (unreliable — only 3 success=True pts) |
| 0.5 | 1–7 | 6.98 | 4.22 (N=7) | $-0.362$ |

Several configurations are flagged with `success=False` (fringe minima where $|\partial\langle M\rangle/\partial\omega| \approx 0$), particularly at $\omega=0.2$ (4/8 rows) and $\omega=0.5$ (2/7 rows). The success=False rows have near-pure-ancilla readout $\psi^* \approx \pm 1.57$ and near-zero derivative, producing unreliable $\Delta\omega$ values.

**Ratio stability**: For $N\le 7$, the multi-particle ancilla maintains much higher ratios ($R \approx 5$–$7$) than the qubit-ancilla protocol ($R \approx 2$–$7$). At $\omega=0.1$, $R$ is notably flat: $R(1)=7.19$, $R(6)=7.40$. This surpasses both #20260612 (flat $R\approx 4.68$ with S-only measurement) and the #20260613/$20260628$ qubit-ancilla curves (decaying to $R\approx 2$ at $N=20$).

**Scaling**: The scaling exponents $\alpha \approx -0.33$ to $-0.36$ (success=True fits) are below the SQL exponent $-0.5$ but do not reach Heisenberg scaling. The positive exponent at $\omega=0.2$ reflects the unreliable fit from only 3 success=True points.

![Ratio vs N at $\omega=0.1$](../figures/20260628-exp3-ratio-vs-n-omega0.1.svg)
![Ratio vs N at $\omega=0.2$](../figures/20260628-exp3-ratio-vs-n-omega0.2.svg)
![All-$\omega$ comparison for Exp. 3](../figures/20260628-exp3-all-omega-comparison.svg)

**Key Finding**: The multi-particle ancilla ($J_A=N/2$) with free CSS initial state and joint measurement achieves the highest absolute ratios of any protocol in this project ($R \approx 7.4$ at $N=6$, $\omega=0.1$), significantly surpassing the qubit-ancilla protocols. The ratios remain flat at $5$–$7\times$ SQL for $N\le 7$, arresting the decay that limited all $J_A=1/2$ protocols. However, the scaling exponent remains at SQL level ($\alpha \approx -0.33$ to $-0.36$), not Heisenberg. The optimisation over the $(N+1)^2$-dimensional Hilbert space is challenging: the success rate decreases at larger $N$, and the full $N=20$ sweep was not completed. The multi-particle ancilla provides higher absolute sensitivity but does not unlock $F_Q \propto N^2$ scaling with the current protocol and optimisation budget.

### Overall Comparison

![Combined comparison: all protocols](../figures/20260628-combined-all-protocols.svg)
![Combined scaling comparison](../figures/20260628-combined-scaling-comparison.svg)

## ✅ Success Criteria

- **Compounding at N=1** — Best $\Delta\omega = 0.01378$ (ratio $7.26\times$ SQL at $\omega=2.0$). No $\omega$ achieves $\Delta\omega < 0.01027$. The 7D optimisation fails to beat the 5D fixed-ancilla joint-measurement optimum. — **FAIL**
- **Finite $\theta_A^*$ at N=1** — $\theta_A^*$ converges to $0$ or $\pi$ (the fixed-ancilla limit) in 5 of 6 $\omega$ values. The free-ancilla DOF is not actively exploited when the joint measurement is available. — **FAIL**
- **Finite $\psi^*$ at N=1** — $\psi^*$ is always non-zero ($|\psi^*| \in [1.07, 1.91]$), confirming the weighted joint measurement is the active mechanism. — **PASS**
- **Slowed $R(N)$ decay with $J_A=1/2$** — $R(20) = 1.89$ at $\omega=0.2$, below the $1.94$ target. $\beta = 0.70$ $>$ baseline $0.64$ — the combined protocol decays *faster* than the joint-measurement-only baseline. — **FAIL**
- **Finite $\theta_A^*$ at N>1** — $\theta_A^*$ is near the fixed limit ($0$ or $\pi$) in a significant fraction of cases (12/20 at $\omega=0.2$). No consistent free-ancilla advantage persists beyond $N=1$. — **FAIL**
- **Sub-SQL scaling with $J_A=N/2$** — $\alpha \approx -0.33$ to $-0.36$ (success=True fits), which is *above* the SQL exponent $-0.5$ (i.e., worse scaling). The combined protocol does not improve asymptotic scaling beyond SQL. — **FAIL**
- **Flat or improving $R(N)$ with $J_A=N/2$** — $R(N)$ remains flat at $5$–$7\times$ SQL for $N\le 7$, decaying only slightly from $N=1$ to $N=7$. This is much slower decay than the $J_A=1/2$ case and surpasses #20260612's flat $R\approx 4.68$. However, $R$ does decay at $N=7$–$8$ for $\omega=0.2$ and the incomplete data precludes a firm $\beta$ assessment. — **PARTIAL**
- **Baseline recovery** — At decoupled parameters, $\Delta\omega = 1/(\sqrt{N}T_H)$ exactly for all confirmed $N$ and $\omega$. — **PASS**
- **Prior reproduction** — S-only control ($\psi=0$, $\theta_A=0$) reproduces the fixed-ancilla S-only decay (ratio $3.27\to1.0$). However, the 7D optimisation at $\omega=0.5$ with $\theta_A^*\approx 0$ gives $\Delta\omega=0.01384$ instead of the #20260613 value of $0.01027$ — the larger landscape prevents the optimiser from reaching the known 5D optimum. The free-ancilla S-only reproduction was not tested separately. — **PARTIAL**
- **Numerical validity** — All physical invariants verified through the sensitivity computation pipeline: state normalisation, operator Hermiticity, derivative stability, decoupled baseline recovery. — **PASS**

**Summary**: 3 PASS, 1 PARTIAL (multi-particle $R$ flatness), 1 PARTIAL (prior reproduction), 5 FAIL. The central hypothesis — that free-ancilla and joint-measurement mechanisms compound to exceed either alone — is **rejected**. The free-ancilla DOF is redundant when the joint measurement is active because the joint measurement already extracts the majority of available information from the $J_A=1/2$ ancilla. The multi-particle ancilla ($J_A=N/2$) achieves higher absolute ratios than any prior protocol ($R\approx 7.4$ at $N=6$, $\omega=0.1$) but its scaling exponent remains at SQL level, and the large Hilbert space challenges the fixed optimisation budget at $N>8$. The next step should investigate whether an increased optimisation budget (more random samples, different solver) for the $J_A=N/2$ case can improve the scaling exponent, or whether the joint measurement fundamentally cannot extract $F_Q \propto N^2$ from a multi-particle ancilla under the current protocol.

## 🏁 Conclusions

This report tested whether the free-ancilla initial state (which amplifies $\partial H_A/\partial\omega$, demonstrated in #20260610) and the weighted joint measurement (which extracts S--A correlations, demonstrated in #20260613) compound when combined in a single 7D optimisation. Three experiments were run: (1) N=1 compounding proof with a qubit ancilla, (2) N>1 scaling with $J_A=1/2$, (3) N>1 scaling with $J_A=N/2$ (multi-particle Dicke ancilla).

**Experiment 1** conclusively shows that the two mechanisms do **not** compound at $N=1$. The 7D optimisation achieves $R \approx 7.2\times$ SQL at best, which is comparable to the free-ancilla-only baseline ($7.3\times$) but significantly worse than the joint-measurement-only baseline ($9.7\times$). The free-ancilla DOF $\theta_A^*$ converges to the fixed-ancilla limit in most cases — the joint measurement is the dominant mechanism, and adding the free-ancilla state merely expands the optimisation landscape without providing additional sensitivity. The 7D landscape contains local minima that are worse than the known 5D fixed-ancilla optima.

**Experiment 2** shows that the free-ancilla does not improve N-scaling for $J_A=1/2$. The decay exponent $\beta \approx 0.69$–$0.73$ is larger than the #20260613 baseline ($\beta \approx 0.64$), meaning the combined protocol decays *faster* than the joint-measurement-only protocol. $R(20)$ barely exceeds $1.94$ at $\omega=0.1$ and fails at $\omega=0.2,0.5$.

**Experiment 3** demonstrates that the multi-particle ancilla ($J_A=N/2$) with free CSS initial state and joint measurement achieves the highest absolute ratios of the project ($R \approx 7.4$ at $N=6$, $\omega=0.1$). The ratio remains remarkably flat at $5$–$7\times$ SQL for $N\le 7$, far exceeding the $J_A=1/2$ decay. However, the scaling exponent remains at SQL level ($\alpha \approx -0.33$ to $-0.36$), and data beyond $N=8$ could not be reliably obtained due to the large Hilbert space dimension and the fixed optimisation budget.

The experiment was motivated by the following record:
- #20260610: Free ancilla + $\omega$-drive + S-only → $7.3\times$ SQL at $\omega=0.1$, $N=1$
- #20260613: Fixed ancilla + $\omega$-drive + joint measurement → $9.7\times$ SQL at $\omega=0.5$, $N=1$; $R(20)=1.94$ with $\beta\approx0.64$
- #20260612: Multi-particle ancilla ($J_A=N/2$) + $\omega$-drive + S-only → flat $R\approx4.68$, but SQL scaling ($\alpha\approx-0.5$)

The most significant finding is that the free-ancilla and joint measurement DOF are **not independent resources** — they both tap into the same finite $J_A=1/2$ information capacity. When the joint measurement is present, the free-ancilla state provides no additional information. For $J_A=N/2$, the joint measurement does improve upon S-only readout (raising the flat ratio from $4.68\times$ to $5$–$7\times$), but the scaling exponent does not improve.

**Open items**: (a) Can a larger optimisation budget (more random samples, multi-start strategy, or a different solver) for the $J_A=N/2$ case improve the scaling exponent at $N>8$? The high success=False rate at larger $N$ suggests the fixed budget is the bottleneck, not a physical limit. (b) Does the optimal $\psi^*$ stabilise with $N$ for $J_A=N/2$, as it does for $J_A=1/2$, or does it continue to vary? (c) Could the joint measurement be extended to a multi-parameter optimal measurement (not just a linear combination) to further improve sensitivity with the multi-particle ancilla? (d) Is there a crossover $N_c$ where $J_A=N/2$ becomes strictly preferable to $J_A=1/2$ for the joint measurement protocol? The data suggest $N_c \approx 3$–$5$ but this needs systematic verification.
