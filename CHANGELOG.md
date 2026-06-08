# Changelog

All notable experiments and infrastructure changes in this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
with weekly groupings corresponding to experimental campaigns.

## Week 23 (Jun 1–7)

### New Report
- **Heisenberg-Limit MZI with NOON and Twin-Fock States** (#20260601) — CFI vs error-propagation in the standard MZI shows $\langle J_z\rangle_{\text{out}} = 0$ degeneracy for NOON $N\ge 2$. CFI from the full $P(m|\theta)$ distribution resolves the degeneracy. Both NOON ($F_C = H_t^2 N^2$, $\alpha=-1.0$) and Twin-Fock ($F_C = H_t^2 N(N+2)/2$, $\alpha=-0.907$) saturate their QFI bounds. CFI is $\theta$-independent for both states. Number-difference measurement is optimal in the balanced MZI.

### Infrastructure
- **CHANGELOG.md** — Created CHANGELOG.md with weekly groupings (Keep a Changelog format). Backlog of 19 items with priority emojis and `Next step:` pipeline annotations. Partially completed reports tracked separately. Updated all 4 skill files and research-assistant.md to reference the CHANGELOG and removed stale `## [Unreleased]` references. Consolidated "New Report" / "Report Variation" categories, removed monthly section headers, and lowered all backlog priorities by one level.

## Week 20 (May 11–17)

### New Report
- **Scaling Survey: Fock-Basis MZI Models** (#20260511) — Systematic scaling analysis of coherent, NOON, and squeezed-vacuum states under Markovian decoherence. Coherent states achieve SQL ($\alpha=-0.5$). Squeezed-vacuum achieves Heisenberg-like scaling ($\alpha\to-1.0$) with $F_Q = 2\langle N\rangle(\langle N\rangle+1)$, beating the NOON-state HL prefactor by $2\times$. NOON collapses to SQL under any loss; two-body loss causes complete collapse ($\alpha\to0$) for all states at large $N$.
- **Scaling Survey: Collective-Spin / Dicke Basis** (#20260511) — OAT spin squeezing achieves $\alpha=-2/3$ at optimal $t_{\text{opt}}\propto N^{-1/3}$. Balanced Dicke (Twin-Fock) achieves $\alpha=-1.0$ (Heisenberg) with $F_Q = N(N+2)/3$. Non-Gaussian states have fractional exponents $\alpha\approx-0.75$ to $-0.85$.
- **Ancilla-Assisted vs Two-System-Particle Comparison** (#20260511) — Fixed 2 total particles: ancilla-assisted configuration (1 S + 1 A) cannot beat two-system-particle configuration (2 S). Ratio $\mathcal{R} = \Delta\theta_A / \Delta\theta_B = 2.02$. Fundamental limitation: $J=1/2$ probe cannot behave like a $J=1$ probe.
- **Single-Particle MZI Holding-Time Scaling** (#20260512) — Baseline reference: $\Delta\theta = 1/T_H$ (SQL) exactly for $T_H\in[10^{-2},10^3]$, log-log scaling exponent $\alpha=-1.0$. Analytical and finite-difference derivatives agree to machine precision. At fringe extrema ($\sin(\theta T_H)=0$) the sensitivity diverges as expected.
- **Ancilla-Assisted Metrology: Full-Parameter Optimization** (#20260512) — Nelder--Mead optimization over all four bilinear interaction coefficients $(\alpha_{xx},\alpha_{xz},\alpha_{zx},\alpha_{zz})$ in a two-qubit ancilla-assisted MZI. Optimal configuration is always $\alpha_{ij}=0$, recovering the SQL. Non-zero $\alpha_{ij}$ generally degrades sensitivity when measuring $J_z^S$ alone. See #20260511.
- **Ancilla-Assisted Metrology with Joint Measurement** (#20260515) — Joint measurement $M = J_z^S + J_z^A$ at $\alpha\neq0$ recovers sensitivity that S-only measurement loses, but never beats the SQL ($\Delta\theta = 1/T_H$ at best). QFI bound $F_Q \leq T_H^2$ is fundamental. See #20260512.

## Week 21 (May 18–24)

### New Report
- **Ancilla-Drive-Enhanced Metrology** (#20260518) — Active ancilla drive $H_A = a_xJ_x^A + a_yJ_y^A + a_zJ_z^A$ with Ising interaction $H_{\text{int}} = a_{zz}J_z^S\otimes J_z^A$. No SQL violation across 2500 4D random-search evaluations, 10 complete 2D slice grids, or 250 Nelder--Mead refinements. The $J=1/2$ spectral radius bound is absolute for $\theta$-independent ancilla drive. See #20260515.
- **Weighted Joint Measurement: N-M Generalization** (#20260518) — Joint measurement $M = aJ_z^S + bJ_z^A$ with $a^2+b^2=1$ and CSS initial states at $N$ system and $M$ ancilla particles. With free interaction coefficients, the optimiser turns the interaction off ($\nu=0.085$, 95% CI $[-0.27,0.24]$, far below SQL $\nu=0.5$). With constrained interaction ($\alpha_{xx}\neq0$), weighted measurement beats S-only up to $85\%$ at $\alpha_{xx}=-1.8$. See #20260515.
- **Ancilla-Drive Phase-Modulated Metrology** (#20260519) — Making the ancilla drive $\theta$-dependent ($H_A = \theta\cdot\mathbf{a}\cdot\mathbf{J}^A$) unlocks SQL violation: best $\Delta\theta = 0.02036$, which is $4.91\times$ below the SQL at $\theta\approx0.2$. Mechanism: static BCH cross-term $[\theta J_z^S,\, a_{zz} J_z^S\otimes J_z^A]$ generates effective $J_z^A$ contribution. Violation requires both $a_{zz}\neq0$ and small $\theta$. See #20260518.
- **XX-Coupling Ancilla Metrology** (#20260520) — Symmetric phase encoding ($H_A = \theta J_z^A$) with transverse coupling $H_{\text{int}} = \alpha_{xx} J_x^S \otimes J_x^A$ and S-only measurement after trace. No SQL violation: $\alpha_{xx}=0$ always optimal for all $\theta$ and $\alpha_{xx}\in[0,20]$. XX coupling creates transverse entanglement that is destroyed by the trace.
- **General-Interaction Ancilla Metrology** (#20260521) — Symmetric encoding with full four-parameter interaction $(\alpha_{xx},\alpha_{xz},\alpha_{zx},\alpha_{zz})$ and L-BFGS-B optimization. Best $\Delta\theta = 0.0690$ ($0.690\times$ SQL) at $\theta=3.8$. Dominant couplings are $\alpha_{zx}$ and $\alpha_{zz}$, generating SQL violation through higher-order BCH cross-terms with $H_0 = \theta(J_z^S + J_z^A)$. See #20260518, #20260520.
- **Multi-Particle XX-Coupling Dual-MZI** (#20260522) — $N\in[1,20]$ particles in both S and A, dual MZI (BS on both before and after hold), $H_{\text{int}} = \alpha_{xx} J_x^S \otimes J_x^A$ only, S-only measurement after trace. No SQL violation at any $N$: $\alpha_{xx}^*=0$ always. Dual MZI does not unlock XX-coupling advantage. See #20260520.
- **Four-Parameter Coupling Multi-Particle Dual-MZI** (#20260523) — Full $(\alpha_{xx},\alpha_{xz},\alpha_{zx},\alpha_{zz})$ interaction with $N\in[1,10]$ and dual MZI. Weak SQL violation: best ratio $\Delta\theta/\text{SQL}=0.916$. Dual MZI symmetrization weakens the static cross-term generation that worked at $N=1$. See #20260522.
- **Phase-Diffusion Robustness of the Phase-Modulated Drive** (#20260524) — Sub-SQL advantage of the phase-modulated drive protocol survives phase diffusion up to $\gamma_\phi^*\approx0.08$ at $\theta=0.2$ and $\gamma_\phi^*\approx0.12$ at $\theta=1.0$. Optimal drive amplitudes $(a_x,a_y,a_z)$ decrease with $\gamma_\phi$ while $\alpha_{zz}$ increases to compensate. See #20260519.
- **Multi-Particle XX-Coupling Dual-MZI with Optimized Joint Measurement** (#20260525) — Joint measurement $M = m_s J_z^S + m_a J_z^A$ with $m_s^2+m_a^2=1$, jointly optimized with $\alpha_{xx}$. Best ratio $\Delta\theta/\text{SQL}_{2N}=0.509$ at $N=5$, $\theta=0.5$. Asymptotic scaling $\Delta\theta \propto 1/N^{0.75}$ (between SQL $\alpha=-0.5$ and HL $\alpha=-1.0$). Optimal weights are always unequal ($m_s\neq m_a$), with ancilla weight $m_a$ dominating at large $\alpha_{xx}$. See #20260522, #20260518.

## Week 22 (May 25–31)

### New Report
- **Drive-Component Analysis in Ancilla-Enhanced Metrology** (#20260527) — Systematic 2D slice analysis of $(a_x,a_{zz})$, $(a_y,a_{zz})$, and $(a_z,a_{zz})$ at 5 $\theta$ values each. All three achieve min ratio $=1.0$ (SQL). The commuting $(a_z,a_{zz})$ slice is qualitatively different: 0% of valid points show degradation, vs 78--87% for non-commuting $(a_x,a_{zz})$ and $(a_y,a_{zz})$. Norm-ball envelope up to $R=10$: zero of 250K samples beat SQL; best_ratio($r$) is non-increasing and saturates at $1.0$ for large $r$. Barycentric heatmaps confirm yellow (R+G) dominance. See #20260519, #20260521.
- **Free Ancilla Initial State in Driven-Ancilla Metrology** (#20260528) — Freeing the ancilla to $(\theta_A,\phi_A)$ does not enable SQL violation. All four scenarios (A: fixed-ancilla baseline, B: free+drive+interaction, C: free+interaction-only, D: free+drive-only) converge to exactly $\Delta\theta=0.1$ at all 5 $\theta$ values. The $(\theta_A,a_{zz})$ slice is flat at SQL across 40,401 grid points. The $J=1/2$ bound is absolute for product initial states. See #20260518, #20260519, #20260521.

### Infrastructure
- **Norm-ball Monte Carlo sampler** — Marsaglia 3-ball sampling for $\|\mathbf{a}\|\le R$ with envelope curve extraction, validated via Kolmogorov--Smirnov test against $(r/R)^3$ CDF.
- **Barycentric sensitivity heatmap** — Combined RGB visualization that encodes each drive component's $\log_{10}(\Delta\theta/\text{SQL})$ contribution into separate colour channels.
- **Free ancilla state preparation** — Generalized initial state $|\psi_A\rangle = \cos(\theta_A/2)|1,0\rangle + e^{i\phi_A}\sin(\theta_A/2)|0,1\rangle$ as a 4D complex vector, accepted by the existing circuit evolution pipeline.
- **Cross-scenario comparison framework** — Four-scenario dispatcher with common random-search and Nelder--Mead refinement pipeline, norm-envelope overlay, and summary statistics.
- **2D slice $(\theta_A, a_{zz})$** — Generic 201$\times$201 grid scan supporting any controllable parameter against $a_{zz}$, enabling comparisons with existing $(a_x,a_{zz})$, $(a_y,a_{zz})$, $(a_z,a_{zz})$ slices.

---

# Backlog

## Partially Completed

Reports that have been started but are not yet fully complete (i.e., not all pipeline steps finished: plan report, implement code, review implementation, generate raw data and figures, write final report).

- 🟡 **High-Order Squeezing Plan** (#20260507) — Hypothesis and theoretical model written. Next step: implement code (build the bosonic+spin oscillator simulation, Lindblad master equation solver with order-$n$ spin-dependent forces).
- 🟢 **Non-Markovian Bath with Ancilla Protection** (#20260509) — Infrastructure complete and validated (75 unit tests). Pseudomode representation of Lorentzian bath, tripartite evolution, and QFI preservation pipeline all operational. Next step: generate raw data and figures (parameter sweeps over coupling $g$ and bath correlation $\lambda$).
- 🟡 **Advanced Architecture Surveys** (#20260511) — Theoretical framework documented for six models (non-Markovian bath, thermal Langevin noise, cavity-enhanced MZI, distributed sensor arrays, dynamical decoupling, tilt-to-length noise). Next step: implement code (build model-specific simulators and figure generators).

## Ancilla-Enhanced Metrology

- 🟠 **3D drive-component landscape** — Beyond 20260527's one-at-a-time 2D slices: study the sensitivity landscape with all three drive components $(a_x, a_y, a_z)$ active simultaneously. Does the commuting $a_z$ drive "protect" against degradation from $a_x$ and $a_y$? Are there interference effects when all three are non-zero? See #20260527.
- 🟠 🚫 **Stratified norm-ball sampling at small $\|\mathbf{a}\|$** — The Marsaglia method under-samples the small-$r$ regime (only $\sim 1\%$ of samples at $\|\mathbf{a}\| \le 2.15$). Use stratified sampling (uniform in $r$) to resolve the envelope at $\|\mathbf{a}\| \le 1$ and determine whether the best ratio at small drive is meaningfully worse than SQL. Blocked by Infrastructure & Tooling: Stratified Monte Carlo sampler. See #20260527.
- 🟠 **Free ancilla with $\theta$-modulated drive** — Combine the free-ancilla initial state (20260528, Scenario B) with the $\theta$-modulated drive mechanism (20260519: $H_A = \theta\cdot\mathbf{a}\cdot\mathbf{J}^A$). The 20260519 protocol achieved $4.91\times$ below SQL with a fixed ancilla $|1,0\rangle$; can a free ancilla $(\theta_A,\phi_A)$ improve this further? Does the optimal ancilla state depend on $\theta$? See #20260519, #20260528.
- 🟡 **Entangled initial S--A state in driven-ancilla metrology** — Replace the product initial state $|1,0\rangle_S \otimes |\psi_A\rangle$ with a maximally entangled Bell state $|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt{2}$ and test whether the $J=1/2$ bound still holds when S and A are entangled from the start.
- 🟡 **Multi-particle ancilla ($J_A > 1/2$) with free initial state** — Test whether an ancilla with $J_A = 1, 3/2, 2$ (M=2,3,4 particles) can circumvent the single-qubit SQL bound through its larger Hilbert space.
- 🟡 **Simple evolution structures for Heisenberg-limit scaling with an ancilla** — Read reports/findings/interferometric_sensitivity_improvements.md and relevant reports. Suggest simple evolution structures (Hamiltonians, measurement schemes, encoding strategies) that would unblock Heisenberg-limit scaling with an ancilla system, while the original system setup could not beat SQL scaling.
- 🟡 **Joint measurement with driven ancilla** — Combine the $\theta$-modulated drive (20260519) with an optimized weighted joint measurement $M = aJ_z^S + bJ_z^A$ (20260525) to test whether the two enhancement mechanisms compound. See #20260519, #20260525.
- 🟡 **Phase-modulated drive with larger $N$** — Extend the 20260519 phase-modulated drive mechanism to $N>1$ system particles. Does the SQL-violation ratio $(\Delta\theta_{\text{SQL}}/\Delta\theta)$ improve with system size, or does it saturate? See #20260519.
- 🟢 **Free ancilla with joint measurement** — Apply the free-ancilla initial state (20260528) to the weighted joint measurement protocol (20260518 NM generalization) to test whether a free ancilla unlocks better joint-readout sensitivity than a fixed $|1,0\rangle$ ancilla. See #20260518, #20260528.

## Squeezed-State & Non-Gaussian Metrology

- 🟡 **Heisenberg-limited MZI with squeezed-vacuum and OAT** — Extend the CFI analysis (#20260601) to squeezed-vacuum input and OAT-generated spin-squeezed states; confirm $\alpha\to-1.0$ and compare prefactor with NOON/Twin-Fock. See #20260601.

## Foundational Analysis

- 🟢 **General POVM beyond linear combinations** — How much sensitivity can an informationally complete measurement on the full $(N+1)(M+1)$-dimensional space extract, compared to the weighted linear $M = aJ_z^S + bJ_z^A$? See #20260518.
- 🟢 **Analytical derivation: $(a_z, a_{zz})$ flatness** — Derive why $[a_z J_z^A, H] = 0$ leads to exactly SQL-level sensitivity for the full parameter space (observed in 20260527). See #20260527.

## Infrastructure & Tooling

- 🟠 **$\phi$ vs $\theta$ semantic audit** — Find all mentions of $\phi$ and $\theta$ in the codebase (`src/`, `pages/`, `reports/`). Clarify the semantic usage between the two: $\theta$ is the unknown phase rate to be estimated, $\phi$ is the measurement weight angle. Highlight any semantic inconsistencies (e.g., swapped usage, ambiguous variable names, conflicting conventions in different modules).
- 🟡 **3D slice visualization** — Heatmap infrastructure currently supports 2D slices only. For studying all three drive components simultaneously, 3D volumetric plots or 2D projections of 3D landscapes are needed.
- 🟡 **Stratified Monte Carlo sampler** — Implement stratified sampling on $r = \|\mathbf{a}\|$ (uniform in $[0,R]$) layered with uniform direction sampling on the 2-sphere, to resolve the small-$r$ regime with adequate density.
- 🟡 **CI/CD pipeline** — Automated test/lint/type-check on push and PR; automated CHANGELOG management.
- 🟢 **Performance benchmarks** — Automated per-function timing to catch regressions exceeding the 100 ms per-simulation budget.
- 🟢 **Advanced architecture simulation functions** — Implement six model-specific simulators and figure generators for the PENDING advanced architecture surveys (non-Markovian bath, thermal noise, cavity-enhanced MZI, distributed arrays, dynamical decoupling, tilt-to-length noise).
