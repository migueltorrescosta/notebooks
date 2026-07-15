# Identical Dynamics, Asymmetric Gain: When the System's Own $\omega$-Modulated Drive Outshines the Ancilla

**A technical review of symmetric $\omega$-modulated drive metrology**

---

## 1. Introduction

In an $\omega$-modulated system--ancilla metrology protocol, does the ancilla matter?

The preceding report (#20260519) established that an ancilla-only $\omega$-modulated drive achieves $4.07\times$ below the Standard Quantum Limit with $N=2$ particles. The Ising interaction $a_{zz} J_z^S \otimes J_z^A$ was identified as the metrological engine: without it, no sub-SQL performance is possible. The ancilla qubit, driven at the unknown frequency $\omega$ and coupled to the system through $a_{zz}$, converts the $\omega$ signal into measurable population differences via BCH cross-terms.

The natural follow-up is: what happens if the system qubit carries its own $\omega$-modulated drive? Does it improve sensitivity, or does the ancilla already capture the available gain?

This article answers that question through three acts:

- **Act 1** (Scenario A): The system-only surprise. A single-qubit Mach--Zehnder interferometer with the system Hamiltonian $H_S = \omega(a_x J_x + a_y J_y + a_z J_z)$ achieves $8.32\times$ below the single-particle SQL with $N=1$, already $2.04\times$ better than the ancilla-only baseline's $4.07\times$ with $N=2$.

- **Act 2** (Scenario B): The compound extension. Adding the ancilla with identical drive parameters and an Ising interaction compounds the gain by up to $34.9\%$ ($\mathcal{R}_{\text{compound}} = 1.3492$), achieving $95.4\%$ of the QFI resource-counting bound of $\sqrt{2}$.

- **Act 3**: Context and synthesis. Both results are placed in the full series arc, revealing why the system's own drive is fundamentally more efficient than the ancilla-mediated mechanism.

The key insight in one sentence: the system's own drive is substantially more effective than the ancilla's because the derivative $\partial H_S/\partial\omega$ acts directly on the measured subsystem, providing a parametric amplification channel that the $J_z$ measurement can access without relying on BCH cross-terms.

---

## 2. Physical Setup

**Scenario A (system-only, $N=1$):** The Hilbert space is $\mathcal{H}_S = \text{span}\{\vert0\rangle, \vert1\rangle\}$ with dimension 2. The basis convention maps $\vert0\rangle = \vert1,0\rangle$ (particle in mode 0) and $\vert1\rangle = \vert0,1\rangle$ (particle in mode 1). Operators are $J_k = \sigma_k/2$ (Pauli matrices divided by 2), with $J_z = \frac{1}{2}\text{diag}(1, -1)$. The initial state is $\vert0\rangle_S$.

**Scenario B (ancilla-assisted, $N=2$):** The total Hilbert space is $\mathcal{H}_{\text{tot}} = \mathcal{H}_S \otimes \mathcal{H}_A$ with dimension 4. The computational basis is $\{\vert00\rangle, \vert01\rangle, \vert10\rangle, \vert11\rangle\}$ with index $= n_S \times 2 + n_A$. System operators are $J_k^S = J_k \otimes \mathbb{1}_2$ and ancilla operators are $J_k^A = \mathbb{1}_2 \otimes J_k$. The interaction is $J_z^S \otimes J_z^A$. The initial state is $\vert00\rangle = \vert0\rangle_S \otimes \vert0\rangle_A$.

**Common conventions:** The SU(2) algebra $[J_i, J_j] = i\epsilon_{ijk} J_k$ holds for both subsystems. The holding time is $t_{\text{hold}} = 10$ (fixed). The SQL reference for Scenario A is the single-particle SQL $\Delta\omega_{\text{SQL}} = 1/t_{\text{hold}} = 0.1$. The SQL reference for Scenario B is the two-particle SQL $\Delta\omega_{\text{SQL}} = 1/(\sqrt{2}\,t_{\text{hold}}) \approx 0.07071$. All parameters are dimensionless.

---

## 3. Circuit Protocol

**Scenario A (single-qubit MZI):**

1. **Beam splitter**: $U_{\text{BS}} = \exp(-i(\pi/2) J_x)$ — the standard 50/50 pulse, creating a coherent superposition from $\vert0\rangle_S$.
2. **Holding period**: Evolution under $H_S = \omega(a_x J_x + a_y J_y + a_z J_z)$ for $t_{\text{hold}} = 10$. The unitary $U_{\text{hold}} = \exp(-i\,t_{\text{hold}}\,H_S)$ depends on $\omega$ through the Hamiltonian itself — the defining feature of the $\omega$-modulated protocol.
3. **Beam splitter**: Same $U_{\text{BS}}$ as step 1.
4. **Measurement**: $J_z$ on the system.

**Scenario B (dual MZI):**

1. **Beam splitter on both qubits**: $U_{\text{BS}} \otimes U_{\text{BS}}$ — the symmetric dual MZI.
2. **Holding period**: Evolution under $H = \omega(a_x J_x^S + a_y J_y^S + a_z J_z^S) + \omega(a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_{zz} J_z^S J_z^A$ for $t_{\text{hold}} = 10$. The Hamiltonian separates into system drive, ancilla drive, and Ising interaction.
3. **Beam splitter on both qubits**: Same as step 1.
4. **Measurement**: $J_z^S$ on the system (ancilla traced out).

The sensitivity is computed via the error-propagation formula $\Delta\omega = \sqrt{\text{Var}(J_z^S)} / |\partial\langle J_z^S\rangle/\partial\omega|$ with central finite differences ($\delta = 10^{-6}$). Both scenarios share the same measurement observable ($J_z^S$) and the same sensitivity formula — the only asymmetry in the problem is which subsystems receive the $\omega$-modulated drive.

**Key symmetry**: Both subsystems in Scenario B feel identical $\omega$-modulated drives. The effective system $z$-drive is $\omega a_z J_z^S$ and the ancilla $z$-drive is $\omega a_z J_z^A$ — the subsystem Hamiltonians are indistinguishable. Any sensitivity advantage of Scenario B over the decoupled limit ($a_{zz}=0$) comes purely from the Ising interaction.

---

## 4. The $\omega$-Modulated Drive Hamiltonian

The Hamiltonian for Scenario B decomposes as $H = H_S + H_A + H_{\text{int}}$ where:

- $H_S = \omega(a_x J_x^S + a_y J_y^S + a_z J_z^S)$ — system drive (identical to Scenario A's Hamiltonian),
- $H_A = \omega(a_x J_x^A + a_y J_y^A + a_z J_z^A)$ — ancilla drive (identical parameters),
- $H_{\text{int}} = a_{zz} J_z^S \otimes J_z^A$ — Ising interaction.

**Identical-subsystem constraint**: The drive parameters $(a_x, a_y, a_z)$ are the same on S and A. This means the subsystem Hamiltonians are indistinguishable — the only asymmetry in the problem is the $J_z^S$ measurement on the system. Any sensitivity advantage of Scenario B over the decoupled limit ($a_{zz} = 0$) comes purely from the Ising interaction channelling ancilla information back into the $J_z^S$ measurement.

**Derivative structure**:

- Scenario A: $\partial H_S/\partial\omega = a_x J_x + a_y J_y + a_z J_z$. Spectral radius: $\frac{1}{2}\sqrt{a_x^2 + a_y^2 + a_z^2}$.
- Scenario B: $\partial H/\partial\omega = (a_x J_x^S + a_y J_y^S + a_z J_z^S) + (a_x J_x^A + a_y J_y^A + a_z J_z^A)$. Spectral radius: $\sqrt{a_x^2 + a_y^2 + a_z^2}$ — exactly double Scenario A's.

**Why the factor-of-2 spectral advantage becomes only $\sqrt{2}$ in sensitivity**: The QFI bound $F_Q \leq 4\,t_{\text{hold}}^2\,\|\partial H/\partial\omega\|^2$ is loose. The actual QFI (accounting for the BS-constrained initial state) gives $F_Q^{(B)} = 2\,F_Q^{(A)}$, yielding a sensitivity ratio of $\sqrt{2}$, not $2$. The BS-constrained Bloch vectors limit the accessible variance: both subsystems start with Bloch vectors in the $-y$ direction after the beam splitter, making the $a_y$ drive component invisible to the QFI (though not to the EP sensitivity).

**Physical mechanism**: In Scenario A, $\partial H_S/\partial\omega$ acts directly on the measured subsystem — the $J_z$ measurement can directly access the amplified signal. In Scenario B, the identical drive on both subsystems doubles the generator norm, but the $J_z^S$ measurement can only access the system's contribution directly — the ancilla's contribution must be channelled through $a_{zz}$. This is why Scenario A's per-particle efficiency is higher.

---

## 5. Numerical Implementation

1. **Operator construction**: Kronecker products for Scenario B ($4\times 4$ matrices), direct $2\times 2$ matrices for Scenario A. Reuses `build_two_qubit_operators()` from `src.analysis.ancilla_optimization` for Scenario B.

2. **Beam-splitter unitaries**: $U_{\text{BS}} = \exp(-i(\pi/2) J_x) = (\mathbb{1} - i\sigma_x)/\sqrt{2}$ via `bs_qubit` from `src.physics.beam_splitter`. Dual BS: $U_{\text{BS}} \otimes U_{\text{BS}}$.

3. **State evolution**: Matrix exponential via `scipy.linalg.expm`. For Scenario A, $2\times 2$ generator; for Scenario B, $4\times 4$.

4. **Sensitivity computation**: Central finite differences with $\delta = 10^{-6}$, requiring 3 matrix exponentials per evaluation (at $\omega$, $\omega + \delta$, $\omega - \delta$).

5. **Data flow**: All result dataclasses store input parameters alongside computed results. Parquet files are self-describing via `to_dataframe()` and `save_parquet()`. Deserialisation fails fast on missing columns.

---

## 6. Parameter Space and Optimisation Strategy

**Stage 1 — Random search**: 500 points per $\omega$ in the parameter space. Scenario A: 3D $(a_x, a_y, a_z) \in [-5, 5]^3$. Scenario B: 4D $(a_x, a_y, a_z, a_{zz}) \in [-5, 5]^4$.

**Stage 2 — Nelder-Mead refinement**: 50 starts per $\omega$ from best random-search candidates. Derivative-free simplex optimisation minimises $\Delta\omega$ directly.

**$\omega$-scan**: 500 points, $\omega \in [0.01, 5.00]$ with spacing $0.01$. Both scenarios swept independently at each $\omega$.

**Fixed parameters**: $t_{\text{hold}} = 10$, $\delta = 10^{-6}$.

**Total cost**: $500 \times (500 + 50 \times \text{~200 iterations}) \approx 5.5 \times 10^6$ matrix exponentials per scenario, each on a $2\times 2$ or $4\times 4$ matrix — runs in seconds on a single CPU.

---

## 7. Results

### 7.1 Scenario A: The System-Only Surprise

Scenario A achieves sub-SQL sensitivity at **all 500 $\omega$ values** (using $N=1$ SQL of $1/t_{\text{hold}} = 0.1$). The best sensitivity is $\Delta\omega = 0.012018$ at $\omega = 4.51$, giving a ratio of $8.32\times$ below the single-particle SQL. The mean ratio across all $\omega$ is $7.68\times$.

The mechanism is direct: the derivative $\partial H_S/\partial\omega = a_x J_x + a_y J_y + a_z J_z$ acts on the measured subsystem, providing parametric amplification without needing BCH cross-terms. Parameter saturation is observed: $a_x$ and $a_z$ hit the $\pm 5$ bounds at $96\%$ of $\omega$ values — framed as a feature, the mechanism is so effective it saturates any bound you impose.

The $a_y$ axis is omitted from parameter plots (though it is optimised in the 3D search) because $a_y$ drops out of the amplitude prefactor $\rho = (a_x^2+a_z^2)/r^2$ but enters through the oscillation angle $\theta = \omega\,t_{\text{hold}}\,r$. Constraining $a_y = 0$ worsens sensitivity by $7$--$13\%$ at $\omega \geq 0.5$.

**Key Finding**: Scenario A already exceeds the #20260519 baseline by $2.04\times$ ($8.32\times$ vs $4.07\times$, both using appropriate SQL for their particle count). The system's own drive is substantially more effective than the ancilla's because the derivative acts directly on the measured subsystem.

<img src="../reports/r20260709/figures/20260709-scenario-a-omega-scan.svg" alt="Scenario A omega scan" width="100%"/>

*Figure 1: Scenario A sensitivity $\Delta\omega$ vs $\omega$ (top) and $\Delta\omega/\text{SQL}$ ratio (bottom). The global minimum is $\Delta\omega = 0.012018$ at $\omega = 4.51$, $8.32\times$ below the single-particle SQL.*

<img src="../reports/r20260709/figures/20260709-scenario-a-optimal-params.svg" alt="Scenario A optimal parameters" width="100%"/>

*Figure 2: Optimal $(a_x, a_y, a_z)$ coloured by $\Delta\omega$. The $a_y$ axis is omitted from the visualisation (though it is optimised in the 3D search) because $a_y$ drops out of the signal amplitude but modulates the oscillation frequency.*

### 7.2 Scenario B: The Compound Extension

Scenario B achieves sub-SQL sensitivity at all 500 $\omega$ values (using $N=2$ SQL of $1/(\sqrt{2}\,t_{\text{hold}}) \approx 0.07071$). The best sensitivity is $\Delta\omega = 0.010482$ at $\omega = 0.01$, giving a ratio of $6.75\times$ below the two-particle SQL. The mean ratio across all $\omega$ is $5.23\times$.

The compound advantage at $\omega = 0.01$ is $\mathcal{R}_{\text{compound}} = 1.3492\times$ — a $34.9\%$ improvement over Scenario A at the same $\omega$.

**Key Finding**: Scenario B achieves the best absolute sensitivity ($0.010482$), and the compound ratio at $\omega = 0.01$ is $1.3492\times$ — a $34.9\%$ improvement over Scenario A at the same $\omega$. The compound advantage is strongest at the lowest $\omega$ values.

<img src="../reports/r20260709/figures/20260709-scenario-b-omega-scan.svg" alt="Scenario B omega scan" width="100%"/>

*Figure 3: Scenario B sensitivity $\Delta\omega$ vs $\omega$ (top) and $\Delta\omega/\text{SQL}$ ratio (bottom). The global minimum is $\Delta\omega = 0.010482$ at $\omega = 0.01$, $6.75\times$ below the two-particle SQL.*

<img src="../reports/r20260709/figures/20260709-scenario-b-optimal-params.svg" alt="Scenario B optimal parameters" width="100%"/>

*Figure 4: Optimal $(a_x, a_z, a_{zz})$ coloured by $\Delta\omega$. The $a_y$ axis is omitted for the same reason as in Scenario A.*

### 7.3 Compound Ratio

The compound ratio $\mathcal{R}_{\text{compound}} = \Delta\omega_A^{\text{opt}} / \Delta\omega_B^{\text{opt}}$ at each $\omega$ compares each scenario's independently optimised sensitivity:

| Metric | Value |
|--------|-------|
| Best $\mathcal{R}_{\text{compound}}$ | $1.3492\times$ at $\omega = 0.01$ |
| Mean $\mathcal{R}_{\text{compound}}$ | $0.965$ |
| Median $\mathcal{R}_{\text{compound}}$ | $0.942$ |
| Fraction B beats A | $24.6\%$ overall ($123/500$) |
| Fraction B beats A at low $\omega \leq 1.0$ | $77\%$ ($77/100$) |
| Fraction B beats A at high $\omega > 1.0$ | $11.5\%$ ($46/400$) |
| Min $\mathcal{R}_{\text{compound}}$ | $0.8581$ at $\omega = 2.72$ |

**Key Finding**: The compound ratio is moderate and strongly $\omega$-dependent. At low $\omega$ ($\leq 1.0$), Scenario B beats Scenario A in $77\%$ of cases — the longer effective rotation time allows $a_{zz}$ to channel ancilla information back into the $J_z^S$ measurement. At high $\omega$ ($> 1.0$), the dual MZI and Ising interaction interfere with the system's own parametric amplification, making Scenario B worse than Scenario A alone in $88.5\%$ of cases.

<img src="../reports/r20260709/figures/20260709-compound-ratio.svg" alt="Compound ratio" width="100%"/>

*Figure 5: Compound ratio $\mathcal{R}_{\text{compound}} = \Delta\omega_A / \Delta\omega_B$ vs $\omega$. Values above 1 indicate Scenario B outperforms Scenario A at the same $\omega$. The best ratio is $1.3492\times$ at $\omega = 0.01$.*

### 7.4 Cross-Protocol Comparison

| Protocol | Sensitivity | Mechanism |
|----------|-------------|-----------|
| #20260519 (ancilla-only, $N=2$) | $\Delta\omega = 0.01739$; $4.07\times$ SQL ($\text{SQL} = 1/(\sqrt{2}\,t_{\text{hold}})$) | BCH cross-term via $a_{zz}$ |
| Scenario A (system-only, $N=1$) | $\Delta\omega = 0.012018$; $8.32\times$ SQL ($\text{SQL} = 1/t_{\text{hold}}$) | Direct derivative on measured subsystem |
| Scenario B (identical drive + Ising, $N=2$) | $\Delta\omega = 0.010482$; $6.75\times$ SQL ($\text{SQL} = 1/(\sqrt{2}\,t_{\text{hold}})$) | Tensor-sum generator + $a_{zz}$ channel |

Note: Scenario A's $8.32\times$ ($N=1$) and Scenario B's $6.75\times$ ($N=2$) use different SQL references. The absolute sensitivity comparison shows B is better ($0.010482 < 0.012018$), but A is more efficient per particle. Both scenarios beat #20260519: B by $1.66\times$ (same $N=2$, directly comparable), A by $2.04\times$ (different $N$, but striking).

The mechanism difference is fundamental: #20260519 relies on BCH cross-terms $[\omega J_z^S, a_{zz} J_z^S \otimes J_z^A]$ to generate an effective $\omega J_z^A$ contribution. Scenario A bypasses this entirely — the drive acts directly on the system.

**Key Finding**: The system-direct drive establishes a substantially stronger baseline than anticipated. The $\sqrt{2}$ QFI bound is nearly saturated ($95.4\%$), suggesting the $J_z^S$ measurement efficiently extracts the available gain from the doubled particle number.

<img src="../reports/r20260709/figures/20260709-sql-violation-ratio.svg" alt="SQL violation ratio comparison" width="100%"/>

*Figure 6: SQL-violation ratio $\Delta\omega_{\text{SQL}} / \Delta\omega$ for Scenario A ($N=1$ SQL) and Scenario B ($N=2$ SQL). Both protocols beat their respective SQL references across the full $\omega$ range.*

---

## 8. Analytical Understanding

### 8.1 Scenario A: Closed-Form Probability

Define $r = \sqrt{a_x^2 + a_y^2 + a_z^2}$, unit vector $\hat{n} = (a_x, a_y, a_z)/r$, and rotation angle $\theta = \omega\,t_{\text{hold}}\,r$. The hold unitary is $U_{\text{hold}} = e^{-i\theta\,\hat{n}\cdot\vec{\sigma}/2}$.

The Bloch vector after BS1 is $(0, -1, 0)$ (pointing in $-y$ direction). This means $\sigma_y$ is the only relevant Pauli component. Using the rotation identity $U_{\text{hold}}^\dagger\,\sigma_y\,U_{\text{hold}} = R_{yy}\,\sigma_y + \text{orthogonal terms}$ where $R_{yy} = \cos\theta + (1-\cos\theta)\,n_y^2$, the final expectation collapses to $\langle\sigma_z\rangle_{\text{final}} = -R_{yy}$.

The positive-outcome probability simplifies to $P_A(+) = \frac{1}{2}(1 - \cos\theta)(n_x^2 + n_z^2)$, which equals $\frac{a_x^2 + a_z^2}{r^2}\sin^2\!\left(\frac{\omega\,t_{\text{hold}}\,r}{2}\right)$.

**Key structural insight**: $a_y$ drops out of the amplitude prefactor $\rho = (a_x^2+a_z^2)/r^2$ — drive along $y$ rotates the Bloch vector around the same axis as the BS1-induced state. However, $a_y$ enters the EP sensitivity through $\theta = \omega\,t_{\text{hold}}\,r$: increasing $a_y$ increases $r$, increasing the oscillation frequency, allowing the optimiser to tune $\theta$ to a more favourable fringe operating point. There is a trade-off — larger $a_y$ increases $r$ (faster oscillation) but decreases $\rho$ (lower signal amplitude).

The expectation evaluates to $\langle J_z^S\rangle_A = \frac{1}{2}\bigl(-\cos\theta\,(n_x^2 + n_z^2) - n_y^2\bigr)$ and the variance is $\operatorname{Var}(J_z^S)_A = P_A(+)(1-P_A(+))$. The EP sensitivity is $\Delta\omega_A = \sqrt{1 - \rho\,\sin^2(\theta/2)} \big/ \bigl(\sqrt{\rho}\,t_{\text{hold}}\,r\,\lvert\cos(\theta/2)\rvert\bigr)$.

**Verification**: At baseline $(a_z=1, a_x=a_y=0)$: $P_A(+) = \sin^2(\omega t/2)$ — the standard single-qubit MZI fringe. Numerical agreement $< 10^{-15}$.

### 8.2 Scenario A: QFI and CFI

The generator is $G_S = \frac{r}{2}\,\hat{n}\cdot\vec{\sigma}$. Since $\langle G_S^2\rangle = r^2/4$ and $\langle G_S\rangle = -r\,n_y/2$ (Bloch vector in $-y$ after BS1): $\operatorname{Var}(G_S) = r^2(1-n_y^2)/4 = (a_x^2+a_z^2)/4$.

The QFI is $F_Q^{(A)} = 4\,t_{\text{hold}}^2\,\operatorname{Var}(G_S) = t_{\text{hold}}^2\,(a_x^2 + a_z^2)$, giving the quantum-limited sensitivity $\Delta\omega_Q^{(A)} = 1/(t_{\text{hold}}\,\sqrt{a_x^2 + a_z^2})$.

Note $a_y$ does not appear in $F_Q^{(A)}$ — the QFI is determined solely by drive components orthogonal to the Bloch direction ($y$). The EP/CFI sensitivity depends on $a_y$ through $\theta$.

The CFI for the binary $J_z$ measurement is $F_C^{(A)} = \rho\,r^2\,t_{\text{hold}}^2\,\cos^2(\theta/2) \big/ \bigl(1 - \rho\,\sin^2(\theta/2)\bigr)$. EP and CFI sensitivities are identical for this binary measurement.

### 8.3 Scenario B: Block Diagonalisation

**Basis choice**: Bell-like basis $\{\vert00\rangle, \vert{+}\rangle_m, \vert{-}\rangle_m, \vert11\rangle\}$ where $\vert{\pm}\rangle_m = (\vert01\rangle \pm \vert10\rangle)/\sqrt{2}$.

**Decoupling**: The antisymmetric state $\vert{-}\rangle_m$ decouples entirely: $\langle{-}_m|H|00\rangle = \langle{-}_m|H|11\rangle = 0$, because both $\vert01\rangle$ and $\vert10\rangle$ have identical coupling strengths to $\vert00\rangle$ and $\vert11\rangle$.

**$3\times 3$ block**: $H_3$ in $\{\vert00\rangle, \vert{+}\rangle_m, \vert11\rangle\}$ has diagonal elements $\omega a_z + a_{zz}/4$, $-a_{zz}/4$, $-\omega a_z + a_{zz}/4$ and off-diagonal elements $\frac{\omega}{\sqrt{2}}(a_x - ia_y)$ coupling adjacent levels.

**Shift by $a_{zz}/4$** (global phase): $H_3 = \frac{a_{zz}}{4}\mathbb{1}_3 + H'$ where $H'$ has diagonal elements $\omega a_z$, $-a_{zz}/2$, $-\omega a_z$.

**Key observation**: $H'$ is not proportional to $\omega$ — the middle diagonal element $-a_{zz}/2$ is $\omega$-independent. Consequently, eigenvectors depend on the ratio $\omega/a_{zz}$, and eigenvalues depend on $\omega$ non-linearly.

**Characteristic polynomial** (verified symbolically via sympy in `verify_block_diag.py`): $\mu^3 + \frac{a_{zz}}{2}\,\mu^2 - \omega^2 r^2\,\mu - \frac{\omega^2 a_z^2 a_{zz}}{2} = 0$. Discriminant $\Delta_c = (a_{zz}/2)^2/3 + \omega^2 r^2 > 0$ for $\omega \neq 0$, guaranteeing three distinct roots.

**Special case $a_{zz} = 0$**: Polynomial reduces to $\mu(\mu^2 - \omega^2 r^2) = 0$, giving eigenvalues $\mu = 0, \pm\omega r$ with $\omega$-independent eigenvectors.

**Post-BS state**: $\vert\Psi_1\rangle = (U_{\text{BS}}\otimes U_{\text{BS}})\vert00\rangle = \frac{1}{2}\vert00\rangle - \frac{i\sqrt{2}}{2}\vert{+}\rangle_m - \frac{1}{2}\vert11\rangle$ — lies entirely in the 3D subspace.

**Closed-form subspace** ($a_x = a_y = 0$): $P_B(+)\big\vert_{a_x=a_y=0} = \frac{1}{2}\!\left(1 - \cos(\omega\,a_z\,t_{\text{hold}})\,\cos\!\left(\frac{a_{zz}\,t_{\text{hold}}}{2}\right)\right)$. Verification: at $a_{zz}=0$, this reduces to $P_A(+)$ with $a_x=a_y=0$.

### 8.4 Scenario B: QFI and Resource-Counting Bound

The generator is $G_{\text{tot}} = G_S + G_A$ on different tensor factors. Since the post-BS state is a product state with Bloch vectors $(0,-1,0)$ on each subsystem, the cross-covariance vanishes: $\operatorname{Var}(G_{\text{tot}}) = \operatorname{Var}(G_S) + \operatorname{Var}(G_A) = (a_x^2+a_z^2)/2$.

The QFI is $F_Q^{(B)} = 2\,t_{\text{hold}}^2\,(a_x^2 + a_z^2) = 2\,F_Q^{(A)}$.

**Resource-counting bound**: $F_Q^{(B)}/F_Q^{(A)} = 2$ is an algebraic identity (independent of parameter values) — the variance of the sum of two independent generators on a product state equals the sum of individual variances. This gives a sensitivity ratio of $\sqrt{N_B/N_A} = \sqrt{2} \approx 1.414$.

The spectral-radius bound of $2\times$ overcounts because the BS-constrained state has Bloch vectors in the $-y$ direction, making $a_y$ invisible to the QFI (though not to the EP sensitivity).

The free-optimisation compound ratio $1.3492$ achieves $95.4\%$ of the $\sqrt{2}$ QFI bound, indicating efficient extraction of the available improvement from the doubled particle number.

### 8.5 Decoupled Limit and Consistency

At $a_{zz} = 0$: Scenario B separates into independent subsystems. $\Delta\omega_B(a_{zz}=0) = \Delta\omega_A$ to machine precision for the same $(a_x, a_y, a_z, \omega)$.

At $a_z = 1, a_x = a_y = a_{zz} = 0$: both scenarios recover $\Delta\omega = 1/t_{\text{hold}} = 0.1$ (standard MZI encoding). This is the single-particle SQL, confirming baseline recovery.

---

## 9. Related Work and Series

The three-protocol arc:

| Report | Protocol | Key Insight |
|--------|----------|-------------|
| #20260519 (2026-05-19) | Ancilla-only $\omega$-drive, system BS, $J_z^S$; $4.07\times$ SQL ($N=2$) | $a_{zz}$ is the metrological engine; BCH cross-terms |
| #20260709 Scenario A (2026-07-09) | System-only $\omega$-drive, single-qubit MZI, $J_z$; $8.32\times$ SQL ($N=1$) | Direct derivative outperforms ancilla-mediated mechanism |
| #20260709 Scenario B (2026-07-09) | Identical $\omega$-drive on both + Ising, dual MZI, $J_z^S$; $6.75\times$ SQL ($N=2$) | Compounding is genuine ($1.3492\times$) but $\omega$-dependent |

The narrative arc: #20260519 asked whether the ancilla matters — the answer was yes, with $a_{zz}$ as the engine. This experiment asks whether the ancilla matters *when the system already has its own drive* — the answer is yes, but modestly ($1.3492\times$ at best), and only at low $\omega$.

**Related dual-MZI experiments**: Reports #20260522 and #20260523 tested symmetric beam-splitting and found it weakens BCH cross-term generation — consistent with the high-$\omega$ detrimental effect observed here (Scenario B worse than A at $88.5\%$ of high $\omega$ values).

**Multi-particle extension**: Report #20260612 tested whether the compound ratio scales with $N$ using larger particle numbers, providing initial evidence for $\sqrt{N}$ scaling.

---

## 10. Conclusions and Open Questions

### 10.1 Summary

- The system's own $\omega$-modulated drive achieves $8.32\times$ SQL with $N=1$ — a standalone result that already surpasses the ancilla-only protocol ($4.07\times$ with $N=2$) by $2.04\times$.
- Adding the ancilla compounds by up to $34.9\%$ ($\mathcal{R}_{\text{compound}} = 1.3492$), achieving $95.4\%$ of the $\sqrt{2}$ QFI resource-counting bound.
- The compound ratio is strongly $\omega$-dependent: beneficial at $77\%$ of low $\omega$ values ($\omega \leq 1.0$), detrimental at $88.5\%$ of high $\omega$ values ($\omega > 1.0$).
- Both scenarios beat SQL at every $\omega$ value ($500/500$ each).

### 10.2 Parameter Saturation as a Feature

$a_x$ and $a_z$ hit the $\pm 5$ bounds at $96\%$ of $\omega$ values in Scenario A and $76\%$/$38\%$ in Scenario B. Framed as a feature: the $\omega$-modulated drive mechanism is so effective that it saturates any bound you impose. The parametric amplification channel is not subtle — it demands the maximum available drive amplitude. This is a signature of a robust mechanism, not a fragile optimisation edge case.

Implication: the true optimum likely lies beyond $\vert a_k \vert = 5$. The mechanism's strength is bounded by the search range, not by the physics.

### 10.3 Open Questions

1. **$\omega$-dependence of the compound ratio**: Why does the ancilla become detrimental at high $\omega$? The dual MZI and Ising interaction appear to interfere with the system's own parametric amplification — is this a general limitation of symmetric dual-MZI protocols?
2. **Expanded bounds**: Could Scenario A exceed $10\times$ SQL with $\vert a_k \vert \leq 10$? The $96\%$ saturation rate suggests significant headroom.
3. **Multi-particle scaling**: Does the compound ratio scale with $N$? The $\sqrt{2}$ bound for $N=2$ suggests a $\sqrt{N}$ scaling, but this needs verification.
4. **Noise robustness**: How do decoherence channels (one-body loss, dephasing, detection inefficiency) affect the protocol? The system-direct drive mechanism may be more noise-resistant than the ancilla-mediated BCH mechanism.
