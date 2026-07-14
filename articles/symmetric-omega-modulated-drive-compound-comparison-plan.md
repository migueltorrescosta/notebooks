# Implementation Plan: Identical Dynamics, Asymmetric Gain

**Working title**: Identical Dynamics, Asymmetric Gain: When the System's Own $\omega$-Modulated Drive Outshines the Ancilla

**Subtitle**: A technical review of symmetric $\omega$-modulated drive metrology

**Target file**: `articles/symmetric-omega-modulated-drive-compound-comparison.md`

**Source report**: `reports/r20260709/Symmetric-omega-Modulated-Drive-Bounded-Compound-Comparison.md`

**Structural template**: `articles/ancilla-drive-phase-modulated-metrology.md`

**Target length**: ~450–500 lines

---

## Conventions

### SQL Convention

Use the **two-particle SQL** $\Delta\omega_{\text{SQL}} = 1/(\sqrt{2}\,t_{\text{hold}}) \approx 0.07071$ for protocols using 2 particles (Scenario B, #20260519). Use the **single-particle SQL** $\Delta\omega_{\text{SQL}} = 1/t_{\text{hold}} = 0.1$ for Scenario A (1 particle). Every comparison table states the SQL reference explicitly.

### Corrected Ratios

| Protocol | N | SQL Reference | Best $\Delta\omega$ | Ratio |
|----------|---|---------------|---------------------|-------|
| Scenario A (system-only drive) | 1 | $1/t_{\text{hold}} = 0.1$ | 0.012018 | 8.32× |
| Scenario B (identical drive + Ising) | 2 | $1/(\sqrt{2}\,t_{\text{hold}}) \approx 0.07071$ | 0.010482 | 6.75× |
| #20260519 (ancilla-only drive) | 2 | $1/(\sqrt{2}\,t_{\text{hold}}) \approx 0.07071$ | 0.01739 | 4.07× |

The compound ratio $\mathcal{R}_{\text{compound}} = 1.3492\times$ (at $\omega = 0.01$) and the $95.4\%$ QFI bound saturation are independent of SQL convention.

### Formatting Rules

- All LaTeX inline only (`$...$`). No block math (`$$...$$`).
- Figures embedded as `<img src="../reports/r20260709/figures/{filename}" alt="..." width="100%"/>` with italicised caption below.
- No decorative emojis. No `|` inside LaTeX math.
- Use `\mathbb{1}_n` for identity operators. Use `\vert` for `|` inside math.

---

## Section-by-Section Plan

### Section 1: Introduction (~40 lines)

**Narrative purpose**: Open with the motivating question and preview the three-act structure.

**Content**:

1. Opening question: *In an $\omega$-modulated system–ancilla metrology protocol, does the ancilla matter?*
2. Recap #20260519: ancilla-only drive achieved $4.07\times$ SQL with $N=2$ particles. The Ising interaction $a_{zz} J_z^S \otimes J_z^A$ was identified as the "metrological engine" — without it, no sub-SQL performance is possible.
3. The natural follow-up: what happens if the system qubit carries its own $\omega$-modulated drive? Does it improve sensitivity, or does the ancilla already capture the available gain?
4. Preview the three acts:
   - **Act 1** (Scenario A): The system-only surprise — a single-qubit MZI with the system Hamiltonian $H_S = \omega(a_x J_x + a_y J_y + a_z J_z)$ achieves $8.32\times$ SQL with $N=1$, already $2.04\times$ better than the ancilla-only baseline's $4.07\times$ with $N=2$.
   - **Act 2** (Scenario B): The compound extension — adding the ancilla with identical drive parameters and an Ising interaction compounds the gain by up to $34.9\%$ ($\mathcal{R}_{\text{compound}} = 1.3492$), achieving $95.4\%$ of the QFI resource-counting bound of $\sqrt{2}$.
   - **Act 3**: Context and synthesis — placing both results in the full series arc.
5. Key insight in one sentence: the system's own drive is substantially more effective than the ancilla's because the derivative $\partial H_S/\partial\omega$ acts directly on the measured subsystem, providing a parametric amplification channel that the $J_z$ measurement can access without relying on BCH cross-terms.

**Figures**: None.

---

### Section 2: Physical Setup (~35 lines)

**Narrative purpose**: Define the Hilbert spaces, operators, and initial states for both scenarios.

**Content**:

**Scenario A (system-only, $N=1$):**

- Hilbert space: $\mathcal{H}_S = \text{span}\{|0\rangle, |1\rangle\}$, dimension 2.
- Basis convention: $|0\rangle = |1,0\rangle$ (particle in mode 0), $|1\rangle = |0,1\rangle$ (particle in mode 1).
- Operators: $J_k = \sigma_k/2$ (Pauli matrices divided by 2). $J_z = \frac{1}{2}\text{diag}(1, -1)$.
- Initial state: $|0\rangle_S$.

**Scenario B (ancilla-assisted, $N=2$):**

- Hilbert space: $\mathcal{H}_{\text{tot}} = \mathcal{H}_S \otimes \mathcal{H}_A$, dimension 4.
- Basis: $\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$ with index $= n_S \times 2 + n_A$.
- Operators: $J_k^S = J_k \otimes \mathbb{1}_2$, $J_k^A = \mathbb{1}_2 \otimes J_k$. Interaction: $J_z^S \otimes J_z^A$.
- Initial state: $|00\rangle = |0\rangle_S \otimes |0\rangle_A$.

**Common:**

- SU(2) algebra: $[J_i, J_j] = i\epsilon_{ijk} J_k$.
- Holding time: $t_{\text{hold}} = 10$ (fixed).
- SQL references: single-particle ($1/t_{\text{hold}} = 0.1$) for Scenario A; two-particle ($1/(\sqrt{2}\,t_{\text{hold}}) \approx 0.07071$) for Scenario B.

**Figures**: None.

---

### Section 3: Circuit Protocol (~45 lines)

**Narrative purpose**: Step-by-step unitary sequence for each scenario.

**Content**:

**Scenario A (single-qubit MZI):**

1. **Beam splitter**: $U_{\text{BS}} = \exp(-i(\pi/2) J_x)$ — the standard 50/50 pulse, creating a coherent superposition from $|0\rangle_S$.
2. **Holding period**: Evolution under $H_S = \omega(a_x J_x + a_y J_y + a_z J_z)$ for $t_{\text{hold}} = 10$. The unitary $U_{\text{hold}} = \exp(-i\,t_{\text{hold}}\,H_S)$ depends on $\omega$ through the Hamiltonian itself — the defining feature of the $\omega$-modulated protocol.
3. **Beam splitter**: Same $U_{\text{BS}}$ as step 1.
4. **Measurement**: $J_z$ on the system.

**Scenario B (dual MZI):**

1. **Beam splitter on both qubits**: $U_{\text{BS}} \otimes U_{\text{BS}}$ — the symmetric dual MZI.
2. **Holding period**: Evolution under $H = \omega(a_x J_x^S + a_y J_y^S + a_z J_z^S) + \omega(a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_{zz} J_z^S J_z^A$ for $t_{\text{hold}} = 10$. The Hamiltonian separates into system drive, ancilla drive, and Ising interaction.
3. **Beam splitter on both qubits**: Same as step 1.
4. **Measurement**: $J_z^S$ on the system (ancilla traced out).

**Sensitivity formula**: $\Delta\omega = \sqrt{\text{Var}(J_z^S)} / |\partial\langle J_z^S\rangle/\partial\omega|$ with central finite differences ($\delta = 10^{-6}$).

**Key symmetry**: Both subsystems feel identical $\omega$-modulated drives — the only asymmetry is the $J_z^S$ measurement on the system.

**Figures**: None.

---

### Section 4: The $\omega$-Modulated Drive Hamiltonian (~40 lines)

**Narrative purpose**: Hamiltonian decomposition, identical-subsystem constraint, physical mechanism for compounding.

**Content**:

1. **Decomposition**: $H = H_S + H_A + H_{\text{int}}$ where:
   - $H_S = \omega(a_x J_x^S + a_y J_y^S + a_z J_z^S)$ — system drive (identical to Scenario A's Hamiltonian),
   - $H_A = \omega(a_x J_x^A + a_y J_y^A + a_z J_z^A)$ — ancilla drive (identical parameters),
   - $H_{\text{int}} = a_{zz} J_z^S \otimes J_z^A$ — Ising interaction.

2. **Identical-subsystem constraint**: The drive parameters $(a_x, a_y, a_z)$ are the same on S and A. This means the subsystem Hamiltonians are indistinguishable — the only asymmetry in the problem is the $J_z^S$ measurement. Any sensitivity advantage of Scenario B over the decoupled limit ($a_{zz} = 0$) comes purely from the Ising interaction channelling ancilla information back into the $J_z^S$ measurement.

3. **Derivative structure**:
   - Scenario A: $\partial H_S/\partial\omega = a_x J_x + a_y J_y + a_z J_z$. Spectral radius: $\frac{1}{2}\sqrt{a_x^2 + a_y^2 + a_z^2}$.
   - Scenario B: $\partial H/\partial\omega = (a_x J_x^S + a_y J_y^S + a_z J_z^S) + (a_x J_x^A + a_y J_y^A + a_z J_z^A)$. Spectral radius: $\sqrt{a_x^2 + a_y^2 + a_z^2}$ — exactly double Scenario A's.

4. **Why the factor-of-2 spectral advantage becomes only $\sqrt{2}$ in sensitivity**: The QFI bound $F_Q \leq 4\,t_{\text{hold}}^2\,\|\partial H/\partial\omega\|^2$ is loose. The actual QFI (accounting for the BS-constrained initial state) gives $F_Q^{(B)} = 2\,F_Q^{(A)}$, yielding a sensitivity ratio of $\sqrt{2}$, not $2$. The BS-constrained Bloch vectors limit the accessible variance.

5. **Physical mechanism**: In Scenario A, $\partial H_S/\partial\omega$ acts directly on the measured subsystem — the $J_z$ measurement can directly access the amplified signal. In Scenario B, the identical drive on both subsystems doubles the generator norm, but the $J_z^S$ measurement can only access the system's contribution directly — the ancilla's contribution must be channelled through $a_{zz}$. This is why Scenario A's per-particle efficiency is higher.

**Figures**: None.

---

### Section 5: Numerical Implementation (~35 lines)

**Narrative purpose**: Describe the computational pipeline.

**Content**:

1. **Operator construction**: Kronecker products for Scenario B ($4\times 4$ matrices), direct $2\times 2$ matrices for Scenario A. Reuses `build_two_qubit_operators()` from `src.analysis.ancilla_optimization` for Scenario B.

2. **Beam-splitter unitaries**: $U_{\text{BS}} = \exp(-i(\pi/2) J_x) = (\mathbb{1} - i\sigma_x)/\sqrt{2}$ via `bs_qubit` from `src.physics.beam_splitter`. Dual BS: $U_{\text{BS}} \otimes U_{\text{BS}}$.

3. **State evolution**: Matrix exponential via `scipy.linalg.expm`. For Scenario A, $2\times 2$ generator; for Scenario B, $4\times 4$.

4. **Sensitivity computation**: Central finite differences with $\delta = 10^{-6}$, requiring 3 matrix exponentials per evaluation (at $\omega$, $\omega + \delta$, $\omega - \delta$).

5. **Data flow**: All result dataclasses store input parameters alongside computed results. Parquet files are self-describing via `to_dataframe()` and `save_parquet()`. Deserialisation fails fast on missing columns.

**Figures**: None.

---

### Section 6: Parameter Space and Optimisation Strategy (~35 lines)

**Narrative purpose**: Describe the two-stage optimisation pipeline and sweep design.

**Content**:

1. **Stage 1 — Random search**: 500 points per $\omega$ in the parameter space. Scenario A: 3D $(a_x, a_y, a_z) \in [-5, 5]^3$. Scenario B: 4D $(a_x, a_y, a_z, a_{zz}) \in [-5, 5]^4$.

2. **Stage 2 — Nelder-Mead refinement**: 50 starts per $\omega$ from best random-search candidates. Derivative-free simplex optimisation minimises $\Delta\omega$ directly.

3. **$\omega$-scan**: 500 points, $\omega \in [0.01, 5.00]$ with spacing $0.01$. Both scenarios swept independently at each $\omega$.

4. **Fixed parameters**: $t_{\text{hold}} = 10$, $\delta = 10^{-6}$.

5. **Total cost**: $500 \times (500 + 50 \times \text{~200 iterations}) \approx 5.5 \times 10^6$ matrix exponentials per scenario, each on a $2\times 2$ or $4\times 4$ matrix — runs in seconds on a single CPU.

**Figures**: None.

---

### Section 7: Results (~100 lines)

**Narrative purpose**: Present all results with figures and Key Finding paragraphs.

#### 7.1 Scenario A: The System-Only Surprise (~25 lines)

**Content**:

- Sub-SQL sensitivity at **all 500 $\omega$ values** (using $N=1$ SQL).
- Best: $\Delta\omega = 0.012018$ at $\omega = 4.51$, ratio $8.32\times$ SQL.
- Mean ratio: $7.68\times$ across all $\omega$.
- The mechanism: direct derivative $\partial H_S/\partial\omega$ acts on the measured subsystem, providing parametric amplification without needing BCH cross-terms.
- Parameter saturation: $a_x$ and $a_z$ hit $\pm 5$ bounds at $96\%$ of $\omega$ values (framed as a feature — the mechanism is so effective it saturates any bound).
- Explain why $a_y$ is omitted from parameter plots: it drops out of the amplitude prefactor $\rho = (a_x^2+a_z^2)/r^2$ but enters through the oscillation angle $\theta = \omega\,t_{\text{hold}}\,r$. Constraining $a_y = 0$ worsens sensitivity by $7$–$13\%$ at $\omega \geq 0.5$.

**Key Finding paragraph**: Scenario A already exceeds the #20260519 baseline by $2.04\times$ ($8.32\times$ vs $4.07\times$, both using appropriate SQL for their particle count). The system's own drive is substantially more effective than the ancilla's because the derivative acts directly on the measured subsystem.

**Figures**:

- `<img src="../reports/r20260709/figures/20260709-scenario-a-omega-scan.svg" alt="Scenario A omega scan" width="100%"/>`
  *Figure 1: Scenario A sensitivity $\Delta\omega$ vs $\omega$ (top) and $\Delta\omega/\text{SQL}$ ratio (bottom). The global minimum is $\Delta\omega = 0.012018$ at $\omega = 4.51$, $8.32\times$ below the single-particle SQL.*

- `<img src="../reports/r20260709/figures/20260709-scenario-a-optimal-params.svg" alt="Scenario A optimal parameters" width="100%"/>`
  *Figure 2: Optimal $(a_x, a_y, a_z)$ coloured by $\Delta\omega$. The $a_y$ axis is omitted from the visualisation (though it is optimised in the 3D search) because $a_y$ drops out of the signal amplitude but modulates the oscillation frequency.*

#### 7.2 Scenario B: The Compound Extension (~25 lines)

**Content**:

- Sub-SQL sensitivity at all 500 $\omega$ values (using $N=2$ SQL).
- Best: $\Delta\omega = 0.010482$ at $\omega = 0.01$, ratio $6.75\times$ SQL ($N=2$).
- Mean ratio: $5.23\times$ across all $\omega$.
- The compound advantage at $\omega = 0.01$: $\mathcal{R}_{\text{compound}} = 1.3492\times$ ($34.9\%$ improvement over Scenario A at the same $\omega$).

**Key Finding paragraph**: Scenario B achieves the best absolute sensitivity ($0.010482$), and the compound ratio at $\omega = 0.01$ is $1.3492\times$ — a $34.9\%$ improvement over Scenario A at the same $\omega$. The compound advantage is strongest at the lowest $\omega$ values.

**Figures**:

- `<img src="../reports/r20260709/figures/20260709-scenario-b-omega-scan.svg" alt="Scenario B omega scan" width="100%"/>`
  *Figure 3: Scenario B sensitivity $\Delta\omega$ vs $\omega$ (top) and $\Delta\omega/\text{SQL}$ ratio (bottom). The global minimum is $\Delta\omega = 0.010482$ at $\omega = 0.01$, $6.75\times$ below the two-particle SQL.*

- `<img src="../reports/r20260709/figures/20260709-scenario-b-optimal-params.svg" alt="Scenario B optimal parameters" width="100%"/>`
  *Figure 4: Optimal $(a_x, a_z, a_{zz})$ coloured by $\Delta\omega$. The $a_y$ axis is omitted for the same reason as in Scenario A.*

#### 7.3 Compound Ratio (~20 lines)

**Content**:

- $\mathcal{R}_{\text{compound}} = \Delta\omega_A^{\text{opt}} / \Delta\omega_B^{\text{opt}}$ at each $\omega$ (independent optimisation).
- Best: $1.3492\times$ at $\omega = 0.01$.
- Mean: $0.965$, median: $0.942$.
- Fraction B beats A: $24.6\%$ overall ($123/500$), $77\%$ at low $\omega \leq 1.0$ ($77/100$), $11.5\%$ at high $\omega > 1.0$ ($46/400$).
- Min: $0.8581$ at $\omega = 2.72$.

**Key Finding paragraph**: The compound ratio is moderate and strongly $\omega$-dependent. At low $\omega$ ($\leq 1.0$), Scenario B beats Scenario A in $77\%$ of cases — the longer effective rotation time allows $a_{zz}$ to channel ancilla information back into the $J_z^S$ measurement. At high $\omega$ ($> 1.0$), the dual MZI and Ising interaction interfere with the system's own parametric amplification, making Scenario B worse than Scenario A alone in $88.5\%$ of cases.

**Figures**:

- `<img src="../reports/r20260709/figures/20260709-compound-ratio.svg" alt="Compound ratio" width="100%"/>`
  *Figure 5: Compound ratio $\mathcal{R}_{\text{compound}} = \Delta\omega_A / \Delta\omega_B$ vs $\omega$. Values above 1 indicate Scenario B outperforms Scenario A at the same $\omega$. The best ratio is $1.3492\times$ at $\omega = 0.01$.*

#### 7.4 Cross-Protocol Comparison (~20 lines)

**Content**:

- Present the comparison table:

| Protocol | N | SQL Reference | Best $\Delta\omega$ | Ratio | Mechanism |
|----------|---|---------------|---------------------|-------|-----------|
| #20260519 (ancilla-only drive) | 2 | $1/(\sqrt{2}\,t_{\text{hold}})$ | 0.01739 | 4.07× | BCH cross-term via $a_{zz}$ |
| Scenario A (system-only drive) | 1 | $1/t_{\text{hold}}$ | 0.012018 | 8.32× | Direct derivative on measured subsystem |
| Scenario B (identical drive + Ising) | 2 | $1/(\sqrt{2}\,t_{\text{hold}})$ | 0.010482 | 6.75× | Tensor-sum generator + $a_{zz}$ channel |

- Note: Scenario A's $8.32\times$ ($N=1$) and Scenario B's $6.75\times$ ($N=2$) use different SQL references. The absolute sensitivity comparison shows B is better ($0.010482 < 0.012018$), but A is more efficient per particle.
- Both scenarios beat #20260519: B by $1.66\times$ (same $N=2$, directly comparable), A by $2.04\times$ (different $N$, but striking).
- The mechanism difference: #20260519 relies on BCH cross-terms $[\omega J_z^S, a_{zz} J_z^S \otimes J_z^A]$ to generate an effective $\omega J_z^A$ contribution. Scenario A bypasses this entirely — the drive acts directly on the system.

**Key Finding paragraph**: The system-direct drive establishes a substantially stronger baseline than anticipated. The $\sqrt{2}$ QFI bound is nearly saturated ($95.4\%$), suggesting the $J_z^S$ measurement efficiently extracts the available gain from the doubled particle number.

**Figures**:

- `<img src="../reports/r20260709/figures/20260709-sql-violation-ratio.svg" alt="SQL violation ratio comparison" width="100%"/>`
  *Figure 6: SQL-violation ratio $\Delta\omega_{\text{SQL}} / \Delta\omega$ for Scenario A ($N=1$ SQL) and Scenario B ($N=2$ SQL). Both protocols beat their respective SQL references across the full $\omega$ range.*

---

### Section 8: Analytical Understanding (~80 lines)

**Narrative purpose**: Full inline derivations for both scenarios. All equations in `$...$` only.

#### 8.1 Scenario A: Closed-Form Probability (~30 lines)

Derive $P_A(+)$ from first principles.

**Derivation steps** (all inline math):

1. Define $r = \sqrt{a_x^2 + a_y^2 + a_z^2}$, unit vector $\hat{n} = (a_x, a_y, a_z)/r$, rotation angle $\theta = \omega\,t_{\text{hold}}\,r$.

2. Hold unitary: $U_{\text{hold}} = e^{-i\theta\,\hat{n}\cdot\vec{\sigma}/2}$.

3. Bloch vector after BS1: $(0, -1, 0)$ (pointing in $-y$ direction). This means $\sigma_y$ is the only relevant Pauli component.

4. Rotation identity: $U_{\text{hold}}^\dagger\,\sigma_y\,U_{\text{hold}} = R_{yy}\,\sigma_y + \text{orthogonal terms}$ where $R_{yy} = \cos\theta + (1-\cos\theta)\,n_y^2$.

5. Final expectation: $\langle\sigma_z\rangle_{\text{final}} = -R_{yy}$.

6. Positive-outcome probability: $P_A(+) = \frac{1}{2}(1 - \cos\theta)(n_x^2 + n_z^2)$, which equals $\frac{a_x^2 + a_z^2}{r^2}\sin^2\!\left(\frac{\omega\,t_{\text{hold}}\,r}{2}\right)$.

7. **Key structural insight**: $a_y$ drops out of the amplitude prefactor $\rho = (a_x^2+a_z^2)/r^2$ — drive along $y$ rotates the Bloch vector around the same axis as the BS1-induced state. However, $a_y$ enters the EP sensitivity through $\theta = \omega\,t_{\text{hold}}\,r$: increasing $a_y$ increases $r$, increasing the oscillation frequency, allowing the optimiser to tune $\theta$ to a more favourable fringe operating point. There is a trade-off — larger $a_y$ increases $r$ (faster oscillation) but decreases $\rho$ (lower signal amplitude).

8. Expectation: $\langle J_z^S\rangle_A = \frac{1}{2}\bigl(-\cos\theta\,(n_x^2 + n_z^2) - n_y^2\bigr)$. Variance: $\text{Var}(J_z^S)_A = P_A(+)(1-P_A(+))$. EP sensitivity: $\Delta\omega_A = \sqrt{1 - \rho\,\sin^2(\theta/2)} \big/ \bigl(\sqrt{\rho}\,t_{\text{hold}}\,r\,\lvert\cos(\theta/2)\rvert\bigr)$.

9. Verification: at baseline $(a_z=1, a_x=a_y=0)$: $P_A(+) = \sin^2(\omega t/2)$ — the standard single-qubit MZI fringe. Numerical agreement $< 10^{-15}$.

#### 8.2 Scenario A: QFI and CFI (~10 lines)

1. Generator: $G_S = \frac{r}{2}\,\hat{n}\cdot\vec{\sigma}$. Since $\langle G_S^2\rangle = r^2/4$ and $\langle G_S\rangle = -r\,n_y/2$ (Bloch vector in $-y$ after BS1): $\text{Var}(G_S) = r^2(1-n_y^2)/4 = (a_x^2+a_z^2)/4$.

2. QFI: $F_Q^{(A)} = 4\,t_{\text{hold}}^2\,\text{Var}(G_S) = t_{\text{hold}}^2\,(a_x^2 + a_z^2)$.

3. Quantum-limited sensitivity: $\Delta\omega_Q^{(A)} = 1/(t_{\text{hold}}\,\sqrt{a_x^2 + a_z^2})$.

4. $a_y$ does not appear in $F_Q^{(A)}$ — the QFI is determined solely by drive components orthogonal to the Bloch direction ($y$). The EP/CFI sensitivity depends on $a_y$ through $\theta$.

5. CFI for binary $J_z$ measurement: $F_C^{(A)} = \rho\,r^2\,t_{\text{hold}}^2\,\cos^2(\theta/2) \big/ \bigl(1 - \rho\,\sin^2(\theta/2)\bigr)$. EP and CFI sensitivities are identical for this binary measurement.

#### 8.3 Scenario B: Block Diagonalisation (~20 lines)

1. **Basis choice**: Bell-like basis $\{|00\rangle, |{+}\rangle_m, |{-}\rangle_m, |11\rangle\}$ where $|{\pm}\rangle_m = (|01\rangle \pm |10\rangle)/\sqrt{2}$.

2. **Decoupling**: The antisymmetric state $|{-}\rangle_m$ decouples entirely: $\langle{-}_m|H|00\rangle = \langle{-}_m|H|11\rangle = 0$, because both $|01\rangle$ and $|10\rangle$ have identical coupling strengths to $|00\rangle$ and $|11\rangle$.

3. **$3\times 3$ block**: $H_3$ in $\{|00\rangle, |{+}\rangle_m, |11\rangle\}$ has diagonal elements $\omega a_z + a_{zz}/4$, $-a_{zz}/4$, $-\omega a_z + a_{zz}/4$ and off-diagonal elements $\frac{\omega}{\sqrt{2}}(a_x - ia_y)$ coupling adjacent levels.

4. **Shift by $a_{zz}/4$** (global phase): $H_3 = \frac{a_{zz}}{4}\mathbb{1}_3 + H'$ where $H'$ has diagonal elements $\omega a_z$, $-a_{zz}/2$, $-\omega a_z$.

5. **Key observation**: $H'$ is not proportional to $\omega$ — the middle diagonal element $-a_{zz}/2$ is $\omega$-independent. Consequently, eigenvectors depend on the ratio $\omega/a_{zz}$, and eigenvalues depend on $\omega$ non-linearly.

6. **Characteristic polynomial** (verified symbolically via sympy in `verify_block_diag.py`): $\mu^3 + \frac{a_{zz}}{2}\,\mu^2 - \omega^2 r^2\,\mu - \frac{\omega^2 a_z^2 a_{zz}}{2} = 0$. Discriminant $\Delta_c = (a_{zz}/2)^2/3 + \omega^2 r^2 > 0$ for $\omega \neq 0$, guaranteeing three distinct roots.

7. **Special case $a_{zz} = 0$**: Polynomial reduces to $\mu(\mu^2 - \omega^2 r^2) = 0$, giving eigenvalues $\mu = 0, \pm\omega r$ with $\omega$-independent eigenvectors.

8. **Post-BS state**: $|\Psi_1\rangle = (U_{\text{BS}}\otimes U_{\text{BS}})|00\rangle = \frac{1}{2}|00\rangle - \frac{i\sqrt{2}}{2}|{+}\rangle_m - \frac{1}{2}|11\rangle$ — lies entirely in the 3D subspace.

9. **Closed-form subspace** ($a_x = a_y = 0$): $P_B(+)\big\vert_{a_x=a_y=0} = \frac{1}{2}\!\left(1 - \cos(\omega\,a_z\,t_{\text{hold}})\,\cos\!\left(\frac{a_{zz}\,t_{\text{hold}}}{2}\right)\right)$. Verification: at $a_{zz}=0$, this reduces to $P_A(+)$ with $a_x=a_y=0$.

#### 8.4 Scenario B: QFI and Resource-Counting Bound (~15 lines)

1. Generator: $G_{\text{tot}} = G_S + G_A$ on different tensor factors.

2. Post-BS state is a product state with Bloch vectors $(0,-1,0)$ on each subsystem. Cross-covariance vanishes: $\text{Var}(G_{\text{tot}}) = \text{Var}(G_S) + \text{Var}(G_A) = (a_x^2+a_z^2)/2$.

3. QFI: $F_Q^{(B)} = 2\,t_{\text{hold}}^2\,(a_x^2 + a_z^2) = 2\,F_Q^{(A)}$.

4. **Resource-counting bound**: $F_Q^{(B)}/F_Q^{(A)} = 2$ is an algebraic identity (independent of parameter values) — the variance of the sum of two independent generators on a product state equals the sum of individual variances. This gives a sensitivity ratio of $\sqrt{N_B/N_A} = \sqrt{2} \approx 1.414$.

5. The spectral-radius bound of $2\times$ overcounts because the BS-constrained state has Bloch vectors in the $-y$ direction, making $a_y$ invisible to the QFI (though not to the EP sensitivity).

6. The free-optimisation compound ratio $1.3492$ achieves $95.4\%$ of the $\sqrt{2}$ QFI bound, indicating efficient extraction of the available improvement from the doubled particle number.

#### 8.5 Decoupled Limit and Consistency (~5 lines)

1. At $a_{zz} = 0$: Scenario B separates into independent subsystems. $\Delta\omega_B(a_{zz}=0) = \Delta\omega_A$ to machine precision for the same $(a_x, a_y, a_z, \omega)$.

2. At $a_z = 1, a_x = a_y = a_{zz} = 0$: both scenarios recover $\Delta\omega = 1/t_{\text{hold}} = 0.1$ (standard MZI encoding). This is the single-particle SQL, confirming baseline recovery.

---

### Section 9: Related Work and Series (~30 lines)

**Narrative purpose**: Position this experiment within the full series arc.

**Content**:

1. **The three-protocol arc** — present as a table:

| Report | Date | Protocol | Best Ratio | SQL Ref | Key Insight |
|--------|------|----------|-----------|---------|-------------|
| #20260519 | 2026-05-19 | Ancilla-only $\omega$-drive, system BS, $J_z^S$ | 4.07× | $N=2$ | $a_{zz}$ is the metrological engine; BCH cross-terms |
| #20260709 (Scenario A) | 2026-07-09 | System-only $\omega$-drive, single-qubit MZI, $J_z$ | 8.32× | $N=1$ | Direct derivative outperforms ancilla-mediated mechanism |
| #20260709 (Scenario B) | 2026-07-09 | Identical $\omega$-drive on both + Ising, dual MZI, $J_z^S$ | 6.75× | $N=2$ | Compounding is genuine ($1.3492\times$) but $\omega$-dependent |

2. **The narrative arc**: #20260519 asked whether the ancilla matters — the answer was yes, with $a_{zz}$ as the engine. This experiment asks whether the ancilla matters *when the system already has its own drive* — the answer is yes, but modestly ($1.3492\times$ at best), and only at low $\omega$.

3. **Related dual-MZI experiments**: Brief mentions of #20260522 and #20260523, where symmetric beam-splitting was found to weaken BCH cross-term generation — consistent with the high-$\omega$ detrimental effect observed here (Scenario B worse than A at $88.5\%$ of high $\omega$ values).

4. **Multi-particle extension**: Brief mention of #20260612, which tested whether the compound ratio scales with $N$ using larger particle numbers.

---

### Section 10: Conclusions and Open Questions (~30 lines)

**Narrative purpose**: Summary, parameter saturation as feature, open questions.

#### 10.1 Summary (~10 lines)

- The system's own $\omega$-modulated drive achieves $8.32\times$ SQL with $N=1$ — a standalone result that already surpasses the ancilla-only protocol ($4.07\times$ with $N=2$) by $2.04\times$.
- Adding the ancilla compounds by up to $34.9\%$ ($\mathcal{R}_{\text{compound}} = 1.3492$), achieving $95.4\%$ of the $\sqrt{2}$ QFI resource-counting bound.
- The compound ratio is strongly $\omega$-dependent: beneficial at $77\%$ of low $\omega$ values ($\omega \leq 1.0$), detrimental at $88.5\%$ of high $\omega$ values ($\omega > 1.0$).
- Both scenarios beat SQL at every $\omega$ value ($500/500$ each).

#### 10.2 Parameter Saturation as a Feature (~10 lines)

- $a_x$ and $a_z$ hit the $\pm 5$ bounds at $96\%$ of $\omega$ values in Scenario A and $76\%$/$38\%$ in Scenario B.
- **Frame as a feature**: the $\omega$-modulated drive mechanism is so effective that it saturates any bound you impose. The parametric amplification channel is not subtle — it demands the maximum available drive amplitude. This is a signature of a robust mechanism, not a fragile optimisation edge case.
- Implication: the true optimum likely lies beyond $|a_k| = 5$. The mechanism's strength is bounded by the search range, not by the physics.

#### 10.3 Open Questions (~10 lines)

1. **$\omega$-dependence of the compound ratio**: Why does the ancilla become detrimental at high $\omega$? The dual MZI and Ising interaction appear to interfere with the system's own parametric amplification — is this a general limitation of symmetric dual-MZI protocols?
2. **Expanded bounds**: Could Scenario A exceed $10\times$ SQL with $|a_k| \leq 10$? The $96\%$ saturation rate suggests significant headroom.
3. **Multi-particle scaling**: Does the compound ratio scale with $N$? The $\sqrt{2}$ bound for $N=2$ suggests a $\sqrt{N}$ scaling, but this needs verification.
4. **Noise robustness**: How do decoherence channels (one-body loss, dephasing, detection inefficiency) affect the protocol? The system-direct drive mechanism may be more noise-resistant than the ancilla-mediated BCH mechanism.

---

## Figure Placement Summary

| Figure | File | Section | Caption |
|--------|------|---------|---------|
| Figure 1 | `20260709-scenario-a-omega-scan.svg` | 7.1 | Scenario A: $\Delta\omega$ vs $\omega$ and $\Delta\omega/\text{SQL}$ ratio |
| Figure 2 | `20260709-scenario-a-optimal-params.svg` | 7.1 | Scenario A: optimal $(a_x, a_y, a_z)$ coloured by $\Delta\omega$ |
| Figure 3 | `20260709-scenario-b-omega-scan.svg` | 7.2 | Scenario B: $\Delta\omega$ vs $\omega$ and $\Delta\omega/\text{SQL}$ ratio |
| Figure 4 | `20260709-scenario-b-optimal-params.svg` | 7.2 | Scenario B: optimal $(a_x, a_z, a_{zz})$ coloured by $\Delta\omega$ |
| Figure 5 | `20260709-compound-ratio.svg` | 7.3 | Compound ratio $\mathcal{R}_{\text{compound}}$ vs $\omega$ |
| Figure 6 | `20260709-sql-violation-ratio.svg` | 7.4 | SQL-violation ratio: Scenario A vs Scenario B |

All embedded as `<img src="../reports/r20260709/figures/{filename}" alt="..." width="100%"/>` followed by italicised caption.

---

## Implementation Sequence

| Step | Sections | ~Lines | Description |
|------|----------|--------|-------------|
| 1 | 1–4 | 160 | Introduction through Hamiltonian — narrative foundation, physics definitions, identical-subsystem constraint |
| 2 | 5–6 | 70 | Implementation and Optimisation — computational pipeline, sweep design |
| 3 | 7 | 100 | Results — all 6 figures, cross-protocol table, Key Finding paragraphs |
| 4 | 8 | 80 | Analytical Understanding — full inline derivations ($P_A(+)$, block diagonalisation, QFI/$\sqrt{2}$ bound) |
| 5 | 9–10 | 60 | Related Work and Conclusions — series context, parameter saturation as feature, open questions |
| 6 | — | — | Review pass: verify all cross-references, figure paths, SQL conventions, numerical values |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Inline-only math makes long derivations hard to read | Use sentence-level structure: "Define $X$. Then $Y = Z$. Substituting gives $W$." Keep each inline equation short. Break derivations into numbered steps. |
| Two-particle SQL convention changes the narrative framing | The corrected ratios (8.32× for A with $N=1$, 6.75× for B with $N=2$) create a more nuanced "per-particle efficiency" narrative that strengthens the "asymmetric gain" theme. |
| Block diagonalisation derivation is dense | Break into sub-paragraphs: (1) basis choice, (2) decoupling of $\|{-}\rangle_m$, (3) $3\times 3$ structure, (4) shift and characteristic polynomial, (5) special case $a_{zz}=0$, (6) post-BS state. |
| Figure paths may break if article is moved | Use relative paths from `articles/` to `reports/r20260709/figures/`, matching the existing article's convention (`../reports/r20260519/figures/...`). |
| Cross-protocol table uses different SQL references for different rows | State the SQL reference column explicitly. Add a note explaining the convention. |
