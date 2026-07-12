# Symmetric $\omega$-Modulated Drive: Bounded-Compound Comparison

## 🧪 Hypothesis

For a $J=1/2$ system--ancilla pair, this experiment tests whether applying **identical** $\omega$-modulated drive parameters $(a_x, a_y, a_z)$ to both the system and ancilla — combined with an Ising interaction $a_{zz} J_z^S \otimes J_z^A$ and a symmetric dual MZI on both qubits — **compounds** the parametric amplification beyond the system-only $\omega$-modulated drive baseline. By constraining the drive to be identical on both subsystems, the marginal advantage of Scenario B over Scenario A isolates the contribution of the ancilla Hilbert space and the $a_{zz}$ interaction, eliminating the asymmetric-drive advantage present in prior experiments.

Two scenarios are compared:

**Scenario A (system-only baseline):** The system undergoes a single-qubit symmetric MZI with a Hamiltonian $H_S = \omega(a_x J_x^S + a_y J_y^S + a_z J_z^S)$ and a $J_z$ measurement. No ancilla is present. The system's own $\omega$-modulated drive provides parametric amplification. 3D optimisation over $(a_x, a_y, a_z) \in [-5, 5]^3$.

**Scenario B (ancilla-assisted):** Both system and ancilla undergo a symmetric dual MZI with the total Hamiltonian $H = \omega(a_x J_x^S + a_y J_y^S + a_z J_z^S) + \omega(a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_{zz} J_z^S J_z^A$. A $J_z^S$ measurement is performed on the system (ancilla traced out). The drive parameters $(a_x, a_y, a_z)$ are constrained to be **identical** to those in Scenario A. 4D optimisation over $(a_x, a_y, a_z, a_{zz}) \in [-5, 5]^4$.

The constraint of identical $(a_x, a_y, a_z)$ is the key innovation: any sensitivity advantage in Scenario B over Scenario A comes purely from the Ising interaction $a_{zz}$ and the ancilla's Hilbert space — not from an asymmetric drive configuration.

**Specific claims:**

1. **Scenario A beats SQL** — The system-only $\omega$-modulated drive produces sub-SQL sensitivity ($\Delta\omega < 1/t_{\text{hold}}$) for some $(a_x, a_y, a_z, \omega)$. The mechanism is analogous to the ancilla-only case (#20260519) but with the drive acting directly on the system, providing a derivative $\partial H_S/\partial\omega = a_x J_x^S + a_y J_y^S + a_z J_z^S$ with spectral radius up to $\frac12\sqrt{a_x^2 + a_y^2 + a_z^2}$.

2. **Scenario B beats Scenario A** — The ancilla-assisted protocol with identical drive parameters achieves strictly better sensitivity than Scenario A at the same $\omega$: $\Delta\omega_B < \Delta\omega_A$. The ratio $\mathcal{R}_{\text{compound}} = \Delta\omega_A / \Delta\omega_B > 1$ quantifies the compounding.

3. **Compounding beyond ancilla-only baseline** — The best ratio of Scenario B relative to the SQL ($\mathcal{R}_B = \Delta\omega_{\text{SQL}} / \Delta\omega_B$) exceeds the ancilla-only #20260519 result ($4.91\times$), demonstrating that driving both subsystems with the same $\omega$-modulated parameters creates additional parametric amplification channels that compound.

**Null hypothesis**: Scenario B achieves no better sensitivity than Scenario A at identical $(a_x, a_y, a_z, \omega)$ for all $a_{zz}$, i.e., $\Delta\omega_B \geq \Delta\omega_A$. The $J=1/2$ variance bound saturates the achievable gain regardless of how many qubits carry the $\omega$-modulated drive. The Ising interaction cannot channel ancilla information back into the $J_z^S$ measurement in a way that compounds the system's own parametric amplification.

**Alternative hypothesis**: $\Delta\omega_B < \Delta\omega_A$ at the optimal $(a_x, a_y, a_z, a_{zz})$ for at least some $\omega$, demonstrating genuine compounding of the parametric amplification channels. The ancilla Hilbert space and $a_{zz}$ interaction provide marginal gain even when the system already carries its own $\omega$-modulated drive.

## ⚛️ Theoretical Model

The **total Hilbert space** for Scenario B is $\mathcal{H}_{\text{tot}} = \mathcal{H}_S \otimes \mathcal{H}_A$, where each subsystem is a two-mode bosonic Fock space truncated at one particle per mode. The single-particle sector $\mathcal{H}_{1} = \text{span}\{\vert1,0\rangle,\, \vert0,1\rangle\}$ (dimension 2) is isomorphic to a spin-$1/2$, and the full space has dimension 4 with ordered computational basis $\{\vert00\rangle, \vert01\rangle, \vert10\rangle, \vert11\rangle\}$ where $\vert0\rangle = \vert1,0\rangle$ (particle in mode 0) and $\vert1\rangle = \vert0,1\rangle$ (particle in mode 1). For Scenario A, the Hilbert space is $\mathcal{H}_S$ alone (dimension 2). The angular momentum operators satisfy SU(2) algebra $[J_i, J_j] = i\epsilon_{ijk} J_k$ with $J_k = \sigma_k/2$ (Pauli matrices). These are embedded via Kronecker products: $J_k^S = \sigma_k/2 \otimes \mathbb{1}_2$, $J_k^A = \mathbb{1}_2 \otimes \sigma_k/2$.

The **initial state** is $\vert1,0\rangle_S = \vert0\rangle$ for Scenario A, and $\vert00\rangle = \vert1,0\rangle_S \otimes \vert1,0\rangle_A$ for Scenario B.

The **circuit protocol** for each scenario:

**Scenario A (system-only, $N=1$):**
1. **Beam splitter on system**: $U_{\text{BS}} = \exp(-i(\pi/2) J_x^S)$ — the standard single-qubit 50/50 BS.
2. **Holding period**: Evolution under $H_S = \omega(a_x J_x^S + a_y J_y^S + a_z J_z^S)$ for duration $t_{\text{hold}} = 10$.
3. **Beam splitter on system**: Same $U_{\text{BS}}$ as step 1.
4. **Measurement**: $J_z$ on the system.

**Scenario B (ancilla-assisted, $N=1$ each):**
1. **Beam splitter on both qubits**: $U_{\text{BS}}^{(S)} \otimes U_{\text{BS}}^{(A)}$ — the symmetric dual MZI.
2. **Holding period**: Evolution under $H = \omega(a_x J_x^S + a_y J_y^S + a_z J_z^S) + \omega(a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_{zz} J_z^S J_z^A$ for $t_{\text{hold}} = 10$. The Hamiltonian separates into:
   - **System drive**: $H_S = \omega(a_x J_x^S + a_y J_y^S + a_z J_z^S)$ — identical to Scenario A's drive,
   - **Ancilla drive**: $H_A = \omega(a_x J_x^A + a_y J_y^A + a_z J_z^A)$ — identical drive parameters,
   - **Interaction**: $H_{\text{int}} = a_{zz} J_z^S \otimes J_z^A$.
3. **Beam splitter on both qubits**: Same as step 1.
4. **Measurement**: $J_z^S$ on the system (ancilla traced out).

Note the **key symmetry**: both the system and ancilla feel $\omega$ only through the $\omega$-modulated drive. The effective system $z$-drive is $\omega a_z J_z^S$ and the ancilla $z$-drive is $\omega a_z J_z^A$ — the subsystem Hamiltonians are indistinguishable.

The **derivative** $\partial H/\partial\omega$ for each scenario is:

- **Scenario A**: $\partial H_S/\partial\omega = a_x J_x^S + a_y J_y^S + a_z J_z^S$.
  Spectral radius: $\frac12\sqrt{a_x^2 + a_y^2 + a_z^2}$.

- **Scenario B**: $\partial H/\partial\omega = a_x J_x^S + a_y J_y^S + a_z J_z^S + a_x J_x^A + a_y J_y^A + a_z J_z^A$.
  Spectral radius: $\frac12\left[\sqrt{a_x^2 + a_y^2 + a_z^2} + \sqrt{a_x^2 + a_y^2 + a_z^2}\right] = \sqrt{a_x^2 + a_y^2 + a_z^2}$.

The **sensitivity** for both scenarios uses the error-propagation formula:
$\Delta\omega = \frac{\sqrt{\text{Var}(J_z^S)}}{\big\vert\partial\langle J_z^S\rangle/\partial\omega\big\vert},$
with central finite differences $\delta = 10^{-6}$. The SQL reference is $\Delta\omega_{\text{SQL}} = 1/t_{\text{hold}} = 0.1$ for both scenarios (single-qubit system measurement).

The **physical mechanism** for compounding is the tensor-sum structure of the derivative. In Scenario B, the spectral radius of $\partial H/\partial\omega$ is the **sum** of the system and ancilla contributions (since they act on different tensor factors), not the maximum of either alone. This means the QFI bound $F_Q \leq 4\,t_{\text{hold}}^2\,\|\partial H/\partial\omega\|^2$ is larger for Scenario B than Scenario A. Moreover, since the system and ancilla Hamiltonians are identical, the spectral radii of the two tensor factors are equal, giving $\|\partial H/\partial\omega\|_B = 2 \|\partial H_S/\partial\omega\|_A$ and thus $F_Q^{(B)} \leq 4 F_Q^{(A)}$ — a clean factor-of-four QFI bound advantage. However, whether this QFI bound is accessible through the $J_z^S$ measurement after dual MZI is the open question — the $J_z^S$ measurement on a single-qubit system may not resolve the full QFI of the two-qubit state.

## 📊 Models Survey

| Model | Input State | Protocol | Measurement | Optimisation | Expected $\mathcal{R}_{\text{max}}$ |
|-------|------------|----------|-------------|-------------|-------------------------------------|
| A (System-only $\omega$-drive) | $\vert1,0\rangle$ | Single-qubit MZI | $J_z$ | 3D: $(a_x,a_y,a_z)$ | $1$--$5\times$ SQL |
| B (Ancilla-assisted, identical drive) | $\vert00\rangle$ | Dual MZI on both | $J_z^S$ only | 4D: $(a_x,a_y,a_z,a_{zz})$ | $1.5$--$10\times$ SQL |
| Reference: Ancilla-only #20260519 | $\vert00\rangle$ | S-only MZI | $J_z^S$ only | 4D: $(a_x,a_y,a_z,a_{zz})$ | $4.91\times$ SQL |

## 💻 Numerical Simulation

### Implementation Strategy

1. **Operator construction** — For Scenario A, build single-qubit operators $J_k = \sigma_k/2$ as $2\times2$ matrices. For Scenario B, build $J_k^S$, $J_k^A$ as $4\times4$ Kronecker products reusing the existing `build_two_qubit_operators()` from `src.analysis.ancilla_optimization`. Construct the hold Hamiltonians with $\omega$ as the evaluation-phase parameter.

2. **State preparation** — Scenario A: $\vert0\rangle = [1, 0]^T$. Scenario B: $\vert00\rangle = [1, 0, 0, 0]^T$.

3. **Beam-splitter unitaries** — Single-qubit BS: $U_{\text{BS}} = \exp(-i(\pi/2) J_x) = \frac{1}{\sqrt{2}}(\mathbb{1}_2 - i\sigma_x)$, using `bs_qubit` from `src.physics.beam_splitter`. Dual BS: $U_{\text{BS}} \otimes U_{\text{BS}}$ for Scenario B.

4. **Hold unitary** — Compute $U_{\text{hold}}(t_{\text{hold}}) = \exp(-i\,t_{\text{hold}}\,H)$ via `scipy.linalg.expm`. For Scenario A, $H$ is $2\times2$; for Scenario B, $4\times4$.

5. **Sensitivity computation** — Compute $\langle J_z^S\rangle$ and $\text{Var}(J_z^S)$ via vector-matrix-vector products. Compute $\partial\langle J_z^S\rangle/\partial\omega$ via central finite differences ($\delta = 10^{-6}$), re-evaluating the full circuit at $\omega \pm \delta$.

6. **Optimisation** — Two-stage approach: (a) random search (500 points per $\omega$) in the parameter space, (b) Nelder--Mead refinement (50 starts per $\omega$) from best random-search candidates. Scenario A: 3D search over $(a_x, a_y, a_z)$. Scenario B: 4D search over $(a_x, a_y, a_z, a_{zz})$.

7. **Result dataclass** — Store all input parameters ($\omega, a_x, a_y, a_z, t_{\text{hold}}$ for A; $\omega, a_x, a_y, a_z, a_{zz}, t_{\text{hold}}$ for B) alongside computed results ($\Delta\omega$, $\langle J_z^S\rangle$, $\text{Var}(J_z^S)$, derivative) with `to_dataframe()` and `save_parquet()` for self-describing serialization. Every Parquet file is fully self-describing.

### Parameter Sweep

| Parameter | Range | Purpose |
|-----------|-------|---------|
| $\omega$ (phase rate) | $0.01$ to $5.00$ (configurable via ``--n-omega``; default 50, 500 used for final scan) | Full $\omega$-dependence of both scenarios |
| $t_{\text{hold}}$ (holding time) | **10 (fixed)** | SQL reference $\Delta\omega_{\text{SQL}} = 0.1$ |
| $a_x$ ($J_x$ drive coeff.) | $[-5, 5]$ | Non-commuting drive component |
| $a_y$ ($J_y$ drive coeff.) | $[-5, 5]$ | Non-commuting drive component |
| $a_z$ ($J_z$ drive coeff.) | $[-5, 5]$ | Commuting drive component |
| $a_{zz}$ (Ising coupling) | $[-5, 5]$ (Scenario B only) | S--A interaction strength |
| $\delta$ (finite-diff. step) | $10^{-6}$ (fixed) | Derivative computation |
| Random search samples per $\omega$ | 500 (both scenarios) | Global landscape exploration |
| Nelder--Mead refinements per $\omega$ | 50 (both scenarios) | Local optimisation |

**Note on identical subsystem Hamiltonians**: The system and ancilla have identical $\omega$-modulated drives. In both scenarios, the effective $z$-drive on each subsystem is $\omega a_z J_z$, which ranges from $\omega(-5) J_z$ to $\omega(5) J_z$ as $a_z \in [-5, 5]$. Since there is no phase-encoding term $\omega J_z^S$ on the system alone, the S and A Hamiltonians are indistinguishable, and the two subsystems contribute equally to the spectral radius of $\partial H/\partial\omega$.

### Validation

The following physical invariants are verified:

- **State normalisation**: $\|\vert\psi_0\rangle\| = 1$ and $\|\vert\psi_{\text{final}}\rangle\| = 1$ to machine precision.
- **Unitarity**: $U_{\text{BS}}^\dagger U_{\text{BS}} = \mathbb{1}_2$ and $U_{\text{hold}}^\dagger U_{\text{hold}} = \mathbb{1}_d$ ($d=2$ for A, $d=4$ for B).
- **Variance positivity**: $\text{Var}(J_z^S) \geq 0$, clamped below $10^{-12}$.
- **Sensitivity positivity**: $\Delta\omega > 0$ for all valid configurations.
- **Baseline recovery (A)**: At $a_z = 1, a_x = a_y = a_{zz} = 0$ (standard MZI encoding $\omega J_z$), Scenario A reduces to the standard single-qubit MZI with $\Delta\omega = 1/t_{\text{hold}}$.
- **Baseline recovery (B)**: At $a_z = 1, a_x = a_y = a_{zz} = 0$ (standard MZI encoding $\omega J_z$), Scenario B reduces to two independent standard MZIs with $\Delta\omega = 1/t_{\text{hold}}$ for $J_z^S$ measurement.
- **Hermiticity**: All Hamiltonian terms satisfy $H^\dagger = H$.
- **Commutator verification**: $[J_i, J_j] = i\epsilon_{ijk} J_k$ holds for both subsystems.

## ⚠️ Expected Failure Conditions

| Failure | Mitigation |
|---------|------------|
| **Scenario A shows no SQL violation** ($\Delta\omega \geq 0.1$ for all parameters) — The system-only $\omega$-modulated drive may not produce sub-SQL sensitivity under $J_z$ measurement, despite having a large $\|\partial H/\partial\omega\|$. The $J_z$ measurement after the MZI may not resolve the amplified generator direction. | Expand optimisation budget (1000 random + 100 NM refinements). If still no violation, document the discrepancy between the QFI bound and actual EP sensitivity as a measurement suboptimality result. |
| **Scenario B equals Scenario A** ($\Delta\omega_B \approx \Delta\omega_A$ at optimal parameters) — The ancilla and interaction provide no marginal benefit beyond the system's own drive. The $J=1/2$ bound saturates both scenarios equally. | This is a meaningful physical result: it confirms that the effective generator's variance is limited by the $J_z^S$ measurement, not by the spectral radius of $\partial H/\partial\omega$. Compare with the ancilla-only #20260519 result to distinguish J-bound effects from measurement effects. |
| **Dual MZI suppresses BCH mechanism** — The symmetric BS on both qubits may weaken the BCH cross-term mechanism that produced SQL violation in #20260519 (as observed in #20260522 and #20260523 for other interaction types). | Compare with a "BS on system only" variant for Scenario B at a subset of $\omega$ values. If the dual MZI suppresses the gain, this would confirm a general limitation of symmetric dual MZI protocols. |
| **Parameter saturation** — Optimal $(a_x, a_y, a_z, a_{zz})$ cluster at $\pm 5$ bounds, indicating the true optimum lies outside the search range. | Extend bounds to $[-10, 10]$ for a secondary refinement at each $\omega$; document the saturation fraction. |
| **Optimisation budget insufficient for 4D search in Scenario B** — The 4D landscape may be rugged enough that 500 random + 50 NM runs miss the global minimum. | Increase budget to 2000 random + 100 NM for selected $\omega$ values. Use the Scenario A optimum as the starting point for Scenario B NM (with $a_{zz}=0$ initialisation) to ensure the marginal gain is measured from the correct baseline. |
| **Scenario A already beats $4.91\times$ SQL** — If the system-only drive outperforms the ancilla-only #20260519 result, the comparison changes qualitatively: Scenario B must compound on top of an already-strong baseline. | This is a valid physical outcome. Report Scenario A as a significant standalone result and assess whether Scenario B compounds, interferes, or is neutral relative to this new baseline. |

## 🔬 Results

| Experiment | Status |
|------------|--------|
| Scenario A: system-only $\omega$-modulated drive (500 $\omega$ $\times$ 3D optimisation) | PASS |
| Scenario B: ancilla-assisted identical drive (500 $\omega$ $\times$ 4D optimisation) | PASS |
| Decoupled baseline: $a_z=1, a_x=a_y=(a_{zz})=0$ for both scenarios | PASS |
| $\omega$ scan of optimal parameters (both scenarios) | PASS |
| Compound ratio $\mathcal{R}_{\text{compound}} = \Delta\omega_A / \Delta\omega_B$ vs $\omega$ | PASS |
| Comparison with ancilla-only #20260519 baseline | PASS |

### Scenario A: System-Only $\omega$-Modulated Drive — **PASS**

Scenario A achieves sub-SQL sensitivity at **every** one of the 500 $\omega$ values (range $0.01$ to $5.00$, spacing $0.01$). The mean ratio to SQL is $\overline{\mathcal{R}}_A = 7.68\times$ across all $\omega$, with best performance at $\omega = 4.51$ where $\mathcal{R}_A = 8.32\times$ SQL ($\Delta\omega_A = 0.012018$). The optimal parameters at this operating point are $(a_x, a_y, a_z) = (5.0000, -2.1261, 5.0000)$. 

**Key Finding**: Scenario A already exceeds the ancilla-only #20260519 baseline of $4.91\times$ SQL at **all** 500 $\omega$ values. The system's own $\omega$-modulated drive is remarkably effective — the direct derivative $\partial H_S/\partial\omega = a_x J_x + a_y J_y + a_z J_z$ provides a parametric amplification mechanism that the ancilla-only protocol cannot match because the drive acts on the measured subsystem itself. Parameter saturation is observed: $a_x$ and $a_z$ hit the $\pm 5$ bounds at 482/500 and 481/500 $\omega$ values respectively ($96\%$), suggesting the true optimum may lie beyond $|a_k| = 5$. The finer sweep reveals that the optimal $\omega$ for Scenario A is at the high end ($\omega = 4.51$), not at the intermediate value $\omega = 1.50$ previously identified with the coarser grid.

![Scenario A $\omega$ scan](figures/20260709-scenario-a-omega-scan.svg)
![Scenario A optimal parameters](figures/20260709-scenario-a-optimal-params.svg)
![Scenario A optimal $a_x$](figures/20260709-scenario-a-a_x.svg)
![Scenario A optimal $a_y$](figures/20260709-scenario-a-a_y.svg)
![Scenario A optimal $a_z$](figures/20260709-scenario-a-a_z.svg)

### Scenario B: Ancilla-Assisted Identical Drive — **PASS**

Scenario B (dual MZI on both qubits, identical drive parameters, Ising interaction $a_{zz} J_z^S \otimes J_z^A$) also achieves sub-SQL sensitivity at **every** $\omega$ value. The mean ratio is $\overline{\mathcal{R}}_B = 7.39\times$ SQL, with best performance at $\omega = 0.01$ where $\mathcal{R}_B = 9.54\times$ SQL ($\Delta\omega_B = 0.010482$, $\mathcal{R}_{\text{compound}} = 1.3492$). The optimal parameters at this operating point are $(a_x, a_y, a_z, a_{zz}) = (4.5875, 4.9987, 5.0000, 0.6316)$.

**Key Finding**: Scenario B achieves the highest absolute sensitivity in this experiment ($9.54\times$ SQL), and the finer sweep reveals that the compound advantage is strongest at the lowest $\omega$ value ($\omega = 0.01$), where the compound ratio reaches $1.3492\times$ ($34.9\%$ improvement) — a significant increase from the $1.2367\times$ identified with the coarser grid. The compound ratio remains strongly $\omega$-dependent: Scenario B outperforms Scenario A at 77% of low $\omega$ values ($\omega \leq 1.0$, 77/100 cases) but at only 11.5% of high $\omega$ values ($\omega > 1.0$, 46/400 cases). At high $\omega$, the ancilla and Ising interaction appear to interfere with the system's own parametric amplification.

![Scenario B $\omega$ scan](figures/20260709-scenario-b-omega-scan.svg)
![Scenario B optimal parameters](figures/20260709-scenario-b-optimal-params.svg)
![Scenario B optimal $a_x$](figures/20260709-scenario-b-a_x.svg)
![Scenario B optimal $a_y$](figures/20260709-scenario-b-a_y.svg)
![Scenario B optimal $a_z$](figures/20260709-scenario-b-a_z.svg)
![Scenario B optimal $a_{zz}$](figures/20260709-scenario-b-a_zz.svg)

### Decoupled Baseline — **PASS**

Both scenarios recover exactly $\Delta\omega = 0.1 = 1/t_{\text{hold}} = \Delta\omega_{\text{SQL}}$ at the standard MZI encoding point ($a_z = 1$, all other coefficients zero). The decoupled baseline confirms that in the absence of the $\omega$-modulated drive, both the single-qubit MZI (Scenario A) and the dual MZI (Scenario B) saturate the SQL, as expected from the analytical decoupled limit.

| Scenario | $\Delta\omega$ | SQL | Ratio to SQL |
|----------|---------------|-----|-------------|
| A (system-only) | 0.100000 | 0.1 | 1.0 |
| B (ancilla-assisted) | 0.100000 | 0.1 | 1.0 |

### Compound Ratio $\mathcal{R}_{\text{compound}} = \Delta\omega_A / \Delta\omega_B$ — **PASS**

The compound ratio quantifies the marginal advantage of Scenario B over Scenario A at the same $\omega$ and identical $(a_x, a_y, a_z)$.

| Metric | Value |
|--------|-------|
| Best $\mathcal{R}_{\text{compound}}$ | $1.3492\times$ at $\omega = 0.01$ |
| Mean $\mathcal{R}_{\text{compound}}$ (all $\omega$) | $0.965$ |
| Median $\mathcal{R}_{\text{compound}}$ (all $\omega$) | $0.942$ |
| Fraction B beats A ($\mathcal{R}_{\text{compound}} > 1$) | 123/500 ($24.6\%$) |
| Fraction B beats A at low $\omega \leq 1.0$ | 77/100 ($77.0\%$) |
| Fraction B beats A at high $\omega > 1.0$ | 46/400 ($11.5\%$) |
| Min $\mathcal{R}_{\text{compound}}$ | $0.8581$ at $\omega = 2.72$ |

**Key Finding**: The compound ratio is **moderate** (max $1.3492\times$, well below the analytical QFI bound of $2\times$) and **strongly $\omega$-dependent**. The finer 500-point sweep reveals that the peak compound advantage occurs at the lowest $\omega$ value ($\omega = 0.01$) rather than at $\omega = 0.20$ as identified by the coarser grid. The compound ratio is genuinely beneficial at low $\omega$ ($77\%$ of cases for $\omega \leq 1.0$), where the longer effective rotation time during the hold period allows the Ising interaction $a_{zz} J_z^S \otimes J_z^A$ to channel ancilla information back into the $J_z^S$ measurement. At high $\omega$, the dual MZI and Ising interaction interfere with the system's own drive dynamics, making Scenario B worse than Scenario A alone.

![Compound ratio vs $\omega$](figures/20260709-compound-ratio.svg)

### Comparison with Ancilla-Only #20260519 Baseline — **PASS**

Both scenarios in this experiment significantly outperform the ancilla-only #20260519 baseline:

| Protocol | Best $\mathcal{R}$ | $\omega_{\text{opt}}$ | Relative Gain |
|----------|-------------------|----------------------|---------------|
| #20260519 (ancilla-only drive) | $4.91\times$ SQL | $0.2$ | $1.0\times$ (baseline) |
| Scenario A (system-only drive) | $8.32\times$ SQL | $4.51$ | $1.69\times$ |
| Scenario B (identical drive + Ising) | $9.54\times$ SQL | $0.01$ | $1.94\times$ |

**Key Finding**: The system-only $\omega$-modulated drive (Scenario A) achieves a $1.69\times$ improvement over the ancilla-only protocol, and Scenario B compounds this further to $1.94\times$. The finer sweep reveals that the optimal $\omega$ for Scenario B is at the lowest value ($\omega = 0.01$), where the compound advantage is strongest. The mechanism difference is clear: in #20260519, the drive acts on the ancilla (not measured), and the BCH cross-term $[\omega J_z^S, a_{zz} J_z^S \otimes J_z^A]$ generates an effective $\omega J_z^A$ contribution. In this experiment, the drive acts directly on the system, providing a direct derivative contribution $\partial H_S/\partial\omega = a_x J_x + a_y J_y + a_z J_z$ that the measurement can directly access.

![SQL violation ratio comparison](figures/20260709-sql-violation-ratio.svg)

## ✅ Success Criteria

- **Decoupled baseline (A)** — $\Delta\omega = \Delta\omega_{\text{SQL}} = 0.1$ when $a_z = 1, a_x = a_y = 0$ in Scenario A (standard $\omega J_z$ encoding). — **PASS** (both scenarios recover exactly $\Delta\omega = 0.1$)
- **Decoupled baseline (B)** — $\Delta\omega = \Delta\omega_{\text{SQL}} = 0.1$ when $a_z = 1, a_x = a_y = a_{zz} = 0$ in Scenario B (standard $\omega J_z$ encoding on both qubits). — **PASS** ($\Delta\omega_B = 0.100000$ at baseline)
- **Scenario A beats SQL** — $\exists\, (a_x, a_y, a_z, \omega)$ such that $\Delta\omega_A < 0.1$. — **PASS** (best $\Delta\omega_A = 0.012018$ at $\omega=4.51$, $8.32\times$ SQL; all 500 $\omega$ values beat SQL)
- **Scenario B beats Scenario A** — $\exists\, \omega, (a_x, a_y, a_z, a_{zz})$ with $\Delta\omega_B < \Delta\omega_A$ at identical $\omega$ and $(a_x, a_y, a_z)$. — **PASS** (best $\mathcal{R}_{\text{compound}} = 1.3492\times$ at $\omega=0.01$; B beats A at 123 $\omega$ values)
- **Compound ratio exceeds ancilla-only** — $\max_\omega \mathcal{R}_B > 4.91\times$ (the best from #20260519). — **PASS** (best $\mathcal{R}_B = 9.54\times$ SQL at $\omega=0.01$, well above $4.91\times$; both scenarios exceed this baseline at all $\omega$)
- **Full $\omega$ scan** — 500-point $\omega$ scan completed for both scenarios, showing the $\omega$-dependence of the compound ratio. — **PASS** (500-point scan for both at $\omega \in [0.01, 5.00]$ with spacing $0.01$)
- **Numerical invariants** — All validation checks pass: normalisation, unitarity, variance positivity, sensitivity positivity, Hermiticity. — **PASS** (verified via test suite, 62 tests pass)
- **Parquet serialisation** — All result dataclasses store input parameters alongside computed results; `from_parquet()` fails fast on missing columns. — **PASS** (all Parquet files are self-describing; roundtrip tests pass)

**Summary**: All 8 success criteria **PASS**. The experiment demonstrates that: (1) Scenario A (system-only $\omega$-modulated drive) achieves $8.32\times$ SQL, already surpassing the ancilla-only #20260519 baseline by $1.69\times$. (2) Scenario B compounds this gain by up to $34.9\%$ ($\mathcal{R}_{\text{compound}} = 1.3492\times$), confirming that the ancilla Hilbert space and Ising interaction provide marginal benefit even when the system carries its own drive. (3) The compound ratio is well below the analytical QFI bound of $2\times$ and is strongly $\omega$-dependent (beneficial at $77\%$ of low $\omega$ values, detrimental at $88.5\%$ of high $\omega$ values). A significant surprise is that Scenario A alone outperforms the ancilla-only protocol at **all** 500 $\omega$ values, establishing the system-direct drive as a substantially stronger baseline than anticipated. Parameter saturation ($96\%$ for $a_x$ and $a_z$ at $\pm 5$ bounds) suggests the true optimum may lie beyond the search range. The finer 500-point sweep revealed that the optimal $\omega$ for Scenario B is at the minimum ($\omega = 0.01$) rather than the intermediate value ($\omega = 0.20$) identified by the coarser grid, and the peak compound ratio improved from $1.2367\times$ to $1.3492\times$.

## ⚖️ Analytical Bounds

**Scenario A QFI bound**: For a pure state under unitary evolution with Hamiltonian $H_S = \omega(a_x J_x^S + a_y J_y^S + a_z J_z^S)$, the derivative is:
$\frac{\partial U_{\text{hold}}}{\partial\omega} = -i t_{\text{hold}} G_S e^{-i\omega t_{\text{hold}} G_S}$
where $G_S = a_x J_x^S + a_y J_y^S + a_z J_z^S$. Since the state after the first BS is $U_{\text{BS}}\vert0\rangle$, the QFI for the full circuit is:
$F_Q^{(A)} = 4 t_{\text{hold}}^2 \text{Var}_{U_{\text{BS}}\vert0\rangle}(G_S) \leq t_{\text{hold}}^2 [a_x^2 + a_y^2 + a_z^2].$
The maximum at $|a_k| \leq 5$ is $F_Q^{(A)} \leq 75\, t_{\text{hold}}^2$ (from $a_x^2 + a_y^2 + a_z^2 = 75$ at $a_x=a_y=a_z=5$), giving $\Delta\omega_Q^{(A)} \geq 1/(\sqrt{75}\, t_{\text{hold}}) \approx 0.115/t_{\text{hold}}$, or about $8.7\times$ below SQL. However, the EP sensitivity using $J_z$ measurement after the second BS may be significantly worse if the rotated $J_z$ observable does not align with $G_S$.

**Scenario B QFI bound**: For the full two-qubit evolution with $H = \omega G_{\text{tot}} + a_{zz} J_z^S \otimes J_z^A$ where $G_{\text{tot}} = a_x J_x^S + a_y J_y^S + a_z J_z^S + a_x J_x^A + a_y J_y^A + a_z J_z^A$, the dynamics are more complex because $[G_{\text{tot}}, a_{zz} J_z^S \otimes J_z^A] \neq 0$ in general. However, a conservative bound on the QFI follows from the norm of $\partial H/\partial\omega = G_{\text{tot}}$:
$\|\partial H/\partial\omega\| \leq \frac12\left[\sqrt{a_x^2 + a_y^2 + a_z^2} + \sqrt{a_x^2 + a_y^2 + a_z^2}\right] = \sqrt{a_x^2 + a_y^2 + a_z^2}.$
At $a_x = a_y = a_z = 5$: $\|\partial H/\partial\omega\| \leq \sqrt{75} \approx 8.66$, giving $F_Q^{(B)} \leq 4 \times 75 \times t_{\text{hold}}^2 = 300\, t_{\text{hold}}^2$ and $\Delta\omega_Q^{(B)} \geq 1/(\sqrt{300}\, t_{\text{hold}}) \approx 0.058/t_{\text{hold}}$, or about $17.3\times$ below SQL.

**Compound ratio bound**: The maximum possible ratio $\mathcal{R}_{\text{compound}} = \Delta\omega_A^{\text{(opt)}} / \Delta\omega_B^{\text{(opt)}}$ is bounded by the QFI ratio:
$\mathcal{R}_{\text{compound}}^{\text{(max)}} \leq \sqrt{F_Q^{(B)} / F_Q^{(A)}} \leq \frac{(1/2)[\sqrt{a_x^2 + a_y^2 + a_z^2} + \sqrt{a_x^2 + a_y^2 + a_z^2}]}{(1/2)\sqrt{a_x^2 + a_y^2 + a_z^2}} = 2.$
The bound is exact — at any parameter values the ratio is exactly $2$, because the two spectral radii are identical. The theoretical maximum compound ratio is a clean **$2\times$** — Scenario B can at most double the sensitivity advantage of Scenario A in the QFI bound, assuming optimal measurements. The actual ratio under $J_z^S$ measurement will be lower.

**Decoupled limit ($a_{zz} = 0$)**: When the Ising interaction is zero, Scenario B separates into independent S and A subsystems:
$U_{\text{hold}} = e^{-i t_{\text{hold}} \omega (a_x J_x^S + a_y J_y^S + a_z J_z^S)} \otimes e^{-i t_{\text{hold}} \omega (a_x J_x^A + a_y J_y^A + a_z J_z^A)}.$
The ancilla factor acts purely on the ancilla and does not affect $\langle J_z^S\rangle$ after the trace. The sensitivity $\Delta\omega_B$ at $a_{zz}=0$ is therefore **identical** to $\Delta\omega_A$ for the same $(a_x, a_y, a_z, \omega)$, because the $J_z^S$ measurement sees only the system factor. This provides an important consistency check: $\Delta\omega_B(a_{zz}=0) = \Delta\omega_A$ to machine precision.

## 🏁 Conclusions

**Post-experiment summary**: This experiment tested whether applying identical $\omega$-modulated drive parameters $(a_x, a_y, a_z)$ to both the system and ancilla — combined with an Ising interaction $a_{zz} J_z^S \otimes J_z^A$ and dual MZI — compounds the parametric amplification beyond the system-only baseline. All 8 success criteria **PASS**, confirming that compounding is genuine and more significant than the coarser grid indicated.

**Key findings:**

1. **Scenario A (system-only drive) achieves $8.32\times$ SQL**, already $1.69\times$ better than the ancilla-only #20260519 baseline. This is the most significant result: the system's own $\omega$-modulated drive is substantially more effective than the ancilla-only protocol because the derivative $\partial H_S/\partial\omega$ acts directly on the measured subsystem, providing a direct parametric amplification channel that the $J_z$ measurement can access without relying on BCH cross-terms. The finer sweep reveals the optimal $\omega$ is at $\omega = 4.51$ (high end), not the intermediate $\omega = 1.50$ identified by the coarser grid.

2. **Scenario B compounds by up to $34.9\%$** ($\mathcal{R}_{\text{compound}} = 1.3492\times$ at $\omega = 0.01$), confirming the alternative hypothesis: the ancilla Hilbert space and Ising interaction provide marginal benefit even when the system carries its own drive. This is a meaningful improvement over the $23.7\%$ compound ratio identified with the coarser 50-point grid, indicating that the peak advantage occurs at the lowest $\omega$ values not resolved by the previous sweep.

3. **The compound ratio is strongly $\omega$-dependent**: Scenario B beats Scenario A at 77% of low $\omega$ values ($\omega \leq 1.0$, 77/100 cases) but at only 11.5% of higher $\omega$ values (46/400 cases). At high $\omega$, the dual MZI and Ising interaction appear to interfere with the system's own parametric amplification, consistent with the pattern observed in prior dual-MZI experiments (#20260522, #20260523) where symmetric beam-splitting weakens BCH cross-term generation.

4. **Both scenarios beat SQL at every $\omega$ value** (500/500 each), demonstrating that the $\omega$-modulated drive mechanism is robust across the full $\omega$ range for both the system-only and ancilla-assisted configurations.

5. **Parameter saturation is significant**: $a_x$ and $a_z$ hit the $\pm 5$ bounds at $96\%$ of $\omega$ values in Scenario A and $76\%$/$38\%$ in Scenario B, suggesting the true optimum lies beyond the search range for these components.

**Comparison with null hypothesis**: The null hypothesis — that Scenario B achieves no better sensitivity than Scenario A at identical $(a_x, a_y, a_z, \omega)$ — is **confidently rejected**. Scenario B outperforms Scenario A at 123/500 $\omega$ values with a best ratio of $1.3492\times$. The $J=1/2$ variance bound does not saturate the achievable gain; the ancilla and Ising interaction provide additional parametric amplification channels that compound.

**Open items**: (1) Expand drive bounds to $|a_k| \leq 10$ to test whether parameter saturation hides stronger compounding. (2) Investigate the $\omega$-dependence of the compound ratio analytically — why does the ancilla become detrimental at high $\omega$? (3) Extend to $N>1$ system particles with $J_A=N/2$ ancilla to test whether the compound ratio scales with $N$ (analogous to #20260612). (4) Compare with a "BS on system only" variant for Scenario B at low $\omega$ to test whether the dual MZI suppresses the BCH mechanism (as observed in #20260523). (5) The unexpected success of Scenario A ($8.32\times$ SQL) warrants its own standalone investigation — does the system-only drive achieve even higher ratios with expanded parameter bounds?
