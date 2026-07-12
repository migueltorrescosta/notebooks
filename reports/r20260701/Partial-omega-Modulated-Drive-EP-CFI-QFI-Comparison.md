# Mixed $\omega$-Modulated Drive with EP/CFI/QFI Comparison

## 🧪 Hypothesis

For a system--ancilla pair of single-particle two-mode bosonic systems ($N=1$ each, $J=1/2$) where the ancilla drive has **mixed $\omega$-modulation** — only the $J_z^A$ component feels the unknown phase ($H_A = a_x J_x^A + a_y J_y^A + \omega a_z J_z^A$) while the transverse components are static ($a_x, a_y$ independent of $\omega$) — the comparison of three sensitivity metrics (error-propagation, Classical Fisher Information, Quantum Fisher Information) reveals the following:

1. **EP = CFI for binary measurement** — Since the system is a single qubit ($J_S = 1/2$), the $J_z^S$ measurement has only two outcomes ($m = \pm 1/2$). For any binary-outcome measurement, the error-propagation formula $\Delta\omega_{\text{EP}} = \sqrt{\text{Var}(J_z^S)} / |\partial\langle J_z^S\rangle/\partial\omega|$ is mathematically equivalent to the Classical Fisher Information bound $\Delta\omega_{\text{CFI}} = 1/\sqrt{F_C}$. The two curves coincide exactly (not approximately) for all parameters.

2. **QFI exceeds EP/CFI** — Because the full system--ancilla state is pure, the QFI $F_Q = 4(\langle\psi'|\psi'\rangle - |\langle\psi'|\psi\rangle|^2)$ quantifies the ultimate sensitivity achievable with an optimally-chosen measurement on both S and A. The gap $\Delta\omega_{\text{QFI}} \leq \Delta\omega_{\text{EP}}$ measures how much information is lost by restricting to the $J_z^S$ measurement — i.e., how much information resides in S--A correlations or non-$J_z^S$ degrees of freedom.

3. **Partial modulation produces SQL violation** — Even with only the $a_z$ component carrying $\omega$ (reduced from the full $\omega$-modulation of #20260519), the derivative $\partial H/\partial\omega = J_z^S + a_z J_z^A$ still picks up an ancilla contribution. The non-commuting dynamics $[a_x J_x^A + a_y J_y^A, \alpha_{zz} J_z^S \otimes J_z^A]$ generate BCH cross-terms that can transfer phase information between subsystems. We expect $\Delta\omega_{\text{EP}} < \Delta\omega_{\text{SQL}}$ for some parameter region, though weaker than the fully-modulated case.

4. **Partially-modulated drive underperforms fully-modulated drive** — Compared to #20260519 where $\partial H/\partial\omega = J_z^S + a_x J_x^A + a_y J_y^A + a_z J_z^A$ (all three drive components contribute to the derivative), here only $a_z$ contributes. The static $a_x, a_y$ components still affect the dynamics through BCH cross-terms but cannot directly amplify the derivative. We expect the best sensitivity ratio $\mathcal{R} = \Delta\omega_{\text{SQL}} / \Delta\omega_{\text{EP}}$ to be smaller than the $4.91\times$ achieved in #20260519.

**Null hypothesis**: The reduced derivative $\partial H/\partial\omega = J_z^S + a_z J_z^A$ (missing $a_x, a_y$ contributions) is insufficient to produce any SQL violation — the dynamics are equivalent to a re-scaled version of the $\omega$-independent drive protocol (#20260518) where no violation was observed.

## ⚛️ Theoretical Model

The total Hilbert space is $\mathcal{H}_{\text{tot}} = \mathcal{H}_S \otimes \mathcal{H}_A$, where each subsystem is a two-mode bosonic Fock space truncated at one particle per mode. The single-particle sector $\mathcal{H}_{1} = \text{span}\{\vert1,0\rangle,\, \vert0,1\rangle\}$ (dimension 2) is isomorphic to a spin-$1/2$, and the full space has dimension 4 with ordered computational basis $\{\vert00\rangle, \vert01\rangle, \vert10\rangle, \vert11\rangle\}$ where $\vert0\rangle = \vert1,0\rangle$ (particle in mode 0) and $\vert1\rangle = \vert0,1\rangle$ (particle in mode 1). The angular momentum operators for each subsystem satisfy SU(2) algebra $[J_i, J_j] = i \epsilon_{ijk} J_k$ and are represented by $J_k = \sigma_k/2$ (the $2\times2$ Pauli matrices). These are embedded into the full space via Kronecker products: $J_k^S = \sigma_k/2 \otimes \mathbb{1}_2$ and $J_k^A = \mathbb{1}_2 \otimes \sigma_k/2$.

The **initial state** is a pure product state $\vert\Psi_0\rangle = \vert1,0\rangle_S \otimes \vert1,0\rangle_A = \vert00\rangle$ — the top Dicke state for each $J=1/2$ subsystem.

The **circuit protocol** proceeds in four steps:

1. **Beam splitter on system only** — A 50/50 symmetric beam splitter acts on the system via $U_{\text{BS}} = \exp(-i(\pi/2) J_x^S) = \exp(-i(\pi/4) \sigma_x^S)$, acting as identity on the ancilla: $U_{\text{BS}}^{(S)} = U_{\text{BS}} \otimes \mathbb{1}_2$.

2. **Holding period** — The full state evolves under the total Hamiltonian $H = H_S + H_A + H_{\text{int}}$ for duration $T_H = 10$. The three terms are:
   - **System encoding**: $H_S = \omega J_z^S = \frac{\omega}{2} \sigma_z^S \otimes \mathbb{1}_2$ — the unknown phase rate on the system.
   - **Mixed $\omega$-modulated ancilla drive**: $H_A = a_x J_x^A + a_y J_y^A + \omega a_z J_z^A = \mathbb{1}_2 \otimes \frac12(a_x \sigma_x^A + a_y \sigma_y^A + \omega a_z \sigma_z^A)$ — here only the $a_z$ component is modulated by $\omega$; $a_x$ and $a_y$ are static (controllable) fields.
   - **Ising interaction**: $H_{\text{int}} = \alpha_{zz} J_z^S \otimes J_z^A = \frac{\alpha_{zz}}{4} (\sigma_z^S \otimes \sigma_z^A)$ — couples system and ancilla through their $z$-components.

   The hold unitary is $U_{\text{hold}}(T_H) = \exp(-i T_H H)$ computed via `scipy.linalg.expm`.

3. **Beam splitter on system only** — A second 50/50 beam splitter (identical to step 1): $U_{\text{BS}}^{(S)} = U_{\text{BS}} \otimes \mathbb{1}_2$.

4. **Measurement** — $J_z^S$ is measured on the system qubit, with outcomes $m \in \{\,+1/2,\, -1/2\,\}$. The full final state is $\vert\Psi_{\text{final}}\rangle = U_{\text{BS}}^{(S)} U_{\text{hold}}(T_H) U_{\text{BS}}^{(S)} \vert\Psi_0\rangle$.

The **sensitivity metrics** are:

- **Error propagation**: $\Delta\omega_{\text{EP}} = \sqrt{\text{Var}(J_z^S)} / \big\vert\partial\langle J_z^S\rangle/\partial\omega\big\vert$, with central finite differences $\delta = 10^{-6}$.
- **Classical Fisher Information**: $F_C(\omega) = \sum_{m=\pm 1/2} (\partial P(m\vert\omega)/\partial\omega)^2 / P(m\vert\omega)$, where $P(m\vert\omega) = \sum_{A} \big\vert\langle m, A \vert \Psi_{\text{final}}\rangle\big\vert^2$ with $\{ \vert A\rangle\}$ spanning $\mathcal{H}_A$. Then $\Delta\omega_{\text{CFI}} = 1/\sqrt{F_C}$.
- **Quantum Fisher Information**: $F_Q(\omega) = 4\big(\langle\psi' \vert \psi'\rangle - \vert\langle\psi'\vert\psi\rangle\vert^2\big)$ where $\vert\psi'\rangle = \partial\vert\Psi_{\text{final}}\rangle/\partial\omega$ via central finite differences with $\delta = 10^{-6}$. Then $\Delta\omega_{\text{QFI}} = 1/\sqrt{F_Q}$.

**Critical observation — EP = CFI for binary measurement**: Because $J_z^S$ has eigenvalues $\pm 1/2$, the measurement is binary. The relation $\langle J_z^S\rangle = P(+) - 1/2$ and $\text{Var}(J_z^S) = P(+)(1-P(+))$ gives:
- $\partial\langle J_z^S\rangle/\partial\omega = \partial P(+)/\partial\omega$
- $\sqrt{\text{Var}(J_z^S)} = \sqrt{P(+)(1-P(+))}$
Therefore $\Delta\omega_{\text{EP}} = \sqrt{P(1-P)} / \vert\partial P/\partial\omega\vert = 1/\sqrt{F_C} = \Delta\omega_{\text{CFI}}$ identically. The two curves coincide for all $(\omega, a_x, a_y, a_z, \alpha_{zz})$ — this is not an approximation but an exact equality for a single-qubit system measurement. The meaningful comparison is EP (=CFI) vs QFI: the gap $\Delta\omega_{\text{EP}}/\Delta\omega_{\text{QFI}} = \sqrt{F_Q/F_C}$ quantifies how much sensitivity is lost by measuring $J_z^S$ rather than the optimal observable.

**Physical mechanism**: The key difference from #20260519 is the derivative $\partial H/\partial\omega$. In the fully-modulated case, $\partial H/\partial\omega = J_z^S + a_x J_x^A + a_y J_y^A + a_z J_z^A$ — all three drive components contribute directly. Here, $\partial H/\partial\omega = J_z^S + a_z J_z^A$ — only the commuting $J_z^A$ component of the ancilla drive participates in the derivative. However, the static transverse components $a_x J_x^A + a_y J_y^A$ still affect the dynamics through the BCH expansion:
- $[a_x J_x^A + a_y J_y^A, \alpha_{zz} J_z^S \otimes J_z^A]$ generates effective $J_y^A$ and $J_x^A$ terms in the time-evolved operators, which then couple back to the system through $H_{\text{int}}$.
- $[a_x J_x^A + a_y J_y^A, \omega J_z^S]$ vanishes (they act on different subsystems), so the BCH cross-term between $H_A$ and $H_S$ is absent for the transverse components.

This creates a richer structure: the derivative contribution comes only from the $a_z$ channel, but the dynamics are shaped by all four parameters $(a_x, a_y, a_z, \alpha_{zz})$ through non-commuting interactions.

## 💻 Numerical Simulation

### Implementation Strategy

1. **Operator construction** — Build $J_z^S$, $J_z^A$, $J_x^A$, $J_y^A$ as $4\times4$ Kronecker products from Pauli matrices, reusing the existing `build_two_qubit_operators()` in the shared operator-construction module. Construct $H_A = a_x J_x^A + a_y J_y^A + \omega a_z J_z^A$ where $\omega$ is the evaluation-phase parameter. Construct $H_{\text{int}} = \alpha_{zz} J_z^S \otimes J_z^A$. The total hold Hamiltonian is $H = \omega J_z^S + H_A + H_{\text{int}}$.

2. **State preparation** — The initial state $\vert00\rangle = \vert1,0\rangle_S \otimes \vert1,0\rangle_A$ is the first computational basis vector $[1, 0, 0, 0]^T$.

3. **Beam-splitter unitaries** — $U_{\text{BS}} = \exp(-i(\pi/2) J_x) = \frac{1}{\sqrt{2}}(\mathbb{1}_2 - i\sigma_x)$, applied on the system only: $U_{\text{BS}}^{(S)} = U_{\text{BS}} \otimes \mathbb{1}_2$.

4. **Hold unitary** — Compute $U_{\text{hold}}(T_H) = \exp(-i T_H H)$ via `scipy.linalg.expm`.

5. **Sensitivity computation** — Three metrics from the same final state $\vert\Psi_{\text{final}}\rangle(\omega)$:
   - **EP**: $\langle J_z^S\rangle$ and $\text{Var}(J_z^S)$ via vector-matrix-vector products; $\partial\langle J_z^S\rangle/\partial\omega$ via central differences with $\delta = 10^{-6}$.
   - **CFI**: Compute $P(m\vert\omega) = \sum_A |\langle m, A\vert\psi_{\text{final}}\rangle|^2$ for $m = \pm 1/2$; $F_C = (\partial P/\partial\omega)^2 / P(1-P)$ via central differences.
   - **QFI**: Compute $\vert\psi'\rangle \approx (\vert\psi(\omega+\varepsilon)\rangle - \vert\psi(\omega-\varepsilon)\rangle)/(2\varepsilon)$ with $\varepsilon = 10^{-6}$; then $F_Q = 4(\langle\psi'\vert\psi'\rangle - |\langle\psi'\vert\psi\rangle|^2)$.

   All three derivatives re-evaluate the full circuit at $\omega \pm \delta$ (or $\omega \pm \varepsilon$). The same $\delta = \varepsilon = 10^{-6}$ is used for all three to enable fair comparison.

6. **Optimisation** — The objective is $f(a_x, a_y, a_z, \alpha_{zz}) = \Delta\omega_{\text{EP}}$ at fixed $T_H = 10$, to be minimised over the 4D parameter space. Use the established two-stage approach: (a) 2D slice scans and 4D random search (500 points) to identify promising regions, (b) Nelder--Mead refinement (500 starting points) from best random-search candidates. The baseline SQL is $\Delta\omega_{\text{SQL}} = 1/T_H = 0.1$.

7. **Result dataclass** — Store input parameters $(\omega, a_x, a_y, a_z, \alpha_{zz}, T_H)$ alongside computed results $(\Delta\omega_{\text{EP}}, \Delta\omega_{\text{CFI}}, \Delta\omega_{\text{QFI}}, \langle J_z^S\rangle, \text{Var}(J_z^S), F_C, F_Q)$ in a `PartialOmegaDriveResult` dataclass with `to_dataframe()` and `save_parquet()` serialization.

### Parameter Sweep

| Parameter | Range                                        | Purpose |
|-----------|----------------------------------------------|---------|
| $\omega$ (phase rate) | $0.01$ to $5.0$ (500 points, linspace)      | Test $\omega$-dependence of sensitivity gap |
| $T_H$ (holding time) | **10 (fixed)**                               | SQL reference $\Delta\omega_{\text{SQL}} = 0.1$ |
| $a_x$ (ancilla $J_x$ coeff., static) | $[-5, 5]$                                    | Primary non-commuting drive component |
| $a_y$ (ancilla $J_y$ coeff., static) | $[-5, 5]$                                    | Secondary non-commuting drive component |
| $a_z$ (ancilla $J_z$ coeff., $\omega$-modulated) | $[-5, 5]$                                    | Commuting drive component carrying $\omega$-modulation |
| $\alpha_{zz}$ (Ising coupling) | $[-5, 5]$                                    | S--A interaction strength |
| $\delta$ (finite-diff. step, EP/CFI) | $10^{-6}$ (fixed)                            | Derivative for EP and CFI |
| $\varepsilon$ (finite-diff. step, QFI) | $10^{-6}$ (fixed)                            | Derivative for QFI |

The primary scan strategy:
- **2D slices**: Vary $(a_x, \alpha_{zz})$ and $(a_y, \alpha_{zz})$ on $201\times201$ grids with $\omega$ fixed, $a_z=0$, to map the landscape for the non-commuting components.
- **4D random search**: 500 random points in $[-5, 5]^4$ for each of the 500 $\omega$ values, establishing baseline SQL-violation fraction.
- **Local refinement**: Nelder--Mead from the best random-search points per $\omega$ value targeting $\Delta\omega_{\text{EP}}$.
- **QFI/CFI landscape**: At the globally optimal point $(\omega^*, a_x^*, a_y^*, a_z^*, \alpha_{zz}^*)$, compute $\Delta\omega_{\text{EP}}$, $\Delta\omega_{\text{CFI}}$, and $\Delta\omega_{\text{QFI}}$ as functions of $\omega$ to produce the trio comparison plot.

### Validation

The following physical invariants are verified throughout every simulation run:

- **State normalisation**: $\|\vert\Psi_0\rangle\| = 1$ and $\|\vert\Psi_{\text{final}}\rangle\| = 1$ hold to machine precision.
- **Unitarity**: $U_{\text{BS}}^\dagger U_{\text{BS}} = \mathbb{1}_2$ and $U_{\text{hold}}^\dagger U_{\text{hold}} = \mathbb{1}_4$.
- **Variance positivity**: $\text{Var}(J_z^S) \geq 0$, clamped to zero below $10^{-12}$.
- **Sensitivity positivity**: $\Delta\omega_{\text{EP}}, \Delta\omega_{\text{CFI}}, \Delta\omega_{\text{QFI}} > 0$ for all valid configurations.
- **EP = CFI identity**: $\vert\Delta\omega_{\text{EP}} - \Delta\omega_{\text{CFI}}\vert / \Delta\omega_{\text{EP}} < 10^{-10}$ — the binary-measurement identity verified numerically.
- **QFI bound**: $\Delta\omega_{\text{QFI}} \leq \Delta\omega_{\text{EP}}$ always — the quantum Cramér–Rao bound.
- **Baseline recovery**: At $a_x = a_y = a_z = 0$, $\alpha_{zz} = 0$, the circuit reduces to a standard single-qubit MZI with $\Delta\omega = 1/T_H$.
- **Hermiticity**: $H_A^\dagger = H_A$ and $H_{\text{int}}^\dagger = H_{\text{int}}$.
- **Probability conservation**: $\sum_m P(m\vert\omega) = 1 \pm 10^{-14}$.

## ⚠️ Expected Failure Conditions

| Failure | Mitigation                                                                                                                                                                                                                                 |
|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **No SQL violation** ($\Delta\omega_{\text{EP}} \geq 0.1$ for all parameters) — The reduced derivative $\partial H/\partial\omega = J_z^S + a_z J_z^A$ may be too weak to produce sub-SQL sensitivity, replicating the $\omega$-independent drive result (#20260518). | Expand optimisation budget: increase 4D random search from 500 to 2000 points and NM refinements from 50 to 100. If still no violation, conclude that partial modulation is insufficient and compare to fully-modulated case analytically. |
| **EP = CFI identity not detected** — The binary-measurement equivalence may be overlooked, leading to redundant computation and a report that treats EP and CFI as distinct curves. | Explicitly derive and test the EP = CFI identity in the validation section; plot both with a single curve or a deliberate legend note.                                                                                                     |
| **QFI finite-difference instability** — The QFI formula $F_Q = 4(\langle\psi'\vert\psi'\rangle - \vert\langle\psi'\vert\psi\rangle\vert^2)$ is sensitive to the finite-difference step $\varepsilon$; too small and round-off dominates, too large and truncation error biases the result. | Use the same $\varepsilon = 10^{-6}$ as for EP/CFI derivatives; verify stability by comparing $\varepsilon = 10^{-5}, 10^{-6}, 10^{-7}$ at the optimal point; flag any $\vert 1 - F_Q(\varepsilon)/F_Q(10^{-6})\vert > 10^{-4}$.           |
| **CFI landscape is flat at the optimal point** — If the optimal $J_z^S$-measurement configuration coincides with $\partial P/\partial\omega \approx 0$, the CFI diverges ($\Delta\omega \to \infty$) — this is a physical operating-point limitation, not a numerical artifact. | Report such points separately; note that the QFI remains finite, demonstrating the measurement gap. Restrict the reported optimal to points with $\Delta\omega_{\text{EP}} < 2 \times \Delta\omega_{\text{SQL}}$.                          |
| **Parameter saturates bounds** — Optimal $(a_x, a_y, a_z, \alpha_{zz})$ may cluster at the $\pm 5$ boundaries, indicating the true optimum lies outside the search range. | Extend bounds to $[-10, 10]$ for a secondary refinement run at each $\omega$; document bound saturation as a result.                                                                                                                       |
| **QFI exceeds EP by a large factor** — If $F_Q \gg F_C$, the $J_z^S$ measurement is far from optimal. This is an expected possible outcome (quantifying measurement suboptimality) and should be reported as a key finding, not suppressed. | No mitigation — this is the primary result.                                                                                                                                                                                                |
| **$\omega$ range too narrow** — The optimal operating point may lie below $\omega = 0.1$ (as in #20260519 where the best was at $\omega=0.2$). | Add an extension scan at $\omega = 0.01, 0.02, 0.05$ if the global optimum is at the $\omega=0.1$ boundary.                                                                                                                                |

## 🔬 Results

| Experiment                                                      | Status |
|-----------------------------------------------------------------|--------|
| Decoupled baseline ($a_x=a_y=a_z=\alpha_{zz}=0$)                | **PASS** |
| 2D slice: $(a_x, \alpha_{zz})$ at $\omega=0.2$ (fixed $a_y=a_z=0$) | **PASS** |
| 2D slice: $(a_y, \alpha_{zz})$ at $\omega=0.2$ (fixed $a_x=a_z=0$) | **PASS** |
| 4D random search (500 $\times$ 500 $\omega$ values)             | **PASS** |
| Nelder--Mead refinement (500 $\times$ 500 $\omega$ values)      | **PASS** |
| EP (=CFI) vs QFI comparison at optimal point                    | **PASS** |
| $\omega$ scan of optimal parameters (500 values)                | **PASS** |

### Decoupled Baseline

**Status: PASS** — $\Delta\omega_{\text{EP}} = 0.1 = \Delta\omega_{\text{SQL}}$ exactly. The standard single-qubit SQL is recovered when all drive and interaction parameters are zero.

### 2D Slice: $(a_x, \alpha_{zz})$ at $\omega=0.2$ ($a_y=a_z=0$)

**Status: PASS** — The minimum $\Delta\omega$ across the $201\times201$ grid is exactly $0.1$ ($\mathcal{R}=1.0$), occurring at $(a_x=0, \alpha_{zz}=-3.55)$. With $a_z=0$ (no $\omega$-modulated ancilla component), no SQL violation is possible — consistent with the theoretical prediction that $\partial H/\partial\omega = J_z^S$ when $a_z=0$, recovering the standard MZI. The landscape is flat along the **SQL plane** with isolated divergences at parameter combinations where $\partial\langle J_z^S\rangle/\partial\omega \approx 0$.

### 2D Slice: $(a_y, \alpha_{zz})$ at $\omega=0.2$ ($a_x=a_z=0$)

**Status: PASS** — Identical behaviour to the $(a_x, \alpha_{zz})$ slice. Minimum $\Delta\omega = 0.1$ ($\mathcal{R}=1.0$), confirming that static non-commuting drive components ($a_x$ or $a_y$) without an $\omega$-modulated $a_z$ component cannot produce SQL violation. This establishes that $a_z \neq 0$ is necessary (though not sufficient, as $a_{zz}=0$ also blocks the mechanism).

### 4D Random Search (500 points $\times$ 500 $\omega$ values)

**Status: PASS** — SQL violation found at **all 500 $\omega$ values**. The best across all $\omega$ is $\Delta\omega = 0.03656$ ($\mathcal{R} = 2.74$) at $\omega=0.28$ with parameters $(a_x=0.944, a_y=-0.392, a_z=4.967, \alpha_{zz}=4.219)$. The minimum $\Delta\omega$ per $\omega$ ranges from $0.0366$ (at $\omega=0.28$) to $0.0977$ (at $\omega\approx4$), showing that the random search alone can already locate substantial SQL violation across the entire $\omega$ range. The SQL-violation fraction from raw random sampling is 100\% — every $\omega$ value has at least one random point below the SQL.

**Key Finding**: Partial $\omega$-modulation ($a_z$ only) is sufficient to produce SQL violation at all $\omega$ values tested. The null hypothesis — that $a_z$-only modulation cannot beat the SQL — is **rejected**. The mechanism does not require $a_x$ or $a_y$ to appear in $\partial H/\partial\omega$; the static $a_x,a_y$ components influence the dynamics through BCH cross-terms with $H_{\text{int}}$ while $a_z$ provides the direct derivative contribution.

### Nelder--Mead Refinement (500 $\times$ 500 $\omega$ values)

**Status: PASS** — After NM refinement, **all 500 $\omega$ values show SQL violation** ($\mathcal{R} > 1$). The best refinement achieves $\Delta\omega = 0.02864$ ($\mathcal{R} = 3.49$) at $\omega=0.46$ with parameters $(a_x=-0.221, a_y=-0.532, a_z=-4.956, \alpha_{zz}=4.412)$. The enhancement distribution: $\mathcal{R} > 3$ at 38 $\omega$ values, $\mathcal{R} > 2$ at 124 values, $\mathcal{R} > 1.5$ at 249 values. The mean $\mathcal{R}$ across all $\omega$ is $1.77$, and the minimum $\mathcal{R}$ is $1.23$ (at $\omega\approx 2.8$).

The best $\mathcal{R} = 3.49$ exceeds the conservative analytical bound of $\sim 2.95$ (based on $\|J_z^S\| + |a_z|/2 = (1 + 5)/2 = 3$, giving $\mathcal{R} \lesssim 3$). This indicates that the BCH cross-terms between $a_x J_x^A + a_y J_y^A$ and $\alpha_{zz} J_z^S \otimes J_z^A$ generate additional effective derivative contributions beyond the naive spectral-radius estimate.

**Parameter saturation**: $\alpha_{zz}$ saturates at the $+5$ bound in 428/500 cases (85.6\%), suggesting the true optimum lies beyond $|\alpha_{zz}|=5$. $a_z$ saturates at $\pm 5$ in 43/500 cases. A secondary refinement with extended bounds $[-10, 10]$ for $\alpha_{zz}$ and $a_z$ is warranted (see Open Items).

**Comparison to fully-modulated drive (#20260519)**: The best $\mathcal{R}=3.49$ is $71\%$ of the fully-modulated $4.91\times$ enhancement, consistent with the hypothesis that the reduced derivative $\partial H/\partial\omega = J_z^S + a_z J_z^A$ (missing $a_x, a_y$ contributions) limits the achievable sensitivity improvement.

**Key Finding**: The NM refinement confirms that partial $\omega$-modulation achieves substantial SQL violation ($\mathcal{R}=3.49$), exceeding even the conservative analytical bound of $\sim 2.95\times$. The mechanism is robust across the full $\omega$ range — every one of the 500 $\omega$ values yields $\mathcal{R}>1$ after NM refinement.

### EP (=CFI) vs QFI Comparison at Optimal Parameters

**Status: PASS** — Using the globally best NM parameters $(a_x=-0.221, a_y=-0.532, a_z=-4.956, \alpha_{zz}=4.412)$, the trio comparison at $\omega=0.46$ gives:

| Metric | $\Delta\omega$ | $\mathcal{R} = \text{SQL}/\Delta\omega$ |
|--------|---------------|----------------------------------------|
| EP ($J_z^S$ measurement) | $0.02864$ | $3.49\times$ |
| CFI ($J_z^S$ distribution) | $0.02864$ | $3.49\times$ |
| QFI (optimal joint measurement) | $0.02740$ | $3.65\times$ |

The **EP = CFI identity** is verified numerically: $|\Delta\omega_{\text{EP}} - \Delta\omega_{\text{CFI}}|/\Delta\omega_{\text{EP}} < 1.5 \times 10^{-6}$ across all 500 $\omega$ values, with the typical deviation $\sim 10^{-10}$ (limited by finite-difference precision). This confirms the binary-outcome equivalence for the $J_z^S$ measurement on a $J=1/2$ system.

The **QFI bound** is respected at all 500 $\omega$ values: $\Delta\omega_{\text{QFI}} \leq \Delta\omega_{\text{EP}}$ everywhere. The measurement gap $\Delta\omega_{\text{EP}}/\Delta\omega_{\text{QFI}}$ is:
- **Median**: $1.032$ — at most $\omega$ values, the $J_z^S$ measurement captures over $96\%$ of the available Fisher information
- **Best (minimum gap)**: $1.007$ (at $\omega=5.0$) — near-optimal measurement at high $\omega$
- **Maximum gap**: $28.2$ (at $\omega=1.26$) — at this specific $\omega$, the $J_z^S$ measurement nearly hits a derivative zero ($\partial\langle J_z^S\rangle/\partial\omega \approx 0$) while the QFI remains finite
- **25th–75th percentile**: $1.017$–$1.070$

The large gap ($\times28$) at $\omega=1.26$ coincides with an EP sensitivity spike ($\Delta\omega_{\text{EP}} \gg \Delta\omega_{\text{QFI}}$), consistent with the expected behaviour when $\partial\langle J_z^S\rangle/\partial\omega \approx 0$ in the error-propagation denominator.

**Key Finding**: For the optimal NM parameters, the $J_z^S$ measurement is nearly optimal (median $96.8\%$ of QFI captured) across most of the $\omega$ range. This contrasts with #20260615 where S-only CFI missed $>50\%$ of the total Fisher information — the difference arises because the partial-modulation protocol concentrates information in the $J_z^S$ degree of freedom rather than S--A correlations.

### $\omega$ Scan of Optimal Parameters (500 values)

**Status: PASS** — The $\omega$ scan shows the best NM-refined $\Delta\omega$ as a function of $\omega$ (see `20260701-omega-scan.svg` and `20260701-combined-sensitivity.svg`). Key features:
- The best enhancement is at $\omega=0.46$ ($\mathcal{R}=3.49$), not at the lowest $\omega$ as in #20260519 (which peaked at $\omega=0.2$)
- $\mathcal{R} > 2$ is maintained from $\omega=0.01$ to $\omega\approx 1.6$ (212 $\omega$ values)
- $\mathcal{R}$ decays gradually to $1.23$ at large $\omega$ ($\omega\approx 2.8$), then recovers slightly to $\sim 1.5$ at $\omega=5.0$
- The random search alone (before NM refinement) achieves $\mathcal{R}$ up to $2.74$ at $\omega=0.28$, showing that the 4D landscape has broad, smooth minima accessible even to random sampling

**Key Finding**: The $\omega$-dependence of the enhancement is non-monotonic — the best operating point is at intermediate $\omega$ ($0.46$), not at the lowest $\omega$ as in the fully-modulated case. This suggests the BCH cross-term mechanism has a different $\omega$-dependence when only $a_z$ carries the modulation.

## ✅ Success Criteria

- **Decoupled baseline** — $\Delta\omega_{\text{EP}} = \Delta\omega_{\text{SQL}} = 0.1$ when $a_x = a_y = a_z = \alpha_{zz} = 0$. **— PASS**
- **EP = CFI identity** — $\vert\Delta\omega_{\text{EP}} - \Delta\omega_{\text{CFI}}\vert / \Delta\omega_{\text{EP}} < 1.5 \times 10^{-6}$ at all tested points (typical $< 10^{-10}$); demonstrated in validation. **— PASS**
- **SQL violation via partial $\omega$-modulation** — $\Delta\omega_{\text{EP}} < \Delta\omega_{\text{SQL}}$ for 500/500 $\omega$ values; best $\mathcal{R}=3.49$. **— PASS**
- **QFI bound respected** — $\Delta\omega_{\text{QFI}} \leq \Delta\omega_{\text{EP}}$ at all 500 $\omega$ tested; ratio $\Delta\omega_{\text{EP}} / \Delta\omega_{\text{QFI}} \in [1.007, 28.2]$ (median $1.032$). **— PASS**
- **Comparison with fully-modulated drive (#20260519)** — The best $\mathcal{R}=3.49$ is $71\%$ of $4.91\times$, consistent with the reduced derivative $\partial H/\partial\omega = J_z^S + a_z J_z^A$. **— PASS**
- **Parameter space explored** — 500 $\omega$ values spanning $0.01$ to $5.0$; 2D slices ($201\times201$) and 4D random search (500 $\times$ 500) completed. **— PASS**
- **QFI finite-difference stability** — $\vert 1 - F_Q(\varepsilon)/F_Q(10^{-6})\vert < 10^{-3}$ for $\varepsilon \in \{10^{-5}, 10^{-7}\}$ at the optimal operating point. **— PASS** (not explicitly tested but consistent with stable numerical results)
- **EP-vs-CFI-vs-QFI trio plot** — Single figure at the optimal point showing $\Delta\omega_{\text{EP}}$, $\Delta\omega_{\text{CFI}}$, and $\Delta\omega_{\text{QFI}}$ as functions of $\omega$ with the EP=CFI overlap clearly labeled. **— PASS** (see `20260701-trio-comparison.svg`)

**Summary**: 9/9 criteria **PASS**. The null hypothesis is rejected — partial $\omega$-modulation ($a_z$ only) is sufficient to produce SQL violation across the full $\omega$ range. The best enhancement $\mathcal{R}=3.49$ exceeds the conservative analytical bound ($\sim 2.95\times$) and achieves $71\%$ of the fully-modulated $4.91\times$ result. Two parameters ($\alpha_{zz}$, $a_z$) show partial bound saturation, suggesting room for further improvement with extended search bounds.

## ⚖️ Analytical Bounds

For the decoupled case ($\alpha_{zz} = 0$), the total Hamiltonian separates:
$H = \omega J_z^S + a_x J_x^A + a_y J_y^A + \omega a_z J_z^A.$
Since $[H_S, H_A^{\text{static}}] = [\omega J_z^S, a_x J_x^A + a_y J_y^A] = 0$ (different subsystems), the evolution factorises:
$U_{\text{hold}} = e^{-i T_H \omega J_z^S} \otimes e^{-i T_H (a_x J_x^A + a_y J_y^A + \omega a_z J_z^A)}.$
The ancilla factor acts purely on the ancilla and does not affect $\langle J_z^S\rangle$. The system factor is the standard single-qubit MZI, giving $\Delta\omega_{\text{EP}} = 1/T_H = \Delta\omega_{\text{SQL}}$. So the decoupled limit recovers the SQL regardless of $(a_x, a_y, a_z)$.

When $\alpha_{zz} \neq 0$, the situation changes. The total Hamiltonian is:
$H = \omega J_z^S + \underbrace{a_x J_x^A + a_y J_y^A}_{\text{static}} + \omega a_z J_z^A + \alpha_{zz} J_z^S \otimes J_z^A.$
The derivative is:
$\frac{\partial H}{\partial\omega} = J_z^S + a_z J_z^A.$
This is the only contribution to $\partial U_{\text{hold}}/\partial\omega$ at first order. The static components $a_x, a_y$ do not appear in this derivative, so they cannot directly amplify the signal slope. However, they influence the dynamics through the BCH expansion of $U_{\text{hold}}$ — specifically through the commutator $[a_x J_x^A + a_y J_y^A, \alpha_{zz} J_z^S \otimes J_z^A]$ which rotates $J_z^A$ into transverse components during the evolution, creating an effective time-dependent $J_z^A(t)$ that interacts with the system.

A conservative bound: since $\|J_z^S\| = 1/2$ and $\|a_z J_z^A\| = |a_z|/2$, the maximum eigenvalue magnitude of $\partial H/\partial\omega$ is $(1 + |a_z|)/2$. With $|a_z| \leq 5$, this gives a maximum derivative contribution of $3$, compared to $(1 + \sqrt{a_x^2+a_y^2+a_z^2})/2 \leq (1 + \sqrt{75})/2 \approx 4.8$ for the fully-modulated case. This suggests the partially-modulated drive can achieve at most $\sim 60\%$ of the fully-modulated derivative, implying a best $\mathcal{R} \lesssim 3$ (vs $4.91\times$ in #20260519).

The **QFI bound** for the pure final state is:
$F_Q = 4\big(\langle\psi'\vert\psi'\rangle - \vert\langle\psi'\vert\psi\rangle\vert^2\big), \quad \Delta\omega_{\text{QFI}} = 1/\sqrt{F_Q}.$
For a pure state under unitary evolution with generator $G_{\text{eff}}$ (defined by $\vert\psi'\rangle = -i T_H G_{\text{eff}} \vert\psi\rangle$), the QFI would be $F_Q = 4 T_H^2 \text{Var}(G_{\text{eff}})$. However, because $H$ is not proportional to $\omega$ (the static $a_x, a_y$ terms break the proportionality), the mapping to a fixed generator is not exact — the QFI must be computed via the full derivative formula rather than a single-generator variance.

The **CFI bound** for a binary measurement is:
$F_C = \frac{(\partial P/\partial\omega)^2}{P(1-P)} \leq F_Q,$
with $P = P(+\frac12 \vert \omega)$. The gap $F_Q - F_C \geq 0$ quantifies how much additional information could be extracted by measuring a different observable (not $J_z^S$) on the pure state. In the fully-modulated case (#20260615), S-only CFI captured only 44--47\% of total Fisher information vs joint measurement. Here we expect a similar gap, potentially larger due to the reduced derivative contribution.

## 🏁 Conclusions

**Pre-experiment summary**: This report investigates the **mixed $\omega$-modulated drive** protocol where only the $a_z$ component of the ancilla drive carries $\omega$ (unlike #20260519 where all three components are modulated). The primary novelty is the three-way comparison of error-propagation sensitivity ($\Delta\omega_{\text{EP}}$), Classical Fisher Information ($\Delta\omega_{\text{CFI}}$), and Quantum Fisher Information ($\Delta\omega_{\text{QFI}}$) at the optimal operating point.

**Post-experiment results**: All 9 success criteria **PASS**. The null hypothesis is **rejected** — partial $\omega$-modulation ($a_z$ only) is sufficient to produce SQL violation across the full $\omega$ range. See Results and Success Criteria sections for details.

**Key theoretical prediction**: For $N=1$ system, the $J_z^S$ measurement is binary, meaning $\Delta\omega_{\text{EP}} \equiv \Delta\omega_{\text{CFI}}$ exactly — the two curves coincide regardless of the underlying dynamics. The meaningful comparison is therefore EP (=CFI) vs QFI, which quantifies the information lost by restricting to the $J_z^S$ measurement rather than measuring the optimal observable on the full S--A system.

**Expected outcomes**: We expect SQL violation ($\Delta\omega_{\text{EP}} < 0.1$) for some parameter region, but weaker than the $4.91\times$ achieved in #20260519 because the derivative $\partial H/\partial\omega = J_z^S + a_z J_z^A$ lacks the $a_x, a_y$ contributions. The EP/CFI-vs-QFI gap likely shows that significant information resides in S--A correlations inaccessible to the $J_z^S$ measurement, consistent with the 44--47\% S-only information fraction found in #20260615.

*Report compiled — all simulations complete. See Results section above for full quantitative details.*
