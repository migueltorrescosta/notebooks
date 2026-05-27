# Multi-Particle XX-Coupling Dual-MZI with Optimized System–Ancilla Joint Measurement

## 🧪 Hypothesis

For a system--ancilla pair of $N$-particle two-mode bosonic systems where both the system S and the ancilla A couple to the unknown phase rate $\theta$ via $H_S = \theta J_z^S$ and $H_A = \theta J_z^A$, the system--ancilla interaction is the transverse (XX) type $H_{\text{int}} = \alpha_{xx} \, J_x^S \otimes J_x^A$, and **both** subsystems undergo a full Mach--Zehnder sequence (50/50 beam splitter before and after the hold), the optimal **joint measurement** on the final system+ancilla state
$M = m_s J_z^S + m_a J_z^A$ with $m_s^2 + m_a^2 = 1$
can yield a sensitivity $\Delta\theta$ (error-propagation uncertainty in estimating $\theta$ via $M$) that **beats** the $2N$-particle standard quantum limit $\Delta\theta_{\text{SQL}} = 1/(\sqrt{2N}\, T_H)$ for some $N \in [1, 20]$, $\theta \in [0.1, 5.0]$, and $\alpha_{xx} > 0$. The holding time is fixed at $T_H = 10$ for all experiments, giving an SQL reference of $\Delta\theta_{\text{SQL}} = 1/(\sqrt{2N} \cdot 10)$.

**Key differences from the 2026-05-22 report (which traced out the ancilla and measured only $J_z^S$):**

- **2026-05-22**: Traced out the ancilla, measured only $J_z^S$ on the reduced system. Compared against the $N$-particle SQL $1/(\sqrt{N}\,T_H)$. Found $\alpha_{xx}^* = 0$ for all $(\theta, N)$ — no SQL violation.
- **This report**: Keeps the ancilla and measures the **optimized** linear combination $M = m_s J_z^S + m_a J_z^A$. Both the coupling $\alpha_{xx}$ and the measurement coefficients $(m_s, m_a)$ are jointly optimized per $(\theta, N)$ pair. The fair sensitivity benchmark is the $2N$-particle SQL $\Delta\theta_{\text{SQL}} = 1/(\sqrt{2N}\,T_H)$, since measurements access both subsystems.

**Physical rationale**: At $\alpha_{xx}=0$, the optimal separable measurement is $\phi = \pi/4$ ($m_s = m_a = 1/\sqrt{2}$), which gives $\Delta\theta = 1/(\sqrt{2N}\,T_H)$ — exactly the $2N$-SQL. This is the **separable baseline**: two independent MZIs saturate the $2N$-SQL. The key question is whether the XX coupling can generate entanglement that pushes the sensitivity **below** this $2N$-SQL baseline.

The central hypothesis decomposes into two specific, testable claims:

1. **XX coupling beats the $2N$-SQL**: There exists $(\theta, N)$ such that with optimal $(\alpha_{xx}, \phi)$ we have $\Delta\theta < 1/(\sqrt{2N}\, T_H)$. This is the non-trivial claim — the XX coupling must generate genuinely useful entanglement beyond what two independent MZIs can achieve.

2. **XX coupling is genuinely beneficial**: There exists $(\theta, N)$ for which the optimal sensitivity at $\alpha_{xx}^* > 0$ is strictly better than the optimal sensitivity at $\alpha_{xx}=0$ (with $\phi$ optimized in both cases). Equivalently, the XX interaction yields a metrological advantage measured by $r_{\text{XX}} = \Delta\theta_{\text{opt}}(\alpha_{xx}^*, \phi^*) \,/\, \Delta\theta(\alpha_{xx}=0, \phi=\pi/4) < 1$.

**Null hypothesis**: For all $(\theta, N, \alpha_{xx}, \phi)$, the optimal sensitivity $\Delta\theta_{\text{opt}}$ is achieved at $\alpha_{xx}^* = 0$ with $\phi^* = \pi/4$, giving $\Delta\theta_{\text{opt}} = \Delta\theta_{\text{SQL}} = 1/(\sqrt{2N} T_H)$. The XX coupling never improves sensitivity beyond the separable $2N$-SQL baseline.

## ⚛️ Theoretical Model

The total Hilbert space is $\mathcal{H}_{\text{tot}} = \mathcal{H}_S \otimes \mathcal{H}_A$, where each subsystem is a **two-mode bosonic Fock space** of $N$ particles symmetrically distributed across two modes. The symmetric subspace is the Dicke basis $|J, m\rangle$ with total spin $J = N/2$ and magnetic quantum number $m \in \{-J, -J+1, \dots, J\}$, giving dimension $d = N+1$ per subsystem. The full space $\mathcal{H}_{\text{tot}}$ therefore has dimension $(N+1)^2$. The ordered basis is $\{|m_S, m_A\rangle = |J, m_S\rangle_S \otimes |J, m_A\rangle_A\}$ with both $m_S$ and $m_A$ descending from $+J$ to $-J$.

The **collective angular momentum operators** for each subsystem satisfy the SU(2) algebra $[J_i, J_j] = i \epsilon_{ijk} J_k$. In the Dicke basis:
- $J_z$ is diagonal: $J_z |J, m\rangle = m |J, m\rangle$,
- $J_x$ has matrix elements $\langle J, m' | J_x | J, m \rangle = \frac12 \sqrt{J(J+1) - m(m\pm 1)}\, \delta_{m', m\pm 1}$,
- $J_y$ is related by $[J_z, J_x] = i J_y$.

The operators are embedded into the full space via Kronecker products: $J_k^S = J_k \otimes \mathbb{1}_{N+1}$ and $J_k^A = \mathbb{1}_{N+1} \otimes J_k$, where $J_k$ is the $(N+1) \times (N+1)$ Dicke-basis representation.

The **initial state** is a pure product state $|\Psi_0\rangle = |N,0\rangle_S \otimes |N,0\rangle_A$, which in the Dicke basis is $|J, J\rangle_S \otimes |J, J\rangle_A$ — the column vector $[1, 0, \dots, 0]^T$ of length $(N+1)^2$.

The **circuit protocol** proceeds in six steps:

1. **Prepare initial state**: $|\Psi_0\rangle = |J, J\rangle_S \otimes |J, J\rangle_A$.

2. **Beam splitter on both subsystems**: A 50/50 symmetric beam splitter acts independently on each subsystem, generated by $J_x$ with angle $\pi/2$:
   $U_{\text{BS}} = \exp(-i (\pi/2) J_x^S) \otimes \exp(-i (\pi/2) J_x^A).$
   Both single-subsystem BS unitaries are $(N+1) \times (N+1)$ matrix exponentials computed via `scipy.linalg.expm`, and the combined unitary is their Kronecker product.

3. **Holding period with simultaneous phase encoding and XX interaction**: The full state evolves under the total Hamiltonian $H = H_S + H_A + H_{\text{int}}$ for duration $T_H = 10$. The three terms are:
   - $H_S = \theta J_z^S = \theta \, J_z \otimes \mathbb{1}_{N+1}$,
   - $H_A = \theta J_z^A = \theta \, \mathbb{1}_{N+1} \otimes J_z$,
   - $H_{\text{int}} = \alpha_{xx} \, J_x^S \otimes J_x^A$.

   The total Hamiltonian is:
   $H = \theta (J_z^S + J_z^A) + \alpha_{xx} J_x^S J_x^A.$

   The hold unitary is $U_{\text{hold}}(T_H) = \exp(-i T_H H)$, computed via `scipy.linalg.expm`. The matrix dimension is $(N+1)^2 \times (N+1)^2$, ranging from $4\times4$ ($N=1$) to $441\times441$ ($N=20$).

4. **Second beam splitter on both subsystems**: An identical 50/50 BS: $U_{\text{BS}}$ (same as step 2).

5. **Construct the measurement operator**: The joint observable is
   $M(\phi) = \cos\phi \, J_z^S + \sin\phi \, J_z^A,$
   parameterized by $\phi \in [-\pi, \pi]$. The coefficients automatically satisfy $m_s^2 + m_a^2 = 1$ with $m_s = \cos\phi$, $m_a = \sin\phi$. The measurement operator $M$ is an $(N+1)^2 \times (N+1)^2$ Hermitian matrix in the full space.

6. **Measure $M$ on the full final state**: The expectation value and variance are computed directly from the pure final state vector $|\Psi_{\text{final}}\rangle$:
   $\langle M \rangle = \langle\Psi_{\text{final}}| M |\Psi_{\text{final}}\rangle,$
   $\text{Var}(M) = \langle M^2 \rangle - \langle M \rangle^2.$
   No partial trace is performed — all information from both S and A is retained.

The **complete evolution** is:
$|\Psi_{\text{final}}\rangle = U_{\text{BS}} \, U_{\text{hold}}(T_H) \, U_{\text{BS}} \, |\Psi_0\rangle.$

The **sensitivity** via **error propagation** is:
$\Delta\theta(\alpha_{xx}, \phi) = \frac{\sqrt{\text{Var}(M)}}{|\partial\langle M\rangle / \partial\theta|},$
where the derivative is computed via central finite differences with step $\delta = 10^{-6}$:
$\frac{\partial\langle M\rangle}{\partial\theta} \approx \frac{\langle M\rangle(\theta+\delta) - \langle M\rangle(\theta-\delta)}{2\delta}.$

The **$2N$-particle standard quantum limit** for all $2N$ particles with holding time $T_H$ is:
$\Delta\theta_{\text{SQL}} = \frac{1}{\sqrt{2N} \, T_H}.$
This is the relevant baseline when accessing both subsystems. The $N$-particle SQL ($1/(\sqrt{N} T_H)$) from 2026-05-22 was the correct benchmark when only the system was measured after tracing out the ancilla; here, with access to all $2N$ particles, the $2N$-SQL is the fair comparison.

**Decoupled limit ($\alpha_{xx} = 0$)**: When the XX coupling vanishes, the evolution factorises into independent MZIs on S and A. The state is a product: $|\Psi\rangle = |\psi_S\rangle \otimes |\psi_A\rangle$, where each subsystem undergoes an identical MZI. For the joint measurement $M = \cos\phi \, J_z^S + \sin\phi \, J_z^A$, the expectation and variance evaluate to:
$\langle M \rangle = (\cos\phi + \sin\phi) \, \langle J_z \rangle_{\text{single}},$
$\text{Var}(M) = \text{Var}(J_z)_{\text{single}}.$

Since $m_s^2 + m_a^2 = 1$, the variance is independent of $\phi$ at $\alpha_{xx}=0$: it equals the single-MZI variance $N/4$ (for a CSS). The derivative is:
$\frac{\partial \langle M \rangle}{\partial \theta} = (\cos\phi + \sin\phi) \, \frac{\partial \langle J_z \rangle_{\text{single}}}{\partial \theta}.$

The sensitivity at $\alpha_{xx}=0$ is therefore:
$\Delta\theta(\alpha_{xx}=0, \phi) = \frac{\sqrt{\text{Var}(J_z)_{\text{single}}}}{|\cos\phi + \sin\phi| \, |\partial \langle J_z \rangle_{\text{single}}/\partial\theta|}.$

At $\phi = 0$ ($M = J_z^S$), we recover the $N$-particle SQL $\Delta\theta = 1/(\sqrt{N} T_H)$, which is **worse** than the $2N$-SQL by a factor $\sqrt{2}$. At $\phi = \pi/4$ ($m_s = m_a = 1/\sqrt{2}$), we have $|\cos\phi + \sin\phi| = \sqrt{2}$, giving:
$\Delta\theta(\alpha_{xx}=0, \phi=\pi/4) = \frac{1}{\sqrt{2N}\, T_H} = \Delta\theta_{\text{SQL}}.$
The optimal $\phi$ at $\alpha_{xx}=0$ is $\pi/4$, and the separable measurement **exactly saturates** the $2N$-SQL.

**Decoupled limit summary**: At $\alpha_{xx}=0$, the optimal joint measurement ($\phi=\pi/4$) achieves $\Delta\theta = \Delta\theta_{\text{SQL}}$ — it does not beat the $2N$-SQL, it matches it exactly. The $2N$-SQL is the **separable baseline**: two independent MZIs cannot do better. The interesting question is whether $\alpha_{xx} > 0$ can improve sensitivity **below** this baseline, indicating genuine XX-generated entanglement.

**XX coupling activation**: The commutator of $M$ with $H_{\text{int}}$ determines whether the interaction affects the measurement:
$[M, H_{\text{int}}] = \alpha_{xx} \left[\cos\phi \, J_z^S + \sin\phi \, J_z^A,\; J_x^S J_x^A\right]$
$= \alpha_{xx} \left( \cos\phi \, [J_z^S, J_x^S] J_x^A + \sin\phi \, J_x^S [J_z^A, J_x^A] \right)$
$= i \alpha_{xx} \left( \cos\phi \, J_y^S J_x^A + \sin\phi \, J_x^S J_y^A \right) \neq 0.$

The commutator is non-zero for $\alpha_{xx} \neq 0$ and generic $\phi$, so the XX coupling actively modifies the measurement dynamics. The strength of this modification depends on $\phi$: at $\phi = 0$ ($M=J_z^S$), it couples through $J_y^S J_x^A$, while at $\phi = \pi/2$ ($M=J_z^A$), it couples through $J_x^S J_y^A$. The optimal $\phi$ may shift away from $\pi/4$ in the presence of non-zero $\alpha_{xx}$ to exploit this structure.

**Mechanism for possible XX advantage**: The XX interaction can generate entanglement between S and A during the hold. When $M$ is measured on the full entangled state, the entanglement can increase the signal derivative $\partial \langle M \rangle / \partial \theta$ beyond the separable case, or reduce the variance $\text{Var}(M)$, or both. The competition between these effects determines whether $\alpha_{xx} > 0$ yields a net improvement over the optimal separable measurement.

## 💻 Numerical Simulation

### Implementation Strategy

1. **Operator construction** — Build $J_z$, $J_x$, $J_y$ as $(N+1) \times (N+1)$ Dicke-basis matrices using existing `dicke_basis.jz_operator(N)`, etc. Embed into the combined space via Kronecker products, as in 2026-05-22. Additionally, build the identity $\mathbb{1}_{(N+1)^2}$ for full-space operations.

2. **State preparation** — The initial state $|J, J\rangle_S \otimes |J, J\rangle_A$ is the first computational basis vector $[1, 0, \dots, 0]^T$ of length $(N+1)^2$.

3. **Beam-splitter unitaries** — Identical to the 2026-05-22 implementation: $U_{\text{BS}} = \exp(-i\pi/2 J_x) \otimes \exp(-i\pi/2 J_x)$.

4. **Hold unitary** — $U_{\text{hold}}(T_H) = \exp(-i T_H H)$ on the $(N+1)^2 \times (N+1)^2$ space. $H$ is Hermitian-symmetrised after construction.

5. **Full-state measurement** — Construct $M(\phi) = \cos\phi \, J_z^S + \sin\phi \, J_z^A$ as a full-space operator. Compute $\langle M \rangle = \langle\Psi| M |\Psi\rangle$ and $\langle M^2 \rangle = \langle\Psi| M^2 |\Psi\rangle$ directly from the final state vector $|\Psi\rangle$, without any partial trace. The variance is $\text{Var}(M) = \langle M^2 \rangle - \langle M \rangle^2$, clamped to zero when below $10^{-12}$ due to numerical round-off.

6. **Sensitivity computation** — Compute $\Delta\theta(\alpha_{xx}, \phi) = \sqrt{\text{Var}(M)} / |\partial\langle M\rangle/\partial\theta|$ via central finite differences with $\delta = 10^{-6}$. The full circuit (including all operators) is re-evaluated at $\theta \pm \delta$.

7. **2D optimization over $(\alpha_{xx}, \phi)$** — For each $(\theta, N)$ pair, minimise $\Delta\theta(\alpha_{xx}, \phi)$ using L-BFGS-B (bounded) with 20 random starting points. The optimization variables are:
   - $\alpha_{xx} \in [0, 20]$ (bounded),
   - $\phi \in [-\pi, \pi]$ (bounded, periodic).
   
   **Procedure per $(\theta, N)$**:
   - Generate 20 random seeds: $\alpha_{xx}^{(0)} \sim U[0, 20]$, $\phi^{(0)} \sim U[-\pi, \pi]$.
   - For each seed, run L-BFGS-B (via `scipy.optimize.minimize`) with numerical gradient (finite differences, step $10^{-6}$ in each parameter).
   - Record the best $(\alpha_{xx}^*, \phi^*)$ across all 20 starts, along with the achieved $\Delta\theta_{\text{opt}}$.
   - **Convergence diagnostic**: Flag any $(\theta, N)$ where the best result is found by only 1 start (possible local minimum issue).

   A supporting **coarse 2D grid** ($\alpha_{xx} \times \phi$) is also evaluated for representative $(\theta, N)$ points to validate that the BFGS landscape search is reliable.

8. **Data serialisation** — For each $(\theta, N)$ pair, store a row containing $\theta$, $N$, $T_H$, $\alpha_{xx}^*$, $\phi^*$, $m_s^*$, $m_a^*$, $\Delta\theta_{\text{opt}}$, $\Delta\theta_{\text{SQL}}$, the ratio $\Delta\theta_{\text{opt}} / \Delta\theta_{\text{SQL}}$, $\langle M \rangle$, $\text{Var}(M)$, and $\partial\langle M\rangle/\partial\theta$. The full dataset is stored as a single Parquet file with all metadata fields required on deserialisation.

**Computational cost estimate**: $50 \, \theta \times 20 \, N \times 20$ random starts $\times \sim 80$ BFGS iterations $\times 3$ circuit evals (central diff) $= 4.8$M circuit evaluations. For $N=20$ ($441\times441$ matrix exponentials at $\sim 10$ ms), this is $\sim 13$ hours for the largest $N$. Smaller $N$ values are much faster ($N=1$: $4\times4$, $\ll 1$ ms), giving a weighted total of roughly $3$--$5$ hours. The sweep is parallelisable over $(\theta, N)$ pairs.

### Parameter Sweep

| Parameter | Range | Purpose |
|-----------|-------|---------|
| $\theta$ (phase rate) | $0.1$ to $5.0$ in steps of $0.1$ (50 points) | Test $\theta$-dependence of $2N$-SQL violation |
| $N$ (particle number per subsystem) | $1$ to $20$ in integer steps (20 points) | Extract scaling exponent $\alpha$ from $\Delta\theta \propto N^\alpha$ |
| $T_H$ (holding time) | **10 (fixed)** | SQL reference $\Delta\theta_{\text{SQL}} = 1/(\sqrt{2N} \cdot 10)$ |
| $\alpha_{xx}$ (XX coupling) | $[0, 20]$ (optimised) | Coupling strength |
| $\phi$ (measurement angle) | $[-\pi, \pi]$ (optimised) | Measurement direction: $m_s = \cos\phi$, $m_a = \sin\phi$ |
| Random starts | 20 per $(\theta, N)$ pair | Avoid local minima in 2D landscape |
| $\delta$ (finite-diff. step) | $10^{-6}$ (fixed) | Derivative computation |

Each $(\theta, N)$ pair receives $20$ L-BFGS-B optimisations, giving $50 \times 20 \times 20 = 20{,}000$ optimisation runs. A supporting coarse grid ($21 \times 21 = 441$ points) is evaluated for a handful of representative $(\theta, N)$ to validate the landscape shape.

A **decoupled baseline** run with $\alpha_{xx} = 0$ and $\phi$ optimised at each $(\theta, N)$ pair verifies that the $\phi$-dependent sensitivity matches the analytical prediction: $\Delta\theta(\phi=\pi/4) = 1/(\sqrt{2N} T_H) = \Delta\theta_{\text{SQL}}$ (the separable baseline is exactly the $2N$-SQL), and $\Delta\theta(\phi=0) = 1/(\sqrt{N} T_H)$ (the $N$-SQL, worse by $\sqrt{2}$).

### Validation

The following physical invariants are verified throughout every simulation run:

- **State normalisation**: $\||\Psi_0\rangle\| = 1$ and $\||\Psi_{\text{final}}\rangle\| = 1$ hold to machine precision.
- **Unitarity**: $U_{\text{BS}}^\dagger U_{\text{BS}} = \mathbb{1}_{N+1}$ (single-subsystem BS) and $U_{\text{hold}}^\dagger U_{\text{hold}} = \mathbb{1}_{(N+1)^2}$ (hold unitary).
- **Variance positivity**: $\text{Var}(M) \geq 0$, with numerical round-off clamped to zero when below $10^{-12}$.
- **Sensitivity positivity**: $\Delta\theta > 0$ for all valid configurations.
- **Decoupled baseline (no interaction)**: At $\alpha_{xx} = 0$, the sensitivity must satisfy $\Delta\theta(\phi=0) = 1/(\sqrt{N} T_H)$ ($N$-SQL, worse than $2N$-SQL) and $\Delta\theta(\phi=\pi/4) = 1/(\sqrt{2N} T_H) = \Delta\theta_{\text{SQL}}$ ($2N$-SQL exactly). Verified analytically and numerically for all $(\theta, N)$.
- **Hermiticity**: $H_{\text{int}}$, $H$, and $M(\phi)$ satisfy $A^\dagger = A$ to machine precision.
- **Commutation relations**: $[J_z^S, J_x^S] = i J_y^S$ and the equivalent for A are verified.
- **Derivative stability**: The central-difference derivative must produce $\Delta\theta$ values stable under changes to $\delta$ (e.g., $\delta \in [10^{-7}, 10^{-5}]$ produces the same $\Delta\theta$ to within $10^{-6}$ relative tolerance).
- **Traced-out equivalence ($\phi=0$)**: Measuring $M = J_z^S$ ($\phi=0$) on the full state gives $\Delta\theta = 1/(\sqrt{N} T_H)$, identical to the 2026-05-22 traced-out protocol. This is **above** the $2N$-SQL, confirming that discarding ancilla information degrades sensitivity.
- **Separable baseline saturation ($\phi=\pi/4$, $\alpha_{xx}=0$)**: The optimal separable measurement achieves $\Delta\theta = 1/(\sqrt{2N} T_H)$, exactly saturating the $2N$-SQL.

#### 🔧 Implementation Status (Planned)

- **Operator construction** — $J_z$, $J_x$, $J_y$ as $(N+1)\times(N+1)$ Dicke-basis matrices — reuse existing `dicke_basis.py`.
- **XX Interaction Hamiltonian** — $H_{\text{int}} = \alpha_{xx} J_x^S \otimes J_x^A$ in the $(N+1)^2$ space — reuse existing code.
- **State preparation** — Fixed $|N,0\rangle_S \otimes |N,0\rangle_A$ initial state — reuse existing code.
- **Beam-splitter unitaries** — $U_{\text{BS}} = \exp(-i\pi/2 J_x) \otimes \exp(-i\pi/2 J_x)$ — reuse existing code.
- **Holding unitary** — $\exp(-i T_H [\theta(J_z^S + J_z^A) + \alpha_{xx} J_x^S \otimes J_x^A])$ — reuse existing code.
- **Full-state measurement** — $M(\phi) = \cos\phi \, J_z^S + \sin\phi \, J_z^A$ as a full-space operator, compute $\langle M\rangle$ and $\text{Var}(M)$ without partial trace — **new**.
- **Sensitivity** — $\Delta\theta = \sqrt{\text{Var}(M)} / |\partial\langle M\rangle/\partial\theta|$ via central finite differences ($\delta = 10^{-6}$) — adapted from existing code.
- **$(\alpha_{xx}, \phi)$ optimisation** — L-BFGS-B with 20 random starts per $(\theta, N)$ pair — **new**.
- **Decoupled baseline** — $\alpha_{xx}=0$, $\phi$-sweep verification — adapted from existing code.
- **Scaling analysis** — Log-log fit $\log(\Delta\theta) = \alpha \log(N) + \log(C)$ for each $\theta$ — reuse.
- **Validation helpers** — Hermiticity, unitarity, variance positivity, SQL baseline recovery, derivative stability — adapted.

**Tests**: The companion `test_local.py` module will provide test cases covering operator construction (dimension, Hermiticity, commutation relations), full-state measurement (variance positivity, trace equivalence at $\phi=0$), sensitivity ($N$-SQL recovery at $\phi=0$, $2N$-SQL saturation at $\phi=\pi/4$, $\alpha_{xx}=0$), 2D optimisation (finite return, bounds, convergence), full sweep (small run, metadata), decoupled baseline (all $\phi$-dependent predictions), scaling analysis ($2N$-SQL exponent), Parquet roundtrip (metadata + fail-fast), and physical invariants.

## ⚠️ Expected Failure Conditions

| Failure | Mitigation |
|---------|------------|
| **XX coupling never beats the $2N$-SQL** — the optimal sensitivity achieves $\Delta\theta_{\text{opt}} = \Delta\theta_{\text{SQL}}$ at $\alpha_{xx}^* = 0$, $\phi^* = \pi/4$ for all $(\theta, N)$ | Accept the null hypothesis. Document that the XX coupling generates no metrologically useful entanglement beyond the separable $2N$-SQL, even with optimised joint measurements. |
| **Optimal $\phi$ is always $\pi/4$** — even at $\alpha_{xx} > 0$, the measurement never benefits from an imbalanced weighting of S and A | At $\alpha_{xx} > 0$, a shifted $\phi^*$ would indicate that the XX coupling creates an asymmetry between S and A that can be exploited by an unbalanced measurement. If $\phi^*$ remains $\pi/4$ throughout, the XX coupling induces a symmetric modification of both subsystems. |
| **Derivative instability at large $N$** — the central-difference derivative $\delta = 10^{-6}$ becomes unstable when the $(N+1)^2$-dimensional Hamiltonian has large eigenvalue spread | Test derivative stability with $\delta \in [10^{-8}, 10^{-4}]$ for $N=20$ at representative $(\theta, \alpha_{xx}, \phi)$ points. Adjust $\delta$ adaptively if needed. |
| **Optimiser failure — many local minima** — the 2D $(\alpha_{xx}, \phi)$ landscape has multiple local minima, and 20 random starts are insufficient to find the global optimum | Increase to 40 random starts for a subset of $(\theta, N)$ and compare results. If the best result is consistently found by only 1--2 starts, flag these points and increase the start count. Use the coarse 2D grid to map the landscape shape for representative points. |
| **Fringe extremum (vanishing derivative)** — some $(\alpha_{xx}, \phi, \theta)$ combinations yield $\partial\langle M\rangle/\partial\theta \approx 0$, giving $\Delta\theta \to \infty$ | L-BFGS-B naturally avoids these regions as the objective diverges. Flag configurations with $\Delta\theta = \infty$ and exclude from analysis. |
| **Computational time for N=20** — the $441\times441$ matrix exponential at each call may be prohibitive for 20 random starts $\times \sim 80$ iterations | Pre-compute and cache the BS unitary for each $N$. Use a two-stage approach: evaluate a coarse 2D grid ($21\times21$ points) for each $(\theta, N)$ pair, then use L-BFGS-B refinement from the best grid point plus 5 random starts. This reduces the per-pair cost drastically. |
| **Decoupled baseline violated** — $\Delta\theta(\phi=\pi/4, \alpha_{xx}=0) \neq 1/(\sqrt{2N} T_H)$ | This indicates a bug in the full-state measurement implementation. Debug the operator construction, measurement expectation, and derivative computation before proceeding. Start with $N=1$ where the analytical result is simplest. |

## 🔬 Results

### Pre-Experiment Status

All experiments are **PENDING** — the report has not yet been run.

| Experiment | Status | Description |
|------------|--------|-------------|
| Decoupled baseline verification | PENDING | $\alpha_{xx}=0$: verify $\Delta\theta(\phi=0) = 1/(\sqrt{N} T_H)$ ($N$-SQL) and $\Delta\theta(\phi=\pi/4) = 1/(\sqrt{2N} T_H)$ ($2N$-SQL) across all $(\theta, N)$ |
| $(\alpha_{xx}, \phi)$ optimisation sweep | PENDING | 50 $\theta \times 20 N \times 20$ random starts; record $(\alpha_{xx}^*, \phi^*, \Delta\theta_{\text{opt}})$ |
| XX advantage analysis | PENDING | Test $r_{\text{XX}} < 1$: compare $\Delta\theta_{\text{opt}}(\alpha_{xx}^*, \phi^*)$ vs $2N$-SQL baseline $\Delta\theta_{\text{SQL}}$ |
| $N$-scaling analysis | PENDING | Log-log fit $\Delta\theta_{\text{opt}} \propto N^\alpha$ for each $\theta$; compare $2N$-SQL ($-0.5$), $N$-SQL ($-0.5$ but offset), HL ($-1.0$) |
| $\theta$-dependence analysis | PENDING | $\Delta\theta_{\text{opt}}(\theta)$ at fixed $N$; identify oscillatory structure from XX coupling |
| Landscape visualisation | PENDING | 2D contour $\Delta\theta(\alpha_{xx}, \phi)$ for 3 representative $(\theta, N)$ |
| Traced-out comparison | PENDING | Compare $\Delta\theta_{\text{opt}}$ (this report, measured against $2N$-SQL) vs 2026-05-22 traced-out $\Delta\theta$ (measured against $N$-SQL) |

### Data and Figure Files

| File | Description |
|------|-------------|
| `2026-05-25-optimised-measurement-sweep.parquet` | Full $(\theta, N, \alpha_{xx}^*, \phi^*, \Delta\theta_{\text{opt}}, \text{SQL}_{2N})$ data |
| `2026-05-25-optimised-measurement-decoupled-baseline.parquet` | $\alpha_{xx}=0$, $\phi$-optimised baseline |
| `2026-05-25-optimised-measurement-scaling.parquet` | Scaling exponents $\alpha(\theta)$, $R^2$ |
| `2026-05-25-ratio-heatmap.svg` | $\Delta\theta_{\text{opt}} / \Delta\theta_{\text{SQL}}$ heatmap |
| `2026-05-25-alpha-opt-heatmap.svg` | $\alpha_{xx}^*$ heatmap |
| `2026-05-25-phi-opt-heatmap.svg` | $\phi^*$ heatmap |
| `2026-05-25-n-scaling-theta{0.3,1.0,3.0}.svg` | N-scaling at 3 $\theta$ values |
| `2026-05-25-scaling-exponents.svg` | Exponent $\alpha$ vs $\theta$ panel |
| `2026-05-25-comparison-traced-out.svg` | $\Delta\theta_{\text{opt}}$ (this report, $2N$-SQL) vs 2026-05-22 traced-out ($N$-SQL) comparison |
| `2026-05-25-landscape-N{1,5,20}-theta{0.5,2.0,4.0}.svg` | 2D contour slices |

### Figures

1. **Sensitivity ratio heatmap** — $r = \Delta\theta_{\text{opt}} / \Delta\theta_{\text{SQL}}$ vs $\theta \times N$ 2D heatmap with a contour at $r = 1.0$ (the $2N$-SQL baseline). Regions where $r < 1$ indicate genuine metrological gain from the XX coupling. The null hypothesis predicts $r = 1$ uniformly.

2. **Optimal $\alpha_{xx}^*$ heatmap** — $\alpha_{xx}^*$ vs $\theta \times N$ 2D heatmap. Regions where $\alpha_{xx}^* > 0$ indicate that the XX coupling is genuinely beneficial. Uniform $\alpha_{xx}^* = 0$ indicates the null hypothesis.

3. **Optimal $\phi^*$ heatmap** — $\phi^*$ vs $\theta \times N$ 2D heatmap. At $\alpha_{xx}=0$, the optimal $\phi$ is $\pi/4$. Shifts away from $\pi/4$ at $\alpha_{xx}^* > 0$ reveal how the XX interaction tilts the optimal measurement direction.

4. **N-scaling at selected $\theta$** — Log-log $\Delta\theta_{\text{opt}}$ vs $N$ for $\theta \in \{0.3, 1.0, 3.0\}$, with $2N$-SQL and $N$-SQL reference lines. Shows whether the scaling exponent changes with $\theta$ or deviates from the $-0.5$ $2N$-SQL exponent.

5. **Comparison with traced-out protocol** — Scatter comparison of $r_{\text{joint}} = \Delta\theta_{\text{opt}} / (1/(\sqrt{2N} T_H))$ (this report) vs $r_{\text{trace}} = \Delta\theta_{\text{trace}} / (1/(\sqrt{N} T_H))$ (2026-05-22). Both axes normalised to their respective SQL, enabling a fair comparison of the added value from optimised joint measurement vs. partial trace.

6. **Landscape slices** — 2D contour $\Delta\theta(\alpha_{xx}, \phi)$ for 3 representative $(\theta, N)$ points: $(N=1, \theta=0.5)$, $(N=5, \theta=2.0)$, $(N=20, \theta=4.0)$. Visualises the optimisation surface and confirms the global minimum is found. The $2N$-SQL contour line shows where $\Delta\theta = \Delta\theta_{\text{SQL}}$.

7. **Scaling exponent panel** — Exponent $\alpha$ vs $\theta$ from $\log\Delta\theta_{\text{opt}} = \alpha \log N + \log C$, with $R^2$ subpanel. Reference lines at $\alpha = -0.5$ ($2N$-SQL, also $N$-SQL) and $\alpha = -1.0$ (HL).

## ✅ Success Criteria

- **Decoupled baseline ($\phi=0$, $\alpha_{xx}=0$)** — $\Delta\theta = 1/(\sqrt{N} T_H)$, matching the $N$-SQL (worse than $2N$-SQL by $\sqrt{2}$) — **PENDING**
- **Decoupled baseline ($\phi=\pi/4$, $\alpha_{xx}=0$)** — $\Delta\theta = 1/(\sqrt{2N} T_H) = \Delta\theta_{\text{SQL}}$, exactly saturating the $2N$-SQL — **PENDING**
- **$2N$-SQL violation via XX coupling** — $\exists (\theta, N)$ such that $\Delta\theta_{\text{opt}} < 1/(\sqrt{2N} T_H)$ — **PENDING**
- **Genuine XX advantage** — $\exists (\theta, N)$ such that $\Delta\theta_{\text{opt}}(\alpha_{xx}^*>0, \phi^*) < \Delta\theta(\alpha_{xx}=0, \phi=\pi/4)$ — **PENDING**
- **Finite optimal $\alpha_{xx}$** — $\alpha_{xx}^* > 0$ for at least some $(\theta, N)$ pairs — **PENDING**
- **Optimal $\phi$ deviates from $\pi/4$** — $\phi^* \neq \pi/4$ for some $(\theta, N)$ where $\alpha_{xx}^* > 0$ — **PENDING**
- **State normalisation** — All intermediate and final state norms equal 1 to machine precision — **PENDING**
- **Unitarity** — $U_{\text{BS}}^\dagger U_{\text{BS}} = \mathbb{1}_{N+1}$ and $U_{\text{hold}}^\dagger U_{\text{hold}} = \mathbb{1}_{(N+1)^2}$ — **PENDING**
- **Numerical validity** — Hermiticity, variance positivity, sensitivity positivity, derivative stability — **PENDING**
- **Parquet roundtrip** — All metadata fields survive serialisation/deserialisation roundtrip; fail-fast on missing columns — **PENDING**

## ⚖️ Analytical Bounds

The total Hamiltonian is:
$H = \theta (J_z^S + J_z^A) + \alpha_{xx} J_x^S J_x^A.$

The measurement operator is $M(\phi) = \cos\phi \, J_z^S + \sin\phi \, J_z^A$. The commutator with $H_{\text{int}}$ is:
$[M, H_{\text{int}}] = i \alpha_{xx} \left( \cos\phi \, J_y^S J_x^A + \sin\phi \, J_x^S J_y^A \right) \neq 0.$

The XX coupling is therefore **active** for any $\phi$ — it does not commute with $M$. This is structurally different from the traced-out case in 2026-05-22, where the ancilla information was discarded and the effective measurement was on the reduced state only.

**Decoupled analytical baseline ($\alpha_{xx}=0$)**: For the standard dual MZI (product state, independent MZIs on S and A), the final state is $|\Psi\rangle = |\psi_S\rangle \otimes |\psi_A\rangle$ where each single-subsystem MZI acts on an $N$-particle CSS. The expectation values are:
$\langle J_z \rangle_{\text{single}} = -\frac{N}{2} \sin(\theta T_H),$
$\text{Var}(J_z)_{\text{single}} = \frac{N}{4}.$

For $M = \cos\phi \, J_z^S + \sin\phi \, J_z^A$:
$\langle M \rangle = (\cos\phi + \sin\phi) \langle J_z \rangle_{\text{single}},$
$\text{Var}(M) = \frac{N}{4}$ (independent of $\phi$),
$\frac{\partial \langle M \rangle}{\partial \theta} = (\cos\phi + \sin\phi) \left(-\frac{N}{2} T_H \cos(\theta T_H)\right).$

The sensitivity is:
$\Delta\theta(\alpha_{xx}=0, \phi) = \frac{1}{|\cos\phi + \sin\phi|} \cdot \frac{1}{\sqrt{N} T_H} \cdot \frac{1}{|\cos(\theta T_H)|}.$

At the optimal $\phi = \pi/4$: $\max_\phi |\cos\phi + \sin\phi| = \sqrt{2}$, giving:
$\Delta\theta_{\text{baseline}} = \frac{1}{\sqrt{2N} \, T_H \, |\cos(\theta T_H)|}.$

For $\theta$ near $0$ or where $|\cos(\theta T_H)| \approx 1$, this simplifies to $1/(\sqrt{2N} T_H)$. The $|\cos(\theta T_H)|$ denominator causes fringe-divergence at $\theta T_H = \pi/2, 3\pi/2, \dots$, where the signal derivative vanishes.

**$2N$-SQL benchmark**: The $2N$-particle standard quantum limit is:
$\Delta\theta_{\text{SQL}} = \frac{1}{\sqrt{2N} T_H}.$
This is the maximum sensitivity achievable with $2N$ unentangled particles in a classical interferometer. The **XX advantage ratio** is:
$r_{\text{XX}} = \frac{\Delta\theta_{\text{opt}}(\alpha_{xx}^*, \phi^*)}{\Delta\theta_{\text{SQL}}}.$
Values $r_{\text{XX}} < 1$ indicate genuine metrological gain from the XX interaction, below the separable $2N$-SQL.

For reference, the $N$-particle SQL (used in 2026-05-22 when only the system was measured) is $1/(\sqrt{N} T_H) = \sqrt{2} \cdot \Delta\theta_{\text{SQL}}$. The squared ratio between them reflects the factor-of-2 difference in total particle number: $\Delta\theta_{\text{SQL}}(2N) / \Delta\theta_{\text{SQL}}(N) = 1/\sqrt{2}$.

**QFI bound for the full pure state**: For the full pure state $|\Psi(\theta)\rangle$, the Quantum Fisher Information is $F_Q = 4 (\langle G_{\text{eff}}^2 \rangle - \langle G_{\text{eff}} \rangle^2)$, where $G_{\text{eff}}$ is the effective generator $G_{\text{eff}} = -i (\partial U_{\text{total}}/\partial \theta) U_{\text{total}}^\dagger$. The error-propagation sensitivity with an optimised measurement satisfies $\Delta\theta \geq \Delta\theta_Q = 1/\sqrt{F_Q}$. If the optimised $M$ is chosen optimally, it may approach but not exceed the QFI bound: $\Delta\theta_{\text{opt}} \geq 1/\sqrt{F_Q}$.

The QFI for the decoupled case ($\alpha_{xx}=0$) with the full state is:
$F_Q = 4\,\text{Var}(T_H (J_z^S + J_z^A)) = T_H^2 \cdot 4 \left( \text{Var}(J_z^S) + \text{Var}(J_z^A) \right) = T_H^2 \cdot 4 \cdot \frac{2N}{4} = 2 N T_H^2,$
giving $\Delta\theta_Q = 1/\sqrt{2N} T_H = \Delta\theta_{\text{SQL}}$, confirming that the separable $2N$-SQL is QFI-saturating.

If the XX coupling generates entanglement, the QFI can increase beyond $2N T_H^2$, enabling $\Delta\theta_Q < \Delta\theta_{\text{SQL}}$. Our error-propagation sensitivity with optimal $M$ may approach this bound if $M$ is chosen near the optimal observable, but the bound $r_{\text{XX}} \geq 1/\sqrt{F_Q / (2N T_H^2)}$ applies.

## 🏁 Conclusions

*To be completed after results are generated.*

The results of this report will determine whether a jointly optimised system--ancilla measurement can activate the XX coupling to beat the $2N$-particle SQL — a genuinely non-trivial benchmark. Unlike the 2026-05-22 report (where the SQL was trivially violated because ancilla information was discarded), the $2N$-SQL cannot be beaten by separable parallel MZIs. Any violation must come from XX-generated entanglement.

Expected outcome scenarios:

1. **Null hypothesis confirmed** ($\alpha_{xx}^* = 0$, $r_{\text{XX}} = 1$ always): The XX coupling is genuinely inactive for metrology, even with optimised joint measurements. The optimal measurement is always $\phi = \pi/4$ (equal weighting of S and A), exactly saturating the $2N$-SQL. The joint measurement matches, but never exceeds, what two independent MZIs can achieve.

2. **Partial activation** ($\alpha_{xx}^* > 0$ for some $(\theta, N)$, but $r_{\text{XX}} \approx 1$): The XX coupling is non-zero at the optimum but does not significantly improve beyond the $2N$-SQL. The optimal $\phi$ may shift slightly away from $\pi/4$, but the sensitivity remains at the $2N$-SQL within numerical precision.

3. **Genuine XX advantage** ($r_{\text{XX}} < 1$ for some $(\theta, N)$): The XX-generated entanglement improves sensitivity below the $2N$-SQL. The optimal $\phi^*$ deviates from $\pi/4$ to exploit interaction-induced correlations. Scaling exponents would deviate from $-0.5$. This would be the first demonstration of metrologically useful XX-coupling with a dual-MZI protocol.

**Open items**: (a) If the XX coupling alone remains inactive even with joint measurements, would adding the off-diagonal coupling terms ($H_y, H_{\text{diff}}$) from 2026-05-21 amplify the effect when combined with an optimised joint measurement? (b) Could the joint measurement approach activate the ZZ (Ising) interaction $H_{\text{int}} = \alpha_{zz} J_z^S \otimes J_z^A$ — where commutator $[M, H_{\text{int}}] = 0$ at $\phi = \pi/4$ but $[M, H_{\text{int}}] \neq 0$ for other $\phi$, potentially enabling higher-order metrological gain? (c) For a non-linear measurement (e.g., parity, photon-number correlations) rather than a linear combination of $J_z$ operators, could the XX coupling be harnessed more effectively? (d) What is the effect of initial entanglement between S and A (rather than a product state) combined with the optimised joint measurement?
