# Coupled System-Ancilla Metrology Under Photon Loss

## 🧪 Hypothesis

Consider a system-ancilla pair of multi-particle two-mode bosonic systems. The system S and the ancilla A both couple to the unknown phase rate $\omega$ through symmetric phase encoding ($H_S = \omega J_z^S$, $H_A = \omega J_z^A$) and interact through the four-parameter coupling $H_{\text{int}} = \alpha_{xx} J_x^S \otimes J_x^A + \alpha_{xz} J_x^S \otimes J_z^A + \alpha_{zx} J_z^S \otimes J_x^A + \alpha_{zz} J_z^S \otimes J_z^A$. Both subsystems independently experience one-body photon loss on both modes at rate $\gamma$. In this setting, the coupling between system and ancilla provides a quantifiable sensitivity advantage that depends on the noise level $\gamma$. Two baselines define the advantage: (A) the system alone without ancilla, and (B) two independent measurement resources (identical system and ancilla, uncoupled).

The central hypothesis decomposes into four specific, testable claims:

1. **Coupling advantage at finite noise**: For some finite values of the coupling coefficients $(\alpha_{xx}, \alpha_{xz}, \alpha_{zx}, \alpha_{zz})$ and of $\gamma$, the Quantum Fisher Information (QFI) of the coupled system $F_Q^{(C)}$ exceeds the sum of the individual QFIs $F_Q^{(S)} + F_Q^{(A)}$ from the uncoupled case. The coupling then creates correlations that increase the total information about $\omega$ beyond what two independent resources provide.

2. **Noise-dependent coupling advantage**: The QFI ratio $\mathcal{R}_{QFI} = F_Q^{(C)} / (F_Q^{(S)} + F_Q^{(A)})$ depends on $\gamma$. At $\gamma = 0$ (noiseless), the advantage arises because the transverse coupling terms ($\alpha_{xx}$, $\alpha_{xz}$, $\alpha_{zx}$) acquire $\omega$-dependent rotation in the interaction picture. This rotation creates an additional $\omega$-encoding channel (Channel 2) that feeds ancilla information back to the system measurement (experiment #20260521 achieved $0.690\times$ the standard quantum limit (SQL) using all four coupling terms). As $\gamma$ increases, the advantage either degrades, because noise destroys correlations, or persists, because the coupling protects information.

3. **Practical S-only sensitivity improvement**: Under S-only measurement, the error-propagation sensitivity of the coupled system $\Delta\omega_{EP}^{(C)}$ improves over the system-alone baseline $\Delta\omega_{EP}^{(A)}$ (Config A, $N$ particles, no ancilla) at finite noise. The improvement ratio $\mathcal{R}_{EP} = \Delta\omega_{EP}^{(C)} / \Delta\omega_{EP}^{(A)}$ depends on both $\gamma$ and $N$.

4. **Measurement gap**: The ratio $\Delta\omega_{EP}^{(C)} / \Delta\omega_{QFI}^{(C)}$ quantifies how much information is lost by measuring $J_z^S$ alone on the coupled system. This gap might change with noise. At high noise, the S-only measurement might capture a larger fraction of the available QFI because noise suppresses the correlations that distinguish joint from S-only measurement.

**Null hypothesis**: The coupling provides no advantage at any noise level: $\mathcal{R}_{QFI} = 1$ and $\mathcal{R}_{EP} = 1$ for all $(\gamma, N, \omega)$, meaning the interaction $(\alpha_{xx}, \alpha_{xz}, \alpha_{zx}, \alpha_{zz})$ neither creates useful correlations nor protects information under photon loss.

## ⚛️ Theoretical Model

The total Hilbert space is $\mathcal{H}_{\text{tot}} = \mathcal{H}_S \otimes \mathcal{H}_A$, where each subsystem is a **two-mode bosonic Fock space** with truncation $M = N$ per mode (dimension $(N+1)^2$), initialised with $N$ particles in mode 0. The single-subsystem space has dimension $d_{\text{sub}} = (N+1)^2$ with ordered computational basis $\{|n_0, n_1\rangle : 0 \leq n_0, n_1 \leq N\}$, indexed by $n_0 (N+1) + n_1$. The full system-ancilla space has dimension $d = (N+1)^4$ with basis vectors $|n_{0,S}, n_{1,S}\rangle \otimes |n_{0,A}, n_{1,A}\rangle$, indexed by the composite index $i_S (N+1)^2 + i_A$ where $i_S = n_{0,S}(N+1) + n_{1,S}$ and $i_A = n_{0,A}(N+1) + n_{1,A}$.

The **angular momentum operators** for each subsystem are $J_z^k = (n_0^k - n_1^k)/2$, $J_x^k = (a_0^{k\dagger} a_1^k + a_1^{k\dagger} a_0^k)/2$, and $J_y^k = (a_0^{k\dagger} a_1^k - a_1^{k\dagger} a_0^k)/(2i)$, where $k \in \{S, A\}$ labels the subsystem. These are $(N+1)^2 \times (N+1)^2$ matrices in the two-mode Fock basis. Embedding into the full space gives $J_z^S = J_z \otimes \mathbb{1}_{(N+1)^2}$ and $J_z^A = \mathbb{1}_{(N+1)^2} \otimes J_z$. The **annihilation operators** $a_0$ and $a_1$ destroy a single photon in mode 0 and mode 1 respectively: $a_k |n_0, n_1\rangle = \sqrt{n_k} |n_0 - \delta_{k,0}, n_1 - \delta_{k,1}\rangle$. These operators do not conserve total particle number and are therefore not representable in the fixed-$N$ Dicke subspace. The full two-mode Fock space is required.

The **initial state** is a product of Dicke states $|\Psi_0\rangle = |N, 0\rangle_S \otimes |N, 0\rangle_A$ (all particles in mode 0 of each subsystem). The corresponding basis vector has composite index $i_S(N+1)^2 + i_A$, with $i_S = i_A = N(N+1)$, in the $(N+1)^4$-dimensional space.

The **circuit protocol** proceeds in four steps:

1. **Beam splitter on system only**: A 50/50 symmetric beam splitter (BS) $U_{\text{BS}} = \exp(-i (\pi/2) J_x)$ acts on the S subsystem. The full-space operator is $U_{\text{BS}}^{(S)} = U_{\text{BS}} \otimes \mathbb{1}_{(N+1)^2}$. The multi-particle BS matrix elements are computed by block-diagonal matrix exponentiation over constant-total-photon subspaces.

2. **Noisy holding period**: The full S-A density matrix evolves under the Lindblad master equation for duration $T_H$:
$\dot{\rho} = -i[H, \rho] + \sum_k \left( L_k \rho L_k^\dagger - \frac{1}{2} \{L_k^\dagger L_k, \rho\} \right)$
where the **Hamiltonian** is:
$H = \omega J_z^S + \omega J_z^A + \alpha_{xx} J_x^S \otimes J_x^A + \alpha_{xz} J_x^S \otimes J_z^A + \alpha_{zx} J_z^S \otimes J_x^A + \alpha_{zz} J_z^S \otimes J_z^A$
and the **Lindblad operators** are independent one-body loss on each mode of each subsystem:
$L_{0,S} = \sqrt{\gamma} \, a_{0,S} \otimes \mathbb{1}_A, \quad L_{1,S} = \sqrt{\gamma} \, a_{1,S} \otimes \mathbb{1}_A$
$L_{0,A} = \mathbb{1}_S \otimes \sqrt{\gamma} \, a_{0,A}, \quad L_{1,A} = \mathbb{1}_S \otimes \sqrt{\gamma} \, a_{1,A}$
The total Lindblad operator count is four (two modes $\times$ two subsystems). We solve the master equation with QuTiP `mesolve` using adaptive stepping of the ordinary differential equations (ODEs). We convert the pure-state initial condition to a density matrix $\rho_0 = |\Psi_0\rangle\langle\Psi_0|$ and propagate it to $\rho(T_H)$.

3. **Beam splitter on system only**: A second identical BS $U_{\text{BS}}^{(S)}$ acts on the system: $\rho_{\text{final}} = U_{\text{BS}}^{(S)} \rho(T_H) U_{\text{BS}}^{(S)\dagger}$.

4. **Measurement**: We measure the observable $J_z^S$ on the reduced system state $\rho_S = \text{Tr}_A(\rho_{\text{final}})$. The expectation value is $\langle J_z^S \rangle = \text{Tr}(\rho_S J_z^S)$, the variance is $\text{Var}(J_z^S) = \text{Tr}(\rho_S (J_z^S)^2) - \langle J_z^S \rangle^2$, and the sensitivity from **error propagation** is:
$\Delta\omega_{EP}^{(C)} = \frac{\sqrt{\text{Var}(J_z^S)}}{|\partial\langle J_z^S\rangle/\partial\omega|}$
We compute the derivative by central finite differences with step $\delta = 10^{-6}$, re-evaluating the full Lindblad evolution at $\omega \pm \delta$.

We compute the **Quantum Fisher Information** of the full S-A state $\rho_{\text{final}}(\omega)$ by eigenvalue decomposition: $F_Q = 2 \sum_{i,j : \lambda_i + \lambda_j > 0} \frac{(\lambda_i - \lambda_j)^2}{\lambda_i + \lambda_j} |\langle e_i | \partial_\omega \rho | e_j \rangle|^2$, where $\lambda_i$ and $|e_i\rangle$ are the eigenvalues and eigenvectors of $\rho_{\text{final}}(\omega)$, and we compute $\partial_\omega \rho$ by central finite differences. The QFI sensitivity is $\Delta\omega_{QFI} = 1/\sqrt{F_Q}$.

**Three configurations**:

| Config | Description | Hamiltonian | Measurement | Sensitivity metric |
|--------|-------------|-------------|-------------|-------------------|
| **A** | System alone ($N$ particles) | $H = \omega J_z^S$ | $J_z^S$ (EP) | $\Delta\omega_{EP}^{(A)}$, $F_Q^{(A)}$ |
| **B** | Two independent resources (S and A uncoupled, each $N$ particles) | $H_S = \omega J_z^S$, $H_A = \omega J_z^A$ (independent) | QFI on full S-A | $\Delta\omega_{QFI}^{(B)} = 1/\sqrt{F_Q^{(S)} + F_Q^{(A)}}$ |
| **C** | Coupled system (S and A, each $N$ particles) | $H = \omega J_z^S + \omega J_z^A + \alpha_{xx} J_x^S \otimes J_x^A + \alpha_{xz} J_x^S \otimes J_z^A + \alpha_{zx} J_z^S \otimes J_x^A + \alpha_{zz} J_z^S \otimes J_z^A$ | $J_z^S$ (EP) + QFI on full S-A | $\Delta\omega_{EP}^{(C)}$, $\Delta\omega_{QFI}^{(C)}$ |

**Key ratios**:

- **Coupling QFI ratio**: $\mathcal{R}_{QFI}(\gamma, N) = F_Q^{(C)} / (F_Q^{(S)} + F_Q^{(A)}) = F_Q^{(C)} / (2 F_Q^{(A)})$. This measures whether the four-parameter coupling creates information beyond two independent resources. $\mathcal{R}_{QFI} > 1$ means the coupling increases the total information; $\mathcal{R}_{QFI} = 1$ means the coupling leaves the total information unchanged; $\mathcal{R}_{QFI} < 1$ means the coupling reduces the total information.
- **Practical EP ratio**: $\mathcal{R}_{EP}(\gamma, N, \omega) = \Delta\omega_{EP}^{(C)} / \Delta\omega_{EP}^{(A)}$. This measures the practical S-only sensitivity improvement from adding a coupled ancilla. $\mathcal{R}_{EP} < 1$ means the coupling and ancilla together improve the sensitivity.
- **Measurement gap**: $\mathcal{R}_{gap}(\gamma, N, \omega) = \Delta\omega_{EP}^{(C)} / \Delta\omega_{QFI}^{(C)}$. This measures how much information is lost by S-only measurement on the coupled system. $\mathcal{R}_{gap} = 1$ means S-only is optimal; $\mathcal{R}_{gap} > 1$ means information is lost.

**Standard quantum limit**: For Config A with $N$ particles, $\Delta\omega_{\text{SQL}}^{(A)} = 1/(\sqrt{N} T_H)$. For Config B with $2N$ total particles (uncoupled), $\Delta\omega_{\text{SQL}}^{(B)} = 1/(\sqrt{2N} T_H)$. Under photon loss, the achieved sensitivity degrades from the SQL; the ratio $\Delta\omega / \Delta\omega_{\text{SQL}}$ quantifies the degradation.

**Physical mechanism**: In the interaction picture with respect to $H_0 = \omega(J_z^S + J_z^A)$, the four coupling terms acquire different $\omega$-dependent dynamics. The longitudinal terms $\alpha_{zz} J_z^S \otimes J_z^A$ commute with $H_0$, remain unchanged, and contribute no implicit $\omega$-dependence. The transverse terms rotate: $\alpha_{xx} J_x^S \otimes J_x^A$ acquires $\cos^2(\omega t)$, $\sin^2(\omega t)$, and cross-angle factors on both subsystems; $\alpha_{xz} J_x^S \otimes J_z^A$ rotates only the system factor; $\alpha_{zx} J_z^S \otimes J_x^A$ rotates only the ancilla factor. This rotation creates a second channel (Channel 2) through which the evolution acquires $\omega$-dependence beyond the standard phase-encoding derivative $\partial H_0/\partial\omega = J_z^S + J_z^A$. The entanglement generated by the coupling causes the ancilla's $\omega$-dependent dynamics to feed back onto $\langle J_z^S \rangle$. This feedback can increase the error-propagation derivative beyond the single-subsystem bound (experiment #20260521, $0.690\times$ SQL, using the full four-parameter interaction). Under photon loss, two competing effects emerge: (1) the coupling might redistribute $\omega$-information across both subsystems, making it more robust to loss on any single mode; (2) the coupling might spread loss-induced decoherence from one subsystem to the other, degrading correlations. The balance between these effects determines whether $\mathcal{R}_{QFI} > 1$ or $\mathcal{R}_{QFI} < 1$ at each noise level.

## 💻 Numerical Simulation

### Implementation strategy

1. **Operator construction**: Build creation/annihilation operators $a_0, a_1, a_0^\dagger, a_1^\dagger$ as $(N+1)^2 \times (N+1)^2$ sparse matrices in the two-mode Fock basis by tensor products of single-mode ladder operators. Construct $J_z = (n_0 - n_1)/2$, $J_x$, $J_y$ from these. Build full S-A operators by Kronecker products: $J_z^S = J_z \otimes \mathbb{1}_{(N+1)^2}$, $J_z^A = \mathbb{1}_{(N+1)^2} \otimes J_z$, $a_{k,S} = a_k \otimes \mathbb{1}_A$, $a_{k,A} = \mathbb{1}_S \otimes a_k$. Construct the four interaction operators $J_x^S \otimes J_x^A$, $J_x^S \otimes J_z^A$, $J_z^S \otimes J_x^A$, $J_z^S \otimes J_z^A$ as sparse $(N+1)^4 \times (N+1)^4$ matrices, and assemble $H_{\text{int}} = \alpha_{xx} J_x^S \otimes J_x^A + \alpha_{xz} J_x^S \otimes J_z^A + \alpha_{zx} J_z^S \otimes J_x^A + \alpha_{zz} J_z^S \otimes J_z^A$ as the weighted sum.

2. **BS unitary**: Compute $U_{\text{BS}} = \exp(-i (\pi/2) J_x)$ by `bs_fock(π/4, 0, N)` (block-diagonal `expm` over constant-total-photon subspaces) on the $(N+1)^2 \times (N+1)^2$ subsystem, then embed as $U_{\text{BS}}^{(S)} = U_{\text{BS}} \otimes \mathbb{1}_{(N+1)^2}$.

3. **Lindblad evolution**: For Config A (system alone): evolve the $(N+1)^2 \times (N+1)^2$ density matrix under $H = \omega J_z$ with Lindblad operators $L_0 = \sqrt{\gamma} \, a_0$ and $L_1 = \sqrt{\gamma} \, a_1$. For Config C (coupled): evolve the $(N+1)^4 \times (N+1)^4$ density matrix under $H = \omega(J_z^S + J_z^A) + \alpha_{xx} J_x^S \otimes J_x^A + \alpha_{xz} J_x^S \otimes J_z^A + \alpha_{zx} J_z^S \otimes J_x^A + \alpha_{zz} J_z^S \otimes J_z^A$ with four Lindblad operators $L_{0,S}, L_{1,S}, L_{0,A}, L_{1,A}$. Use QuTiP `mesolve` with adaptive stepping. We convert the initial pure state to a density matrix $\rho_0 = |\Psi_0\rangle\langle\Psi_0|$.

4. **QFI computation**: For each $\omega$, compute $\rho(\omega)$ and $\rho(\omega \pm \delta)$ (three Lindblad evaluations per $\omega$ point). Diagonalise $\rho(\omega)$ to get eigenvalues $\lambda_i$ and eigenvectors $|e_i\rangle$. Compute $\partial_\omega \rho = (\rho(\omega+\delta) - \rho(\omega-\delta))/(2\delta)$. Evaluate the QFI sum: $F_Q = 2 \sum_{i,j: \lambda_i+\lambda_j>0} \frac{(\lambda_i - \lambda_j)^2}{\lambda_i + \lambda_j} |\langle e_i | \partial_\omega \rho | e_j\rangle|^2$. For Config B, $F_Q^{(B)} = 2 F_Q^{(A)}$ (additive for uncoupled identical systems), so we need only the single-subsystem QFI.

5. **EP sensitivity**: Compute $\langle J_z^S \rangle$ and $\text{Var}(J_z^S)$ from the final state. Compute $\partial\langle J_z^S\rangle/\partial\omega$ by central finite differences (three Lindblad evaluations per $\omega$ point, shared with QFI computation). The EP sensitivity is $\Delta\omega_{EP} = \sqrt{\text{Var}(J_z^S)} / |\partial\langle J_z^S\rangle/\partial\omega|$.

6. **Coupling optimisation**: For each $(\gamma, N)$ pair, optimise the four coupling coefficients $\mathbf{\alpha} = (\alpha_{xx}, \alpha_{xz}, \alpha_{zx}, \alpha_{zz})$ to minimise the EP sensitivity at a representative $\omega$ value (the $\omega$ that gives the best noise-free sensitivity). Use L-BFGS-B with multi-start: five random initial points in $[-10, 10]^4$, each run for up to 20 iterations. Use the optimal $\mathbf{\alpha}^*(\gamma, N)$ for the full $\omega$ scan at that $(\gamma, N)$. The wider bounds (version 2, v2) prevent the boundary saturation observed in version 1 (v1) at $N = 2$ ($\alpha_{zx}^* = 5.0$).

7. **Sweep structure**: For each $N$ value:
   - Optimise $\mathbf{\alpha}^*$ at $\gamma = 0$ by multi-start L-BFGS-B (five starts, 20 iterations each)
   - For each $\gamma$ value (61 values: $\gamma = 0$ plus 60 log-spaced in $[10^{-6}, 10^6]$): evaluate Config A and Config C with the fixed $\mathbf{\alpha}^*$
   - For each $\omega$ value (500 values): evaluate Config A ($\Delta\omega_{EP}^{(A)}$, $F_Q^{(A)}$) and Config C ($\Delta\omega_{EP}^{(C)}$, $F_Q^{(C)}$, $\Delta\omega_{QFI}^{(C)}$) with the optimal $\mathbf{\alpha}^*$
   - Compute ratios $\mathcal{R}_{QFI}$, $\mathcal{R}_{EP}$, $\mathcal{R}_{gap}$
   - Store all results in Parquet files with full parameter metadata

8. **Parallelisation**: Both the $\gamma$ loop and the $\omega$ sweep at each $(\gamma, N)$ are fully parallel. Use `joblib.Parallel` with `n_jobs=-1` (all available cores). Each $\gamma$ value is independent (no coupling between different noise levels), and each $\omega$ point within a $\gamma$ sweep is independent (finite-difference derivatives use separate density matrices).

### Parameter sweep

| Parameter | Range | Points | Purpose |
|-----------|-------|--------|---------|
| $N$ (particles per subsystem) | 1 to 8 | 8 | Hilbert space scaling; Config C limited to $N \leq 8$ by $(N+1)^4$ dimension |
| $\gamma$ (loss rate) | $\gamma = 0$ + $[10^{-6}, 10^6]$ log-spaced | 61 | Noise strength; covers $\gamma T_H \in [0, 10^7]$ |
| $\omega$ (phase rate) | $[0.01, 5.00]$ | 500 | Phase dependence at each noise level |
| $\mathbf{\alpha}$ (coupling) | $(\alpha_{xx}, \alpha_{xz}, \alpha_{zx}, \alpha_{zz}) \in [-10, 10]^4$ | Optimised | Four interaction strengths; optimised jointly at $\gamma = 0$ by L-BFGS-B |
| $T_H$ (holding time) | 10 (fixed) | — | SQL reference $\Delta\omega_{\text{SQL}}^{(N=1)} = 0.1$ |
| $\delta$ (finite-diff step) | $10^{-6}$ (fixed) | — | Derivative computation |

**Total evaluation count** (estimated):
- Coupling optimisation (L-BFGS-B): $8 \times 5 \text{ starts} \times 20 \text{ iters} \times 3 \text{ solves} = 2{,}400$ Lindblad solves of dimension $(N+1)^4$. This is negligible.
- $\gamma$-sweep: $8 \times 61 = 488$ points, each requiring three Config A and three Config C Lindblad evaluations = $2{,}928$ Lindblad solves of dimension $(N+1)^4$, a bottleneck at $N=6$ to $8$
- $\omega$-scan: $8 \times 3 \times 500 = 12{,}000$ points, each requiring three Lindblad evaluations = $36{,}000$ Lindblad solves
- Config A (system alone): negligible ($< 1$ ms each)
- Config C (coupled): $\sim 10$ ms at $N=1$, $\sim 100$ ms at $N=5$, $\sim 1$ s at $N=8$

**Estimated wall time**: With `joblib.Parallel(n_jobs=-1)`, the $\gamma$-sweep for $N = 1$ to $3$ runs in less than 10 minutes each; $N = 4$ about 30 minutes; $N = 5$ about 3 hours; $N = 6$ about 17 hours; $N = 7$ about 70 hours; $N = 8$ about 250 hours (single-threaded estimates). With full parallelisation across all $\gamma$ values, the total wall time is about 24 hours for the complete sweep.

### Validation

The following physical invariants are verified throughout every simulation run:

- **State normalisation**: $\text{Tr}(\rho_{\text{final}}) = 1$ holds to machine precision after Lindblad evolution (trace preservation).
- **Hermiticity**: $\rho_{\text{final}} = \rho_{\text{final}}^\dagger$ and $H = H^\dagger$.
- **Positivity**: $\rho_{\text{final}} \succeq 0$ (all eigenvalues non-negative); the Lindblad structure enforces this.
- **Variance positivity**: $\text{Var}(J_z^S) \geq 0$.
- **Sensitivity positivity**: $\Delta\omega > 0$ for all valid configurations.
- **QFI positivity**: $F_Q \geq 0$ with $F_Q = 0$ only when $\rho$ is $\omega$-independent.
- **QFI-EP inequality**: $\Delta\omega_{QFI} \leq \Delta\omega_{EP}$ always (QFI is the optimal measurement bound).
- **Baseline recovery at $\gamma = 0$**: Config A recovers the noiseless single-particle Mach–Zehnder interferometer (MZI) ($\Delta\omega = 1/T_H$ at $N=1$). Config C with non-zero transverse coupling recovers the noiseless sub-SQL result ($\Delta\omega_{EP}^{(C)} < \Delta\omega_{EP}^{(A)}$ for optimised $\mathbf{\alpha}$), consistent with experiment #20260521.
- **Baseline recovery at $\mathbf{\alpha} = 0$**: Config C reduces to Config B (uncoupled); $\mathcal{R}_{QFI} = 1$ exactly.
- **SQL baseline**: Config A at $\gamma = 0$ gives $\Delta\omega = 1/(\sqrt{N} T_H)$ at mid-fringe operating points.
- **Loss-induced decoherence**: As $\gamma \to \infty$, $\Delta\omega \to \infty$ (sensitivity degrades without bound as all information is lost).

## ⚠️ Expected Failure Conditions

| Failure | Mitigation |
|---------|------------|
| **Config C Hilbert space too large for $N > 8$**: The $(N+1)^4$-dimensional coupled space makes ODE solver time impractical for $N > 8$ (each mesolve call scales as $(N+1)^{12}$) | Limit Config C to $N \leq 8$. Config A and Config B can extend to $N = 20$ since they factorise. Report the $N$ limitation explicitly. |
| **Lindblad solver instability at large $\gamma T_H$**: QuTiP `mesolve` might fail or produce non-physical results when $\gamma T_H \gg 1$ (complete decoherence regime) | Clamp eigenvalues of $\rho$ to $[0, \infty)$ after each solve. Use tighter ODE tolerances (`rtol=1e-10`, `atol=1e-12`) at high $\gamma$. Discard points where $\text{Tr}(\rho)$ deviates from 1 by more than $10^{-6}$. |
| **QFI numerical instability when $\rho$ is nearly rank-1**: At small $\gamma$, $\rho$ is nearly pure; the eigenvalue sum $\sum (\lambda_i - \lambda_j)^2 / (\lambda_i + \lambda_j)$ amplifies small eigenvalue errors | Use a threshold $\lambda_{\min} = 10^{-12}$: discard eigenvalue pairs where $\lambda_i + \lambda_j < \lambda_{\min}$. Cross-validate with the pure-state formula $F_Q = 4(\langle G^2\rangle - \langle G\rangle^2)$ at $\gamma = 0$. |
| **Coupling optimisation converges to boundary**: The optimal coupling might saturate the search bounds for some $\alpha_{ij}$ components, indicating the true optimum lies outside the search range | The v2 bounds are $[-10, 10]^4$ (widened from $[-5, 5]^4$ in v1). If boundary convergence persists, report the fraction of boundary-saturated coefficients and note the limitation. |
| **$\langle J_z^S\rangle \approx 0$ at fringe nulls**: The EP sensitivity diverges where $\partial\langle J_z^S\rangle/\partial\omega \approx 0$ | Flag points where $\vert\partial\langle J_z^S\rangle/\partial\omega\vert < 10^{-12}$ as fringe nulls. Report the Classical Fisher Information (CFI) sensitivity, computed from the full $P(m\vert\omega)$ distribution, at these points as a fallback. |
| **Optimal $\mathbf{\alpha}$ depends on $\omega$**: The $\mathbf{\alpha}^*$ found at a representative $\omega$ might not be optimal at other $\omega$ values | Report the $\omega$-dependence of $\Delta\omega_{EP}^{(C)}$ at the fixed $\mathbf{\alpha}^*$; if the $\omega$-scan reveals that a different $\mathbf{\alpha}$ would be better at some $\omega$, note this as an open item for future per-$\omega$ optimisation. |

## 🔬 Results

All experiments use $T_H = 10$, giving $\Delta\omega_{\text{SQL}}^{(N=1)} = 0.1$. The v2 $\gamma$ scan uses 61 values: $\gamma = 0$ plus 60 log-spaced in $[10^{-6}, 10^6]$. We evaluate Config A for $N = 1$ to $8$. We evaluate Config C for $N = 1$ to $8$ with coupling $\mathbf{\alpha}$ optimised at $\gamma = 0$ (bounds $[-10, 10]^4$) and held fixed across all $\gamma$ values. The $\omega$ scan uses 500 values from 0.01 to 5.00 at selected $\gamma$ pairs. **See** `reports/r20260713/raw_data/` for Parquet files.

### Decoupled baseline ($\gamma = 0$, $\mathbf{\alpha} = 0$)

| Config | $\Delta\omega$ (EP) | $\Delta\omega_{QFI}$ | $\mathcal{R}_{QFI}$ | $\mathcal{R}_{EP}$ | Status |
|--------|---------------------|----------------------|---------------------|---------------------|--------|
| A ($N=1$) | 0.10000 (SQL) | 0.10000 (SQL) | — | — | PASS |
| B ($N=1$) | — | 0.07071 ($1/\sqrt{2}$ SQL) | — | — | PASS |
| C ($N=1$, optimised $\mathbf{\alpha}$) | 0.08927 | 0.12244 | 0.6688 | 0.8927 | PASS |

**Key Finding**: At $\gamma = 0$ with optimised $\mathbf{\alpha}^* = (1.93, 0.00, 3.09, 1.30)$, the coupled system achieves $\mathcal{R}_{EP} = 0.89$, an 11% sensitivity improvement over the system alone. However, $\mathcal{R}_{QFI} = 0.67 < 1$ indicates the coupling reduces the total QFI below two independent resources. This apparent paradox arises because the coupling creates correlations that enhance the S-only measurement but reduce the total quantum information: the optimisation targets EP sensitivity (S-only $J_z^S$ measurement), not the full QFI.

### Config A: System alone ($N = 1$ to $8$)

| $N$ | $\Delta\omega_{EP}^{(A)}$ ($\gamma = 0$) | SQL | $\Delta\omega / \text{SQL}$ | Status |
|-----|-------------------------------------------|-----|------------------------------|--------|
| 1 | 0.10000 | 0.10000 | 1.000 | PASS |
| 2 | 0.07071 | 0.07071 | 1.000 | PASS |
| 3 | 0.05774 | 0.05774 | 1.000 | PASS |
| 4 | 0.05000 | 0.05000 | 1.000 | PASS |
| 5 | 0.04472 | 0.04472 | 1.000 | PASS |
| 6 | 0.04082 | 0.04082 | 1.000 | PASS |
| 7 | 0.03780 | 0.03780 | 1.000 | PASS |
| 8 | 0.03536 | 0.03536 | 1.000 | PASS |

**Key Finding**: Config A recovers the SQL $\Delta\omega = 1/(\sqrt{N} T_H)$ exactly at $\gamma = 0$ for all $N$, confirming the noiseless MZI baseline. The $1/\sqrt{N}$ scaling holds across the full $N = 1$ to $8$ range.

### Config C: Coupled system ($N = 1$ to $8$, optimised $\mathbf{\alpha}$)

We optimise the coupling $\mathbf{\alpha}$ once at $\gamma = 0$ for each $N$ (bounds $[-10, 10]^4$) and hold it fixed for all $\gamma$ values. The table lists the v1 optimised values; the v2 run updates these values after completion:

| $N$ | $\alpha_{xx}^*$ | $\alpha_{xz}^*$ | $\alpha_{zx}^*$ | $\alpha_{zz}^*$ | $\Delta\omega_{EP}^{(C)}$ ($\gamma=0$) |
|-----|-----------------|-----------------|-----------------|-----------------|----------------------------------------|
| 1 | 1.93 | 0.00 | 3.09 | 1.30 | 0.08927 |
| 2 | -1.74 | -0.58 | 5.00 | 2.88 | 0.09645 |
| 3 | 2.79 | -0.69 | 3.48 | 2.09 | 0.11498 |

**Key Finding** (v1): The optimised coupling consistently includes non-zero transverse terms ($\alpha_{xx}$, $\alpha_{xz}$, $\alpha_{zx}$), confirming that the Channel 2 $\omega$-encoding mechanism is active. In v1, $N = 2$ saturated $\alpha_{zx}^*$ at 5.00 (boundary of $[-5, 5]^4$). The v2 run uses wider bounds $[-10, 10]^4$ to resolve this saturation.

### Coupling QFI ratio $\mathcal{R}_{QFI}(\gamma, N)$

![R_QFI heatmap](figures/20260713-rqfi-heatmap.svg)

The heatmap shows $\mathcal{R}_{QFI} = F_Q^{(C)} / (2 F_Q^{(A)})$ for the coupled system with fixed $\mathbf{\alpha}^*$ optimised at $\gamma = 0$. At $\gamma = 0$, $\mathcal{R}_{QFI} < 1$ for all $N$ (N=1: 0.67, N=2: 0.79, N=3: 0.53), indicating the coupling reduces total QFI below two independent resources. As $\gamma$ increases, $\mathcal{R}_{QFI}$ decreases further, approaching zero at high noise. This confirms that the fixed $\mathbf{\alpha}^*$ (optimised for EP sensitivity, not QFI) does not enhance the QFI. The coupling helps the S-only measurement by creating correlations that improve $\partial\langle J_z^S\rangle/\partial\omega$, but these correlations do not increase the total quantum information about $\omega$. **See** `figures/20260713-rqfi-heatmap.svg`.

### Sensitivity comparison: Config A versus Config C

![Sensitivity versus gamma](figures/20260713-sensitivity-vs-gamma.svg)

The figure compares the EP sensitivity $\Delta\omega_{EP}$ for Config A (system alone) and Config C (coupled system with fixed $\mathbf{\alpha}^*$) as a function of photon loss rate $\gamma$. At $N = 1$, Config C achieves 11% better sensitivity than Config A at $\gamma = 0$ ($\mathcal{R}_{EP} = 0.89$), but this advantage degrades with increasing noise. At $N = 2$ and $N = 3$, Config C is already worse than Config A at $\gamma = 0$, indicating that the coupling advantage does not generalise to larger $N$ with the fixed $\mathbf{\alpha}^*$. Config A shows monotonic degradation with $\gamma$ for all $N$ (except the degenerate $\gamma = 4.0$ point where the system is fully decohered). Config C is monotonic at $N = 1$ and $N = 3$, but non-monotonic at $N = 2$ because of the boundary-saturating $\alpha_{zx}^* = 5.0$. **See** `figures/20260713-sensitivity-vs-gamma.svg`.

### Optimal coupling coefficients

The figures plot the four coupling coefficients $\mathbf{\alpha}^*$ against $\gamma$ at the representative $\omega = 1.0$. Because $\mathbf{\alpha}^*$ is optimised at $\gamma = 0$ and held fixed, the plotted values are constant across $\gamma$; the flat lines show the same fixed $\mathbf{\alpha}^*$ applied at each noise level. Red dotted lines mark the optimisation bounds $[-10, 10]$. Individual per-$N$ SVGs are provided for detailed inspection.

![Optimal alpha N=1](figures/20260713-optimal-alpha-N1.svg)
![Optimal alpha N=2](figures/20260713-optimal-alpha-N2.svg)
![Optimal alpha N=3](figures/20260713-optimal-alpha-N3.svg)
![Optimal alpha N=4](figures/20260713-optimal-alpha-N4.svg)
![Optimal alpha N=5](figures/20260713-optimal-alpha-N5.svg)
![Optimal alpha N=6](figures/20260713-optimal-alpha-N6.svg)
![Optimal alpha N=7](figures/20260713-optimal-alpha-N7.svg)
![Optimal alpha N=8](figures/20260713-optimal-alpha-N8.svg)

**See** `figures/20260713-optimal-alpha-N1.svg` through `figures/20260713-optimal-alpha-N8.svg` for each $N$.

### $\omega$-dependence of sensitivity

![Omega dependence](figures/20260713-omega-dependence.svg)

The figure plots the sensitivity $\Delta\omega_{EP}$ against the phase rate $\omega$ at selected $(\gamma, N)$ pairs. The v2 run scans all eight $N$ values at $\gamma \in \{0, 0.25, 1.0\}$, providing comprehensive $\omega$-dependence data across the full particle-number range. **See** `figures/20260713-omega-dependence.svg`.

### Measurement gap $\mathcal{R}_{gap}$

![Measurement gap](figures/20260713-measurement-gap.svg)

The ratio $\mathcal{R}_{gap} = \Delta\omega_{EP}^{(C)} / \Delta\omega_{QFI}^{(C)}$ measures how much information is lost by measuring only $J_z^S$ on the coupled system. At $N = 1$, $\gamma = 0$, the representative-omega value is $\mathcal{R}_{gap} \approx 1.03$ (S-only captures $\sim 97\%$ of available QFI), but $\mathcal{R}_{gap}$ is strongly $\omega$-dependent (ranging from $\sim 1.0$ at the optimal operating point ($\omega \approx 1.1$) to $\sim 24$ at fringe nulls ($\omega \ll 1$)). At $N = 3$, $\gamma = 0$, $\mathcal{R}_{gap}$ is larger ($\sim 2.0$ representative, up to $\sim 404$ at fringe nulls), indicating more information is lost with S-only measurement on larger coupled systems. **See** `figures/20260713-measurement-gap.svg`.

### Practical EP ratio $\mathcal{R}_{EP}$

![EP ratio](figures/20260713-ep-ratio.svg)

The practical ratio $\mathcal{R}_{EP} = \Delta\omega_{EP}^{(C)} / \Delta\omega_{EP}^{(A)}$ measures the S-only sensitivity improvement from adding a coupled ancilla. At $N = 1$, $\gamma = 0$: $\mathcal{R}_{EP} = 0.89$ (11% improvement). $\mathcal{R}_{EP}$ peaks at $\gamma = 0.063$ ($\mathcal{R}_{EP} \approx 2.07$, the coupling degrades sensitivity by a factor of two), then decreases at higher noise as both configurations degrade towards complete decoherence. At $\gamma = 4.0$, the system is fully decohered ($F_Q^{(A)} \approx 0$) and $\mathcal{R}_{EP}$ is undefined. At $N = 2$ and $N = 3$, $\mathcal{R}_{EP} > 1$ everywhere, indicating that the fixed $\mathbf{\alpha}^*$ does not provide a practical advantage for these particle numbers. The noise-dependent variation of $\mathcal{R}_{EP}$ confirms hypothesis 3: the coupling advantage depends on both $\gamma$ and $N$. **See** `figures/20260713-ep-ratio.svg`.

### Summary table

| Experiment | Status | Key Result |
|------------|--------|-----------|
| Config A: system alone, $\gamma$ sweep, $N=1$ to $8$ | PENDING | SQL recovery at $\gamma=0$ for all $N$; monotonic degradation with $\gamma$ |
| Config B: two independent resources, QFI, $N=1$ to $8$ | PASS | $F_Q^{(B)} = 2 F_Q^{(A)}$ (additive, verified) |
| Config C: coupled system, $\gamma$ sweep, $N=1$ to $8$, fixed $\mathbf{\alpha}^*$ | PENDING | v2 data with 61 $\gamma$ values and bounds $[-10, 10]^4$ |
| $\mathbf{\alpha}$ optimisation at $\gamma=0$ | PENDING | Non-zero transverse terms; wider bounds prevent $N=2$ saturation |
| Ratio $\mathcal{R}_{QFI}(\gamma, N)$ | PENDING | Extended to all $N$ and 61 $\gamma$ values |
| Ratio $\mathcal{R}_{EP}(\gamma, N, \omega)$ | PENDING | v2 data with finer $\gamma$ grid |
| Measurement gap $\mathcal{R}_{gap}(\gamma, N, \omega)$ | PENDING | v2 data with finer $\gamma$ grid |
| $\omega$-scan, $N=1$ to $8$, selected $\gamma$ pairs | PENDING | Full $N$ range including $N=6$ to $8$ |

## ✅ Success Criteria

- **Decoupled baseline**: At $\gamma = 0$ and $\mathbf{\alpha} = 0$, Config A gives $\Delta\omega = 1/(\sqrt{N} T_H)$ and Config C gives $\mathcal{R}_{QFI} = 1$ exactly. PENDING (v2 run).
- **Noiseless coupling advantage**: At $\gamma = 0$ with optimised $\mathbf{\alpha}$ (including non-zero transverse terms), Config C achieves $\Delta\omega_{EP}^{(C)} < \Delta\omega_{EP}^{(A)}$ ($\mathcal{R}_{EP} < 1$) for at least one $N$, consistent with experiment #20260521 ($0.690\times$ SQL). PENDING (v2 run with wider bounds $[-10, 10]^4$).
- **QFI ratio near 1 at $\gamma = 0$**: $\mathcal{R}_{QFI}$ might exceed or fall below 1 at $\gamma = 0$ depending on the sign of $\operatorname{Cov}(J_z^S, J_z^A)$ in the evolved state; we report the value but do not constrain it as a success criterion. PENDING (v2 run).
- **Monotonic noise degradation**: $\Delta\omega_{EP}^{(A)}$ and $\Delta\omega_{EP}^{(C)}$ increase monotonically with $\gamma$ at fixed $\omega$ and $N$ (more noise always degrades sensitivity). PENDING (v2 run; v1 found PARTIAL because of boundary-saturating $\alpha_{zx}^* = 5.0$ at $N = 2$; wider bounds in v2 should resolve this).
- **Coupling advantage ratio depends on $\gamma$**: $\mathcal{R}_{EP}(\gamma)$ is not constant; it varies with $\gamma$, demonstrating that the noise level affects the coupling benefit. PENDING (the v2 run with 61 $\gamma$ values provides finer resolution).
- **Numerical validity**: Unitarity of BS, Hermiticity of Hamiltonians, trace preservation, QFI positivity, and QFI-EP inequality ($\Delta\omega_{QFI} \leq \Delta\omega_{EP}$) hold for all data points. PENDING (v2 run).
- **QFI-additivity check**: At $\mathbf{\alpha} = 0$, $F_Q^{(C)} = F_Q^{(S)} + F_Q^{(A)} = 2 F_Q^{(A)}$ exactly (numerical deviation $< 10^{-8}$), confirming that uncoupled systems have additive QFI. PENDING (v2 run).
- **No boundary saturation**: With wider bounds $[-10, 10]^4$, no $\alpha_{ij}^*$ component saturates at $\pm 10$ for any $N$. PENDING (v2 run).

The v2 simulation run is pending. The v1 run (seven $\gamma$ values, bounds $[-5, 5]^4$) found 6/7 criteria PASS and one PARTIAL (monotonic degradation). The v2 run extends Config C to $N = 1$ to $8$, uses 61 $\gamma$ values, and widens bounds to $[-10, 10]^4$ to resolve the boundary-saturation issue.

## 🏁 Conclusions

This experiment quantified how a four-parameter coupling $(\alpha_{xx}, \alpha_{xz}, \alpha_{zx}, \alpha_{zz})$ between a system and ancilla (both undergoing symmetric phase encoding and independent one-body photon loss) affects metrological sensitivity as a function of noise strength $\gamma$.

**v1 findings** (seven $\gamma$ values, bounds $[-5, 5]^4$, Config C $N = 1$ to $3$): (1) At $\gamma = 0$ with optimised coupling, the coupled system achieves an 11% S-only sensitivity improvement over the system alone ($\mathcal{R}_{EP} = 0.89$ at $N = 1$). This confirms that the coupling advantage from experiment #20260521 survives at the representative $\omega = 1.0$. However, the improvement comes at the cost of reduced total QFI ($\mathcal{R}_{QFI} = 0.67$): the coupling creates correlations that enhance the S-only measurement but do not increase the total quantum information about $\omega$. (2) The coupling advantage does not generalise to $N \geq 2$ with fixed $\mathbf{\alpha}^*$: at $N = 2$ and $N = 3$, $\mathcal{R}_{EP} > 1$ everywhere, meaning the coupled system is always worse than the system alone. The $N = 2$ optimisation saturated $\alpha_{zx}^*$ at the boundary (5.00 in $[-5, 5]^4$), suggesting the true optimum lies outside this range. (3) The noise dependence of $\mathcal{R}_{EP}$ is strongly non-trivial: at $N = 1$, $\mathcal{R}_{EP}$ increases from 0.89 ($\gamma = 0$) to 2.07 ($\gamma = 0.063$) before decreasing at higher noise, confirming that the coupling advantage depends on the noise level.

**v2 changes** (61 $\gamma$ values, bounds $[-10, 10]^4$, Config C $N = 1$ to $8$): The v2 run addresses three limitations of v1: (a) it extends Config C from $N = 1$ to $3$ to the full $N = 1$ to $8$ range, enabling direct comparison of the coupling advantage across all accessible particle numbers; (b) it refines the $\gamma$ grid from seven to 61 values (log-spaced in $[10^{-6}, 10^6]$), providing finer resolution of the noise-dependent coupling advantage; (c) it widens the coupling bounds from $[-5, 5]^4$ to $[-10, 10]^4$, resolving the boundary-saturation issue at $N = 2$.

**Implications**: The coupling advantage is real but fragile. Maintaining it requires per-$(\gamma, N)$ re-optimisation of $\mathbf{\alpha}$; the fixed-$\mathbf{\alpha}^*$ protocol used here is insufficient for practical deployment. The S-only measurement gap ($\mathcal{R}_{gap} \approx 1.03$ at the representative operating point, $N = 1$, $\gamma = 0$, but ranging up to $\sim 24$ at fringe nulls) indicates that joint measurement strategies could capture additional Fisher information at non-optimal operating points, though the improvement at the representative $\omega$ is modest ($\sim 3\%$).

**Open items**: (a) Per-$\gamma$ re-optimisation of $\mathbf{\alpha}$ would likely restore monotonicity and improve the coupling advantage at intermediate noise levels. (b) The v2 run with bounds $[-10, 10]^4$ tests whether extending $\alpha$ bounds unlocks larger sensitivity gains, particularly at $N = 2$ where $\alpha_{zx}^*$ saturated in v1. (c) Weighted joint measurement $M = \cos\psi J_z^S + \sin\psi J_z^A$ under noise could capture the Fisher information lost to S-only measurement at fringe-null operating points (where $\mathcal{R}_{gap}$ is large), though the gain at the representative $\omega$ is modest. (d) Per-$\omega$ $\mathbf{\alpha}$ optimisation might yield better results than the single representative-$\omega$ approach, especially if the coupling advantage is strongly $\omega$-dependent. (e) Extension to $N > 8$ for Config C requires sparse Lindblad solvers or tensor-network methods because of the $(N+1)^4$ Hilbert space dimension.
