# Phase-Diffusion Robustness of the Phase-Modulated Ancilla Drive Protocol

## 🧪 Hypothesis

For the phase-modulated ancilla drive protocol that achieves $\Delta\theta = 0.02036$ (4.91$\times$ below SQL at $T_H=10$, $\theta=0.2$) in ideal decoherence-free conditions, the introduction of **phase diffusion noise** on both qubits at rate $\gamma_\phi$ degrades the sensitivity continuously. The central question is whether the protocol retains a **sub-SQL advantage** for finite noise rates, and at what critical rate $\gamma_\phi^*$ the advantage is lost.

Three specific, testable claims:

1. **Finite-noise survival** — There exists a range of phase diffusion rates $0 < \gamma_\phi < \gamma_\phi^*$ where the protocol (with re-optimised parameters) still achieves $\Delta\theta < \Delta\theta_{\text{SQL}} = 1/T_H$.

2. **Critical noise rate** — The threshold rate $\gamma_\phi^*$ depends on the true phase $\theta$. At small $\theta$ (where the noise-free advantage is largest), the protocol is **more fragile** (smaller $\gamma_\phi^*$); at larger $\theta$, the advantage is more modest but may persist to higher $\gamma_\phi^*$.

3. **Optimal parameter shift** — The optimal coefficients $(\alpha_{zz}^*, a_x^*, a_y^*, a_z^*)$ shift systematically with $\gamma_\phi$. In particular, the drive amplitudes $(a_x, a_y, a_z)$ are expected to **decrease** with increasing noise (the optimiser trades off parametric gain for noise resilience), while the interaction $\alpha_{zz}$ may increase to compensate via the static BCH cross-term mechanism identified in the 2026-05-21 report.

**Null hypothesis**: Even infinitesimal phase diffusion $\gamma_\phi \to 0^+$ collapses the protocol to or above SQL. The advantage is a fragile effect that disappears in any open-system setting.

## ⚛️ Theoretical Model

The total Hilbert space is $\mathcal{H}_{\text{tot}} = \mathcal{H}_S \otimes \mathcal{H}_A$, where each subsystem is a spin-$1/2$ (single-particle two-mode bosonic subspace). The full space has dimension 4 with ordered computational basis $\{\vert00\rangle, \vert01\rangle, \vert10\rangle, \vert11\rangle\}$ where $\vert0\rangle = \vert1,0\rangle$ (particle in mode 0) and $\vert1\rangle = \vert0,1\rangle$ (particle in mode 1). Angular momentum operators satisfy SU(2) algebra and are embedded as $J_k^S = \sigma_k/2 \otimes \mathbb{1}_2$ and $J_k^A = \mathbb{1}_2 \otimes \sigma_k/2$.

The **initial state** is the pure product state $\rho_0 = \vert00\rangle\langle00\vert$ (density matrix form, since noise evolution requires mixed-state treatment).

The **circuit** proceeds in three steps:

1. **Beam splitter on system only**: $\rho \to U_{\text{BS}}^{(S)} \, \rho \, U_{\text{BS}}^{(S)\dagger}$, where $U_{\text{BS}}^{(S)} = U_{\text{BS}} \otimes \mathbb{1}_2$ with $U_{\text{BS}} = \exp(-i (\pi/2) J_x^S)$.

2. **Holding period with simultaneous encoding, theta-modulated ancilla drive, interaction, and phase diffusion**: The state evolves under the **Lindblad master equation** for duration $T_H$:
   $\dot{\rho} = -i[H, \rho] + \sum_{k \in \{S, A\}} \left( L_k \rho L_k^\dagger - \frac12\{L_k^\dagger L_k, \rho\} \right),$
   where the Hamiltonian is unchanged from the noise-free protocol:
   $H = \theta J_z^S + \theta (a_x J_x^A + a_y J_y^A + a_z J_z^A) + \alpha_{zz} J_z^S \otimes J_z^A,$
   and the phase diffusion Lindblad operators are:
   $L_S = \sqrt{\gamma_\phi} \, J_z^S = \sqrt{\gamma_\phi} \, (\sigma_z/2 \otimes \mathbb{1}_2), \qquad L_A = \sqrt{\gamma_\phi} \, J_z^A = \sqrt{\gamma_\phi} \, (\mathbb{1}_2 \otimes \sigma_z/2).$

   Each $L_k$ dephases the respective qubit at rate $\gamma_\phi$. The two Lindblad operators are independent, modelling uncorrelated phase noise on the system and ancilla.

3. **Beam splitter on system only**: $\rho \to U_{\text{BS}}^{(S)} \, \rho \, U_{\text{BS}}^{(S)\dagger}$ (same as step 1).

The **measurement** is $M = J_z^S$ on the system qubit. For a mixed state:
$\langle M \rangle = \operatorname{Tr}(M \rho_{\text{final}}), \quad \operatorname{Var}(M) = \operatorname{Tr}(M^2 \rho_{\text{final}}) - \operatorname{Tr}(M \rho_{\text{final}})^2.$

The **sensitivity** via error propagation:
$\Delta\theta = \frac{\sqrt{\operatorname{Var}(J_z^S)}}{|\partial \langle J_z^S \rangle / \partial \theta|},$
where the derivative is computed via central finite differences with step $\delta = 10^{-6}$, re-evaluating the full noisy circuit at $\theta \pm \delta$. The **standard quantum limit** is $\Delta\theta_{\text{SQL}} = 1/T_H = 0.1$.

**Units**: Dimensionless throughout. $\theta$ is the unknown phase rate, $T_H = 10$ is the holding time, $\gamma_\phi$ is the phase diffusion rate (in units of inverse time), and all Hamiltonian coefficients $(a_x, a_y, a_z, \alpha_{zz})$ are real.

## 💻 Numerical Simulation

### Implementation Strategy

1. **Liouvillian construction** — For the 4-dimensional Hilbert space, the Lindblad master equation is solved by constructing the Liouvillian superoperator $\mathcal{L}$ as a $16 \times 16$ matrix. Using **column-major vectorisation** ($\operatorname{vec}(\rho)$ stacks columns, so $\operatorname{vec}(\rho)[i + d \cdot j] = \rho[i, j]$):
   $\mathcal{L} = -i(I \otimes H - H^\mathsf{T} \otimes I) + \sum_{k \in \{S, A\}} \left[ L_k^* \otimes L_k - \frac12 \bigl(I \otimes L_k^\dagger L_k + (L_k^\dagger L_k)^\mathsf{T} \otimes I\bigr) \right].$
   The evolution is $\operatorname{vec}(\rho(T_H)) = e^{\mathcal{L} T_H} \operatorname{vec}(\rho_0)$, computed via `scipy.linalg.expm`. This is exact (no Trotter error) and avoids adaptive integrator overhead for the small Hilbert space.

2. **State preparation** — The initial density matrix is $\rho_0 = \vert00\rangle\langle00\vert$, a $4 \times 4$ positive Hermitian matrix with trace 1.

3. **Circuit evaluation** — Implement as a function `evolve_noisy_drive_circuit(rho0, T_BS, T_H, theta, gamma_phi, a_x, a_y, a_z, a_zz, ops)` that returns $\rho_{\text{final}}$:
   - `rho0` is the initial $4 \times 4$ density matrix,
   - `ops` is a dictionary of two-qubit operators from `build_two_qubit_operators()`,
   - Vectorise $\rho_0$, apply $U_{\text{BS}}^{(S)}$ via conjugation,
   - Exponentiate Liouvillian and apply to vectorised density matrix,
   - Reshape to $4 \times 4$, apply $U_{\text{BS}}^{(S)}$ again.

4. **Sensitivity computation** — At each parameter point $(\theta, a_x, a_y, a_z, \alpha_{zz}, \gamma_\phi)$:
   - Compute $\rho_{\text{final}}$ at $\theta$,
   - Compute $\langle J_z^S \rangle = \operatorname{Tr}(J_z^S \rho_{\text{final}})$ and $\langle (J_z^S)^2 \rangle = \operatorname{Tr}((J_z^S)^2 \rho_{\text{final}})$,
   - Compute $\rho_{\text{final}}$ at $\theta \pm \delta$ via two additional circuit evaluations,
   - Central difference: $\partial\langle J_z^S\rangle/\partial\theta \approx (\langle J_z^S\rangle_{\theta+\delta} - \langle J_z^S\rangle_{\theta-\delta})/(2\delta)$,
   - Return $\Delta\theta = \sqrt{\operatorname{Var}} / |\partial\langle J_z^S\rangle/\partial\theta|$.

5. **Parameter optimisation** — For each $(\theta, \gamma_\phi)$ pair, re-optimise the four free parameters $(a_x, a_y, a_z, \alpha_{zz})$ to minimise $\Delta\theta$:
   - Stage 1: Coarse 4D random search ($N_{\text{samples}}$ points in $[-5, 5]^4$),
   - Stage 2: Local Nelder-Mead refinement from top $N_{\text{refine}}$ candidates,
   - Fixed: $T_H = 10$, $T_{\text{BS}} = \pi/2$, initial state $|00\rangle$, measurement $J_z^S$.

6. **Result dataclass** — The dataclass `DriveNoiseScanResult` stores all input parameters ($\theta$ values, $\gamma_\phi$ values, $T_H$, SQL, optimisation hyperparameters `n_random`, `n_nm_refine`, `maxiter`, `bounds`, `fd_step`, `seed`) alongside computed results (optimal parameters per point, best $\Delta\theta$ per point, optimiser diagnostics). Serialised via `to_dataframe()` / `save_parquet()` with full self-describing metadata — every Parquet file stores all input parameters alongside computed arrays.

### Parameter Sweep

| Parameter | Range | Purpose |
|-----------|-------|---------|
| $\theta$ (true phase) | 10 values: 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0 | Map phase-dependence of noise collapse |
| $\gamma_\phi$ (phase diffusion rate) | 15 values log-spaced: $10^{-4}$ to $10^{1}$ | Span from near-ideal to strongly dephased |
| $(a_x, a_y, a_z, \alpha_{zz})$ | $[-5, 5]^4$ (re-optimised per $(\theta, \gamma_\phi)$) | 4D parameter space for Nelder-Mead |
| Random search samples | $N_{\text{samples}} = 1000$ per $(\theta, \gamma_\phi)$ | Coverage of 4D space |
| Nelder-Mead refinements | $N_{\text{refine}} = 25$ per $(\theta, \gamma_\phi)$ | Local optimisation from top candidates |

Total optimisations: $10 \times 15 = 150$ independent optimisation runs. Each run: 1000 random evaluations + 25 Nelder-Mead refinements (up to $\sim$5000 iterations each).

### Validation

- **Trace preservation**: $\operatorname{Tr}(\rho_{\text{final}}) = 1 \pm 10^{-8}$ for all $\gamma_\phi$, verified at every evaluation.
- **Hermiticity**: $\rho_{\text{final}} = \rho_{\text{final}}^\dagger \pm 10^{-8}$.
- **Positivity**: All eigenvalues of $\rho_{\text{final}} \ge -10^{-8}$.
- **Baseline recovery**: At $\gamma_\phi = 0$ and $(a_x, a_y, a_z, \alpha_{zz}) = 0$, the noisy circuit recovers the decoupled baseline $\Delta\theta = 1/T_H$ to machine precision. At $\gamma_\phi > 0$, the decoupled baseline is **degraded** (worse than SQL) because system-qubit phase diffusion ($L_S = \sqrt{\gamma_\phi} J_z^S$) destroys the single-qubit Ramsey coherences that the second BS converts into a population difference. Even without system--ancilla entanglement, the system qubit alone is dephased during the hold.
- **Noise-free reproduction**: At $\gamma_\phi = 10^{-4}$ (negligible noise) with re-optimised parameters, $\Delta\theta / \Delta\theta_{\text{SQL}} < 0.75$ confirming sub-SQL performance persists.
- **CSS limit**: At $\gamma_\phi \to \infty$, the evolution fully dephases the state and $\Delta\theta \to \infty$ (no information survives).

## 📊 Models Survey

| Model | Input State | Noise | Expected Behaviour | Implementation Status |
|-------|-------------|-------|--------------------|-----------------------|
| Decoupled baseline ($a_k = \alpha_{zz} = 0$) | $\vert00\rangle$ | Phase diffusion $\gamma_\phi$ | $\Delta\theta > 1/T_H$ for $\gamma_\phi > 0$ — system-qubit phase diffusion destroys single-qubit Ramsey coherences even without entanglement | PASS (implementation) |
| Optimal noise-free ($\gamma_\phi = 0$) | $\vert00\rangle$ | None | $\Delta\theta / \Delta\theta_{\text{SQL}} = 0.204$ (best case) | PASS (2026-05-19) |
| Noisy optimal ($\gamma_\phi > 0$, re-optimised $a_k, \alpha_{zz}$) | $\vert00\rangle$ | Phase diffusion $\gamma_\phi$ | $\Delta\theta / \Delta\theta_{\text{SQL}}$ increases with $\gamma_\phi$; unknown critical $\gamma_\phi^*$ | PASS (code ready, data PENDING) |
| Noisy with noise-free params ($\gamma_\phi > 0$, fixed $a_k^*, \alpha_{zz}^*$) | $\vert00\rangle$ | Phase diffusion $\gamma_\phi$ | Worse than re-optimised; diagnostic for parameter shift importance | PASS (code ready, data PENDING) |

## ⚠️ Expected Failure Conditions

| Failure | Mitigation |
|---------|------------|
| **Liouvillian positivity violation** — For large $\gamma_\phi$, the Liouvillian matrix exponential may accumulate floating-point errors $> 10^{-8}$ in positivity or trace. | Use `scipy.linalg.expm` (Pade approximation, stable for small matrices). Verify trace/Hermiticity/positivity post-evolution. Fall back to QuTiP `mesolve` with adaptive stepping if numerical issues arise. |
| **Optimiser collapse at high noise** — At large $\gamma_\phi$, many parameter points give $\Delta\theta \to \infty$, making Nelder-Mead simplex shrink to a degenerate valley. | Increase random search density at high $\gamma_\phi$. Add explicit $\Delta\theta$ upper bound ($10^3$) to prevent infinities from breaking the optimiser. Use $\texttt{penalty_scale}$ for bound enforcement. |
| **Finite-difference instability at high noise** — The derivative $\partial\langle J_z^S\rangle/\partial\theta$ becomes very small at large $\gamma_\phi$ (signal washed out), leading to $\Delta\theta$ ratios with large relative error. | The code returns $\infty$ when the derivative magnitude falls below $10^{-12}$ (numerical noise floor). Adaptive $\delta$ is not implemented — the fixed step $\delta = 10^{-6}$ suffices for the tested range because the return-$\infty$ behaviour prevents false finite $\Delta\theta$ from noisy derivatives. |
| **No critical threshold** — If the protocol collapses to SQL at the smallest $\gamma_\phi$, the null hypothesis is confirmed and no $\gamma_\phi^*$ exists. | This is a valid outcome. Report the minimum $\gamma_\phi$ tested ($10^{-4}$) and state the bound. |
| **Re-optimisation overkill** — If the optimal parameters barely shift with $\gamma_\phi$, the two-stage optimisation is wasteful. | Compare re-optimised vs fixed-parameter results. If they coincide within $1\%$ for all $\gamma_\phi$, reduce optimisation budget for future sweeps. |

## 🔬 Results

*Pre-experiment: all checks marked PENDING.*

| Check | Status |
|-------|--------|
| Finite-noise survival — $\gamma_\phi^* > 0$ exists | PENDING |
| Critical rate $\gamma_\phi^*(\theta)$ is finite and $\theta$-dependent | PENDING |
| Optimal parameters shift systematically with $\gamma_\phi$ | PENDING |
| Re-optimised beats fixed-parameters at every $\gamma_\phi$ | PENDING |
| Noise-free reproduction ($\gamma_\phi = 10^{-4}$ sub-SQL) | PASS (unit test) |
| Baseline degraded at $\gamma_\phi > 0$ (system-qubit dephasing) | PASS (unit test) |
| Trace, Hermiticity, positivity pass for all $(\theta, \gamma_\phi)$ | PASS (unit tests) |
| Parquet roundtrip with full metadata | PASS (unit tests) |
| Fail-fast on missing Parquet columns (core + diagnostics) | PASS (unit tests) |
| CSS-limit sensitivity diverges at $\gamma_\phi \to \infty$ | PASS (unit test) |

## ✅ Success Criteria

- **Finite-noise survival** — At least one $\gamma_\phi > 0$ exists where the re-optimised protocol achieves $\Delta\theta / \Delta\theta_{\text{SQL}} < 0.99$ (a $1\%$ improvement over SQL). — PENDING
- **Critical noise rate** — A clear threshold $\gamma_\phi^*$ can be identified where $\Delta\theta / \Delta\theta_{\text{SQL}} = 1.0$, and this threshold varies with $\theta$ by at least a factor of 2 across the tested $\theta$ range. — PENDING
- **Parameter trajectory** — At least two of the four optimal parameters $(a_x^*, a_y^*, a_z^*, \alpha_{zz}^*)$ show a monotonic trend with $\gamma_\phi$ over at least half the tested range. — PENDING
- **Re-optimisation benefit** — The re-optimised $\Delta\theta$ at $\gamma_\phi = 0.1$ beats the fixed-parameter (noise-free optimal) $\Delta\theta$ by at least $5\%$. — PENDING
- **Numerical validity** — Trace, Hermiticity, and positivity pass for $>99\%$ of all circuit evaluations. — PASS (unit tests over 36 parameter combinations)

If all five criteria pass, the protocol is declared **noise-robust** for the tested $\gamma_\phi$ range. If the first criterion fails (not even $1\%$ improvement at any finite $\gamma_\phi$), the null hypothesis is confirmed: the drive protocol is **fragile** under phase diffusion. A mixed result (some criteria pass, some fail) motivates a follow-up investigation into alternative dephasing models or noise-mitigation strategies.

## 🏁 Conclusions

*Pre-experiment: no conclusions yet.*

**Open items** — This report tests only phase diffusion at N=1. Natural extensions include: (a) combining phase diffusion with one-body loss (a more realistic noise model), (b) testing the protocol at N $>$ 1 with phase diffusion (connecting to the multi-particle scaling open problem), (c) comparing Lindblad phase diffusion with a stochastic-parameter (Gaussian dephasing) model to verify that the Markovian approximation does not miss essential physics.
