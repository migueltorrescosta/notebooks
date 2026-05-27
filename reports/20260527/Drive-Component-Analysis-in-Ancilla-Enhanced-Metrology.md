# Drive-Component Analysis in Ancilla-Enhanced Metrology: 2D Slices and Norm-Constrained Landscape

## 🧪 Hypothesis

The original driven-ancilla experiment (2026-05-18) found that **no configuration** of $(a_x, a_y, a_z, a_{zz})$ can beat the $N=1$ SQL $\Delta\theta = 1/T_H$ when measuring $J_z^S$ on the system. The present report asks two questions about that negative result: which drive components are "important" (i.e., most strongly affect the sensitivity), and how does the available drive magnitude $\|\mathbf{a}\|$ control the best achievable sensitivity?

For a system--ancilla pair of single-particle two-mode bosonic systems with the driven-ancilla protocol (system-only BS, hold Hamiltonian $H = \theta J_z^S + a_x J_x^A + a_y J_y^A + a_z J_z^A + a_{zz} J_z^S \otimes J_z^A$, measurement $J_z^S$), the hypotheses are:

1. **Non-commuting vs commuting drive have qualitatively different sensitivity landscapes.** The 2D slice $(a_z, a_{zz})$ with $a_x = a_y = 0$ (commuting drive, $[a_z J_z^A, J_z^A] = 0$) should show a qualitatively different $\Delta\theta$ landscape from the $(a_x, a_{zz})$ and $(a_y, a_{zz})$ slices (non-commuting drive, $[a_x J_x^A, J_z^A] \neq 0$). In particular, the commuting $a_z$ drive should produce a simpler, more structured landscape because it does not generate time-dependent $J_z^A(t)$ during the hold. The best achievable ratio $\min \Delta\theta/\text{SQL}$ per slice is expected to be 1.0 (SQL) for all three slice types, consistent with the original null result.

2. **Larger drive magnitude $\|\mathbf{a}\|$ does not enable beating the SQL.** The best achievable sensitivity ratio $\min_{\|\mathbf{a}\| \leq r, \, a_{zz} \in [-5,5]} \Delta\theta/\text{SQL}$ is a non-increasing function of $r$ (larger drive budget cannot hurt), but it never drops below 1.0 (SQL) for any $r \leq 10$. The curve saturates at $\text{ratio}(r) = 1.0$ for all $r$, confirming that the SQL ceiling is absolute regardless of drive strength.

3. **Drive components are symmetric in their effect.** All three 2D slices yield identical best-ratio statistics: the minimum $\Delta\theta/\text{SQL}$ is always 1.0 (attained at $a_{zz}=0$ or when the net effect on $J_z^S$ vanishes), and the fraction of points at SQL is comparable across slice types. The differences between slices manifest only in the spatial pattern of degraded-sensitivity regions (the heatmap texture), not in the extremal values.

**Null hypothesis**: The three drive components $(a_x, a_y, a_z)$ are all equivalent in their effect on the sensitivity: no slice type achieves $\Delta\theta/\text{SQL} < 1$, and the norm-constrained envelope never dips below 1.0 for any $r \in [0, 10]$.

## ⚛️ Theoretical Model

The total Hilbert space is $\mathcal{H}_{\text{tot}} = \mathcal{H}_S \otimes \mathcal{H}_A$, where each subsystem is a **two-mode bosonic Fock space** truncated at one particle per mode. The single-particle sector $\mathcal{H}_{1} = \text{span}\{\vert1,0\rangle,\, \vert0,1\rangle\}$ (dimension 2) is isomorphic to a spin-$1/2$, and the full space has dimension 4 with ordered computational basis $\{\vert00\rangle, \vert01\rangle, \vert10\rangle, \vert11\rangle\}$ where $\vert0\rangle = \vert1,0\rangle$ (particle in mode 0) and $\vert1\rangle = \vert0,1\rangle$ (particle in mode 1). The **angular momentum operators** for each subsystem satisfy SU(2) algebra $[J_i, J_j] = i \epsilon_{ijk} J_k$ and are represented by $J_k = \sigma_k/2$ (the $2\times2$ Pauli matrices). These are embedded into the full space via Kronecker products: $J_k^S = \sigma_k/2 \otimes \mathbb{1}_2$ and $J_k^A = \mathbb{1}_2 \otimes \sigma_k/2$.

The **initial state** is the fixed product state $\vert\Psi_0\rangle = \vert1,0\rangle_S \otimes \vert1,0\rangle_A = \vert00\rangle$, the first computational basis vector.

The **circuit protocol** proceeds in four steps: (1) a 50/50 beam-splitter on the system only, $U_{\text{BS}} = \exp(-i\pi/2 J_x^S)$ acting as $U_{\text{BS}} \otimes \mathbb{1}_2$; (2) a holding period of duration $T_H = 10$ under $H = \theta J_z^S + H_A + H_{\text{int}}$, where $H_A = a_x J_x^A + a_y J_y^A + a_z J_z^A$ is the ancilla drive and $H_{\text{int}} = a_{zz} J_z^S \otimes J_z^A$ is the Ising interaction; (3) a second 50/50 system-only beam-splitter; (4) measurement of $J_z^S$ on the system.

The complete evolution is $\vert\Psi_{\text{final}}\rangle = U_{\text{BS}}^{(S)} U_{\text{hold}}(T_H) U_{\text{BS}}^{(S)} \vert\Psi_0\rangle$ with $U_{\text{hold}}(T_H) = \exp(-i T_H H)$. The **sensitivity** via error propagation is $\Delta\theta = \sqrt{\text{Var}(J_z^S)} / \vert\partial\langle J_z^S\rangle/\partial\theta\vert$, computed via central finite differences with step $\delta = 10^{-6}$. The **standard quantum limit** for $N=1$ particle is $\Delta\theta_{\text{SQL}} = 1/T_H = 0.1$, which serves as the reference ratio denominator throughout.

The **drive vector norm** is $\|\mathbf{a}\| = \sqrt{a_x^2 + a_y^2 + a_z^2}$. The norm-ball $\{\mathbf{a} \in \mathbb{R}^3 \mid \|\mathbf{a}\| \leq R\}$ constrains the total drive magnitude while allowing the direction (relative weighting of $x$, $y$, $z$ components) to vary arbitrarily. The interaction coefficient $a_{zz}$ is **not** constrained by the norm; it varies independently in $[-5, 5]$.

## 💻 Numerical Simulation

### Implementation Strategy

1. **Reuse existing infrastructure** — The core operators, circuit evolution, sensitivity computation, and 2D-slice scanning are already implemented in `src.analysis.ancilla_drive_metrology`. The existing `drive_2d_slice()` function supports `slice_type='ax'` and `slice_type='ay'`. A new `slice_type='az'` must be added (trivially: set $a_x = a_y = 0$, scan $a_z$ against $a_{zz}$).

2. **Norm-ball sampling** — For each of 50 $\theta$ values in $\{0.1, 0.2, \dots, 5.0\}$, generate $N_{\text{samp}} = 5000$ random configurations:
   - $\mathbf{a} = (a_x, a_y, a_z)$ sampled uniformly from the 3-ball $\|\mathbf{a}\| \leq R = 10$ using **Marsaglia's method** (generate three i.i.d. standard normal variates, divide by their norm, multiply by $R \cdot u^{1/3}$ where $u \sim U[0,1]$).
   - $a_{zz}$ sampled uniformly from $[-5, 5]$.
   - Evaluate $\Delta\theta$ for each configuration.
   
   This yields $50 \times 5000 = 250\,000$ evaluations total.

3. **Envelope curve extraction** — From the collected data, for each $\theta$ and each $r$ in a fine grid $\{0.1, 0.2, \dots, 10.0\}$, compute:
   - $\text{best\_ratio}(r) = \min_{i: \|\mathbf{a}_i\| \leq r} (\Delta\theta_i / \text{SQL})$, the best achievable ratio among all samples whose drive norm does not exceed $r$.
   - The envelope is a non-increasing function of $r$ (a larger drive budget includes all configurations from smaller budgets).

4. **2D slices** — Generate three sets of 201$\times$201 heatmaps:
   - $(a_x, a_{zz})$ at $\theta \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$ (replicates existing experiment)
   - $(a_y, a_{zz})$ at the same $\theta$ values (replicates existing experiment)
   - $(a_z, a_{zz})$ at the same $\theta$ values (new — no existing slice)
   
   All slices use drive coefficient range $[-5, 5]$ and $a_{zz}$ range $[-5, 5]$.

5. **Metadata recording** — Every result entry records:
   - $\theta$, $T_H$, $\|\mathbf{a}\|$, $a_x, a_y, a_z, a_{zz}$
   - $\Delta\theta$, $\langle J_z^S \rangle$, $\text{Var}(J_z^S)$, $\partial\langle J_z^S\rangle/\partial\theta$
   - $\Delta\theta/\text{SQL}$ ratio, fringe-extremum flag (whether $\Delta\theta = \infty$)
   - Experiment type (`slice_ax`, `slice_ay`, `slice_az`, `normball`)
   - Norm-ball constraint $R$ (for `normball` experiments)

### Parameter Sweep

| Parameter | Range | Purpose |
|-----------|-------|---------|
| $\theta$ (phase rate) | $0.1, 0.2, \dots, 5.0$ (50 values, step 0.1) | $\theta$-dependence of sensitivity landscape |
| $T_H$ (holding time) | 10 (fixed) | SQL reference $\Delta\theta_{\text{SQL}} = 0.1$ |
| $a_x, a_y, a_z$ (2D slices) | $[-5, 5]$ (201 pts), 2 of 3 set to 0 | Per-component slice scans |
| $a_z$ (new slice) | $[-5, 5]$ (201 pts), $a_x = a_y = 0$ | Commuting-drive slice |
| $a_{zz}$ (interaction) | $[-5, 5]$ (201 pts in slices; uniform in normball) | Ising coupling strength |
| $\|\mathbf{a}\|$ (drive norm) | $\leq 10$ (ball radius; Marsaglia sampling) | Norm-constrained landscape |
| $a_{zz}$ (normball) | $[-5, 5]$ (uniform) | Interaction in normball experiments |
| $N_{\text{samp}}$ per $\theta$ | 5000 | Norm-ball Monte Carlo density |
| $\delta$ (finite-diff. step) | $10^{-6}$ (fixed) | Derivative computation |

### Validation

- **State normalisation**: $\|\vert\Psi_0\rangle\| = 1$ and $\|\vert\Psi_{\text{final}}\rangle\| = 1$ to machine precision — verified by existing `evolve_drive_circuit`.
- **Unitarity**: $U_{\text{BS}}^\dagger U_{\text{BS}} = \mathbb{1}_2$ and $U_{\text{hold}}^\dagger U_{\text{hold}} = \mathbb{1}_4$ — verified in existing implementation.
- **Baseline recovery**: At $(a_x, a_y, a_z, a_{zz}) = (0,0,0,0)$, $\Delta\theta = 0.1$ exactly (SQL) — already verified in the original report.
- **Fringe-extremum exclusion**: Configurations with $|\partial\langle J_z^S\rangle/\partial\theta| < 10^{-12}$ are flagged as $\Delta\theta = \infty$ and excluded from best-ratio statistics.
- **Norm-ball uniformity**: The Marsaglia sampling is validated by verifying that the empirical distribution of $\|\mathbf{a}\|$ matches $P(\|\mathbf{a}\| \leq r) = (r/R)^3$ for the 3-ball within statistical tolerance (Kolmogorov--Smirnov test at 5% significance).
- **Slice consistency**: The $(a_x, a_{zz})$ and $(a_y, a_{zz})$ slices reproduce the original report's results to within $10^{-10}$.

#### 🔧 Implementation Status

To be built during the implementation phase:
- **`slice_type='az'` support** in `drive_2d_slice()` — 1-line logical addition (set $a_x = a_y = 0$, scan $a_z$).
- **`norm_ball_sampling()`** — A new function in `reports/20260527/local.py` implementing Marsaglia's method for the 3-ball, driving `drive_sensitivity_objective` for each sample, and returning a structured array with all metadata.
- **`extract_envelope_curve()`** — Post-processing: given all norm-ball data, compute $\text{best\_ratio}(r)$ for each $r$ and $\theta$, producing the envelope plot.
- **Plot: 2D slice heatmaps** — All 15 SVG heatmaps (3 slice types $\times$ 5 $\theta$ values) reusing the existing `plot_drive_2d_slice_heatmap`.
- **Plot: Norm-envelope curve** — New figure: $\min \Delta\theta/\text{SQL}$ vs $r$, with separate curves for each $\theta$ and an overall minimum across $\theta$.
- **Plot: Best-ratio-by-slice bar chart** — Comparing the minimum $\Delta\theta/\text{SQL}$ across the three slice types for each $\theta$.

Test count target: ~30 new test functions covering norm-ball sampling, envelope extraction, $(a_z, a_{zz})$ slice, floating-point stability of best-ratio computation, and Parquet roundtrip for the new dataclasses.

## ⚠️ Expected Failure Conditions

| Failure | Mitigation |
|---------|------------|
| **SQL bound holds for all slices** — The $(a_z, a_{zz})$ slice, like $(a_x, a_{zz})$ and $(a_y, a_{zz})$, never yields $\Delta\theta/\text{SQL} < 1$. | This is the expected outcome, confirming the original null result extends to the commuting-drive slice. Report the max-degradation patterns instead. |
| **Norm envelope is flat at 1.0** — The best-ratio curve equals 1.0 for all $r \in [0, 10]$, even at the largest drive magnitude. | This confirms that increasing drive amplitude does not unlock SQL violation. The envelope plot becomes a horizontal line at 1.0, which is a valid (null) result. |
| **Insufficient samples at small $\|\mathbf{a}\|$** — Marsaglia sampling yields few points with $\|\mathbf{a}\| \leq 1$ (expected fraction $\sim 10^{-3}$), making the envelope noisy at small $r$. | Accept the noisy small-$r$ region; resample with a stratified approach (uniform in $r$) if higher resolution is needed. The large-$r$ region ($r \geq 3$) will have adequate coverage. |
| **Fringe extremum dominates** — For many $\theta$ values and large $a_{zz}$, the derivative $\partial\langle J_z^S\rangle/\partial\theta$ vanishes, producing $\Delta\theta = \infty$ for most samples. | Flag and exclude fringe-extremum points. Report the fraction of valid (finite) points per $\theta$ and $r$. The envelope is computed only over finite-$\Delta\theta$ configurations. |
| **Optimal at decoupled limit** — The best ratio is always achieved at $a_{zz} = 0$, regardless of $\|\mathbf{a}\|$ or $\theta$. | This would indicate the ancilla drive is always detrimental when the interaction is active, consistent with the original report. Report best-ratio curves both with and without the $a_{zz}=0$ configuration included. |

## 🔬 Results

All experiments described below are **PENDING** — this section will be populated with actual data and figures after implementation and simulation runs.

### Pre-Experiment Summary

| Experiment | Status | Expected Outcome |
|------------|--------|------------------|
| 2D slice: $(a_x, a_{zz})$ at 5 $\theta$ | PENDING | Reproduce original report: min ratio = 1.0 |
| 2D slice: $(a_y, a_{zz})$ at 5 $\theta$ | PENDING | Identical to $(a_x, a_{zz})$ by symmetry |
| 2D slice: $(a_z, a_{zz})$ at 5 $\theta$ | PENDING | New: expected same as $(a_x, a_{zz})$ |
| Norm-ball sampling (50 $\theta$, $R=10$) | PENDING | Envelope at 1.0 for all $r$ |
| Envelope curve extraction | PENDING | Flat line at ratio = 1.0 |
| Best-ratio-by-slice comparison | PENDING | All three slices equal at ratio = 1.0 |

## ✅ Success Criteria

- **Slice equivalence** — All three slice types ($a_x$, $a_y$, $a_z$) achieve minimum $\Delta\theta/\text{SQL} = 1.0$ to within numerical precision ($10^{-8}$ relative).
- **Norm envelope bound** — $\min_{\|\mathbf{a}\| \leq r} \Delta\theta/\text{SQL} \geq 1.0$ for all $r \in [0, 10]$ and all $\theta \in [0.1, 5.0]$, confirming the SQL cannot be beaten regardless of drive magnitude.
- **Envelope monotonicity** — The curve $\text{best\_ratio}(r)$ is non-increasing (or flat at 1.0) as $r$ increases, confirming that larger drive budgets do not harm the best achievable sensitivity.
- **Commuting drive equivalence** — The $(a_z, a_{zz})$ slice's best ratio and SQL-achieving fraction are comparable to the $(a_x, a_{zz})$ slice (within 10% relative).
- **Reproducibility** — The $(a_x, a_{zz})$ and $(a_y, a_{zz})$ slices reproduce the original 2026-05-18 results: same min ratio (1.0), same SQL-achieving points count, same degradation patterns.
- **Numerical validity** — All unitarity, Hermiticity, positivity, and normalisation assertions pass. Marsaglia sampling distribution validated against the analytical $P(\|\mathbf{a}\| \leq r) = (r/R)^3$ CDF.

## 🏁 Conclusions

This report will be completed after the experiments are run. The expected conclusion is that the original null result (no SQL violation) holds across all three drive components individually and across all drive magnitudes up to $\|\mathbf{a}\| \leq 10$. The envelope curve $\text{best\_ratio}(r)$ is expected to be flat at 1.0 for all $r$, confirming that the SQL ceiling is absolute regardless of drive strength. The $(a_z, a_{zz})$ slice is expected to show a qualitatively different heatmap texture (due to the commuting drive generating different dynamics) but identical extremal statistics.

**Open items**: If the envelope curve shows any systematic deviation from 1.0 (either below, indicating a SQL violation, or above, indicating that weak drives are strictly worse), that would warrant further investigation. The norm-ball data also enables future analysis of how the **fraction** of fringe-extremum points varies with $\|\mathbf{a}\|$, potentially revealing a connection between drive amplitude and the size of the singular region in parameter space.
