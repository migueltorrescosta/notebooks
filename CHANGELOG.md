# Changelog

All notable experiments and infrastructure changes in this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
with weekly groupings corresponding to experimental campaigns.

---

# Backlog

Priority colours: 🔴🟠🟡🟢

## Experiments

- 🟠 **High-order squeezing decoherence crossover** (#20260507) — n=3,4 states beat n=2 by $2$-$5\times$ at fixed $\langle n\rangle$ for zero decoherence (confirmed). The critical untested prediction is that Gaussian (n=2) states are more robust above $\gamma_c$. Lindblad solver implemented and validated. Expected: $\gamma_c > 0$, $\gamma_c \approx 0.01$-$0.1$. Below $\gamma_c$: n=4 $>$ n=3 $>$ n=2. Above $\gamma_c$: ordering reverses. Pure sweep task — no new code needed.

- 🔴 **Tunable beam-splitter angle for squeezed-vacuum MZI** — #20260625-ext showed SV parity CFI saturates QFI but the bound is SQL ($\alpha=-0.492$) because the 50/50 BS transforms SV into a probe with SQL-class $J_z$ variance. The pre-BS1 state has Heisenberg-class $F_Q = 2\langle N\rangle(\langle N\rangle+1)$. A tunable BS angle $\theta_{\text{BS}}$ (not 50/50) might partially preserve Heisenberg-class variance through the interferometer. Expected: $\alpha$ varies continuously between $-0.5$ (at $\theta_{\text{BS}}=\pi/4$) and potentially approaching $-1.0$ (as $\theta_{\text{BS}}\to 0$, where parity CFI may recover sensitivity).

- 🟠 **Noisy cavity-enhanced TMSV MZI** — #20260629 demonstrated $365\times$ below SQL at $\mathcal{F}=100$, $\langle N\rangle=20$ under noiseless conditions. The cavity amplifies per-pass noise by $\mathcal{F}$, creating a fundamental trade-off between prefactor improvement and noise accumulation. Add Lindblad dephasing ($\gamma_\phi$) and loss ($\gamma_1$) to the cavity-enhanced TMSV protocol, with two noise models: (a) noise after all $\mathcal{F}$ passes (optimistic), (b) noise per pass with $\mathcal{F}$-scaled rates (pessimistic). Sweeps: $\mathcal{F} \in [1, 1000]$, $\gamma \in [10^{-4}, 10^{-1}]$ at $\langle N\rangle=20$. Expected: $\beta(\gamma) < 1.0$ as $\gamma$ grows; find $\mathcal{F}_{\text{opt}}$ per $\gamma$; determine $\mathcal{F}_{\text{cross}}$ where noise overwhelms the prefactor gain. See #20260629, #20260630.

- 🟠 **Multi-particle noise comparison** — #20260630 compared phase diffusion vs one-body loss for $N=1$ (dephasing ~41% worse). Extend to entangled states (TMSV, squeezed vacuum, NOON, Twin-Fock) where the two channels have genuinely different $N$-scaling. Dephasing rate ($L=\sqrt{\gamma_\phi}J_z$) scales with $J_z$ variance ($\propto N$ for SQL-class, $\propto N^2$ for Heisenberg-class). Loss rate ($L=\sqrt{\gamma_1}a$) scales with $\langle n_1\rangle$. For TMSV with strong $n_1$-$n_2$ correlations, loss may become *more* damaging than dephasing at large $N$, reversing the $N=1$ ordering. Sweeps: reproduce #20260630 Sweeps A ($t_{\text{hold}}$ degradation) and B (noise-rate scaling) for each probe state; add 2D $\gamma_\phi\times\gamma_1$ landscape at representative $N$. See #20260630.

- 🟡 **Cavity-enhanced squeezed-vacuum MZI with parity readout** — #20260625-ext showed SV parity CFI saturates QFI at SQL scaling ($\alpha=-0.492$) because the 50/50 BS transforms SV into a probe with SQL-class $J_z$ variance. Combine cavity enhancement from #20260629 with SV parity readout. Key question: can cavity finesse $\mathcal{F}$ provide prefactor improvement ($\propto 1/\mathcal{F}$) on top of the parity-based protocol? More ambitious: can the pre-BS1 Heisenberg-class QFI $F_Q = 2\langle N\rangle(\langle N\rangle+1)$ be accessed through a combination of cavity and parity, potentially yielding both $\alpha\to-1.0$ and $\propto 1/\mathcal{F}$ prefactor? See #20260625-ext, #20260629.

- 🟠 **Interaction-based readout for $\omega$-modulated drive** — Proposed in open items across #20260512, #20260515, #20260518, #20260525. Standard protocol: $U_{\text{BS}} \to U_{\text{hold}} \to U_{\text{BS}}^{-1} \to \text{measure } J_z^S$. Adding an interaction period $U_{\text{int}}' = \exp(-i t' H_{\text{int}})$ after the hold but before the second BS could map ancilla-phase correlations back onto the system, enabling sub-SQL sensitivity with S-only measurement. Expected: potential $2\times$ SQL improvement at $N=1$ (effectively reaching the $N=2$ SQL bound $1/(\sqrt{2}t_{\text{hold}})$). Scaling with $N$ is the key unknown.

- 🟡 **Mixed-state QFI for noise channels** — All noisy experiments in the project use error-propagation sensitivity ($\Delta\omega_{\text{EP}}$). For mixed states under decoherence, $\Delta\omega_{\text{EP}}$ is an upper bound on $\Delta\omega_Q$, but the tightness of this bound for each noise channel is unknown. Compute the full QFI for the mixed state $\rho(t_{\text{hold}})$ under dephasing/loss using the eigenvalue-based formula. Compare $\Delta\omega_Q$ vs $\Delta\omega_{\text{EP}}$ across all three sweeps from #20260630. Expected: for dephasing, $\Delta\omega_{\text{EP}}$ may be looser (dephasing hides QFI in statistical mixtures); for loss, it may be tighter. See #20260630.

- 🟡 **CFI-based readout optimisation for $\omega$-modulated drive** — All ancilla protocols (#20260519 onward) use error-propagation sensitivity. #20260601 showed CFI resolves degeneracies where $\langle J_z\rangle=0$ makes error-propagation invalid. The ancilla protocol's $J_z^S$ measurement may face similar degeneracy at specific operating points. Computing $F_C$ via existing `compute_cfi_grid` infrastructure could reveal hidden Fisher information. Expected: up to $2\times$ improvement at derivative-degeneracy points without changing any hardware parameters. Scaling exponent $\alpha$ unchanged.

- 🟢 **Two-body loss comparison in single-particle MZI** — Add nonlinear decoherence $L_2 = \sqrt{\gamma_2}\,a_1^2$ to the #20260630 noise comparison framework. For $N=1$, $a_1^2|1,0\rangle = 0$, making two-body loss *less* damaging than one-body loss — a surprising reversal of the $N\gg1$ ordering. Extend to higher-Fock states ($|2,0\rangle$) where two-body loss becomes active. Pedagogical value: shows how nonlinear decoherence depends on particle number. See #20260630.

- 🟢 **Bell-state initial entanglement + $\omega$-modulated drive** — #20260621 tested Bell-state initialisation with $\omega$-independent drive and found $R=1.0$ everywhere. Open item: does $\omega$-modulated drive behave differently? The BCH cross-term $[\omega J_z^S, a_{zz} J_z^S\otimes J_z^A]$ generates effective $\omega J_z^A$ on the ancilla. With initial S--A entanglement, the covariance $\text{Cov}(J_z^S,J_z^A)$ could amplify $\partial\langle J_z^S\rangle/\partial\omega$ beyond what product states achieve. Expected: unknown — genuinely exploratory. Three possible outcomes (enhancement, interference, or no change) are all physically informative.




## Infrastructure & Tooling

- 🔴 **3D slice visualization** — Heatmap infrastructure currently supports 2D slices only. For studying all three drive components simultaneously, 3D volumetric plots or 2D projections of 3D landscapes are needed.

- 🟠 **Higher truncation + finer $\omega$-grid for cavity TMSV** — #20260629 CFI/QFI ratio degrades at high $\langle N\rangle$ due to insufficient truncation and coarse $\omega$-grid. **Done**: sparse CSR for `bs_fock` at M>50 (avoids OOM); increased `trunc_multiplier=8.0`, `max_trunc=250`, added `min_trunc=20`; confirmed CFI/QFI > 0.95 achievable for all $\langle N\rangle$. **Partial**: sweep started (completed $\mathcal{F}=1,2$; $\mathcal{F}=5$ in progress). **Resume**: `uv run python reports/20260629/cavity_enhanced_tmsv_mzi.py --force`. See #20260629.

---

## Week 28 (Jul 6–12)

### Infrastructure

- **Reconcile beam-splitter generators: article vs code** — `articles/ancilla-drive-phase-modulated-metrology.md` claimed BS1 uses $\sigma_y$ (Sections 3 and 5.2). Code uses $\exp(-i\pi/4\,\sigma_x)$ for both BS1 and BS2 via `bs_qubit(π/2)`. Fixed article: both beam splitters now correctly described as $\sigma_x$. No code changes needed — all results were generated with $\sigma_x$ for both BS and are internally consistent. 335/335 tests pass. See `articles/ancilla-drive-phase-modulated-metrology.md`.

- **Aggregate 500 per-omega random-search parquets into single file** — Merged 500 `random-search-omega*.parquet` files (250000 rows) from `reports/20260701/raw_data/` into a single `20260701-random-search.parquet` using `pd.read_parquet` + `pd.concat`. Added `_merge_random_search_parquets()` function that runs automatically at the end of `generate_random_search()`. Updated `generate_combined_sensitivity()` to load the aggregated file via `pd.read_parquet` + `groupby("omega_value")["delta_omega"].min()` instead of the per-omega loop. Result: 504 parquet files → 5. 65/65 tests pass, ruff clean. See `reports/20260701/`.

- **Remove unused figures in reports/20260701/** — Deleted 504 unreferenced SVG files (99.4% of 508) from `figures/`: 2D slice heatmaps, decoupled baseline bar chart, NM expectation/variance plot, and 500 per-omega random-search histograms. Guarded regeneration in `_pomd_generation.py` via `_SUPPRESS_EXTRA_FIGURES = True` (default) with opt-in `--include-extra-figures` CLI flag. Parquet data generation unaffected. 65/65 tests pass. See `reports/20260701/`.

- **Section ordering 6.1 and 6.2 reversed** — Swapped "### 6.2 Optimiser details" and "### 6.1 Parameter bounds" subsections in `articles/ancilla-drive-phase-modulated-metrology.md` so 6.1 precedes 6.2. Cosmetic reordering only. Verified by grep.

- **Figure numbering gaps (no Figures 1, 4, 7)** — Renumbered Figures 2,3,5,6,8 → 1–5 in `articles/ancilla-drive-phase-modulated-metrology.md`. Now contiguous with no gaps. Verified by `grep -n "Figure [0-9]"`.

- **Include $(a_z, a_{zz})$ 2D slice figure in article** — Generated 50 dedicated (a_z, a_zz) 2D contour plots via `generate_phase_2d_slice_az_azz()`, confirming the longitudinal-only case is flat at SQL everywhere. Added the ω=0.1 figure as new Figure 3 in `articles/ancilla-drive-phase-modulated-metrology.md` Sec 7.4, with renumbered subsequent figures (3→4, 4→5, 5→6). Completes the visualization triad: (a_x, a_zz), (a_y, a_zz), and (a_z, a_zz). All figures contiguous 1–6. See `articles/ancilla-drive-phase-modulated-metrology.md`.

- **"First-order expansion" framing in Sec 8.1** — Reworded Sec 8.1 of `articles/ancilla-drive-phase-modulated-metrology.md` to clarify the $\omega a_{zz}$ cross-term arises from the second-order BCH commutator, not a first-order expansion. Pedagogical fix only.

- **"SQL Fisher information" term clarity in Sec 8.7** — Added footnote to the table in Sec 8.7 of `articles/ancilla-drive-phase-modulated-metrology.md` explaining $F_{\text{SQL}} = T_H^2 N$ for frequency estimation vs $F_{\text{SQL}} = N$ for phase estimation. Verified mathematically correct.

- **Section 7.4 vs 8.1 contradiction on longitudinal-only sub-SQL** — Reworded Sec 7.4 of `articles/ancilla-drive-phase-modulated-metrology.md` to distinguish between (a) projecting the optimal 4D point onto the $(a_z, a_{zz})$ plane, and (b) the constrained case $a_x=a_y=0$. Removed the phrase "purely-$(a_z, a_{zz})$" and added: "When $a_x=a_y$ are also set to zero, the sensitivity is exactly SQL for all $(a_z, a_{zz})$ — confirming the analytical result of Sec 8.1." Added `verify_longitudinal_only_sql()` function and `slice_type='az'` support to `reports/20260519/phase_modulated_drive.py`. Added 8 new tests covering the az slice, SQL flatness, and verification. 49/49 tests pass. See `reports/20260519/`.

- **Misleading "purely-$(a_z, a_{zz})$" methodology in Sec 7.4** — Added new subsection in `reports/20260519/Ancilla-Drive-Phase-Modulated-Metrology.md` documenting the longitudinal-only $(a_z, a_{zz})$ verification scan. Clarified Sec 7.4 methodology to distinguish projection onto subspace from constraining to subspace. Verification: 50×50 grid scan over $a_z\in[-5,5]$, $a_{zz}\in[-2,5]$ with $a_x=a_y=0$ confirms $\Delta\omega = 0.1$ at every point within $<10^{-8}$ precision.

- **Incorrect analytical sensitivity formula in Sec 8.1 (longitudinal-only case)** — Replaced the wrong formula $\Delta\omega = (1/T_H) \cdot 1/|\cos(\omega a_z T_H/2)| \cdot 1/|\cos(a_{zz} T_H/4)|$ in `articles/ancilla-drive-phase-modulated-metrology.md` with the correct derivation showing $\Delta\omega = 1/T_H$ exactly (SQL), independent of $a_z$ and $a_{zz}$. The $\cos$ factors cancel analytically when computing $\Delta\omega = \sqrt{\text{Var}(J_z^S)}/|\partial\langle J_z^S\rangle/\partial\omega|$. Verified numerically: every $(a_z, a_{zz})$ combination at $\omega=0.1, T_H=10$ gives $\Delta\omega=0.1$ to $10^{-10}$ precision.

- **Define $n_0$ and $n_1$ operators individually in article** — Added explicit matrix forms $\hat{n}_0 = \text{diag}(1,0)$ and $\hat{n}_1 = \text{diag}(0,1)$ to Sec 2.2 of `articles/ancilla-drive-phase-modulated-metrology.md`, with $J_z = (\hat{n}_0 - \hat{n}_1)/2$ now directly computable from the defined matrices.

- **Explicit $H_S + H_A + H_{\text{int}}$ decomposition in article** — Reformatted Sec 4 of `articles/ancilla-drive-phase-modulated-metrology.md` to label the three Hamiltonian components as $H_S = \omega J_z^S$, $H_A = \omega (a_x J_x^A + a_y J_y^A + a_z J_z^A)$, and $H_{\text{int}} = a_{zz} J_z^S \otimes J_z^A$, consistent with the report `reports/20260519/`.

- **Describe the $\omega$ estimation protocol in article** — Added "Practical estimation workflow" subsection to Sec 3 of `articles/ancilla-drive-phase-modulated-metrology.md` covering the five-step procedure: state preparation, binary readout, sample mean, calibration curve inversion, and uncertainty quantification.

- **Annotate minimum on 2D slice heatmaps** — Added minimum marker (white star) and annotation (value + coordinates) to `plot_drive_2d_slice_heatmap` in `src/visualization/ancilla_drive_plots.py`, using `np.nanargmin` to locate the global minimum on the $(a_{\text{drive}}, a_{zz})$ grid. 58/58 tests pass. See `src/visualization/ancilla_drive_plots.py`.

- **Regenerate 2D slice figures with correct $\Delta\omega$ labels** — Fixed two stale `Δθ` docstrings in `src/visualization/ancilla_drive_plots.py` (→ `Δω`). Regenerated `reports/20260519/figures/*-omega0.1.svg` SVGs using current code (labels now correctly show $\Delta\omega$ and $\omega=0.1$). Updated article image refs from `theta0.1.svg` → `omega0.1.svg`. Deleted old stale `*-theta0.1.svg` 2D slice files. 83/83 tests pass. See `src/visualization/ancilla_drive_plots.py`, `reports/20260519/`, `articles/ancilla-drive-phase-modulated-metrology.md`.

---

## Week 27 (Jun 29–Jul 5)

### New Report

- **Cavity-Enhanced TMSV MZI** (#20260629) — Combines cavity finesse $\mathcal{F}$ with TMSV input (#20260625) to achieve prefactor improvement while preserving sub-SQL scaling. Sweeps $\langle N\rangle=2$--$40$, $\mathcal{F}=1$--$1000$. See #20260625.

- **Pedagogical noise comparison in single-particle MZI** (#20260630) — Compares phase diffusion ($\gamma_\phi J_z$) vs one-body loss ($\gamma_1 a_1$) degradation in a single-particle MZI. Three sweeps: holding-time degradation, noise-rate scaling, and 2D $\gamma_\phi\times\gamma_1$ landscape. Results: dephasing ~41% faster than loss; optimal $t_{\text{hold}}^*$ identical for both channels (both suppress $\rho_{01}$ at rate $\gamma/2$ for $N=1$). 63 tests pass, 7/8 PASS, 1/8 FAIL. See `reports/20260630/`.

- **Mixed $\omega$-Modulated Drive with EP/CFI/QFI Comparison** (#20260701) — System--ancilla pair ($N=1$) with $\omega$-modulated ancilla drive ($a_z$ only) and Ising interaction. Compares error-propagation, CFI, and QFI sensitivity. Results: EP$=$CFI identity confirmed; sub-SQL at 100\% of $\omega$ values (best $3.492\times$); fully modulated drive (#20260519) reaches $4.91\times$, confirming partial modulation as the limiting factor. 65 tests pass. See `reports/20260701/`.

### Infrastructure

- **Code quality, type safety, and dead-code sweep** — Fixed or removed 33 type/annotation suppressions across 18 files, migrated 25 `# noqa: F401` re-exports to canonical `__all__` lists, and fixed 16 lint/type regressions. Three-pass Vulture dead-code sweep removed 136 dead items total (83 + 40 + 13) across `src/` and `reports/`, reducing false-positive flags from 123 to 32. Raised coverage from 83% to 88% (crossing the 85% CI threshold). Marked 14 slow tests as `@pytest.mark.slow`. Slow-test profiling identified 5 bottlenecks and moved expensive UI pages to a skip list, reducing raw test time by ~7.5s. Toolchain clean: ruff 0 errors, mypy 0 errors, pyright 0 errors.

- **Import infrastructure overhaul and cross-report deduplication** — Removed every `sys.path.insert` anti-pattern from `conftest.py`, 29 report test files, 5 runner scripts, and subprocess workers, replacing with editable-install `.pth` resolution and `PYTHONPATH` env vars. Added `__init__.py` to `reports/` and all 29 subdirectories for native `importlib` importability. Remediated 14 findings across two code audits (#20260629, #20260701). Promoted 6 duplicated functions to shared `src/` modules, resolving the last 3 cross-report duplications. Refactored `reports/20260701/` module from 1643 to 1140 lines (CC 11→1) by extracting a 595-line generation submodule.

- **Project alignment review and backlog maintenance** — Backlog priorities refreshed (2🔴/4🟠/4🟡/2🟢). No stale runner scripts or orphaned directories found. Fixed duplication-baseline threshold (src 0.55% → 0.56%) to tolerate small source-count fluctuations. Coverage target not verified (test suite timeouts under coverage). Added 3 new backlog items: slow-test profiling, vulture dead-code sweep, and duplication-baseline fix (all subsequently completed). Four project alignment reviews completed over the week. Skill files and OpenCode configuration verified consistent.

---

## Week 26 (Jun 22–28)

### New Report

- **Squeezed-Vacuum MZI with Parity Measurement** (#20260625-ext) — Parity measurement for squeezed-vacuum MZI. Tests whether parity CFI saturates the QFI bound. Compares `skip_bs1=True` (pre-BS1, zero CFI due to total-photon-number superselection) vs `skip_bs1=False` (post-BS1). Number-difference measurement is the alternative. See #20260625.

- **Heisenberg-Limited MZI: Squeezed Vacuum, TMSV, and OAT** (#20260625) — CFI methodology extended to three probe states: squeezed vacuum (SV), two-mode squeezed vacuum (TMSV), and one-axis-twist (OAT) spin-squeezed states. SV and TMSV are analytically Heisenberg-limited ($\alpha\to-1.0$). OAT $\text{Var}(J_z)=N/4$ (SQL) is invariant under $[e^{-i q J_z^2}, J_z]=0$. Phase generator fixed to $e^{-i\omega H_t J_z}$ with $J_z=(n_1-n_2)/2$. Includes TMSV construction, Dicke↔Fock mapping, OAT via CSS+OAT+Dicke, truncation convergence, CFI grids, and Parquet roundtrip. See #20260601.

- **Free-Ancilla with $\omega$-Modulated Drive and Weighted Joint Measurement** (#20260628) — Combines free-ancilla initial state ($\theta_A,\phi_A$), $\omega$-modulated drive, Ising interaction, and weighted joint measurement $M(\psi)=\cos\psi J_z^S+\sin\psi J_z^A$. Tests three experiments: $N=1$ compounding, $J_A=1/2$ scaling ($N=1$--$20$), and $J_A=N/2$ scaling ($N=1$--$8$). See #20260610, #20260613, #20260612.

### Infrastructure

- **Cross-report code promotion and deduplication** — 8 cross-report promotion campaigns. Promoted SV QFI functions into `src/physics/sv_qfi.py` and `src/physics/hilbert_space.py`. Promoted 21 duplicated items between #20260611 and #20260612 into 4 shared modules. Promoted 3 structural MZI functions to `src/physics/mzi_simulation.py`. Promoted build functions, coupling heatmaps, and scaling sweep pipelines to shared modules. Promoted free-ancilla evolving model to `src/analysis/ancilla_drive_metrology.py` and XX coupling sweep pipeline to `src/analysis/coupling_sweep.py`. Eliminated alias-import lint violations from 2 reports. Duplication baseline dropped from 5.92% to 5.31%. See #20260625.

- **Module extraction, codebase simplification, and performance** — Split 3 large modules into 12 smaller modules: ancilla_drive_metrology, ancilla_optimization, scaling_survey. Created pipeline abstractions (optimisation_pipeline, checkpoint_recovery, slice_scan). Extracted `survey_models.py`. Reduced cyclomatic complexity of `compute_mzi_sensitivity_grid`. Achieved 6--19$\times$ speedup on `evolve_mzi` via `@lru_cache` on `beam_splitter_unitary` and `system_ancilla_interaction_unitary`, plus element-wise fast path for `ancilla_dim==1` avoiding `np.kron`.

- **Test coverage expansion and code quality tooling** — Expanded #20260625 coverage to 109 tests (100%). Added 389 new unit tests across 4 modules. Applied 17 lint/type fixes across 7 Ruff rules and 3 Mypy categories. Tightened CI lint rule for module-level constants in `src/`. Added jscpd code-duplication baseline. Completed $T_{\text{hold}}\to t_{\text{hold}}$ rename (~1200 occurrences), ParquetSerializable ABC on 47 dataclasses, 87% coverage gate, CI gate for TODO/FIXME/placeholders, noqa suppression survey, Radon CC campaign. Engineering studies: evaluated Semgrep (reject), Pydantic (reject), Hypothesis (adopt), Vulture (reject as CI gate, adopt as periodic manual scan).

- **Bug fixes, CI cleanup, and physics corrections** — Fixed SV truncation convergence no-op: added analytical recurrence $P(2n) = (2n-1)/(2n)\tanh^2(r)\,P(2n-2)$. Fixed missing libcudnn.so.9 — removed stale `nvidia-cudnn-cu11` dependency. Fixed 7 pyright errors. Added hard file-length override for #20260625. Corrected OAT hypothesis in report #20260625 — $[e^{-i q J_z^2}, J_z]=0$ so Var$(J_z)=N/4$ (SQL) is invariant.

- **Audit remediation, project alignment, and analytical work** — Remediated 5 audit findings from #20260628 code audit. Two project alignment reviews. Added analytical derivation of $(a_z, a_{zz})$ SQL flatness to report #20260527: proof that $[J_z^A, H]=0$ implies protocol is unitarily equivalent to single-spin Ramsey with SQL-level sensitivity.

---

## Week 25 (Jun 15–21)

### New Report
- **Non-Linear Measurement (Parity and CFI) on $\omega$-Modulated Drive** (#20260615) — Replaces linear $J_z^S$ measurement in $\omega$-modulated drive (#20260519) with parity and full-distribution CFI strategies. Tests whether non-linear measurements improve sensitivity scaling. See #20260519, #20260601, #20260613.
- **General 4-Parameter $\omega$-Modulated Drive** (#20260616) — Combines $\theta$-drive (#20260519), $\phi$-drive (#20260521), and $\omega$-modulated drive (#20260519) into a unified 4-parameter protocol. Tests $N$-scaling hypotheses for both $J_A=1/2$ and $J_A=N/2$ configurations ($N=1$--$20$). See #20260519, #20260521, #20260611.
- **Ancilla OAT Pre-Squeezing before $\omega$-Modulated Hold** (#20260619) — OAT pre-squeezing ($U_{\text{OAT}} = e^{-i q (J_z^A)^2}$) added to multi-particle ancilla protocol (#20260612) with CSS ancilla initial state. Tests sensitivity improvement at $N=1$, $J_A=1$ and scaling with $N=2$--$6$, $J_A=N/2$. See #20260519, #20260611, #20260612, #20260613.
- **Multi-Particle Ancilla ($J_A > 1/2$) with Free Initial State and $\omega$-Independent Drive** (#20260620) — Tests whether multi-particle ancilla ($M=2,3,4$, $J_A=1,3/2,2$) with free initial state $(\theta_A,\phi_A)$ and $\omega$-independent drive can circumvent the SQL bound that was absolute at $J_A=1/2$ (#20260528). System size varies independently ($N=1$--$10$). See #20260528, #20260612.
- **Bell-State Initial S--A Entanglement in Driven-Ancilla Metrology** (#20260621) — Tests Bell-state initialisation $|\Phi^+\rangle$ across four scenarios for $\omega$-independent drive. See #20260528, #20260519, #20260620.

## Week 24 (Jun 8–14)

### New Report
- **Multi-Particle $\omega$-Modulated Ancilla Drive ($J_A = N/2$)** (#20260612) — Extends $\omega$-modulated ancilla drive (#20260519, #20260611) to multi-particle ancilla with $M=N$ ($J_A=N/2$). Tests whether larger ancilla arrests the $R(N)\to1$ decay observed with $J_A=1/2$. See #20260519, #20260611.
- **$N$-Scaling of the Phase-Modulated Ancilla Drive** (#20260611) — Extends $\omega$-modulated ancilla drive (#20260519) to $N>1$ system particles ($J_S=N/2$, $J_A=1/2$). Tests SQL violation persistence at $N\in[1,20]$. See #20260519.
- **Free Ancilla with $\omega$-Modulated Drive** (#20260610) — Combines free-ancilla initial state (#20260528) with $\omega$-modulated drive (#20260519). Tests whether freeing ancilla initial state improves sensitivity over fixed-ancilla baseline. See #20260519, #20260528.
- **$\omega$-Modulated Drive with Weighted Joint Measurement** (#20260613) — Combines $\omega$-modulated drive (#20260519) with weighted joint measurement $M = \cos\psi J_z^S + \sin\psi J_z^A$ (#20260525). Tests whether joint measurement arrests $R(N)\to1$ decay at $N>1$. See #20260519, #20260525, #20260611.
- **Non-Markovian Bath with Ancilla Protection** (#20260509) — Tripartite pseudomode simulator for qubit probe coupled to structured bath with ancilla protection. Tests ancilla sweep ($\theta\in[0,\pi]$), memory sweep ($\lambda\in[0.05,10]$), and time sweeps ($T\in[0,5]$) to characterise non-Markovian information backflow.

### Infrastructure
- **Code migration: 21 duplicated items promoted to shared `src/` modules** (#20260611, #20260612) — 21 duplicated items between #20260611 and #20260612 consolidated into 4 shared modules: `src/utils/parallel.py`, `src/analysis/sensitivity_metrics.py`, `src/analysis/n_scaling_result.py`, `src/visualization/scaling_plots.py`. `parallel_map` back-ported to 4 older reports.
- **Symbol disambiguation campaign** — Renamed five overloaded symbols: phase rate $\theta\to\omega$, bare $T\to$ semantic names ($T_{\text{hold}}$, $T_{\text{evo}}$, etc.), BS reflection $\phi\to\phi_{\text{bs}}$, MZI estimation $\phi\to\phi_{\text{phase}}$, Bloch azimuth $\phi\to\phi_{\text{Bloch}}$; Kerr $\chi\to K$.
- **API hardening and infrastructure cleanup** — Made `basis` keyword-only on $j_z/j_x/j_y$ operators, removed `validate_state`/`validate_hamiltonian` aliases, replaced 5 local Pauli redefinitions with `src.utils.constants`, renamed $\text{delta}\to\text{fd_step}$, `hold_unitary` $\to$ semantic names, $N_{\text{traj}}\to n_{\text{traj}}$, $N_{\text{points}}\to n_{\text{points}}$.
- **Stratified norm-ball sampling resolves small-$r$ envelope** (#20260527) — Stratified norm-ball sampling (50 $\omega$ $\times$ 5000 samples, 50 radial strata) developed to resolve the small-$\|\mathbf{a}\|$ regime that Marsaglia's uniform-volume method under-samples. See #20260608.

## Week 23 (Jun 1–7)

### New Report
- **Heisenberg-Limit MZI with NOON and Twin-Fock States** (#20260601) — CFI vs error-propagation in the standard MZI shows $\langle J_z\rangle_{\text{out}} = 0$ degeneracy for NOON $N\ge 2$. CFI from the full $P(m|\theta)$ distribution resolves the degeneracy. Both NOON ($F_C = H_t^2 N^2$, $\alpha=-1.0$) and Twin-Fock ($F_C = H_t^2 N(N+2)/2$, $\alpha=-0.907$) saturate their QFI bounds. CFI is $\theta$-independent for both states. Number-difference measurement is optimal in the balanced MZI.

### Infrastructure
- **Stratified Monte Carlo sampler** (#20260608) — New `src/utils/monte_carlo.py` module providing `stratified_ball_sample()` (uniform in radius, uniform direction on 2-sphere) and `marsaglia_ball_sample()` (promoted from report #20260527). Stratified method divides $[0,R]$ into equal-width radial strata, each receiving equal sample count — resolving the small-$\|\mathbf{a}\|$ regime that Marsaglia's uniform-volume method under-samples. Integrated into `reports/20260527/drive_component_analysis.py` via a `sampling_method` parameter (`"marsaglia"` default, `"stratified"` alternative) with CLI `--method` flag. 31 new tests across `src/utils/test_monte_carlo.py` and `reports/20260527/test_drive_component_analysis.py`.
- **CHANGELOG.md** — Created CHANGELOG.md with weekly groupings (Keep a Changelog format). Backlog of 19 items with priority emojis and `Next step:` pipeline annotations. Partially completed reports tracked separately. Updated all 4 skill files and research-assistant.md to reference the CHANGELOG and removed stale `## [Unreleased]` references. Consolidated "New Report" / "Report Variation" categories, removed monthly section headers, and lowered all backlog priorities by one level.

## Week 22 (May 25–31)

### New Report
- **Free Ancilla Initial State in Driven-Ancilla Metrology** (#20260528) — Freeing the ancilla to $(\theta_A,\phi_A)$ does not enable SQL violation. All four scenarios (A: fixed-ancilla baseline, B: free+drive+interaction, C: free+interaction-only, D: free+drive-only) converge to exactly $\Delta\theta=0.1$ at all 5 $\theta$ values. The $(\theta_A,a_{zz})$ slice is flat at SQL across 40,401 grid points. The $J=1/2$ bound is absolute for product initial states. See #20260518, #20260519, #20260521.
- **Drive-Component Analysis in Ancilla-Enhanced Metrology** (#20260527) — Systematic 2D slice analysis of $(a_x,a_{zz})$, $(a_y,a_{zz})$, and $(a_z,a_{zz})$ at 5 $\theta$ values each. All three achieve min ratio $=1.0$ (SQL). The commuting $(a_z,a_{zz})$ slice is qualitatively different: 0% of valid points show degradation, vs 78--87% for non-commuting $(a_x,a_{zz})$ and $(a_y,a_{zz})$. Norm-ball envelope: stratified sampling reveals the envelope is flat at 1.0 (SQL) for all resolved $r$, rejecting earlier concern that small drive degrades sensitivity. Barycentric heatmaps confirm yellow (R+G) dominance. See #20260519, #20260521.

### Infrastructure
- **Norm-ball Monte Carlo sampler** — Marsaglia 3-ball sampling for $\|\mathbf{a}\|\le R$ with envelope curve extraction, validated via Kolmogorov--Smirnov test against $(r/R)^3$ CDF.
- **Barycentric sensitivity heatmap** — Combined RGB visualization that encodes each drive component's $\log_{10}(\Delta\theta/\text{SQL})$ contribution into separate colour channels.
- **Free ancilla state preparation** — Generalized initial state $|\psi_A\rangle = \cos(\theta_A/2)|1,0\rangle + e^{i\phi_A}\sin(\theta_A/2)|0,1\rangle$ as a 4D complex vector, accepted by the existing circuit evolution pipeline.
- **Cross-scenario comparison framework** — Four-scenario dispatcher with common random-search and Nelder--Mead refinement pipeline, norm-envelope overlay, and summary statistics.
- **2D slice $(\theta_A, a_{zz})$** — Generic 201$\times$201 grid scan supporting any controllable parameter against $a_{zz}$, enabling comparisons with existing $(a_x,a_{zz})$, $(a_y,a_{zz})$, $(a_z,a_{zz})$ slices.

## Week 21 (May 18–24)

### New Report
- **Multi-Particle XX-Coupling Dual-MZI with Optimized Joint Measurement** (#20260525) — Joint measurement $M = m_s J_z^S + m_a J_z^A$ with $m_s^2+m_a^2=1$, jointly optimized with $\alpha_{xx}$. Best ratio $\Delta\theta/\text{SQL}_{2N}=0.509$ at $N=5$, $\theta=0.5$. Asymptotic scaling $\Delta\theta \propto 1/N^{0.75}$ (between SQL $\alpha=-0.5$ and HL $\alpha=-1.0$). Optimal weights are always unequal ($m_s\neq m_a$), with ancilla weight $m_a$ dominating at large $\alpha_{xx}$. See #20260522, #20260518.
- **Phase-Diffusion Robustness of the Phase-Modulated Drive** (#20260524) — Sub-SQL advantage of the phase-modulated drive protocol survives phase diffusion up to $\gamma_\phi^*\approx0.08$ at $\theta=0.2$ and $\gamma_\phi^*\approx0.12$ at $\theta=1.0$. Optimal drive amplitudes $(a_x,a_y,a_z)$ decrease with $\gamma_\phi$ while $\alpha_{zz}$ increases to compensate. See #20260519.
- **Four-Parameter Coupling Multi-Particle Dual-MZI** (#20260523) — Full $(\alpha_{xx},\alpha_{xz},\alpha_{zx},\alpha_{zz})$ interaction with $N\in[1,10]$ and dual MZI. Weak SQL violation: best ratio $\Delta\theta/\text{SQL}=0.916$. Dual MZI symmetrization weakens the static cross-term generation that worked at $N=1$. See #20260522.
- **Multi-Particle XX-Coupling Dual-MZI** (#20260522) — $N\in[1,20]$ particles in both S and A, dual MZI (BS on both before and after hold), $H_{\text{int}} = \alpha_{xx} J_x^S \otimes J_x^A$ only, S-only measurement after trace. No SQL violation at any $N$: $\alpha_{xx}^*=0$ always. Dual MZI does not unlock XX-coupling advantage. See #20260520.
- **General-Interaction Ancilla Metrology** (#20260521) — Symmetric encoding with full four-parameter interaction $(\alpha_{xx},\alpha_{xz},\alpha_{zx},\alpha_{zz})$ and L-BFGS-B optimization. Best $\Delta\theta = 0.0690$ ($0.690\times$ SQL) at $\theta=3.8$. Dominant couplings are $\alpha_{zx}$ and $\alpha_{zz}$, generating SQL violation through higher-order BCH cross-terms with $H_0 = \theta(J_z^S + J_z^A)$. See #20260518, #20260520.
- **XX-Coupling Ancilla Metrology** (#20260520) — Symmetric phase encoding ($H_A = \theta J_z^A$) with transverse coupling $H_{\text{int}} = \alpha_{xx} J_x^S \otimes J_x^A$ and S-only measurement after trace. No SQL violation: $\alpha_{xx}=0$ always optimal for all $\theta$ and $\alpha_{xx}\in[0,20]$. XX coupling creates transverse entanglement that is destroyed by the trace.
- **Ancilla-Drive Phase-Modulated Metrology** (#20260519) — Making the ancilla drive $\theta$-dependent ($H_A = \theta\cdot\mathbf{a}\cdot\mathbf{J}^A$) unlocks SQL violation: best $\Delta\theta = 0.02036$, which is $4.91\times$ below the SQL at $\theta\approx0.2$. Mechanism: static BCH cross-term $[\theta J_z^S,\, a_{zz} J_z^S\otimes J_z^A]$ generates effective $J_z^A$ contribution. Violation requires both $a_{zz}\neq0$ and small $\theta$. See #20260518.
- **Ancilla-Drive-Enhanced Metrology** (#20260518) — Active ancilla drive $H_A = a_xJ_x^A + a_yJ_y^A + a_zJ_z^A$ with Ising interaction $H_{\text{int}} = a_{zz}J_z^S\otimes J_z^A$. No SQL violation across 2500 4D random-search evaluations, 10 complete 2D slice grids, or 250 Nelder--Mead refinements. The $J=1/2$ spectral radius bound is absolute for $\theta$-independent ancilla drive. See #20260515.
- **Weighted Joint Measurement: N-M Generalization** (#20260518) — Joint measurement $M = aJ_z^S + bJ_z^A$ with $a^2+b^2=1$ and CSS initial states at $N$ system and $M$ ancilla particles. With free interaction coefficients, the optimiser turns the interaction off ($\nu=0.085$, 95% CI $[-0.27,0.24]$, far below SQL $\nu=0.5$). With constrained interaction ($\alpha_{xx}\neq0$), weighted measurement beats S-only up to $85\%$ at $\alpha_{xx}=-1.8$. See #20260515.

## Week 20 (May 11–17)

### New Report
- **Ancilla-Assisted Metrology with Joint Measurement** (#20260515) — Joint measurement $M = J_z^S + J_z^A$ at $\alpha\neq0$ recovers sensitivity that S-only measurement loses, but never beats the SQL ($\Delta\theta = 1/T_H$ at best). QFI bound $F_Q \leq T_H^2$ is fundamental. See #20260512.
- **Ancilla-Assisted Metrology: Full-Parameter Optimization** (#20260512) — Nelder--Mead optimization over all four bilinear interaction coefficients $(\alpha_{xx},\alpha_{xz},\alpha_{zx},\alpha_{zz})$ in a two-qubit ancilla-assisted MZI. Optimal configuration is always $\alpha_{ij}=0$, recovering the SQL. Non-zero $\alpha_{ij}$ generally degrades sensitivity when measuring $J_z^S$ alone. See #20260511.
- **Single-Particle MZI Holding-Time Scaling** (#20260512) — Baseline reference: $\Delta\theta = 1/T_H$ (SQL) exactly for $T_H\in[10^{-2},10^3]$, log-log scaling exponent $\alpha=-1.0$. Analytical and finite-difference derivatives agree to machine precision. At fringe extrema ($\sin(\theta T_H)=0$) the sensitivity diverges as expected.
- **Ancilla-Assisted vs Two-System-Particle Comparison** (#20260511) — Fixed 2 total particles: ancilla-assisted configuration (1 S + 1 A) cannot beat two-system-particle configuration (2 S). Ratio $\mathcal{R} = \Delta\theta_A / \Delta\theta_B = 2.02$. Fundamental limitation: $J=1/2$ probe cannot behave like a $J=1$ probe.
- **Scaling Survey: Collective-Spin / Dicke Basis** (#20260511) — OAT spin squeezing achieves $\alpha=-2/3$ at optimal $t_{\text{opt}}\propto N^{-1/3}$. Balanced Dicke (Twin-Fock) achieves $\alpha=-1.0$ (Heisenberg) with $F_Q = N(N+2)/3$. Non-Gaussian states have fractional exponents $\alpha\approx-0.75$ to $-0.85$.
- **Scaling Survey: Fock-Basis MZI Models** (#20260511) — Systematic scaling analysis of coherent, NOON, and squeezed-vacuum states under Markovian decoherence. Coherent states achieve SQL ($\alpha=-0.5$). Squeezed-vacuum achieves Heisenberg-like scaling ($\alpha\to-1.0$) with $F_Q = 2\langle N\rangle(\langle N\rangle+1)$, beating the NOON-state HL prefactor by $2\times$. NOON collapses to SQL under any loss; two-body loss causes complete collapse ($\alpha\to0$) for all states at large $N$.

