---
name: physics-conventions
description: Reference for consistent physics symbol usage across the codebase. Load alongside build-simulation, audit-code, specify-experiment, or generate-results to ensure new variables, dataclass fields, and LaTeX symbols follow established conventions.
---

# Rules

1. Import Pauli matrices from `src.utils.constants` — do not redefine them locally.
2. Use the `OperatorBasis` enum (`DICKE` vs `FOCK`) explicitly when constructing angular momentum operators; never hard-code the ordering convention.
3. Never use bare `|` inside LaTeX inline math `$...$` — it is ambiguous with the Markdown table column delimiter. Use `\vert ` (with trailing space) for conditional probability (`$P(m\vert \omega)$`), absolute values (`$\vert \psi\rangle$`), and set-builder notation (`$\{x\vert x>0\}$`). This applies everywhere in `.md` files, not just inside tables.

# Reference: Physics Variables

## Interferometry and Phase Estimation

- **θ** (theta) — Beam-splitter rotation angle (`π/4` = 50/50); also polar angle on Bloch sphere for CSS / initial states. Also Pegg-Barnett phase angle (`lindblad_solver.py`), squeezing phase (`High_Order_Squeezing.py`), binomial probability (`Fisher_information.py`).
- **φ** (phi) — MZI phase shift (the parameter being estimated via the interferometer). `phi_bs` is BS reflection phase; `phi_Bloch` is Bloch-sphere azimuth. Measurement weight renamed to `psi_opt`.
- **ω** (omega) — Unknown phase rate (the parameter to be estimated in ancilla-driven metrology). Sensitivity reported as `Δω`. Also Rabi frequency (ω_k) and squeezing rate (Ω_n).
- **Φ** (Phi, capital) — CSS azimuthal angle (distinct from lower-case φ). Preserved unchanged during φ→ψ rename.
- **ψ** (psi) — Quantum state vector / wavefunction. Also measurement weight (`psi_opt`).
- **Δω** — Sensitivity (standard deviation of estimate) via error propagation. SQL comparison: `Δω_SQL = 1/t_hold`.
- **F_Q** — Quantum Fisher Information.
- **F_C** — Classical Fisher Information (from full probability distribution).
- **SQL** — Standard Quantum Limit: `1/√N` (particle scaling) or `1/t_hold` (time scaling). Document which definition applies.
- **HL** — Heisenberg Limit: `1/N`.

## Particle Number and Angular Momentum

- **N** — Number of particles / atoms (Dicke dimension `N+1`); also Fock truncation (max photons). Both control Hilbert space size.
- **J** — Total angular momentum `J = N/2`; also 1D Heisenberg coupling strength.
- **J_x, J_y, J_z** — Angular momentum operators (`J_k = σ_k/2` for qubit). Canonical definitions in `src.utils.constants` as `J_X, J_Y, J_Z`.
- **m** — Dicke basis eigenvalue (`m = -J..+J`, descending order). Also mass in `Numerical_Quantum_Time_Evolution.py`.
- **σ** (sigma) — Pauli matrices (`σ_x, σ_y, σ_z`). Canonical definitions in `src.utils.constants` as `SIGMA_X, SIGMA_Y, SIGMA_Z`. Do not redefine locally.
- **k** — Ancilla Fock level index; momentum (quantum time evolution).
- **n̄** (n_bar) — Mean photon number `|α|²`.

## Noise and Dissipation

- **γ** (gamma) — Loss/dissipation rate: `γ₁` (one-body), `γ₂` (two-body), `γ_φ` (phase diffusion). Also pseudomode decay rate. Always subscripted to disambiguate channel. Dimensionless as `γ·T` product.
- **η** (eta) — Detection efficiency (`NoiseConfig.eta`, `η ∈ [0,1]`).
- **λ** (lambda) — Decay rate; wavelength; system-ancilla coupling; expansion coefficient. Heavily overloaded; disambiguate by context.
- **T_decay** — Decoherence / dissipation evolution time (Lindblad solver).

## Time Parameters

- **t_hold** — Holding (evolution) time under the hold Hamiltonian. Sets the SQL: `Δω_SQL = 1/t_hold`. Lower-case for PEP 8 compliance.
- **T_BS** — Beam-splitter pulse duration (defines rotation angle).
- **T_kerr** — Kerr evolution / squeezing time.
- **T_evo** — Generic evolution time (TDVP, TWA, BEC). Used only when no specific mechanism dominates.
- **T_dd** — Dynamical decoupling interval.
- **temp** — Temperature (thermal state / PSD context). Plain name, not `T_temp`.
- **t** — Generic evolution / squeezing time (lower-case).
- **dt** — Time step (integration / Trotter).

## Hamiltonian Coefficients and Drive

- **a_x, a_y, a_z** — Ancilla drive amplitudes (linear Hamiltonian coefficients: `H_A = a_x J_x^A + a_y J_y^A + a_z J_z^A`). Used consistently across all ancilla-metrology modules.
- **a_zz** — Ising interaction coefficient (`H_int = a_zz J_z^S ⊗ J_z^A`).
- **α_xx, α_xz, α_zx, α_zz** — Bilinear interaction coefficients in the general ancilla-system Hamiltonian. Convention: first subscript = system operator, second = ancilla operator.
- **j_s, δ_s** — System parameters: `j_s` = transverse field (σ_x coeff), `δ_s` = on-site energy (σ_z coeff). Used uniformly in `delta_estimation.py` and ancilla metrology modules.
- **χ** (chi) — OAT (one-axis twisting) squeezing strength (nonlinear coupling `χ J_z²`). Also Kerr nonlinearity coefficient (use `K` instead in `kerr_mzi.py`).
- **K** — Kerr coefficient (alternative to χ). Only in `kerr_mzi.py`.
- **δ** (delta) — Detuning / on-site energy (`δ_S`, `δ_A`). Also finite-difference step (`fd_delta`).

## Context-Dependent Symbols

- **α** (alpha) — Most overloaded symbol: coherent amplitude (`|α|² = n̄`), scaling exponent (`Δφ ∝ N^α`), bilinear interaction coefficients (`α_xx`), generic coupling in hybrid systems. Always comment its meaning.
- **ε** (epsilon) — Numerical tolerance / clipping threshold; SVD truncation threshold.
- **κ** (kappa) — Concentration parameter for von Mises prior distribution. Only in `src/analysis/bayesian_statistics.py`.
- **ξ** (xi) — Squeezing parameter: `ξ² = Var_min/(N/4)` (spin squeezing); `ξ = √(Var_min/(N/4))`. Also used in truncated Wigner context.
- **r_n** — High-order squeezing parameter `r_n = Ω_n · t_sqz`. Only in `High_Order_Squeezing.py`.
- **Ω_n** — High-order squeezing rate (Hamiltonian strength). Only in `High_Order_Squeezing.py`.


