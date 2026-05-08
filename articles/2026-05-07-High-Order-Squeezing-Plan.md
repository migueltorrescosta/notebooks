# High-Order Non-Gaussian Squeezing under Decoherence

## Hypothesis

Non-Gaussian squeezed states generated via high-order spin-dependent forces (n ≥ 3) can
outperform Gaussian squeezed states (n = 2) for phase estimation in Mach-Zehnder
interferometers **at fixed mean oscillator photon number ⟨a†a⟩ at the input to the MZI**,
provided decoherence rates are below a critical threshold.

There exists a critical decoherence rate γ_c such that:
- For γ < γ_c: F_Q(n=3 or 4) > F_Q(n=2) at fixed ⟨a†a⟩
- For γ > γ_c: F_Q(n=2) ≥ F_Q(n=4) (Gaussian states more robust)

---

## Physical Model

### Hybrid Oscillator-Spin System

A bosonic oscillator (Fock basis, |n=0…N⟩) couples to a two-level spin (|↓⟩, |↑⟩). The
combined Hilbert space dimension is 2(N+1). Index ordering: n × 2 + s (s=0 for |↓⟩, s=1
for |↑⟩).

### Hamiltonian: Two Simultaneous Spin-Dependent Forces

Two spin-dependent forces (SDFs) at different detunings produce an effective time-independent
Hamiltonian under the rotating wave approximation with m = 1 - n:

- **n=2 (squeezing)**: σ_z ⊗ (a² + a†²)
- **n=3 (trisqueezing)**: σ_⊥ ⊗ (a³ + a†³)
- **n=4 (quadsqueezing)**: σ_z ⊗ (a⁴ + a†⁴)

The effective bosonic rate Ω_n ∝ Ω_{α'} Ω_α^{n-1} / Δ^{n-1}.

### Decoherence Channels

| Channel | Lindblad Operator | Rate |
|---------|-------------------|------|
| One-body loss | √γ₁ (a ⊗ I₂) | γ₁ |
| Phase diffusion | √γ_φ (I_osc ⊗ σ_z/2) | γ_φ |
| Two-body loss | √γ₂ (a² ⊗ I₂) | γ₂ (optional) |

Master equation: Lindblad form dρ/dt = -i[H, ρ] + Σ_k γ_k D[L_k]ρ.

### Input States and MZI Protocol

1. Start with |0⟩ (or |α⟩) ⊗ |↓⟩.
2. Apply U_n(t) = exp(-i H_n t) for squeezing time t_sqz, giving parameter r_n = Ω_n · t_sqz.
3. Embed into two-mode MZI: ρ₂₋ₘₒ𝚍ₑ = ρ_hybrid ⊗ |0⟩⟨0|_vacuum.
4. Apply 50:50 BS → phase shift φ on mode 1 → 50:50 BS.
5. QFI is computed on the full post-MZI hybrid state (spin kept as ancilla). Phase generator:
   G = n₁ ⊗ I_spin.

---

## Numerical Simulation Techniques

### Adaptive Fock Truncation

N_osc = min(N_max, ceil(|α|² + n·r_n + 10√(|α|² + n·r_n + 1))). After each step, verify
⟨a†a⟩ ≤ 0.9 · N_osc.

### Evolution Methods

- **Unitary (no decoherence)**: Matrix exponentiation of time-independent H.
- **Lindblad (with decoherence)**: Integrate master equation with RK4 or scipy ODE solvers.
  Verify trace conservation, Hermiticity, and positivity at each step.

### Observables

- Photon number statistics ⟨n⟩, ⟨(Δn)²⟩
- Quadrature variances (for n=2 comparison with analytical formulas)
- Wigner function (must show negativity for n≥3)
- QFI for MZI phase estimation (SLD formulation for mixed states)

---

## Success Criteria

1. n=2 results match analytical Gaussian squeezing formulas.
2. n≥3 states show Wigner negativity.
3. QFI(n=3,4) > QFI(n=2) at zero decoherence, same ⟨n⟩.
4. QFI curves cross at some γ_c > 0 (decoherence crossover).
5. Numerical stability: trace, Hermiticity, positivity conserved.

---

## Current Status

| Test | Result |
|------|--------|
| n=2 physics validation | PASS — ⟨n⟩ = sinh²(r) matches analytical |
| Wigner negativity for n≥3 | PASS — confirmed for n=3 |
| QFI scaling | PARTIAL — QFI(3) > QFI(2), QFI(4) scaling under investigation |
| Decoherence crossover | READY — hybrid Lindblad solver implemented |
| Numerical stability | PASS |

**Open issue**: QFI(4) < QFI(3) contradicts the hypothesis. Possible causes: n=4 Hamiltonian
requires 20–40× longer evolution than n=2, truncation may be insufficient for a⁴ operators,
or n=4 may genuinely be less useful for metrology.

---

## Summary

This model generates high-order non-Gaussian squeezed states via two simultaneous
spin-dependent forces on a hybrid oscillator-spin system. The core hypothesis is that
non-Gaussian states provide metrological advantage over Gaussian states at fixed mean
photon number, but this advantage is lost above a critical decoherence rate. The simulation
implements time evolution under the hybrid Hamiltonian, Lindblad decoherence, MZI readout
with the spin as an ancilla, and QFI computation for hypothesis validation.
