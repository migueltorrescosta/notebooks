# High-Order Non-Gaussian Squeezing under Decoherence: Theoretical Plan

## Hypothesis

**Core Claim**: Non-Gaussian squeezed states generated via high-order spin-dependent forces (n ≥ 3) can outperform Gaussian squeezed states (n = 2) for phase estimation in Mach-Zehnder interferometers **at fixed mean oscillator photon number ⟨a†a⟩ at the input to the MZI**, provided decoherence rates are below a critical threshold.

**Predicted Crossover**: There exists a critical decoherence rate γ_c such that:
- For γ < γ_c: F_Q(n=3 or 4) > F_Q(n=2) at fixed ⟨a†a⟩ (measured before MZI input)
- For γ > γ_c: F_Q(n=2) ≥ F_Q(n=4) (Gaussian states are more decoherence-robust)

This tests the fundamental trade-off between **non-Gaussian metrological advantage** and **decoherence susceptibility** in higher-order squeezed states.

---

## 1. Theoretical Model

### 1.1 Hybrid Oscillator-Spin System

The system consists of a bosonic oscillator (e.g., motional mode of a trapped ion) coupled to a two-level spin (qubit).

| Component | Basis | Dimension | Description |
|-----------|-------|------------|-------------|
| Oscillator | Fock states \|n⟩, n = 0…N | N+1 | Truncated bosonic Hilbert space |
| Spin | \|↓⟩, \|↑⟩ (σ_z eigenstates) | 2 | Pauli matrices σ_x, σ_y, σ_z |
| Combined | \|n⟩ ⊗ \|σ⟩ | 2(N+1) | Tensor product hybrid space |

**State ordering**: Index = n × 2 + s, where s=0 for \|↓⟩, s=1 for \|↑⟩.

### 1.2 Hamiltonian: Two Spin-Dependent Forces (SDFs)

The hybrid system is driven by two simultaneously applied SDFs:

$$
H(t) = \frac{\hbar\Omega_\alpha}{2}\sigma_\alpha\left(a e^{-i\Delta t} + a^\dagger e^{i\Delta t}\right) + \frac{\hbar\Omega_{\alpha'}}{2}\sigma_{\alpha'}\left(a e^{-im\Delta t + i\phi_{\alpha'}} + a^\dagger e^{-i(m\Delta t + \phi_{\alpha'})}\right)
$$

Where:
- $\Omega_\alpha, \Omega_{\alpha'}$: Drive strengths (dimensionless in ℏ=1 units)
- $\sigma_\alpha, \sigma_{\alpha'}$: Spin operators (σ_x, σ_y, σ_z, or σ_φ = cosφ σ_x + sinφ σ_y)
- Δ: Detuning from oscillator frequency ω_osc
- m: Order parameter for the second SDF

**Rotating frame transformation**: Transform to the frame rotating at ω_osc. After applying the Rotating Wave Approximation (RWA) assuming Δ ≫ Ω, the time-averaged Hamiltonian becomes:

$$
H_{\text{SDF}} = \frac{\hbar}{2}\left(\Omega_\alpha \sigma_\alpha \otimes a + \Omega_{\alpha'}\sigma_{\alpha'} \otimes a e^{-i(m-1)\Delta t - i\phi_{\alpha'}} + \text{h.c.}\right)
$$

**Resonance condition for n-th order squeezing**: Set m = 1 - n, yielding time-independent effective dynamics:

| n | Interaction Type | m | σ_α | σ_α' | Final Time-Independent H_n (ℏ=1, after RWA + nΔ rotating frame) |
|---|------------------|---|-----|------|---------------------------------------------------------------|
| 2 | Squeezing (Gaussian) | -1 | σ_φ | σ_{φ+π/2} | $\propto \sigma_z \otimes \left(a^2 e^{-i\theta_2} + a^{\dagger 2} e^{i\theta_2}\right)$ |
| 3 | Trisqueezing (Non-Gaussian) | -2 | σ_φ | σ_z | $\propto \sigma_{\phi+\pi/2} \otimes \left(a^3 e^{-i\theta_3} + a^{\dagger 3} e^{i\theta_3}\right)$ |
| 4 | Quadsqueezing (Non-Gaussian) | -3 | σ_φ | σ_{φ+π/2} | $\propto \sigma_z \otimes \left(a^4 e^{-i\theta_4} + a^{\dagger 4} e^{i\theta_4}\right)$ |

The time dependence from the RWA step ($e^{i n \Delta t}$) is removed via an additional rotating frame transformation at frequency $n\Delta$, yielding the time-independent Hamiltonian above.

The effective **purely bosonic** Hamiltonian (after adiabatic elimination of the spin, for reference) is:

$$
H_n = \frac{\hbar\Omega_n}{2}\left(a^n e^{-i\theta} + a^{\dagger n} e^{i\theta}\right), \quad \Omega_n \propto \frac{\Omega_{\alpha'}\Omega_\alpha^{n-1}}{\Delta^{n-1}}
$$

Where θ is an effective phase and Ω_n is the n-th order squeezing rate.

### 1.3 Decoherence Model (Lindblad Master Equation)

Applied to the full hybrid system (oscillator + spin):

| Channel | Lindblad Operator L | Rate | Physical Origin |
|---------|-------------------|------|-----------------|
| One-body loss | √γ₁ a ⊗ I₂ | γ₁ | Photon/phonon loss from oscillator |
| Phase diffusion | √γ_φ I_osc ⊗ σ_z/2 | γ_φ | Spin dephasing |
| (Optional) Two-body loss | √γ₂ a² ⊗ I₂ | γ₂ | Pair loss (not primary focus) |

**Master equation**:

$$
\frac{d\rho}{dt} = -i[H_{\text{SDF}}, \rho] + \sum_k \gamma_k\left(L_k\rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\}\right)
$$

### 1.4 Input States for Metrology

| State Type | Oscillator Part | Spin Part | Preparation |
|------------|----------------|-----------|--------------|
| Squeezed vacuum | Apply Uₙ to \|0⟩ | \|↓⟩ | H_SDF evolution for time t_sqz |
| Squeezed coherent | Apply Uₙ to \|α⟩ | \|↓⟩ | Prepare coherent state, then apply Uₙ |

**Squeezing parameter**: rₙ = Ωₙ · t_sqz (defined for each order n).

---

## 2. Simulation Requirements

### 2.1 Adaptive Truncation Rule

The oscillator truncation N must accommodate the squeezed state's photon number distribution. Use:

$$
N_{\text{osc}} = \min\left(N_{\max}, \lceil |\alpha|^2 + n \cdot r_n + 10\sqrt{|\alpha|^2 + n \cdot r_n + 1} \rceil\right)
$$

Where:
- |α| is the coherent state amplitude (0 for vacuum)
- rₙ is the effective squeezing parameter
- n is the squeezing order
- N_max is a safety upper bound

**Validation**: After each evolution step, verify ⟨a†a⟩ ≤ 0.9 · N_osc.

### 2.2 Evolution Methods

**Unitary evolution** (no decoherence):
- Solve |ψ(t)⟩ = U(t) |ψ(0)⟩ where U(t) = exp(-i H_SDF t)
- Use exact matrix exponentiation for time-independent H (or time-ordered integration if resonance condition not perfectly met)

**Lindblad evolution** (with decoherence):
- Integrate the master equation using a stable numerical method (e.g., Runge-Kutta 4)
- Ensure trace conservation, Hermiticity, and positivity at each step

### 2.3 Observables to Compute

1. **Photon number statistics**: ⟨n⟩, ⟨(Δn)²⟩
2. **Quadrature variances**: Var(x), Var(p) for n=2 comparison
3. **Wigner function**: W(x,p) = (1/π) Tr[ρ D(α) (-1)^(a†a) D†(α)]
   - Non-Gaussian states (n ≥ 3) must show Wigner negativity (min(W) < 0)
4. **Quantum Fisher Information (QFI)** for MZI phase estimation
5. **MZI interference fringe**: P₀(φ) vs phase shift φ

### 2.4 MZI Readout Protocol

After preparing the hybrid state ρ_hybrid (oscillator + spin):

1. **Two-mode embedding**: ρ₂₋ₘₒ𝚍ₑ = ρ_hybrid ⊗ \|0⟩⟨0\|_vacuum (oscillator = mode 1, vacuum = mode 2, spin remains as ancilla)
2. **MZI evolution**: 50:50 beam splitter on modes 1-2 → phase shift φ on mode 1 → 50:50 beam splitter on modes 1-2
3. **Measurement**: Output probabilities P(n₁, n₂, σ) (photon numbers n₁, n₂ + spin state σ) or P₀(φ) marginalized over spin/vacuum mode
4. **QFI computation**: Compute QFI on the full post-MZI state (including spin ancilla) using phase shift φ as the estimand.

**Phase generator for QFI**: G = n₁ ⊗ I_spin (photon number operator in mode 1, tensored with identity on spin).

---

## 3. Validation Benchmarks

### 3.1 Gaussian Squeezing (n=2) Benchmark

For n=2 with vacuum input, the (pure) state should approach a **squeezed vacuum state**:

- Quadrature variances: Var(x) = e^{-2r}/2, Var(p) = e^{2r}/2
- QFI for MZI: F_Q ≈ 4⟨n⟩ + 4|α|²e^{-2r} (pure state limit)
- Wigner function: Gaussian with elliptic contours

**Test**: Compare numerical variances and QFI with analytical formulas.

### 3.2 Trisqueezing (n=3) Benchmark

- Wigner function must be **non-Gaussian** (deviation from Gaussian shape)
- Must show **Wigner negativity** for sufficient r₃: min(W) < 0
- QFI should **exceed n=2** at the same photon number cost (hypothesis test)

### 3.3 Quadsqueezing (n=4) Benchmark

- Stronger non-Gaussianity than n=3
- Wigner function may show **multi-peaked structure**
- QFI scaling with r₄

### 3.4 Decoherence Robustness Benchmark

Compare QFI vs γ for all orders n=2,3,4:

| Regime | Prediction |
|--------|------------|
| γ = 0 (no decoherence) | F_Q(n=4) > F_Q(n=3) > F_Q(n=2) |
| Low γ | Non-Gaussian advantage persists |
| High γ | F_Q(n=2) ≥ F_Q(n=4) (Gaussian more robust) |

**Key plot**: QFI(γ) curves for each n — this is the central result testing the hypothesis.

---

## 4. Critical Theoretical Considerations

### 4.1 Spin Entanglement

The hybrid evolution entangles the oscillator with the spin. The full hybrid state (oscillator + spin) is used as input to the MZI, with the spin acting as an ancilla. QFI is computed on the full hybrid state **before tracing out the spin**:
- For pure hybrid states (no decoherence), use the pure-state QFI formula: $F_Q = 4\langle (\Delta G)^2 \rangle$
- For mixed hybrid states (with decoherence), use the symmetric logarithmic derivative (SLD) formulation for QFI on the full density matrix.

### 4.2 Time-Dependence of H_SDF

The Hamiltonian after RWA still contains the factor e^{-i(m-1)Δt} = e^{i n Δ t} for m = 1-n. To achieve a time-independent Hamiltonian:
- Apply an additional rotating frame transformation at frequency nΔ, OR
- Use time-ordered evolution (Trotter splitting) if the residual time-dependence is non-negligible

This final time-independent Hamiltonian is shown in the updated Table 1.2.

### 4.3 Adiabatic Elimination Validity

The effective bosonic Hamiltonian H_n is derived assuming **adiabatic elimination** of the spin degree of freedom. This requires:
- Δ ≫ Ω_α, Ω_α' (already assumed for RWA)
- The spin remains close to its instantaneous eigenstate

For the full hybrid simulation, this approximation is **not** made — the spin dynamics are explicitly simulated.

---

## 5. Success Criteria

The simulation is successful if:

1. ✅ **Physics validation**: n=2 results match analytical Gaussian squeezing formulas
2. ✅ **Non-Gaussian signature**: n≥3 states show Wigner negativity
3. ✅ **Hypothesis test**: QFI(n=3,4) > QFI(n=2) at zero decoherence, same ⟨n⟩
4. ✅ **Decoherence crossover**: QFI curves cross at some γ_c > 0
5. ✅ **Numerical stability**: Trace, Hermiticity, positivity conserved; no truncation artifacts

---

## 6. Implementation Status

### 6.1 Test Results Summary

| Test | Expectation | Result | Status |
|------|-------------|--------|--------|
| 1. Physics validation (n=2) | Matches analytical formulas | ⟨n⟩ = sinh²(r) ✓ | **PASS** |
| 2. Wigner negativity | min(W) < 0 for n≥3 | Formula bug fixed; retest needed | **FIXED** |
| 3. QFI scaling | QFI(4) > QFI(3) > QFI(2) | QFI(3)>QFI(2), but QFI(4)<QFI(3) | **PARTIAL** |
| 4. Decoherence crossover | QFI curves cross at γ_c > 0 | Hybrid Lindblad solver missing | **INCOMPLETE** |
| 5. Numerical stability | Trace, Hermiticity, positivity | All checks passed | **PASS** |

### 6.2 Issues Resolved

**Wigner function bug (Test 2)**: Code computed Husimi Q-function instead of Wigner function. Q(α) = ⟨α|ρ|α⟩/π is always ≥ 0. Fixed in `src/physics/wigner.py`:
```python
# Correct: W(x,p) = (2/π) exp(-2r²) Σ ρ_mn (-1)^n (α*)ᵐ αⁿ / √(m!n!)
```

### 6.3 Open Issues

**QFI scaling (Test 3)**: QFI(4) < QFI(3) contradicts plan. Possible causes:
- n=4 Hamiltonian inefficient (requires t_sqz ≈ 20-40× longer than n=2)
- Truncation N=30 may be insufficient for a⁴ operators
- May be real physics (n=4 less useful than n=3 for metrology)

**Decoherence testing (Test 4)**: Blocked by missing hybrid Lindblad solver. Required operators:
- One-body loss: √γ₁ (a ⊗ I₂)
- Phase diffusion: √γ_φ (I_osc ⊗ σ_z/2)
- Two-body loss: √γ₂ (a² ⊗ I₂)

### 6.4 Priority Actions

| Priority | Task | Status |
|----------|------|--------|
| 1 | Re-test n=3,4 Wigner negativity with fixed formula | Pending |
| 2 | Investigate QFI(4) < QFI(3) scaling | Pending |
| 3 | Implement hybrid Lindblad solver (~5 hours) | Not started |
| 4 | Run decoherence sweep, find γ_c | Blocked by #3 |

---

## Summary

This plan models the generation of **high-order non-Gaussian squeezed states** via two simultaneous spin-dependent forces on a trapped ion (or similar hybrid oscillator-spin system). The core hypothesis is that **non-Gaussian states provide metrological advantage over Gaussian states at fixed mean oscillator photon number ⟨a†a⟩ at the MZI input, but this advantage is lost at high decoherence rates**.

The simulation must model: (1) time evolution under the hybrid Hamiltonian, (2) Lindblad decoherence, (3) MZI readout with full hybrid state (spin as ancilla), and (4) QFI computation for hypothesis validation.
