# Implementation Plan: BEC Interferometer & Ancilla Simulations

## Overview

This plan implements the missing quantum metrology simulations from `/home/miguel/Git/simulations` into `/home/miguel/Git/notebooks`. The implementation is organized into **4 phases** with **12 tasks**, each containing self-contained instructions for a coding agent.

---

## Phase 1: Foundational Physics Modules

### Task 1: Dicke Basis & Collective Spin Operators

**Location**: `src/dicke_basis.py` (new file)

**Goal**: Implement the Dicke basis representation for N-atom two-mode systems, enabling efficient computation of collective spin operators and Lindblad dynamics.

**Physical Model**:
- Hilbert space: Dicke basis $|J, m⟩$ with $J = N/2, m ∈ {-J, ..., J}$, dimension $d = N+1$
- Collective spin operators: $J_x = (a†b + b†a)/2$, $J_y = (a†b - b†a)/(2i)$, $J_z = (a†a - b†b)/2$
- Fock basis mapping: $|n₁, n₂⟩ ↔ |J, m⟩$ where $m = (n₁ - n₂)/2$

**Instructions**:
1. Create `src/dicke_basis.py` with:
   - ` dicke_states(N: int) -> dict`: Generate Dicke basis states mapping m → fock_index
   - `to_dicke_basis(fock_state: np.ndarray, N: int) -> np.ndarray`: Convert Fock to Dicke basis
   - `from_dicke_basis(dicke_state: np.ndarray, N: int) -> np.ndarray`: Convert Dicke to Fock basis
   - `jz_eigenvalues(N: int) -> np.ndarray`: J_z eigenvalues array
   - `jz_operator(N: int) -> np.ndarray`: Dense J_z matrix
   - `jx_operator(N: int) -> np.ndarray`: Dense J_x matrix using lowering/raising operators

2. Add tests in `src/test_dicke_basis.py`:
   - Test unitarity of basis transformations
   - Test J_z eigenvalues: m ∈ [-N/2, N/2] with step 1
   - Test [J_i, J_j] = iε_ijk J_k commutation relations
   - Test J²|m⟩ = J(J+1)|m⟩ for J = N/2

3. Validation:
   - Assert basis dimension is N+1
   - Assert J_z is diagonal with correct eigenvalues
   - Assert J_x is off-diagonal with correct matrix elements

---

### Task 2: Lindblad Master Equation Solver

**Location**: `src/lindblad_solver.py` (new file)

**Goal**: Implement open quantum system dynamics via the Lindblad master equation, enabling simulation of decoherence effects (one-body loss, two-body loss, phase diffusion).

**Physical Model**:
- Master equation: dρ/dt = -i[H, ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
- Lindblad operators: L_1 = √γ₁a, L_2 = √γ₂a², L_φ = √γ_φ J_z
- Time evolution via 4th-order Runge-Kutta or scipy ODE solver

**Instructions**:
1. Create `src/lindblad_solver.py` with:
   ```python
   @dataclass
   class LindbladConfig:
       N: int
       gamma_1: float = 0.0      # one-body loss rate
       gamma_2: float = 0.0      # two-body loss rate
       gamma_phi: float = 0.0    # phase diffusion rate
       chi: float = 0.0          # OAT squeezing strength
   
   def lindblad_liouvillian(rho: np.ndarray, H: np.ndarray, L_ops: list[np.ndarray], 
                            gammas: list[float]) -> np.ndarray:
       """Compute dρ/dt from Lindblad master equation."""
   
   def evolve_lindblad(initial_rho: np.ndarray, config: LindbladConfig, 
                      T: float, dt: float) -> np.ndarray:
       """Time-evolve density matrix under Lindblad equation."""
   
   def steady_state(H: np.ndarray, L_ops: list[np.ndarray], 
                   gammas: list[float]) -> np.ndarray:
       """Compute steady-state density matrix."""
   ```

2. Implement:
   - Vectorized Liouvillian action for performance
   - Sparse matrix support for N > 100
   - Adaptive timestep for stiff equations

3. Add tests in `src/test_lindblad_solver.py`:
   - Test trace preservation: Tr[ρ(t)] = 1 for all t
   - Test hermiticity: ρ = ρ†
   - Test positivity: eigenvalues of ρ ≥ 0
   - Test conservation under no-loss: particle number preserved
   - Test phase diffusion: off-diagonal decay rate matches γ_φ

4. Validation:
   - Run with γ_1=γ_2=γ_φ=0, verify unitary evolution
   - Run with only phase diffusion, verify dephasing timescale

---

### Task 3: Truncated Wigner Approximation (TWA)

**Location**: `src/truncated_wigner.py` (new file)

**Goal**: Implement semi-classical phase-space method for efficient simulation of large-N BEC interferometers without exponential Hilbert space growth.

**Physical Model**:
- SU(2) Wigner function on Bloch sphere
- Stochastic differential equations for Bloch vector (x, y, z) = (⟨J_x⟩/J, ⟨J_y⟩/J, ⟨J_z⟩/J)
- Quantum jumps modeled as noise terms
- Average over N_traj trajectories

**Instructions**:
1. Create `src/truncated_wigner.py` with:
   ```python
   def sample_wigner_sphere(N: int, state_type: str, rng: np.random.Generator,
                            phi: float = 0.0) -> np.ndarray:
       """Sample initial Bloch vector from Wigner function.
       
       Args:
           N: Total atom number.
           state_type: State type ('CSS', 'SSS', 'NOON').
           rng: Random number generator.
           phi: Phase angle for CSS initial state (default 0.0).
       
       Returns:
           Bloch vector of shape (3,).
       """
   
   def wigner_sde_trajectory(J_init: np.ndarray, params: dict, T: float, 
                            dt: float, rng: np.random.Generator) -> dict:
       """Propagate single trajectory via SDEs."""
   
   def compute_twa_expectations(N: int, state_type: str, params: dict, T: float, 
                               N_traj: int = 5000, rng_seed: int = 42) -> dict:
       """Compute ⟨J_z⟩ and Var(J_z) via TWA."""
   ```

2. Implement:
   - CSS initial conditions: uniform distribution on circle (sampling from Wigner for coherent state)
   - SSS initial conditions: Gaussian narrowing in x-direction
   - NOON: bimodal distribution (not reliable, flag warning)
   - SDE integration: Euler-Maruyama method with quantum jump terms

3. Add tests in `src/test_truncated_wigner.py`:
   - Test CSS agreement with Lindblad for small N (N ≤ 20)
   - Test trajectory normalization: Bloch vector length ≈ 1
   - Test output statistics converge with N_traj

4. Validation:
   - Compare ⟨J_z⟩(φ) with Lindblad results for CSS at N=10, tolerance 5%
   - Verify scaling agreement: Δφ_TWA vs Δφ_Lindblad for CSS

---

## Phase 2: Quantum States & Input Preparation

### Task 4: Spin-Squeezed States (One-Axis Twisting)

**Location**: `src/spin_squeezing.py` (new file)

**Goal**: Implement one-axis twisting (OAT) Hamiltonian for generating spin-squeezed states, a key resource for sub-SQL phase estimation.

**Physical Model**:
- OAT Hamiltonian: H = χ J_z²
- Squeezing parameter: ξ = √(N Var(J_⊥))/⟨J_z⟩
- Optimal squeezing time: χt_opt ≈ (6/N)^(1/3)
- Squeezing degree: ξ_squeeze ∝ N^(-2/3)

**Instructions**:
1. Create `src/spin_squeezing.py` with:
   ```python
   def coherent_spin_state(N: int) -> np.ndarray:
       """CSS: all atoms in symmetric superposition |J, -J⟩_x."""
   
   def one_axis_twist(initial_state: np.ndarray, N: int, chi: float, 
                     t: float) -> np.ndarray:
       """Apply OAT: U = exp(-i χ t J_z²)."""
   
   def squeezing_parameter(state: np.ndarray, N: int) -> float:
       """Compute squeezing parameter ξ = √(N Var(J_perp))/|⟨J⟩|."""
   
   def optimal_squeezing_time(N: int, chi: float) -> float:
       """Return t_opt ≈ (6/N)^(1/3) / χ."""
   
   def generate_squeezed_state(N: int, chi: float, t: float) -> np.ndarray:
       """Generate CSS then apply OAT."""
   ```

2. Implement:
   - Efficient J_z² construction in Dicke basis (diagonal)
   - Expm action via diagonal unitary: U|m⟩ = exp(-iχt m²)|m⟩
   - Squeezing parameter computation: perpendicular variance from J_x, J_y

3. Add tests in `src/test_spin_squeezing.py`:
   - Test CSS is unentangled: ξ ≥ 1
   - Test OAT generates squeezing: ξ < 1 for χt > 0
   - Test optimal time scaling: t_opt ∝ N^(-1/3)
   - Test squeezing magnitude: ξ_min ∝ N^(-2/3)

4. Validation:
   - Assert ξ_CSS = 1 exactly
   - Assert ξ_SSS < 1 for t > 0
   - Verify t_opt matches theoretical prediction to 1%

---

### Task 5: Twin-Fock & NOON States

**Location**: `src/mzi_states.py` (new file, extends mzi_simulation.py)

**Goal**: Implement Twin-Fock and NOON states as input resources for Heisenberg-limited interferometry.

**Physical Model**:
- Twin-Fock: |N/2, N/2⟩_Fock = |J, 0⟩_Dicke (balanced two-mode Fock)
- NOON: (|N, 0⟩ + |0, N⟩)/√2 = (|J, J⟩ + |J, -J⟩)/√2
- Twin-Fock Fisher information: F_Q = N²(1+1/N)/2 ≈ N²/2
- NOON achieves Heisenberg limit F_Q = N²

**Instructions**:
1. Create `src/mzi_states.py` with:
   ```python
   def twin_fock_state(N: int) -> np.ndarray:
       """Twin-Fock: |N/2, N/2⟩ in Fock basis."""
   
   def noon_state(N: int) -> np.ndarray:
       """NOON: (|N,0⟩ + |0,N⟩)/√2."""
   
   def coherent_state(alpha: complex, N: int) -> np.ndarray:
       """Coherent state in two-mode basis."""
   
   def input_state_factory(state_type: str, N: int, **kwargs) -> np.ndarray:
       """Factory: 'css', 'sss', 'twin_fock', 'noon', 'coherent'."""
   ```

2. Add tests in `src/test_mzi_states.py`:
   - Test Twin-Fock normalization
   - Test NOON has equal overlap with |N,0⟩ and |0,N⟩
   - Test all states are orthogonal to incorrect Fock states

3. Validation:
   - Assert ⟨J_z⟩ = 0 for Twin-Fock (symmetric)
   - Assert Var(J_z) = N²/4 for NOON
   - Assert Fisher information for NOON = N²

---

### Task 6: Noise Channel Implementations

**Location**: `src/noise_channels.py` (new file)

**Goal**: Implement physical noise channels (one-body loss, two-body loss, phase diffusion, detection noise) with validated Lindblad operators.

**Physical Model**:
| Channel | Lindblad Operator | Rate Symbol |
|---------|------------------|-------------|
| One-body loss | L = √γ₁ a | γ₁ (s⁻¹) |
| Two-body loss | L = √γ₂ a² | γ₂ (s⁻¹ per pair) |
| Phase diffusion | L = √γ_φ J_z | γ_φ (s⁻¹) |
| Detection noise | P(k\|n) = Binomial(n, η) | η (efficiency) |

**Instructions**:
1. Create `src/noise_channels.py` with:
   ```python
   @dataclass
   class NoiseConfig:
       gamma_1: float = 0.0    # one-body loss
       gamma_2: float = 0.0    # two-body loss
       gamma_phi: float = 0.0  # phase diffusion
       eta: float = 1.0        # detection efficiency
   
   def build_lindblad_operators(N: int, config: NoiseConfig) -> list[np.ndarray]:
       """Return list of Lindblad operators for given noise channels."""
   
   def apply_detection_noise(probabilities: np.ndarray, eta: float, 
                             n_trials: int) -> np.ndarray:
       """Convolve with binomial for detection inefficiency."""
   ```

2. Add tests in `src/test_noise_channels.py`:
   - Test one-body loss conserves total probability initially, then decays
   - Test two-body loss rate scales with N(N-1)
   - Test phase diffusion causes off-diagonal decay proportional to m²

3. Validation:
   - Verify Lindblad operators satisfy completeness relation
   - Verify detection noise: Binomial(n, η=1) = identity

---

## Phase 3: Sensitivity Estimation & Analysis

### Task 7: Fisher Information Computation

**Location**: `src/fisher_information.py` (new file)

**Goal**: Implement quantum and classical Fisher information computation for phase sensitivity analysis.

**Physical Model**:
- Classical Fisher Information: F_C(φ) = Σ [∂P(m|φ)/∂φ]² / P(m|φ)
- Quantum Fisher Information (QFI): F_Q = 4 Var(H_gen) for pure states
- Phase sensitivity: Δφ = 1/√F_C (classical), Δφ = 1/√F_Q (quantum)

**Instructions**:
1. Create `src/fisher_information.py` with:
   ```python
   def classical_fisher_information(probabilities: np.ndarray, dphi: float) -> float:
       """Compute F_C via numerical derivative."""
   
   def quantum_fisher_information(state: np.ndarray, generator: np.ndarray) -> float:
       """Compute F_Q for pure state: F_Q = 4 Var(G)."""
   
   def quantum_fisher_information_dm(rho: np.ndarray, generator: np.ndarray) -> float:
       """Full QFI for mixed state via eigen-decomposition."""
   
   def phase_sensitivity_from_fisher(F: float) -> float:
       """Δφ = 1/√F."""
   ```

2. Add tests in `src/test_fisher_information.py`:
   - Test CSS: F_Q = N (SQL)
   - Test NOON: F_Q = N² (Heisenberg)
   - Test Squeezed: F_Q > N (sub-SQL)

3. Validation:
   - Assert F_C ≤ F_Q always
   - Assert F_C(φ) > 0 for all φ where P(m|φ) > 0
   - Verify Δφ_CSS = 1/√N to within 1%

---

### Task 8: Bayesian Phase Estimator

**Location**: `src/bayesian_phase_estimation.py` (new file)

**Goal**: Implement Bayesian inference for phase estimation from measurement outcomes, computing posterior distributions and sensitivity bounds.

**Physical Model**:
- Prior: π(φ) = 1/(2π) uniform on [0, 2π)
- Likelihood: P(m|φ) from interferometer output distribution
- Posterior: P(φ|m₀) ∝ P(m₀|φ)π(φ)
- Sensitivity: Δφ_B = Std[φ|m₀]

**Instructions**:
1. Create `src/bayesian_phase_estimation.py` with:
   ```python
   def bayesian_posterior(measurement_outcome: int, state: np.ndarray, 
                         N: int, prior_range: np.ndarray) -> np.ndarray:
       """Compute P(φ|m₀) over discretized φ grid."""
   
   def bayesian_sensitivity(posterior: np.ndarray, phi_grid: np.ndarray) -> float:
       """Compute posterior standard deviation."""
   
   def sample_measurement_outcomes(state: np.ndarray, phi_true: float, 
                                   n_samples: int, rng: np.random.Generator) -> np.ndarray:
       """Draw measurement outcomes from P(m|φ_true)."""
   
   def bayesian_estimator(outcomes: np.ndarray, state: np.ndarray, N: int) -> dict:
       """Full Bayesian estimation pipeline."""
   ```

2. Add tests in `src/test_bayesian_phase_estimation.py`:
   - Test posterior normalization: Σ P(φ|m) = 1
   - Test prior is recovered when likelihood is flat
   - Test sensitivity decreases with more measurements

3. Validation:
   - Verify posterior peaks near φ_true
   - Compare Δφ_B with Cramér-Rao bound

---

### Task 9: Sensitivity Metrics & Comparison

**Location**: `src/sensitivity_metrics.py` (new file)

**Goal**: Implement all three sensitivity estimators (error propagation, Fisher information, Bayesian) with comparison utilities.

**Physical Model**:
- Error propagation: $Δφ_EP = σ_Jz / |∂⟨J_z⟩/∂φ|$
- Fisher: $Δφ_F = 1/√F_C$
- Bayesian: $Δφ_B = Std[φ|m₀]$

**Instructions**:
1. Create `src/sensitivity_metrics.py` with:
   ```python
   from dataclasses import dataclass
   
   @dataclass
   class SensitivityScalingResult:
       """Result container for sensitivity scaling analysis.
       
       Attributes:
           df: pd.DataFrame with columns N, delta_phi_ep, delta_phi_fc,
               delta_phi_fq, delta_phi_bayes, fisher_quantum, state_type.
           state_type: Input state type tested ('css', 'noon', 'twin_fock', etc.).
           exponents: dict of fitted scaling exponents from log-log fit,
               e.g., {'delta_phi_ep': -0.5, 'delta_phi_fq': -1.0}.
       """
       df: pd.DataFrame
       state_type: str
       exponents: dict
   
   def error_propagation_sensitivity(state_or_rho, N: int, phi_grid: np.ndarray, 
                                dphi: float) -> dict:
       """Compute Δφ_EP over φ range, return minimum."""
   
   def all_sensitivity_metrics(state_or_rho, N: int, phi_true: float, 
                               n_mc: int = 500, rng_seed: int = 42) -> dict:
       """Compute all three metrics for comparison."""
   
   def sensitivity_scaling(state_type: str, N_range: np.ndarray, 
                          noise_config: NoiseConfig) -> SensitivityScalingResult:
       """Compute Δφ vs N for sensitivity scaling analysis.
       
       Returns:
           SensitivityScalingResult containing DataFrame and fitted exponents.
       """
   ```

2. Add tests in `src/test_sensitivity_metrics.py`:
   - Test all methods agree for large sample limit
   - Test CSS: all methods → 1/√N
   - Test NOON: methods agree only for small N

3. Validation:
   - Assert Δφ_EP ≤ Δφ_CramerRao always
   - Verify scaling exponents match theory

---

## Phase 4: Tensor Networks & Advanced Simulations

### Task 10: Tensor Tree Network (TTN)

**Location**: `src/tensor_tree_network.py` (new file)

**Goal**: Implement tree tensor network for efficient representation of bipartite BEC systems with ancilla.

**Physical Model**:
- Binary tree with 2N qubit leaves (N main + N ancilla)
- Bond dimensions controlled via SVD truncation at ε = 10⁻⁸
- Tree structure: [main_subtree]—[ancilla_subtree]
- Local dimension d = 2 per qubit site

**Instructions**:
1. Create `src/tensor_tree_network.py` with:
   ```python
   @dataclass
   class TTNNode:
       tensor: np.ndarray
       left: Optional['TTNNode']
       right: Optional['TTNNode']
       bond_dims: dict
   
   class TensorTreeNetwork:
       def __init__(self, n_sites: int, local_dim: int = 2):
           """Initialize empty TTN."""
       
       @staticmethod
       def from_state_vector(state: np.ndarray, n_sites: int, 
                            local_dim: int, svd_epsilon: float) -> 'TensorTreeNetwork':
           """Construct TTN from flat state vector via SVD."""
       
       def contract(self, ops: list[tuple[int, np.ndarray]]) -> complex:
           """Contract TTN with local operators."""
       
       def max_bond_dimension(self) -> int:
           """Return maximum bond dimension in network."""
       
       def truncate(self, epsilon: float) -> None:
           """Apply SVD truncation to all bonds."""
   ```

2. Implement:
   - Efficient SVD-based tensor decomposition
   - Contraction with expectation values
   - Bond dimension tracking

3. Add tests in `src/test_tensor_tree_network.py`:
   - Test reconstruction fidelity for small systems (N ≤ 4)
   - Test bond dimension growth with entanglement
   - Test truncation preserves normalization

4. Validation:
   - Compare TTN results with exact for N ≤ 6, tolerance 10⁻⁴
   - Verify bond dimensions remain tractable for N ≤ 20

---

### Task 11: TDVP Time Evolution on TTN

**Location**: `src/tdvp.py` (new file)

**Goal**: Implement time-dependent variational principle for TTN, enabling efficient quantum dynamics for many-body systems.

**Physical Model**:
- TDVP projects dynamics onto TTN manifold
- One-site updates: |ψ̇_i⟩ = -i P_i (H - E) |ψ⟩
- Suzuki-Trotter decomposition for Hamiltonian terms
- Checkpointing every 10 time steps

**Instructions**:
1. Create `src/tdvp.py` with:
   ```python
   def tdvp_single_site(ttn: TensorTreeNetwork, site_idx: int, H_eff: np.ndarray, 
                        dt: float) -> TensorTreeNetwork:
       """Apply single-site TDVP update."""
   
   def tdvp_evolution(ttn: TensorTreeNetwork, H: np.ndarray, T: float, 
                      dt: float, n_sites: int) -> dict:
       """Full TDVP evolution with checkpoints."""
   
   def apply_trotter_step(ttn: TensorTreeNetwork, H_terms: list[np.ndarray], 
                          dt: float, order: int = 2) -> TensorTreeNetwork:
       """Apply Trotter decomposition step."""
   ```

2. Add tests in `src/test_tdvp.py`:
   - Test norm preservation during evolution
   - Test energy conservation (within TTN manifold)
   - Test agreement with exact evolution for small systems

3. Validation:
   - Run TDVP + TTN vs QuTiP for N ≤ 6, verify relative error < 10⁻⁴

---

### Task 12: Streamlit UI Pages

**Location**: `pages/BEC_Sensitivity_Scaling.py` and `pages/BEC_Ancilla.py` (new files)

**Goal**: Create interactive Streamlit pages for BEC interferometer simulations, enabling real-time exploration of sensitivity scaling and ancilla-enhanced metrology.

**Instructions**:

1. **pages/BEC_Sensitivity_Scaling.py**:
   ```python
   # Layout: 3-column layout
   # Left: State selection, N sweep range, noise parameters
   # Center: Log-log plot of Δφ vs N for all states
   # Right: Scaling exponent comparison bar chart
   
   # Features:
   - Radio: CSS, SSS, Twin-Fock, NOON (or "All")
   - Slider: N range [10, 1000]
   - Checkboxes: noise channels to include
   - Method toggle: Lindblad vs TWA
   - Export button: Save CSV of results
   ```

2. **pages/BEC_Ancilla.py**:
   ```python
   # Layout: 2-row layout
   # Top: Parameter controls (N, chi, lambda, state type)
   # Bottom: Results comparison (with/without ancilla)
   
   # Features:
   - N slider: 1-20 (TTN tractable range)
   - State dropdown: coherent, noon, hybrid
   - Coupling slider: lambda [0, 5] Hz
   - Toggle: Show TTN bond dimension growth
   - Comparison plot: Δφ_with vs Δφ_without
   ```

3. Add integration tests in `tests/test_pages_bec.py`:
   - Test page loads without errors
   - Test simulation completes within timeout
   - Test plots render correctly

4. Validation:
   - Assert simulation completes within 100ms per point
   - Assert plots display correct scaling laws

---

## Implementation Order & Dependencies

```
Phase 1: Foundational (must precede all others)
├── Task 1: Dicke Basis & Collective Spin Operators
├── Task 2: Lindblad Master Equation Solver
└── Task 3: Truncated Wigner Approximation

Phase 2: Quantum States (depends on Phase 1)
├── Task 4: Spin-Squeezed States (uses Task 1, 2)
├── Task 5: Twin-Fock & NOON States (uses Task 1)
└── Task 6: Noise Channel Implementations (uses Task 1, 2)

Phase 3: Sensitivity Estimation (depends on Phase 2)
├── Task 7: Fisher Information Computation (uses Task 5)
├── Task 8: Bayesian Phase Estimator (uses Task 7)
└── Task 9: Sensitivity Metrics & Comparison (uses Task 7, 8)

Phase 4: Advanced Simulations (independent, parallelizable)
├── Task 10: Tensor Tree Network (new foundation)
├── Task 11: TDVP Time Evolution (uses Task 10)
└── Task 12: Streamlit UI Pages (uses all above)

Testing Strategy:
- Unit tests: src/test_*.py (co-located with modules)
- Integration tests: tests/ (end-to-end pipelines)
- Performance tests: < 100ms per simulation point
- Physics assertions: norm preservation, unitarity, scaling laws
```

---

## Expected Outcomes & Validation Checklist

After completing all tasks, the following should be verified:

- [ ] **CSS achieves SQL**: Δφ ∝ N^(-0.5) for coherent spin states
- [ ] **SSS achieves sub-SQL**: Δφ ∝ N^(-2/3) for optimally squeezed states
- [ ] **Twin-Fock achieves sub-SQL**: Δφ ∝ N^(-0.75)
- [ ] **NOON achieves Heisenberg**: Δφ ∝ N^(-1) for small N, noiseless
- [ ] **Noise degrades appropriately**: sensitivity degrades with noise strength
- [ ] **Lindblad vs TWA agree for CSS**: < 5% discrepancy for N ≤ 100
- [ ] **Ancilla provides advantage**: Δφ_with < Δφ_without for hybrid states
- [ ] **TTN validation passes**: relative error < 10⁻⁴ vs exact for N ≤ 6
- [ ] **All tests pass**: `pytest . --quiet --tb=short`
- [ ] **All lints pass**: `ruff check . --fix && ruff format .`

---

## Estimated Effort

| Task | Complexity | Lines of Code (est.) | Testing Effort |
|------|------------|----------------------|----------------|
| 1. Dicke Basis | Low | 200 | Low |
| 2. Lindblad Solver | High | 400 | Medium |
| 3. TWA | Medium | 300 | Medium |
| 4. Spin Squeezing | Medium | 150 | Low |
| 5. Twin-Fock/NOON | Low | 100 | Low |
| 6. Noise Channels | Medium | 200 | Medium |
| 7. Fisher Information | Medium | 150 | Low |
| 8. Bayesian Estimator | Medium | 200 | Medium |
| 9. Sensitivity Metrics | Medium | 200 | Medium |
| 10. TTN | High | 500 | High |
| 11. TDVP | High | 400 | High |
| 12. UI Pages | Medium | 300 | Low |

**Total estimated**: ~3,100 lines of new code + tests