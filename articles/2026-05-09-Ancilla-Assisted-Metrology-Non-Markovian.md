# Ancilla-Assisted Metrology in Non-Markovian Environments

## 🧪 Hypothesis

Auxiliary qubits (spin ancillae) can systematically protect Quantum Fisher Information
(QFI) against decoherence from **non-Markovian baths** with a Lorentzian spectral density.
There exists an optimal direct-coupling strength between the oscillator probe and the
spin ancilla that maximizes QFI preservation at a given evolution time T, and this
optimum depends on the bath correlation rate λ (degree of non-Markovianity).

Specifically:

1. **QFI preservation**: ℛ(T) = F_Q(T) / F_Q(0) > ℛ_no_ancilla(T) for suitable g,
   where g is the direct-coupling strength n ⊗ σ_x between oscillator and ancilla.

2. **Optimal coupling**: There exists g* > 0 that maximizes ℛ(T) at fixed T, λ, and N.

3. **Non-Markovian advantage**: The QFI preservation improvement grows as the bath
   becomes more non-Markovian (smaller λ), because the ancilla can leverage bath
   memory to help protect phase information.

4. **Crossover regime**: In the Markovian limit (λ → ∞), ancilla assistance provides
   diminishing returns, converging to the standard Lindblad treatment.

## ⚛️ Theoretical Model

The **total Hilbert space** is a tensor product of three subsystems:
$\mathcal{H}_{\text{total}} = \mathcal{H}_{\text{osc}} \otimes \mathcal{H}_{\text{spin}} \otimes \mathcal{H}_{\text{pm}}$.
The **oscillator** is a bosonic probe mode (senses phase φ) in the **Fock basis**
$\vert n\rangle$, $n = 0, \dots, N$, dimension $N + 1$. The **spin** is a two-level ancilla
qubit with basis $\vert{\downarrow}\rangle, \vert{\uparrow}\rangle$, dimension 2. The
**pseudomode** is a damped bosonic mode representing the Lorentzian bath, with **Fock basis**
$\vert k\rangle$, $k = 0, \dots, K$, dimension $K + 1$. The **index ordering convention** is
$\text{Index} = (n \times 2 + s) \times (K + 1) + k$, where $n$ is the oscillator Fock index,
$s \in \{0, 1\}$ is the spin state ($0 = \vert{\downarrow}\rangle$, $1 = \vert{\uparrow}\rangle$),
and $k$ is the pseudomode Fock index. The total dimension is $2 (N + 1) (K + 1)$.

The **total Hamiltonian** comprises four terms: $H = H_{\text{osc}} + H_{\text{sa}} + H_{\text{sp}} + H_{\text{pm}}$.
The **oscillator free energy** is $H_{\text{osc}} = 0$, measured relative to the oscillator
resonance in the chosen working frame — no rotating-wave approximation is made, so no
frequency terms are discarded. The **system-ancilla direct coupling** (the controllable
entangling interaction) is
$H_{\text{sa}} = g_{\text{sa}} \cdot (a^\dagger a) \otimes \sigma_x \otimes \mathbb{1}_{\text{pm}}$,
a dispersive interaction where the ancilla's effective energy shift is proportional to the
oscillator photon number. The **system-pseudomode coupling** (system-bath interaction) is
$H_{\text{sp}} = g_{\text{sp}} \cdot (a + a^\dagger) \otimes \mathbb{1}_{\text{spin}} \otimes (b + b^\dagger)$,
a position-position coupling that generates the Lorentzian spectral density. The **pseudomode
free energy** is $H_{\text{pm}} = \omega_0 \cdot \mathbb{1}_{\text{osc}} \otimes \mathbb{1}_{\text{spin}} \otimes b^\dagger b$,
where $\omega_0$ is the central frequency of the Lorentzian bath.

The **pseudomode is damped** at rate λ (the bath correlation width), giving the sole Lindblad
dissipator $L_{\text{pm}} = \sqrt{\lambda} \cdot \mathbb{1}_{\text{osc}} \otimes \mathbb{1}_{\text{spin}} \otimes b$,
with master equation $d\rho/dt = -i[H, \rho] + \lambda (b \rho b^\dagger - \tfrac12 \{b^\dagger b, \rho\})$.
This single dissipator together with $H_{\text{sp}}$ implements the exact non-Markovian
dynamics of the oscillator coupled to a Lorentzian reservoir with spectral density
$S(\omega) = (g_{\text{sp}}^2 \cdot \lambda) / [(\omega - \omega_0)^2 + \lambda^2]$.
In the **Markovian limit** ($\lambda \to \infty$), the standard Lindblad equation for the
oscillator alone is recovered with effective decay rate $\gamma = g_{\text{sp}}^2 / \lambda$,
and the pseudomode can be adiabatically eliminated. In the **strongly non-Markovian limit**
($\lambda \to 0$), the pseudomode is essentially undamped and the dynamics are fully coherent
within the enlarged oscillator-pseudomode subspace. Additional Markovian noise channels
($\gamma_1$, $\gamma_\phi$) can be added if desired.

The **metrology protocol** proceeds in five steps. (1)
**Initial state**: The oscillator is prepared in a coherent state $\vert\alpha\rangle$ (or
vacuum $\vert0\rangle$ with pre-squeezing), the spin ancilla in $\vert{\downarrow}\rangle$, and
the pseudomode in vacuum $\vert0\rangle$:
$\vert\Psi(0)\rangle = \vert\alpha\rangle \otimes \vert{\downarrow}\rangle \otimes \vert0\rangle$.
(2) **Ancilla entanglement**: The system-ancilla coupling $H_{\text{sa}}$ is activated for
time τ, generating entanglement between oscillator photon number and spin state:
$\vert\Psi(\tau)\rangle = \exp(-i \tau g_{\text{sa}} \cdot a^\dagger a \otimes \sigma_x)
\vert\alpha\rangle \otimes \vert{\downarrow}\rangle \otimes \vert0\rangle$;
in the Fock basis this gives
$\vert\Psi(\tau)\rangle = \sum_n c_n \vert n\rangle \otimes
(\cos(g_{\text{sa}} \tau n) \vert{\downarrow}\rangle - i \sin(g_{\text{sa}} \tau n) \vert{\uparrow}\rangle)
\otimes \vert0\rangle$. (3) **Phase imprint**: The unknown phase φ is encoded on the
oscillator: $U_\phi = \exp(i \phi \cdot a^\dagger a)$. This commutes with $H_{\text{sa}}$
(both are diagonal in the Fock basis), so the order of steps 2 and 3 can be exchanged.
(4) **Non-Markovian evolution** (pulsed ancilla protocol): The Hamiltonian
$H = H_{\text{sp}} + H_{\text{pm}}$ (all terms except $H_{\text{sa}}$) drives the system
for time T under the Lindblad master equation with pseudomode damping λ.
The ancilla coupling $H_{\text{sa}}$ is turned off during decoherence — the ancilla only
entangles with the oscillator during Step 2 (pulsed protection), not continuously,
isolating the ancilla's benefit to its initial entanglement. (5) **QFI evaluation**:
The final state $\rho(T)$ is generally mixed. The **QFI with respect to φ** is computed
via the SLD eigen-decomposition formula:
$F_Q(\phi; \rho(T)) = 2 \sum_{i<j} (\lambda_i - \lambda_j)^2 / (\lambda_i + \lambda_j)
\vert\langle i\vert G\vert j\rangle\vert^2 + 4 \sum_i \lambda_i \Delta G_{ii}^2$,
where $\rho(T) = \sum_i \lambda_i \vert i\rangle\langle i\vert$ is the eigen-decomposition,
$G = a^\dagger a$ is the **phase generator**, and
$\Delta G_{ii} = \langle i\vert G\vert i\rangle - \operatorname{Tr}(\rho G)$.

If the ancilla (spin) is retained, the QFI includes the full oscillator-spin-pseudomode
space. If the ancilla is traced out (to simulate ancilla-less metrology), we compare
$\mathcal{S}_{\text{with}} = F_Q(T; \rho_{\text{osc+sp+pm}})$ (full state) and
$\mathcal{S}_{\text{without}} = F_Q(T; \operatorname{Tr}_{\text{spin}}[\rho_{\text{osc+sp+pm}}])$
(ancilla traced out). The **QFI preservation ratio** is
$\mathcal{R}(T) = F_Q(T) / F_Q(0)$.

The central observable is $F_Q(\theta, T, \lambda, N, K, g_{\text{sp}}, \omega_0, \alpha)$.
Parameter sweeps vary the **ancilla rotation angle** $\theta = g_{\text{sa}} \cdot \tau$
(independent variable), the **bath memory** λ (inverse correlation time), the **decoherence
time** T, the **bath coupling** $g_{\text{sp}}$, and the **probe energy** $\vert\alpha\vert^2$.
Key metrics are the QFI preservation ratio $\mathcal{R}(T)$, the ancilla improvement
$\Delta\mathcal{R} = \mathcal{R}_{\text{with}}(T) - \mathcal{R}_{\text{without}}(T)$, and
the optimal ancilla rotation angle
$\theta^* = \operatorname{argmax} \mathcal{R}(T)$.

## 💻 Numerical Simulation

### Implementation Strategy

1. **Operator construction** — Create pseudomode operators $b, b^\dagger, b^\dagger b$ in a truncated Fock basis (dimension $K+1$) and extend existing hybrid operators to the tripartite space via Kronecker products: $O_{\text{total}} = O_{\text{osc}} \otimes O_{\text{spin}} \otimes \mathbb{1}_{\text{pm}}$.
2. **Hamiltonian assembly** — Build $H_{\text{sa}}$, $H_{\text{sp}}$, and $H_{\text{pm}}$ from the constructed operators, using a configuration dataclass with pseudomode parameters $(K, g_{\text{sp}}, \omega_0, \lambda)$.
3. **Lindblad simulation** — Reuse existing dimension-agnostic RK4 and scipy-based Lindblad integrators; the single dissipator is $L_{\text{pm}} = \sqrt{\lambda} \cdot \mathbb{1}_{\text{osc}} \otimes \mathbb{1}_{\text{spin}} \otimes b$, with trace preservation, Hermiticity, and positivity verified at each step.
4. **QFI computation** — Trace out the pseudomode to obtain the reduced oscillator-spin density matrix, then apply the SLD formulation (eigen-decomposition with eigenvalue threshold $10^{-12}$). The phase generator is $G = a^\dagger a \otimes \mathbb{1}_{\text{spin}}$. Compare two scenarios: with ancilla ($F_Q$ on $\rho_{\text{osc+spin}}$) and without ancilla ($F_Q$ on $\rho_{\text{osc}} = \operatorname{Tr}_{\text{spin}}[\rho_{\text{osc+spin}}]$).
5. **Control parameter handling** — The dimensionless product $\theta = g_{\text{sa}} \cdot \tau$ (ancilla rotation angle) is the primary control. Implement three sweep types: `ancilla_sweep` over $\theta$, `memory_sweep` over $\lambda$, and `time_sweep` over $T$.
6. **Dimension management** — Total Hilbert space dimension $2(N+1)(K+1)$. Choose $K$ adaptively: $K = \max(5, \lceil g_{\text{sp}} \cdot T + 5\sqrt{g_{\text{sp}} \cdot T} + 6 \rceil)$, with monitoring ensuring $\langle b^\dagger b \rangle \leq 0.8\, K$ at all times.

### Parameter Sweep

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| N | 20 | 5–50 | Oscillator Fock truncation |
| K | 10 | 3–30 | Pseudomode Fock truncation (adaptive) |
| $\vert\alpha\vert^2$ | 1.0 | 0.5–5.0 | Mean photon number of probe |
| $\theta$ | 0.0 | 0–π | Ancilla rotation angle $\theta = g_{\text{sa}} \cdot \tau$ (swept) |
| τ | 0.1 | 0.01–0.5 | Ancilla entanglement time (fixed, $\tau \ll 1/g_{\text{sp}}$) |
| $g_{\text{sp}}$ | 0.5 | 0.1–2.0 | System-pseudomode coupling |
| $\omega_0$ | 0.0 | –2 to 2 | Bath central frequency (0 = resonant with oscillator) |
| λ | 1.0 | 0.05–10 | Bath correlation rate |
| T | 2.0 | 0–10 | Evolution time |

### Validation

```python
# — Physical validity —
assert np.isclose(np.trace(rho), 1.0), "Trace must be preserved"
assert np.allclose(rho, rho.conj().T, atol=1e-8), "ρ must be Hermitian"
assert np.min(np.linalg.eigvalsh(rho)) >= -1e-8, "ρ must be positive"
assert 0 <= F_Q(T) <= F_Q(0), "QFI must be non-negative and non-increasing"

# — Baseline recovery —
assert np.allclose(ℛ(T; θ=0), ℛ_no_ancilla(T)), "θ=0 recovers no-ancilla baseline"

# — Pseudomode truncation guard —
assert np.max(b_dag_b_expect) <= 0.8 * K, "Pseudomode occupancy exceeds safe limit"

# — QFI numerical stability —
F_Q = compute_sld_qfi(rho, generator, eigval_threshold=1e-12)
assert np.isfinite(F_Q), "QFI must be finite"

# — Operator consistency (Kronecker order) —
# For a small reference system (N=2, K=2), manually construct the Hamiltonian
# matrix element-by-element and compare against the Kronecker-based builder.
# Both must agree to machine precision.
```

#### 🔧 Implementation Status

The full simulation code described in this plan has been implemented and unit-tested.

| Component | Description |
|-----------|-------------|
| **Configuration** | Dataclass with pseudomode parameters and validation |
| **Operators** | Pseudomode creation/annihilation/number, tripartite Kronecker construction |
| **Hamiltonian** | Builder with `include_sa` flag for entanglement vs. decoherence steps |
| **Lindblad** | Single dissipator $L_{\text{pm}} = \sqrt{\lambda} \cdot I\otimes I\otimes b$ |
| **State preparation** | $\vert\alpha\rangle \otimes \vert{\downarrow}\rangle \otimes \vert0\rangle$ |
| **Entanglement** | $\exp(-i H_{\text{sa}} \cdot \tau)$ with $\sigma_x$ coupling |
| **Evolution** | RK4 and scipy solvers matching existing Lindblad patterns |
| **Partial trace** | Reshape + np.trace for tracing pseudomode and/or spin |
| **QFI** | With-ancilla and without-ancilla variants, preservation ratio |
| **Protocol** | Full 5-step pipeline |
| **Validation** | Density validation, pseudomode occupancy checks |

**Tests**: 75 unit tests covering configuration, operators, Hamiltonian, Lindblad operators,
state preparation, ancilla entanglement, partial trace, Lindblad evolution, QFI computation,
metrology protocol, density validation, pseudomode occupancy, QFI preservation ratio, and
edge cases — all passing.

## ⚠️ Expected Failure Conditions

| Failure | Description | Mitigation |
|---------|-------------|------------|
| Pseudomode truncation artifacts | If $g_{\text{sp}}$ or T is too large, the pseudomode population grows beyond K, causing reflection at the Fock-space boundary that feeds spurious population back into the system. | Fail immediately if $\langle b^\dagger b\rangle > 0.8 \cdot K$ at any time step. Use adaptive K with a safety margin of 2–3× above expected occupation. |
| Hilbert space explosion | The total dimension $2(N+1)(K+1)$ grows as $N \cdot K$. For N = 50 and K = 30, the density matrix has $\sim 10^7$ elements ($\sim 80$ MB), manageable but pushing RK4 to $\sim 10$–100 ms per step. | Keep N ≤ 30, K ≤ 20 for initial sweeps; use sparse methods or vectorized Liouvillian if scaling to larger systems. |
| Kronecker-product ordering mismatch | If the index convention disagrees with the np.kron call order, every Hamiltonian matrix element is silently wrong while dimensions remain correct. | Unit-test the operator builder against a hand-constructed reference for a tiny system (N=2, K=2) verified to machine precision. |
| Over-rotation at strong coupling | For large $\theta = g_{\text{sa}} \cdot \tau$, the ancilla entanglement wraps around ($\theta > \pi/2$), reducing rather than protecting QFI. | Plot $\mathcal{R}(T)$ vs θ; expect a peak at moderate $\theta^*$, not monotonic or flat behavior. |
| Small λ (deeply non-Markovian) regime | When λ is very small, the pseudomode behaves almost coherently, and the combined oscillator-pseudomode system may exhibit coherent oscillations rather than dissipative dynamics, causing QFI to oscillate rather than decay monotonically. | Output the full $\mathcal{R}(t)$ trajectory, not just the endpoint at T. |
| Phase generator ambiguity | The phase generator $G = a^\dagger a \otimes \mathbb{1}_{\text{spin}}$ acts on the oscillator only. When the ancilla is traced out, phase information stored in spin-oscillator correlations is discarded. | Always compare $\mathcal{R}_{\text{full}}$ (ancilla retained) vs $\mathcal{R}_{\text{partial}}$ (ancilla traced out). The difference $\Delta\mathcal{R}$ is the ancilla benefit. |
| QFI numerical instability for highly mixed states | The SLD formula divides by $(\lambda_i + \lambda_j)$, which becomes singular for near-zero eigenvalues. | Threshold eigenvalues at $\epsilon = 10^{-12}$; skip terms where $\lambda_i + \lambda_j < \epsilon$. Check purity $\operatorname{Tr}(\rho^2)$ and warn if $< 0.05$. |

## 🔬 Results

| # | Test | Expectation | Status |
|---|------|-------------|--------|
| 1 | Pseudomode Lindblad reproduces Markovian limit ($\lambda \to \infty$) | $\mathcal{R}(T)$ matches standard Lindblad at $\gamma = g_{\text{sp}}^2 / \lambda$ | ⏳ |
| 2 | Ancilla improves QFI at moderate θ | $\mathcal{R}_{\text{with}} > \mathcal{R}_{\text{without}}$ at fixed $T > 0$ | ⏳ |
| 3 | Optimal $\theta^*$ exists at finite value | $\mathcal{R}(T)$ is concave in θ | ⏳ |
| 4 | Non-Markovian advantage | $\Delta\mathcal{R}$ larger for smaller λ | ⏳ |
| 5 | Numerical validity | Trace, Hermiticity, positivity | ✅ |
| 6 | Over-rotation at strong coupling | $\mathcal{R}(T)$ decreases for $\theta \gg \theta^*$ | ⏳ |

Numerical validity (test 5) has been verified through 75 unit tests — trace preservation,
Hermiticity, and positivity hold throughout all tested scenarios. The remaining five tests
await parameter sweeps.

## ✅ Success Criteria

| # | Check | Expectation |
|---|-------|-------------|
| 1 | **Ancilla benefit** | $\exists\,\theta > 0$ such that $\mathcal{R}_{\text{with}}(T) > \mathcal{R}_{\text{without}}(T)$ at fixed $T > 0$, verified with tolerance-based significance |
| 2 | **Optimal coupling** | $\mathcal{R}(T)$ as a function of $\theta$ shows a clear maximum at some finite $\theta^* > 0$, not at the boundaries ($\theta^*$ not at 0 or $\pi$) |
| 3 | **Non-Markovian scaling** | The ancilla improvement $\Delta\mathcal{R}$ increases as $\lambda$ decreases (more non-Markovian), confirming that bath memory is the resource |
| 4 | **Markovian recovery** | As $\lambda \to \infty$, $\mathcal{R}_{\text{with}}(T) \to \mathcal{R}_{\text{without}}(T)$, converging to the known Lindblad result |
| 5 | **Numerical validity** | Trace preservation, Hermiticity, and positivity hold at all times; $\mathcal{R}(T) \le 1$ for all $T \ge 0$; pseudomode truncation does not cause artifacts ($\langle b^\dagger b\rangle \le 0.8\,K$ at all times) |
| 6 | **MZI compatibility** | The protocol can be embedded into the existing MZI pipeline for a full interferometric readout |

Numerical validity (criterion 5) has been verified through the unit-test suite — trace
preservation, Hermiticity, and positivity hold throughout all tested scenarios. The remaining
five criteria await parameter sweeps. The simulation code is ready: the `ancilla_sweep`,
`memory_sweep`, and `time_sweep` functions are implemented and validated against baseline
recovery ($\theta = 0$ matches the no-ancilla case).

## 🏁 Conclusions

The theoretical model for ancilla-assisted metrology with a non-Markovian pseudomode bath
has been fully implemented and validated at the unit-test level. The five-step protocol
(initial state → ancilla entanglement → phase imprint → non-Markovian evolution → QFI
evaluation) is operational, and the simulation correctly recovers the no-ancilla baseline
when $\theta = 0$.

The core hypotheses — that ancilla coupling can protect QFI, that an optimal coupling
strength exists, and that the benefit grows with non-Markovianity — remain untested
experimentally. The simulation infrastructure is ready for systematic parameter sweeps:
an ancilla sweep over $\theta = g_{\text{sa}} \cdot \tau$ across $[0, \pi]$ to find
$\theta^*$, a memory sweep over $\lambda$ across $[0.05, 10]$ to quantify
$\Delta\mathcal{R}(\lambda)$, and a time sweep over T to study QFI trajectories.
Cross-validation in the $\lambda \to \infty$ limit against standard Lindblad with
$\gamma = g_{\text{sp}}^2 / \lambda$ will confirm the Markovian recovery prediction.

#### 🔍 Open Questions

🔍 **Open items**: (a) What is the optimal initial state (coherent vs squeezed vs Fock) for the oscillator? (b) Does multiple ancilla entanglement (using the spin as a multi-level system) provide additional benefit beyond the two-level case? (c) How does the optimal ancilla coupling depend on the bath central frequency $\omega_0$ (off-resonant vs resonant)? (d) Can the ancilla-assisted protocol be combined with existing high-order squeezing (n=3,4) for further enhancement?
