"""
Visualization utilities for quantum dynamics and state evolution.

This module provides plotting functions for:
- Fourier transforms of time-domain signals
- Population dynamics of quantum states over time
- Density matrix heatmaps showing real and imaginary parts

All functions use Matplotlib for rendering and include appropriate
assertions to validate physical constraints (Hermiticity, normalization).
"""

from typing import List, Optional
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.linalg import expm


def fourier_transform(
    f: Callable[[float], float],
    a: float = -1 * 10**2,
    b: float = 10**2,
    time_domain_n: int = 10**5,
) -> None:
    """Compute and visualize the Fourier transform of a real-valued function.

    Samples the function over the interval [a, b] and displays three panels:
    1. Time domain (original signal)
    2. Frequency domain (magnitude of FFT)
    3. Frequency domain (phase of FFT)

    Args:
        f: Real-valued function to transform. Should be well-behaved
            over the interval [a, b].
        a: Start of time interval (default: -100).
        b: End of time interval (default: 100).
        time_domain_n: Number of sampling points (default: 100000).
            Higher values give better frequency resolution.

    Note:
        The FFT assumes periodic boundary conditions. Non-periodic
        signals will show spectral leakage.
    """
    x_axis = np.linspace(a, b, time_domain_n)
    time_frequency = np.divide(b - a, time_domain_n)
    y = [f(t) for t in x_axis]
    yf = fft(y)
    xf = fftfreq(time_domain_n, time_frequency)

    fig, axs = plt.subplots(3)
    fig.tight_layout(pad=1)
    axs[0].set_title("Time Domain")
    axs[1].set_title("Frequency Domain, absolute value")
    axs[2].set_title("Frequency Domain, angle")
    axs[0].plot(x_axis, y)
    axs[1].plot(xf, [np.sqrt(t.real**2 + t.imag**2) for t in yf])
    axs[2].plot(xf, [np.angle(t) for t in yf])


def finite_dimensional_populations_over_time(
    hamiltonian: np.ndarray,
    rho0: np.ndarray,
    time_window_upper_bound: int = 10,
    labels: Optional[List[str]] = None,
) -> None:
    """Compute and plot quantum state populations over time.

    Uses eigendecomposition to compute time evolution of populations
    (diagonal elements of the density matrix in the energy basis).
    For time-independent Hamiltonians, populations are constants of
    motion when expressed in the energy eigenbasis.

    For a diagonal Hamiltonian H = diag(E₀, E₁, ...), the population
    of eigenstate |i⟩ is time-independent:
        ρ_ii(t) = |⟨φ_i|ψ₀⟩|² = |(V^dagger @ ψ₀)_i|²

    Off-diagonal elements acquire phase factors exp(-i(E_i - E_j)t).

    Args:
        hamiltonian: Square Hermitian matrix representing the system
            Hamiltonian. Must be square and symmetric (Hermitian).
        rho0: Initial density matrix. Must have same dimension as
            Hamiltonian and trace 1.
        time_window_upper_bound: Upper bound of the time interval
            for plotting (default: 10 in dimensionless units).
        labels: Optional list of labels for each state. Must match
            the dimension of rho0.

    Raises:
        AssertionError: If Hamiltonian is not square.
        AssertionError: If Hamiltonian is not Hermitian.
        AssertionError: If rho0 dimension doesn't match Hamiltonian.
        AssertionError: If time_window_upper_bound is not positive.
        AssertionError: If rho0 doesn't have trace 1.
        AssertionError: If number of labels doesn't match dimension.
    """
    n_states = hamiltonian.shape[0]

    assert hamiltonian.shape[1] == n_states, "the hamiltonian is not square"
    assert np.allclose(hamiltonian, hamiltonian.conj().T), (
        "Hamiltonian must be Hermitian"
    )
    assert rho0.shape == hamiltonian.shape, (
        "The initial state ρ0 does not match the hamiltonian's dimension"
    )
    assert time_window_upper_bound > 0, "The time cannot be negative"
    assert np.einsum("ii", rho0) == 1, "The diagonals of rho0 must add up to 1"

    if labels is None:
        labels = [str(i) for i in range(rho0.shape[1])]
    else:
        assert len(labels) == rho0.shape[1], (
            f"We have {len(labels)} labels but {rho0.shape[1]} possible states"
        )

    time_axis = np.linspace(0, time_window_upper_bound, 1000)
    n_time = len(time_axis)

    # Eigendecomposition: H = V @ diag(E) @ V^dagger
    eigvals, eigvecs = np.linalg.eigh(hamiltonian)
    # Project rho0 into eigenbasis: rho0_eigen = V^dagger @ rho0 @ V
    rho0_eigen = eigvecs.conj().T @ rho0 @ eigvecs

    # Populations are the diagonal elements of rho0 in eigenbasis (time-invariant)
    populations = np.abs(np.diag(rho0_eigen)) ** 2

    # Verify populations sum to 1 (trace preservation)
    assert np.isclose(np.sum(populations), 1.0, rtol=1e-5, atol=1e-8), (
        f"Populations must sum to 1, got {np.sum(populations)}"
    )

    # Expand time-invariant populations across time axis for stackplot
    y_arrays = [np.full(n_time, pop) for pop in populations]
    plt.stackplot(time_axis, *y_arrays, baseline="zero", labels=labels)
    plt.legend()


def quantum_state_heatmap(
    hamiltonian: np.ndarray, rho0: np.ndarray, t: float = 0
) -> None:
    """Plot the real and imaginary parts of a time-evolved density matrix.

    Computes the density matrix at time t via unitary evolution:
        ρ(t) = U(t) ρ₀ U(t)⁻¹  where U(t) = exp(-iHt)

    Displays two heatmaps side by side showing the real and imaginary
    components of the density matrix. This is useful for visualizing
    quantum state coherence and off-diagonal elements.

    Args:
        hamiltonian: Square Hermitian matrix representing the system
            Hamiltonian.
        rho0: Initial density matrix at time t=0.
        t: Time at which to evaluate the evolved density matrix
            (default: 0, which shows the initial state).

    Raises:
        AssertionError: If Hamiltonian is not square.
        AssertionError: If Hamiltonian is not Hermitian.
        AssertionError: If rho0 dimension doesn't match Hamiltonian.

    Note:
        This function directly computes expm(-iHt), which becomes
        inefficient for large Hilbert spaces. For large systems,
        consider using the eigendecomposition method in
        `finite_dimensional_populations_over_time`.
    """
    rho_t = expm(-1j * hamiltonian * t) @ rho0 @ expm(1j * hamiltonian * t)
    fig, (ax1, ax2) = plt.subplots(1, 2)

    for ax in (ax1, ax2):
        ax.set_xticks(range(rho_t.shape[0]), range(rho_t.shape[0]))
        ax.set_yticks(range(rho_t.shape[0]), range(rho_t.shape[0]))

    ax1.imshow(rho_t.real, vmin=-1, vmax=1)
    ax2.imshow(rho_t.imag, vmin=-1, vmax=1)
    plt.draw()
