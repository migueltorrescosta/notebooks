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
    """Compute and plot populations over time using eigendecomposition.

    Uses eigendecomposition to avoid computing expm() for each time point.
    For diagonal Hamiltonian H, populations are time-invariant:
        ρ_ii(t) = |⟨φ_i|ψ₀⟩|² = constant
    Off-diagonal elements have phase factor exp(-i(E_i-E_j)t).
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
    rho_t = expm(-1j * hamiltonian * t) @ rho0 @ expm(1j * hamiltonian * t)
    fig, (ax1, ax2) = plt.subplots(1, 2)

    for ax in (ax1, ax2):
        ax.set_xticks(range(rho_t.shape[0]), range(rho_t.shape[0]))
        ax.set_yticks(range(rho_t.shape[0]), range(rho_t.shape[0]))

    ax1.imshow(rho_t.real, vmin=-1, vmax=1)
    ax2.imshow(rho_t.imag, vmin=-1, vmax=1)
    plt.draw()
