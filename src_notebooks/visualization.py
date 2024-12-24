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
    hamiltonian: np.array,
    rho0: np.array,
    time_window_upper_bound: int = 10,
    labels: Optional[List[str]] = None,
) -> None:
    n_states = hamiltonian.shape[0]

    assert hamiltonian.shape[1] == n_states, "the hamiltonian is not square"
    assert (
        rho0.shape == hamiltonian.shape
    ), "The initial state \rho0 does not match the hamiltonian's dimension"
    assert time_window_upper_bound > 0, "The time cannot be negative"
    assert np.einsum("ii", rho0) == 1, "The diagonals of rho0 must add up to 1"

    if labels is None:
        labels = range(rho0.shape[1])
    else:
        assert (
            len(labels) == rho0.shape[1]
        ), f"We have {len(labels)} labels but {rho0.shape[1]} possible states"

    time_axis = np.linspace(0, time_window_upper_bound, 1000)

    def rho(t: float) -> np.array:
        return expm(-1j * hamiltonian * t) @ rho0 @ expm(1j * hamiltonian * t)

    full_rho = np.array(
        [rho(t) for t in time_axis]
    )  # Indexed as (time, state_i, state_j)
    plt.stackplot(
        time_axis,
        [np.vectorize(np.abs)(full_rho[:, t, t]) for t in range(full_rho.shape[1])],
        baseline="zero",
        labels=labels,
    )
    plt.legend()


def quantum_state_heatmap(hamiltonian: np.array, rho0: np.array, t: float = 0) -> None:
    rho_t = expm(-1j * hamiltonian * t) @ rho0 @ expm(1j * hamiltonian * t)
    fig, (ax1, ax2) = plt.subplots(1, 2)

    for ax in (ax1, ax2):
        ax.set_xticks(range(rho_t.shape[0]), range(rho_t.shape[0]))
        ax.set_yticks(range(rho_t.shape[0]), range(rho_t.shape[0]))

    ax1.imshow(np.vectorize(lambda x: x.real)(rho_t), vmin=-1, vmax=1)
    ax2.imshow(np.vectorize(lambda x: x.imag)(rho_t), vmin=-1, vmax=1)
    plt.draw()
