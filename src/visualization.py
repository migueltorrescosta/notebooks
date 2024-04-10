from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq


def fourier_transform(
    f: Callable[[float], float],
    a: float = -np.pi,
    b: float = np.pi,
    time_domain_n: int = 100,
):
    x = np.linspace(a, b, time_domain_n)
    time_frequency = np.divide(b - a, time_domain_n)
    y = [f(t) for t in x]
    yf = fft(y)
    xf = fftfreq(time_domain_n, time_frequency)

    fig, axs = plt.subplots(3)
    fig.tight_layout(pad=1)
    axs[0].set_title("Time Domain")
    axs[1].set_title("Frequency Domain, absolute value")
    axs[1].set_title("Frequency Domain, angle")
    axs[0].plot(x, y)
    axs[1].plot(xf, [np.sqrt(t.real**2 + t.imag**2) for t in yf])
    axs[2].plot(xf, [np.angle(t) for t in yf])

    #
    # plt.plot(xf, 2.0 / time_domain_n * np.abs(yf[0:time_domain_n // 2]))
    # plt.grid()
    # plt.show()
