"""Hilbert-space dimension helpers for interferometer simulations.

Shared across reports to avoid duplicating truncation logic for SV, TMSV,
and definite-N states.
"""

from __future__ import annotations

import numpy as np


def resource_value_to_truncation(
    resource_value: float,
    state_type: str,
    trunc_multiplier: float = 5.0,
    max_trunc: int = 80,
) -> int:
    r"""Compute appropriate Hilbert space truncation for a given resource value.

    For SV/TMSV: ``resource_value`` is the mean photon number, and truncation
    is ``min(ceil(trunc_multiplier * resource_value), max_trunc)``.

    For OAT/definite-N states: truncation = ``resource_value`` (exact).

    Args:
        resource_value: The resource parameter (mean N for SV/TMSV, N for OAT).
        state_type: ``"sv"``, ``"tmsv"``, or ``"oat"``.
        trunc_multiplier: Multiplier for indefinite-N states (default 5.0).
        max_trunc: Maximum truncation per mode (default 80).

    Returns:
        Truncation M (max photons per mode).
    """
    if state_type == "oat":
        return int(resource_value)
    return min(int(np.ceil(trunc_multiplier * resource_value)), max_trunc)
