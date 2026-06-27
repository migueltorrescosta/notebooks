r"""Shared constants and path helpers for the 2026-06-25 report modules.

Exact-match items extracted from ``heisenberg_limit_mzi_sq_oat.py`` and
``squeezed_vacuum_parity.py`` to eliminate cross-module duplication within
the same report directory.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.paths import report_path_fn

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORT_DATE = "20260625"
t_hold: float = 10.0  # Holding time

# Parameter sweep ranges
SV_N_RANGE: list[float] = [float(n) for n in range(1, 21)]  # ⟨N⟩ = 1..20

# Scaling fit
SV_ALPHA_EXPECTED: float = -1.0  # SV = Heisenberg
ALPHA_TOL: float = 0.05  # Tolerance on fitted alpha for PASS/FAIL determination

# Path helpers (module-internal, used via from _shared import _parquet_path, _fig_path)
_parquet_path, _fig_path = report_path_fn(REPORTS_DIR, REPORT_DATE)
