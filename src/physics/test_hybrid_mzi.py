"""
Tests for MZI Embedding for Hybrid Oscillator-Spin System.

Note: Tests for embed_hybrid_in_mzi, mzi_beam_splitter, mzi_phase_shift,
mzi_phase_generator, evolve_hybrid_mzi, mzi_output_probabilities,
mzi_marginal_photon_probs, compute_wigner_for_state have been migrated to
reports/2026-05-07/test_local.py.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# ── Load local.py via importlib for hybrid_coherent_state ────────────────────
_local_path = (
    Path(__file__).resolve().parent.parent.parent
    / "reports"
    / "2026-05-07"
    / "local.py"
)
_spec = importlib.util.spec_from_file_location("report_local", str(_local_path))
assert _spec is not None, "Could not find reports/2026-05-07/local.py"
_report_local = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _report_local
_spec.loader.exec_module(_report_local)
hybrid_coherent_state = _report_local.hybrid_coherent_state

from .hybrid_mzi import (  # noqa: E402
    extract_oscillator_density,
    qfi_hybrid_mzi,
)
from .hybrid_system import hybrid_vacuum_state  # noqa: E402


class TestQFIHybridMZI:
    def test_given_vacuum_then_non_negative(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        fq = qfi_hybrid_mzi(state, N)
        assert fq >= 0.0

    def test_given_vacuum_then_qfi_zero(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        fq = qfi_hybrid_mzi(state, N)
        assert fq == pytest.approx(0.0, abs=1e-6)

    def test_given_more_photons_then_qfi_higher(self) -> None:
        N = 10
        state1 = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")
        state4 = hybrid_coherent_state(N, alpha=2.0 + 0j, spin_state="down")
        fq1 = qfi_hybrid_mzi(state1, N)
        fq4 = qfi_hybrid_mzi(state4, N)
        assert fq4 > fq1


class TestDensityExtraction:
    def test_given_vacuum_state_then_extracted_density_is_vacuum(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        rho_osc = extract_oscillator_density(state, N)
        assert rho_osc[0, 0] == pytest.approx(1.0)
        assert np.sum(np.abs(rho_osc[1:, :])) == pytest.approx(0.0, abs=1e-10)

    def test_given_coherent_state_then_extracted_density_has_unity_trace(self) -> None:
        N = 5
        state = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")
        rho_osc = extract_oscillator_density(state, N)
        trace = np.trace(rho_osc).real
        assert trace == pytest.approx(1.0, rel=1e-3, abs=1e-3)
