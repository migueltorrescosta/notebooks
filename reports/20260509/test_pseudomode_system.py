"""
Tests for the 2026-05-09 ancilla-assisted non-Markovian metrology report.

Verifies all pseudomode_system functions migrated from reports/20260511/
into reports/20260509/local.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

# Load local.py via importlib
# Must register in sys.modules for dataclass machinery to resolve __module__
_local_path = Path(__file__).resolve().parent / "local.py"
_spec = importlib.util.spec_from_file_location("report_2026_05_09", str(_local_path))
assert _spec is not None, f"Could not find local.py at {_local_path}"
_report_local = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["report_2026_05_09"] = _report_local
_spec.loader.exec_module(_report_local)


# =============================================================================
# Tests: pseudomode_system
# =============================================================================


def _make_dim_total(N: int, K: int) -> int:
    return 2 * (N + 1) * (K + 1)


class TestPseudomodeConfig:
    """Configuration dataclass defaults and validation."""

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("N", 5),
            ("K", 3),
            ("alpha", 1.0),
            ("g_sa", 1.0),
            ("tau", 0.1),
            ("g_sp", 0.5),
            ("omega_0", 0.0),
            ("lam", 1.0),
            ("T", 2.0),
            ("dt", 0.01),
        ],
    )
    def test_default_values(self, field: str, expected: object) -> None:
        assert getattr(_report_local.PseudomodeConfig(N=5, K=3), field) == expected

    def test_invalid_N_neg(self) -> None:
        with pytest.raises(ValueError):
            _report_local.PseudomodeConfig(N=-1, K=3)

    def test_invalid_dt_neg(self) -> None:
        with pytest.raises(ValueError):
            _report_local.PseudomodeConfig(N=5, K=3, dt=-0.1)


class TestPseudomodeOperators:
    """Ladder operator and number operator construction."""

    def test_shape(self) -> None:
        b, bd = _report_local.create_pseudomode_operators(5)
        assert b.shape == (6, 6)
        assert bd.shape == (6, 6)

    def test_negative_K_raises(self) -> None:
        with pytest.raises(ValueError):
            _report_local.create_pseudomode_operators(-1)

    def test_number_shape(self) -> None:
        assert _report_local.pseudomode_number_operator(5).shape == (6, 6)


class TestTripartiteOperator:
    """Tripartite Kronecker product construction."""

    def test_correct_dimensions(self) -> None:
        N, K = 5, 3
        op = _report_local.tripartite_operator(
            np.eye(N + 1), np.eye(2), np.eye(K + 1), N, K
        )
        assert op.shape == (_make_dim_total(N, K), _make_dim_total(N, K))

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(AssertionError):
            _report_local.tripartite_operator(np.eye(7), np.eye(2), np.eye(4), 5, 3)


class TestBuildPseudomodeHamiltonian:
    """Hamiltonian construction."""

    def test_shape(self) -> None:
        H = _report_local.build_pseudomode_hamiltonian(
            _report_local.PseudomodeConfig(N=5, K=3)
        )
        assert H.shape == (_make_dim_total(5, 3), _make_dim_total(5, 3))

    def test_hermiticity(self) -> None:
        H = _report_local.build_pseudomode_hamiltonian(
            _report_local.PseudomodeConfig(N=5, K=3)
        )
        assert pytest.approx(H.conj().T, abs=1e-10) == H


class TestPseudomodeInitialState:
    """Initial state preparation."""

    @pytest.mark.parametrize(
        ("alpha", "N"), [(0.0, 5), (0.5, 10), (1.0, 10), (2.0, 25)]
    )
    def test_normalization(self, alpha: float, N: int) -> None:
        state = _report_local.pseudomode_initial_state(
            _report_local.PseudomodeConfig(N=N, K=3, alpha=alpha)
        )
        assert np.sum(np.abs(state) ** 2) == pytest.approx(1.0, abs=1e-3)


class QFIComputation:
    """QFI computation tests."""

    def test_initial_nonzero(self) -> None:
        cfg = _report_local.PseudomodeConfig(
            N=10, K=3, alpha=1.0, g_sa=0.0, g_sp=0.0, lam=0.0
        )
        rho = np.outer(
            _report_local.pseudomode_initial_state(cfg),
            _report_local.pseudomode_initial_state(cfg).conj(),
        )
        assert _report_local.compute_qfi_with_ancilla(rho, cfg.N, cfg.K) > 0

    def test_vacuum_has_zero_qfi(self) -> None:
        cfg = _report_local.PseudomodeConfig(
            N=5, K=3, alpha=0.0, g_sa=0.0, g_sp=0.0, lam=0.0
        )
        rho = np.outer(
            _report_local.pseudomode_initial_state(cfg),
            _report_local.pseudomode_initial_state(cfg).conj(),
        )
        assert _report_local.compute_qfi_with_ancilla(
            rho, cfg.N, cfg.K
        ) == pytest.approx(0.0, abs=1e-10)


class TestRunMetrologyProtocol:
    """Full metrology protocol."""

    _default_kw: ClassVar[dict] = {
        "N": 5,
        "K": 3,
        "alpha": 1.0,
        "g_sa": 0.5,
        "tau": 0.2,
        "g_sp": 0.3,
        "lam": 0.5,
        "T": 0.5,
    }

    def test_protocol_completes(self) -> None:
        result = _report_local.run_metrology_protocol(
            _report_local.PseudomodeConfig(**self._default_kw)
        )
        expected_keys = {
            "rho_final",
            "qfi_with",
            "qfi_without",
            "qfi_initial",
            "ratio_with",
            "ratio_without",
            "pm_occupancy",
            "validation",
        }
        assert expected_keys.issubset(result.keys())

    def test_ratio_between_zero_and_one(self) -> None:
        result = _report_local.run_metrology_protocol(
            _report_local.PseudomodeConfig(**self._default_kw)
        )
        assert 0 <= result["ratio_with"] <= 1.0 + 1e-6
        assert 0 <= result["ratio_without"] <= 1.0 + 1e-6


# =============================================================================
# Tests: Module-level CLI
# =============================================================================


class TestModuleLoading:
    """Test that the module loads correctly with all expected attributes."""

    def test_module_has_main(self) -> None:
        assert hasattr(_report_local, "main")
        assert callable(_report_local.main)

    def test_module_has_report_date(self) -> None:
        assert _report_local.REPORT_DATE == "20260509"

    def test_module_has_expected_classes(self) -> None:
        expected_classes = [
            "PseudomodeConfig",
        ]
        for cls_name in expected_classes:
            assert hasattr(_report_local, cls_name), f"Missing class: {cls_name}"

    def test_module_can_generate_raw_data(self) -> None:
        """Verify the CLI generator functions are callable."""
        assert callable(_report_local._generate_ancilla_nonmarkovian_raw_data)
        assert callable(_report_local._generate_ancilla_nonmarkovian_figures)
