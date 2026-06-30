"""
Tests for the joint measurement ancilla module.

Verifies plotting functions, generate functions (cache/skip, force re-run),
CLI entry point, and module-level constants defined in
``joint_measurement_ancilla.py``.

Run with:
    uv run pytest reports/20260515/test_joint_measurement_ancilla.py -q --tb=short
"""

from __future__ import annotations

import importlib
import sys as _sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from src.analysis.ancilla_optimization_results import (
    AlphaReoptScanResult,
    CovarianceAnalysisResult,
    DecoupledBaselineResult,
    InteractionRobustnessResult,
    OmegaScanResult,
    OptimisationResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# ── Module loading via importlib ────────────────────────────────────────────

_m = importlib.import_module("reports.20260515.joint_measurement_ancilla")

_local_fig_path = _m._fig_path
_local_parquet_path = _m._parquet_path
compute_decoupled_baseline = _m.compute_decoupled_baseline
generate_alpha_reoptimisation = _m.generate_alpha_reoptimisation
generate_covariance_analysis = _m.generate_covariance_analysis
generate_interaction_robustness = _m.generate_interaction_robustness
generate_omega_scan = _m.generate_omega_scan
main = _m.main
plot_alpha_reoptimisation = _m.plot_alpha_reoptimisation
plot_covariance_analysis = _m.plot_covariance_analysis
plot_decoupled_baseline = _m.plot_decoupled_baseline
plot_interaction_robustness = _m.plot_interaction_robustness
plot_omega_scan = _m.plot_omega_scan

# ============================================================================
# Helpers
# ============================================================================


def _check_plot(
    plot_fn: Callable[..., Path],
    result: object,
    tmp_path: Path,
    filename: str = "test.svg",
) -> None:
    """Call *plot_fn(result, save_path)* and verify the SVG was created."""
    save_path = tmp_path / filename
    out = plot_fn(result, save_path)
    assert out == save_path
    assert out.exists(), f"Plot file {out} was not created"
    assert out.stat().st_size > 0, f"Plot file {out} is empty"
    assert out.suffix == ".svg", f"Expected .svg, got {out.suffix}"


def _redirect_to_tmpdir(
    tmp_path: Path,
) -> None:
    """Monkey-patch ``_m.REPORTS_DIR`` so generate functions write to *tmp_path*."""
    _m.REPORTS_DIR = tmp_path
    _m.REPORT_DATE = "20260515"


# ============================================================================
# Test: Module Loading and Constants
# ============================================================================


class TestModuleLoading:
    """Verify the module loads correctly and exposes expected names."""

    def test_module_has_main(self) -> None:
        assert callable(_m.main)

    def test_report_date(self) -> None:
        assert _m.REPORT_DATE == "20260515"

    def test_reports_dir_is_parent(self) -> None:
        assert (_m.REPORTS_DIR / "20260515").resolve() == Path(
            __file__
        ).resolve().parent

    def test_module_has_expected_functions(self) -> None:
        expected = [
            "plot_decoupled_baseline",
            "plot_omega_scan",
            "plot_interaction_robustness",
            "plot_alpha_reoptimisation",
            "plot_covariance_analysis",
            "generate_omega_scan",
            "generate_interaction_robustness",
            "generate_alpha_reoptimisation",
            "generate_covariance_analysis",
            "main",
        ]
        for name in expected:
            assert hasattr(_m, name), f"Missing expected function: {name}"
            assert callable(getattr(_m, name)), f"{name} is not callable"


# ============================================================================
# Test: Plot Functions (direct dataclass object)
# ============================================================================


class TestPlotFunctionsDirect:
    """Verify each plot function runs with a dataclass object and produces SVG."""

    def test_decoupled_baseline(self, tmp_path: Path) -> None:
        result = DecoupledBaselineResult(
            t_hold_values=np.array([0.5, 1.0, 2.0]),
            delta_omega_values=np.array([2.0, 1.0, 0.5]),
            sql_values=np.array([2.0, 1.0, 0.5]),
        )
        _check_plot(plot_decoupled_baseline, result, tmp_path)

    def test_omega_scan(self, tmp_path: Path) -> None:
        omega_vals = [0.1, 0.2]
        all_results: dict[float, list[OptimisationResult]] = {}
        for ow in omega_vals:
            r = OptimisationResult(
                delta_omega_opt=2.0 / ow,
                params_opt=np.array(
                    [0.0, 0.0, 0.0, 0.0, 1.57, 1.57, 2.0, 0.0, 0.0, 0.0, 0.0],
                ),
                omega_true=ow,
                success=True,
                nfev=100,
                message="OK",
                meas_label="Joint M",
                expectation_M=0.5,
                variance_M=0.01,
                covariance_SA=0.005,
            )
            all_results[ow] = [r]
        result = OmegaScanResult(
            results=[r for rr in all_results.values() for r in rr],
            omega_values=np.array(omega_vals),
            best_per_omega=np.array([2.0 / 0.1, 2.0 / 0.2]),
            all_results=all_results,
        )
        _check_plot(plot_omega_scan, result, tmp_path)

    def test_interaction_robustness(self, tmp_path: Path) -> None:
        n_T, n_a = 3, 3
        result = InteractionRobustnessResult(
            t_hold_values=np.array([0.5, 1.0, 2.0]),
            alpha_values=np.array([0.0, 1.0, 2.0]),
            delta_omega_joint=np.ones((n_T, n_a)),
            delta_omega_sonly=np.ones((n_T, n_a)) * 1.5,
        )
        _check_plot(plot_interaction_robustness, result, tmp_path)

    def test_alpha_reoptimisation(self, tmp_path: Path) -> None:
        result = AlphaReoptScanResult(
            alpha_values=np.array([-1.0, 0.0, 1.0]),
            delta_omega_joint=np.array([0.25, 0.20, 0.25]),
            delta_omega_sonly=np.array([0.30, 0.20, 0.30]),
            best_params_joint=[
                np.zeros(11),
                np.zeros(11),
                np.zeros(11),
            ],
            best_params_sonly=[
                np.zeros(11),
                np.zeros(11),
                np.zeros(11),
            ],
        )
        _check_plot(plot_alpha_reoptimisation, result, tmp_path)

    def test_covariance_analysis(self, tmp_path: Path) -> None:
        result = CovarianceAnalysisResult(
            coefficient_names=[r"$\alpha_{xx}$", r"$\alpha_{xz}$"],
            max_covariances=np.array([0.12, 0.08]),
            covariance_signs=np.array([1.0, -1.0]),
        )
        _check_plot(plot_covariance_analysis, result, tmp_path)


# ============================================================================
# Test: Plot Functions (via Parquet path)
# ============================================================================


class TestPlotFunctionsFromPath:
    """Verify each plot function works when passed a parquet file path."""

    def test_decoupled_baseline(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "t_hold": [0.5, 1.0, 2.0],
                "delta_omega": [2.0, 1.0, 0.5],
                "sql": [2.0, 1.0, 0.5],
                "ratio": [1.0, 1.0, 1.0],
            },
        )
        p = tmp_path / "decoupled.parquet"
        df.to_parquet(p, index=False)
        _check_plot(plot_decoupled_baseline, p, tmp_path)

    def test_omega_scan(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "omega": [0.1, 0.2, 0.5],
                "best_delta_omega": [2.0, 1.0, 0.5],
                "sql": [0.5, 0.5, 0.5],
                "vs_sql": [4.0, 2.0, 1.0],
                "spread": [0.1, 0.05, 0.02],
                "t_hold_star": [2.0, 2.0, 2.0],
                "covariance": [0.01, 0.005, 0.002],
                "expectation_M": [0.5, 0.3, 0.2],
                "flag": ["ok", "ok", "fringe"],
            },
        )
        p = tmp_path / "omega_scan.parquet"
        df.to_parquet(p, index=False)
        _check_plot(plot_omega_scan, p, tmp_path)

    def test_interaction_robustness(self, tmp_path: Path) -> None:
        rows = []
        for t in [0.5, 1.0, 2.0]:
            for a in [0.0, 1.0, 2.0]:
                rows.append(
                    {
                        "t_hold": t,
                        "alpha": a,
                        "measurement": "joint",
                        "delta_omega": 1.0,
                    },
                )
                rows.append(
                    {
                        "t_hold": t,
                        "alpha": a,
                        "measurement": "sonly",
                        "delta_omega": 1.5,
                    },
                )
        df = pd.DataFrame(rows)
        p = tmp_path / "interaction.parquet"
        df.to_parquet(p, index=False)
        _check_plot(plot_interaction_robustness, p, tmp_path)

    def test_alpha_reoptimisation(self, tmp_path: Path) -> None:
        param_cols = [
            "theta_S",
            "phi_S",
            "theta_A",
            "phi_A",
            "T_BS1",
            "T_BS2",
            "t_hold",
            "alpha_xx",
            "alpha_xz",
            "alpha_zx",
            "alpha_zz",
        ]
        data: dict[str, object] = {
            "alpha": [-1.0, 0.0, 1.0],
            "delta_omega_joint": [0.25, 0.20, 0.25],
            "delta_omega_sonly": [0.30, 0.20, 0.30],
        }
        for col in param_cols:
            data[f"joint_{col}"] = [0.0, 0.0, 0.0]
            data[f"sonly_{col}"] = [0.0, 0.0, 0.0]
        df = pd.DataFrame(data)
        p = tmp_path / "alpha_reopt.parquet"
        df.to_parquet(p, index=False)
        _check_plot(plot_alpha_reoptimisation, p, tmp_path)

    def test_covariance_analysis(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "coefficient": ["alpha_xx", "alpha_xz"],
                "max_covariance": [0.12, 0.08],
                "sign": [1.0, -1.0],
            },
        )
        p = tmp_path / "covariance.parquet"
        df.to_parquet(p, index=False)
        _check_plot(plot_covariance_analysis, p, tmp_path)


# ============================================================================
# Test: Generate Functions (cache / skip / force)
# ============================================================================


class TestGenerateDecoupledBaseline:
    """Verify the decoupled-baseline generator (from ``decoupled_baseline`` module)."""

    def _call_decoupled_baseline(self, force: bool) -> None:
        """Call the decoupled-baseline generator with the same args as ``main()``."""
        _m.generate_decoupled_baseline(
            force=force,
            parquet_path=_local_parquet_path("decoupled-baseline", date="20260515"),
            fig_path=_local_fig_path("decoupled-baseline", date="20260515"),
            compute_fn=compute_decoupled_baseline,
            compute_kwargs={
                "t_hold_values": np.array([0.5, 1.0]),
            },
            result_cls=DecoupledBaselineResult,
            plot_fn=plot_decoupled_baseline,
            label="decoupled baseline",
        )

    def test_skip_when_exists(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Pre-creating the parquet should trigger the skip path."""
        _redirect_to_tmpdir(tmp_path)
        parquet_p = _local_parquet_path("decoupled-baseline", date="20260515")
        parquet_p.parent.mkdir(parents=True, exist_ok=True)
        # Write a valid decoupled-baseline parquet
        df = pd.DataFrame(
            {
                "t_hold": [0.5, 1.0, 2.0],
                "delta_omega": [2.0, 1.0, 0.5],
                "sql": [2.0, 1.0, 0.5],
                "ratio": [1.0, 1.0, 1.0],
            },
        )
        df.to_parquet(parquet_p, index=False)

        # Create figures dir so plot can save
        fig_dir = _local_fig_path("decoupled-baseline", date="20260515").parent
        fig_dir.mkdir(parents=True, exist_ok=True)

        self._call_decoupled_baseline(force=False)
        captured = capsys.readouterr()
        assert "[skip]" in captured.out, (
            f"Expected [skip] in output, got: {captured.out}"
        )

    def test_force_runs(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """With ``force=True`` and valid compute, the generator should run."""
        _redirect_to_tmpdir(tmp_path)
        self._call_decoupled_baseline(force=True)
        captured = capsys.readouterr()
        assert "[save]" in captured.out, (
            f"Expected [save] in output, got: {captured.out}"
        )


class TestGenerateCovarianceAnalysis:
    """Covariance analysis is fast (no Nelder-Mead) — test with ``force=True``."""

    def test_run_and_save(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        _redirect_to_tmpdir(tmp_path)
        generate_covariance_analysis(force=True)
        captured = capsys.readouterr()
        assert "[save]" in captured.out, (
            f"Expected [save] in output, got: {captured.out}"
        )
        # Verify the parquet was created
        parquet_p = _local_parquet_path("covariance-analysis", date="20260515")
        assert parquet_p.exists(), f"Parquet not created at {parquet_p}"
        # Verify the figure was created
        fig_p = _local_fig_path("covariance-analysis", date="20260515")
        assert fig_p.exists(), f"Figure not created at {fig_p}"

    def test_skip_when_exists(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _redirect_to_tmpdir(tmp_path)
        # Pre-create a valid covariance-analysis parquet
        parquet_p = _local_parquet_path("covariance-analysis", date="20260515")
        parquet_p.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "coefficient": ["alpha_xx"],
                "max_covariance": [0.1],
                "sign": [1.0],
            },
        )
        df.to_parquet(parquet_p, index=False)
        # Pre-create figures dir
        fig_p = _local_fig_path("covariance-analysis", date="20260515")
        fig_p.parent.mkdir(parents=True, exist_ok=True)

        generate_covariance_analysis(force=False)
        captured = capsys.readouterr()
        assert "[skip]" in captured.out, (
            f"Expected [skip] in output, got: {captured.out}"
        )


class TestGenerateInteractionRobustness:
    """Interaction robustness is moderately fast — test with minimal parameters."""

    def test_run_and_save(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        _redirect_to_tmpdir(tmp_path)
        generate_interaction_robustness(force=True)
        captured = capsys.readouterr()
        assert "[save]" in captured.out, (
            f"Expected [save] in output, got: {captured.out}"
        )
        parquet_p = _local_parquet_path("interaction-robustness", date="20260515")
        assert parquet_p.exists()
        fig_p = _local_fig_path("interaction-robustness", date="20260515")
        assert fig_p.exists()

    def test_skip_when_exists(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _redirect_to_tmpdir(tmp_path)
        parquet_p = _local_parquet_path("interaction-robustness", date="20260515")
        parquet_p.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for t in [0.5, 1.0, 2.0]:
            for a in [0.0, 1.0, 2.0]:
                rows.append(
                    {
                        "t_hold": t,
                        "alpha": a,
                        "measurement": "joint",
                        "delta_omega": 1.0,
                    }
                )
                rows.append(
                    {
                        "t_hold": t,
                        "alpha": a,
                        "measurement": "sonly",
                        "delta_omega": 1.5,
                    }
                )
        pd.DataFrame(rows).to_parquet(parquet_p, index=False)
        fig_p = _local_fig_path("interaction-robustness", date="20260515")
        fig_p.parent.mkdir(parents=True, exist_ok=True)
        generate_interaction_robustness(force=False)
        captured = capsys.readouterr()
        assert "[skip]" in captured.out


@pytest.mark.slow
class TestGenerateOmegaScan:
    """Omega scan involves Nelder-Mead — mark as slow."""

    def test_skip_when_exists(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _redirect_to_tmpdir(tmp_path)
        parquet_p = _local_parquet_path("omega-scan", date="20260515")
        parquet_p.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "omega": [0.1],
                "best_delta_omega": [2.0],
                "sql": [0.5],
                "vs_sql": [4.0],
                "spread": [0.1],
                "t_hold_star": [2.0],
                "covariance": [0.01],
                "expectation_M": [0.5],
                "flag": ["ok"],
            },
        )
        df.to_parquet(parquet_p, index=False)
        fig_p = _local_fig_path("omega-scan", date="20260515")
        fig_p.parent.mkdir(parents=True, exist_ok=True)

        generate_omega_scan(force=False)
        captured = capsys.readouterr()
        assert "[skip]" in captured.out


@pytest.mark.slow
class TestGenerateAlphaReoptimisation:
    """Alpha re-optimisation involves Nelder-Mead — mark as slow."""

    def test_skip_when_exists(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _redirect_to_tmpdir(tmp_path)
        parquet_p = _local_parquet_path("alpha-reoptimisation", date="20260515")
        parquet_p.parent.mkdir(parents=True, exist_ok=True)
        param_cols = [
            "theta_S",
            "phi_S",
            "theta_A",
            "phi_A",
            "T_BS1",
            "T_BS2",
            "t_hold",
            "alpha_xx",
            "alpha_xz",
            "alpha_zx",
            "alpha_zz",
        ]
        data: dict[str, object] = {
            "alpha": [0.0],
            "delta_omega_joint": [0.2],
            "delta_omega_sonly": [0.2],
        }
        for col in param_cols:
            data[f"joint_{col}"] = [0.0]
            data[f"sonly_{col}"] = [0.0]
        pd.DataFrame(data).to_parquet(parquet_p, index=False)
        fig_p = _local_fig_path("alpha-reoptimisation", date="20260515")
        fig_p.parent.mkdir(parents=True, exist_ok=True)

        generate_alpha_reoptimisation(force=False)
        captured = capsys.readouterr()
        assert "[skip]" in captured.out


# ============================================================================
# Test: CLI entry point (main)
# ============================================================================


class TestMain:
    """Test the ``main()`` CLI entry point."""

    def test_unknown_dataset_exits(self) -> None:
        """Passing ``--only`` with an unknown name should print an error."""
        _m.REPORTS_DIR = Path(__file__).resolve().parent.parent

        _sys.argv = ["prog", "--only", "nonexistent-dataset"]
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @pytest.mark.slow
    def test_only_covariance_runs(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """``--only covariance-analysis`` should run just that task."""
        _redirect_to_tmpdir(tmp_path)
        _sys.argv = ["prog", "--only", "covariance-analysis", "--force"]
        main()
        captured = capsys.readouterr()
        assert "[save]" in captured.out
        assert "covariance-analysis" in captured.out

    def test_help_message(self) -> None:
        """``--help`` should print usage."""
        _m.REPORTS_DIR = Path(__file__).resolve().parent.parent

        _sys.argv = ["prog", "--help"]
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
