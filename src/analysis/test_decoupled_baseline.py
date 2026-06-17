"""Tests for :mod:`src.analysis.decoupled_baseline`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.decoupled_baseline import (
    generate_decoupled_baseline,
    plot_decoupled_baseline_heatmap,
    verify_decoupled_baseline,
)
from src.analysis.sensitivity_metrics import sql_reference
from src.utils.serialization import ParquetSerializable

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _MockResult(ParquetSerializable):
    """Minimal result dataclass for testing orchestration."""

    x: float
    y: float

    _PARQUET_COLUMNS: ClassVar[list[str]] = ["x", "y"]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"x": [self.x], "y": [self.y]})

    @classmethod
    def from_parquet(cls, path: str | Path) -> _MockResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        return cls(x=float(df["x"].iloc[0]), y=float(df["y"].iloc[0]))


def _mock_compute() -> _MockResult:
    return _MockResult(x=1.0, y=2.0)


def _mock_compute_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2], "b": [3, 4]})


# ──────────────────────────────────────────────────────────────────────
# verify_decoupled_baseline
# ──────────────────────────────────────────────────────────────────────


class TestVerifyDecoupledBaseline:
    """Parameterised decoupled-baseline verification."""

    def test_default_compute_fn_all_pass(self) -> None:
        """Default compute_fn should give all PASS for small (N, ω)."""
        results = verify_decoupled_baseline(
            N_values=[1, 2, 4],
            omega_values=[0.2],
        )
        assert all(results.values()), (
            f"Failed pairs: {[(n, w) for (n, w), v in results.items() if not v]}"
        )

    def test_custom_compute_fn_off_by_factor(self) -> None:
        """A compute_fn returning 1.1×SQL should fail under rtol=1e-10."""

        def _off(N: int, omega: float) -> float:
            return sql_reference(N) * 1.1

        results = verify_decoupled_baseline(
            N_values=[1, 2],
            omega_values=[0.2],
            compute_fn=_off,
            rtol=1e-10,
        )
        assert not any(results.values()), "Should all fail with 10% offset"

    def test_compute_kwargs_forwarded(self) -> None:
        """Extra keyword arguments are unpacked into compute_fn."""

        def _fn(N: int, omega: float, *, factor: float = 1.0) -> float:
            return sql_reference(N) * factor

        # factor=1.0 should pass
        results_pass = verify_decoupled_baseline(
            N_values=[1, 2],
            omega_values=[0.2],
            compute_fn=_fn,
            factor=1.0,
        )
        assert all(results_pass.values())

        # factor=1.1 should fail at strict rtol
        results_fail = verify_decoupled_baseline(
            N_values=[1, 2],
            omega_values=[0.2],
            compute_fn=_fn,
            rtol=1e-10,
            factor=1.1,
        )
        assert not any(results_fail.values())

    def test_empty_omega_values(self) -> None:
        """Empty omega list returns empty results dict."""
        results = verify_decoupled_baseline(
            N_values=[1, 2],
            omega_values=[],
        )
        assert results == {}

    def test_default_values_no_error(self) -> None:
        """Calling with no arguments uses defaults and does not raise."""
        results = verify_decoupled_baseline()
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_custom_rtol_tighter(self) -> None:
        """A tight rtol still passes when delta == SQL exactly."""
        results = verify_decoupled_baseline(
            N_values=[1],
            omega_values=[0.2],
            rtol=1e-14,
        )
        assert results[(1, 0.2)]

    def test_custom_rtol_looser(self) -> None:
        """A loose rtol passes even with a 5% offset."""

        def _off(N: int, omega: float) -> float:
            return sql_reference(N) * 1.05

        results = verify_decoupled_baseline(
            N_values=[1, 2],
            omega_values=[0.2],
            compute_fn=_off,
            rtol=0.1,  # 10% tolerance
        )
        assert all(results.values())


# ──────────────────────────────────────────────────────────────────────
# generate_decoupled_baseline
# ──────────────────────────────────────────────────────────────────────


class TestGenerateDecoupledBaseline:
    """Orchestration behaviour."""

    def test_compute_and_save_dataclass(self, tmp_path: Path) -> None:
        """Dataclass result is saved and returned."""
        p = tmp_path / "test.parquet"
        result = generate_decoupled_baseline(
            parquet_path=p,
            compute_fn=_mock_compute,
            result_cls=_MockResult,
            label="test",
        )
        assert isinstance(result, _MockResult)
        assert result.x == 1.0
        assert p.exists()

    def test_skip_and_load_from_cache(self, tmp_path: Path) -> None:
        """When cached, loads and returns without calling compute_fn."""
        p = tmp_path / "test.parquet"
        # First run — populate cache
        generate_decoupled_baseline(
            parquet_path=p,
            compute_fn=_mock_compute,
            result_cls=_MockResult,
        )
        # Second run — should skip compute
        call_count = 0

        def _compute_with_side_effect() -> _MockResult:
            nonlocal call_count
            call_count += 1
            return _MockResult(x=99.0, y=99.0)

        result = generate_decoupled_baseline(
            parquet_path=p,
            compute_fn=_compute_with_side_effect,
            result_cls=_MockResult,
        )
        assert call_count == 0  # compute was NOT called
        assert isinstance(result, _MockResult)
        assert result.x == 1.0  # original cached value

    def test_force_recomputes(self, tmp_path: Path) -> None:
        """With ``force=True``, recomputes even if cached."""
        p = tmp_path / "test.parquet"
        generate_decoupled_baseline(
            parquet_path=p,
            compute_fn=_mock_compute,
            result_cls=_MockResult,
        )
        result = generate_decoupled_baseline(
            force=True,
            parquet_path=p,
            compute_fn=_mock_compute,
            result_cls=_MockResult,
        )
        assert isinstance(result, _MockResult)
        assert result.x == 1.0

    def test_compute_and_save_dataframe(self, tmp_path: Path) -> None:
        """DataFrame result is saved via to_parquet."""
        p = tmp_path / "test.parquet"
        result = generate_decoupled_baseline(
            parquet_path=p,
            compute_fn=_mock_compute_df,
            result_cls=None,
        )
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert p.exists()

    def test_skip_without_result_cls(self, tmp_path: Path) -> None:
        """When result_cls is None, skip returns None."""
        p = tmp_path / "test.parquet"
        # Populate cache
        generate_decoupled_baseline(
            parquet_path=p,
            compute_fn=_mock_compute_df,
            result_cls=None,
        )
        call_count = 0

        def _compute_side_effect() -> pd.DataFrame:
            nonlocal call_count
            call_count += 1
            return pd.DataFrame({"a": [99]})

        result = generate_decoupled_baseline(
            parquet_path=p,
            compute_fn=_compute_side_effect,
            result_cls=None,
        )
        assert call_count == 0
        assert result is None

    def test_plot_fn_called(self, tmp_path: Path) -> None:
        """Plot function is invoked with result and fig_path."""
        p = tmp_path / "test.parquet"
        f = tmp_path / "test.svg"
        plot_calls: list[tuple] = []

        def _plot(result: object, path: Path) -> None:
            plot_calls.append((result, path))

        generate_decoupled_baseline(
            parquet_path=p,
            fig_path=f,
            compute_fn=_mock_compute,
            result_cls=_MockResult,
            plot_fn=_plot,
        )
        assert len(plot_calls) == 1
        assert isinstance(plot_calls[0][0], _MockResult)
        assert plot_calls[0][1] == f

    def test_raises_on_bad_type(self, tmp_path: Path) -> None:
        """Non-dataclass, non-DataFrame result raises TypeError."""

        def _bad_compute() -> str:  # type: ignore[return]
            return "not a valid result"

        with pytest.raises(
            TypeError, match=r"Expected ParquetSerializable or pd.DataFrame"
        ):
            generate_decoupled_baseline(
                parquet_path=tmp_path / "test.parquet",
                compute_fn=_bad_compute,
            )

    def test_compute_kwargs_forwarded(self, tmp_path: Path) -> None:
        """Keyword arguments are unpacked into compute_fn."""

        def _compute_with_kwargs(a: int, b: int = 0) -> _MockResult:
            return _MockResult(x=float(a), y=float(b))

        p = tmp_path / "test.parquet"
        result = generate_decoupled_baseline(
            parquet_path=p,
            compute_fn=_compute_with_kwargs,
            compute_kwargs={"a": 10, "b": 20},
            result_cls=_MockResult,
        )
        assert result is not None
        assert result.x == 10.0
        assert result.y == 20.0

    def test_missing_parquet_dir_created_for_df(self, tmp_path: Path) -> None:
        """Parent directory is created when saving a DataFrame."""
        p = tmp_path / "sub" / "test.parquet"
        assert not p.parent.exists()
        generate_decoupled_baseline(
            parquet_path=p,
            compute_fn=_mock_compute_df,
            result_cls=None,
        )
        assert p.exists()


# ──────────────────────────────────────────────────────────────────────
# plot_decoupled_baseline_heatmap
# ──────────────────────────────────────────────────────────────────────


class TestPlotDecoupledBaselineHeatmap:
    """Heatmap visualisation."""

    @dataclass
    class _HeatmapResult:
        """Minimal duck-typed result for the heatmap."""

        omega_values: np.ndarray
        N_values: np.ndarray
        ratio: np.ndarray
        t_hold: float

    @pytest.fixture
    def sample_result(self) -> _HeatmapResult:
        """A 2×2 grid of (ω, N) pairs where ratio should be ~1.0."""
        omegas = np.array([0.5, 1.0, 0.5, 1.0], dtype=float)
        Ns = np.array([1, 1, 2, 2], dtype=int)
        ratios = np.array([1.0, 1.01, 0.99, 1.0], dtype=float)
        return self._HeatmapResult(
            omega_values=omegas,
            N_values=Ns,
            ratio=ratios,
            t_hold=10.0,
        )

    def test_heatmap_creates_svg_file(
        self, sample_result: _HeatmapResult, tmp_path: Path
    ) -> None:
        """Output SVG is created."""
        save_path = tmp_path / "heatmap.svg"
        returned = plot_decoupled_baseline_heatmap(
            sample_result, save_path, title_prefix="Test"
        )
        assert save_path.exists()
        assert returned == save_path

    def test_heatmap_all_ones(self, tmp_path: Path) -> None:
        """When ratio=1.0 everywhere, dev_map entries are zero."""
        omegas = np.array([0.5, 1.0, 0.5, 1.0], dtype=float)
        Ns = np.array([1, 1, 2, 2], dtype=int)
        ratios = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
        result = self._HeatmapResult(
            omega_values=omegas, N_values=Ns, ratio=ratios, t_hold=5.0
        )
        save_path = tmp_path / "all_ones.svg"
        plot_decoupled_baseline_heatmap(result, save_path)
        assert save_path.exists()

    def test_heatmap_nan_handling(self, tmp_path: Path) -> None:
        """Non-finite ratios produce NaN entries in dev_map."""
        omegas = np.array([0.5, 1.0], dtype=float)
        Ns = np.array([1, 1], dtype=int)
        ratios = np.array([float("inf"), 1.0], dtype=float)
        result = self._HeatmapResult(
            omega_values=omegas, N_values=Ns, ratio=ratios, t_hold=5.0
        )
        save_path = tmp_path / "nan_handling.svg"
        plot_decoupled_baseline_heatmap(result, save_path)
        assert save_path.exists()

    def test_heatmap_custom_labels(
        self, sample_result: _HeatmapResult, tmp_path: Path
    ) -> None:
        """Custom axis and colorbar labels are accepted."""
        save_path = tmp_path / "custom.svg"
        plot_decoupled_baseline_heatmap(
            sample_result,
            save_path,
            sql_label=r"Custom $|\Delta\omega|$",
            title_prefix="Custom Test",
            omega_label=r"$\Omega$",
            N_label=r"$N_{particles}$",
        )
        assert save_path.exists()
