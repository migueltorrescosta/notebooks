"""Tests for :mod:`src.analysis.checkpoint_recovery`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from src.analysis.checkpoint_recovery import (
    _check_required_columns,
    _omega_sort_key,
    _safe_sort_key,
    load_checkpoints,
    run_pending_groups,
)

# ---------------------------------------------------------------------------
# Fake result dataclasses for testing
# ---------------------------------------------------------------------------


@dataclass
class _FakeResult:
    """Minimal result dataclass for checkpoint recovery testing."""

    N: int
    omega: float
    delta_omega_opt: float
    sql: float = 0.1
    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "N",
        "omega",
        "delta_omega_opt",
        "sql",
    ]


@dataclass
class _FakeScanResult:
    """Minimal scan result wrapping a list of results."""

    results: list[_FakeResult] = field(default_factory=list)

    def save_parquet(self, path: Path) -> None:
        df = pd.DataFrame([vars(r) for r in self.results])
        df.to_parquet(path)


@dataclass
class _FakeResult3Key:
    """Result with three key columns (N, M, omega)."""

    N: int
    M: int
    omega: float
    delta_omega_opt: float
    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "N",
        "M",
        "omega",
        "delta_omega_opt",
    ]


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_fake_ckpt(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write a checkpoint Parquet file with the given rows."""
    pd.DataFrame(rows).to_parquet(path)


def _build_from_row(row_dict: dict[str, Any]) -> _FakeResult:
    return _FakeResult(
        N=int(row_dict["N"]),
        omega=float(row_dict["omega"]),
        delta_omega_opt=float(row_dict["delta_omega_opt"]),
        sql=float(row_dict.get("sql", 0.1)),
    )


def _build_from_dict(rdict: dict[str, Any]) -> _FakeResult:
    return _FakeResult(
        N=int(rdict["N"]),
        omega=float(rdict["omega"]),
        delta_omega_opt=float(rdict["delta_omega_opt"]),
        sql=float(rdict.get("sql", 0.1)),
    )


def _worker_fn(item: tuple[int, float]) -> dict[str, Any]:
    """Pretend to run optimisation; return a fake result dict."""
    N, omega = item
    return {
        "N": N,
        "omega": omega,
        "delta_omega_opt": 0.1 / (N + 1.0),
        "sql": 0.1,
    }


def _worker_inf_fn(item: tuple[int, float]) -> dict[str, Any]:
    """Worker that always returns infinite delta."""
    N, omega = item
    return {
        "N": N,
        "omega": omega,
        "delta_omega_opt": np.inf,
        "sql": 0.1,
    }


# ---------------------------------------------------------------------------
# _check_required_columns
# ---------------------------------------------------------------------------


class TestCheckRequiredColumns:
    def test_passes_with_all_columns(self) -> None:
        df = pd.DataFrame({"N": [1], "omega": [0.5], "delta_omega_opt": [0.01]})
        _check_required_columns(df, ["N", "omega"], Path("test.parquet"))

    def test_raises_on_missing_column(self) -> None:
        df = pd.DataFrame({"N": [1], "omega": [0.5]})
        with pytest.raises(ValueError, match="missing columns"):
            _check_required_columns(df, ["N", "omega"], Path("test.parquet"))

    def test_raises_on_missing_key_column(self) -> None:
        df = pd.DataFrame({"N": [1], "delta_omega_opt": [0.01]})
        with pytest.raises(ValueError, match="missing columns"):
            _check_required_columns(df, ["N", "omega"], Path("test.parquet"))


# ---------------------------------------------------------------------------
# _safe_sort_key
# ---------------------------------------------------------------------------


class TestSafeSortKey:
    def test_single_int(self) -> None:
        assert _safe_sort_key(5) == ((0, 5),)

    def test_single_float(self) -> None:
        assert _safe_sort_key(3.14) == ((1, 3.14),)

    def test_tuple_mixed(self) -> None:
        result = _safe_sort_key((1, 2.5))
        assert result == ((0, 1), (1, 2.5))


# ---------------------------------------------------------------------------
# _omega_sort_key
# ---------------------------------------------------------------------------


class TestOmegaSortKey:
    def test_omega_attribute(self) -> None:
        r = _FakeResult(N=1, omega=0.5, delta_omega_opt=0.01)
        assert _omega_sort_key(r) == 0.5

    def test_fallback_zero(self) -> None:
        r = _FakeResult(N=1, omega=0.0, delta_omega_opt=0.01)
        # omega is 0.0
        assert _omega_sort_key(r) == 0.0


# ---------------------------------------------------------------------------
# load_checkpoints
# ---------------------------------------------------------------------------


class TestLoadCheckpoints:
    def test_empty_directory(self, tmp_path: Path) -> None:
        completed, results = load_checkpoints(tmp_path, _build_from_row, ["N", "omega"])
        assert len(completed) == 0
        assert len(results) == 0

    def test_single_checkpoint(self, tmp_path: Path) -> None:
        _make_fake_ckpt(
            tmp_path / "N_001.parquet",
            [{"N": 1, "omega": 0.5, "delta_omega_opt": 0.05, "sql": 0.1}],
        )
        completed, results = load_checkpoints(tmp_path, _build_from_row, ["N", "omega"])
        assert (1, 0.5) in completed
        assert len(results) == 1
        assert results[0].N == 1
        assert results[0].omega == 0.5
        assert results[0].delta_omega_opt == 0.05

    def test_skips_infinite_delta(self, tmp_path: Path) -> None:
        _make_fake_ckpt(
            tmp_path / "N_001.parquet",
            [
                {"N": 1, "omega": 0.5, "delta_omega_opt": 0.05, "sql": 0.1},
                {"N": 1, "omega": 1.0, "delta_omega_opt": np.inf, "sql": 0.1},
            ],
        )
        completed, results = load_checkpoints(tmp_path, _build_from_row, ["N", "omega"])
        assert (1, 0.5) in completed
        assert (1, 1.0) not in completed
        assert len(results) == 1

    def test_multiple_checkpoint_files(self, tmp_path: Path) -> None:
        for N in range(1, 4):
            _make_fake_ckpt(
                tmp_path / f"N_{N:03d}.parquet",
                [{"N": N, "omega": 0.5, "delta_omega_opt": 0.1 / N, "sql": 0.1}],
            )
        completed, results = load_checkpoints(tmp_path, _build_from_row, ["N", "omega"])
        assert len(completed) == 3
        assert len(results) == 3

    def test_three_key_columns(self, tmp_path: Path) -> None:
        def _build_3k(row_dict: dict[str, Any]) -> _FakeResult3Key:
            return _FakeResult3Key(
                N=int(row_dict["N"]),
                M=int(row_dict["M"]),
                omega=float(row_dict["omega"]),
                delta_omega_opt=float(row_dict["delta_omega_opt"]),
            )

        _make_fake_ckpt(
            tmp_path / "N_001.parquet",
            [{"N": 1, "M": 2, "omega": 0.5, "delta_omega_opt": 0.05}],
        )
        completed, results = load_checkpoints(tmp_path, _build_3k, ["N", "M", "omega"])
        assert (1, 2, 0.5) in completed
        assert len(results) == 1
        assert results[0].M == 2

    def test_missing_key_column_warns_and_skips(self, tmp_path: Path) -> None:
        """Missing columns print a warning and the file is skipped."""
        _make_fake_ckpt(
            tmp_path / "N_001.parquet",
            [{"N": 1, "delta_omega_opt": 0.05}],
        )
        completed, results = load_checkpoints(tmp_path, _build_from_row, ["N", "omega"])
        assert len(completed) == 0
        assert len(results) == 0

    def test_deterministic_order(self, tmp_path: Path) -> None:
        """Loading the same checkpoints twice gives the same results."""
        _make_fake_ckpt(
            tmp_path / "N_001.parquet",
            [{"N": 1, "omega": 0.5, "delta_omega_opt": 0.05, "sql": 0.1}],
        )
        c1, r1 = load_checkpoints(tmp_path, _build_from_row, ["N", "omega"])
        c2, r2 = load_checkpoints(tmp_path, _build_from_row, ["N", "omega"])
        assert c1 == c2
        assert len(r1) == len(r2)
        for r_a, r_b in zip(r1, r2, strict=False):
            assert r_a.N == r_b.N
            assert r_a.omega == r_b.omega


# ---------------------------------------------------------------------------
# run_pending_groups
# ---------------------------------------------------------------------------


class TestRunPendingGroups:
    def test_empty_items(self, tmp_path: Path) -> None:
        results = run_pending_groups(
            items_to_run=[],
            checkpoint_dir=tmp_path,
            worker_fn=_worker_fn,
            build_result_from_dict=_build_from_dict,
            group_key_fn=lambda item: item[0],
            checkpoint_name_fn=lambda key: tmp_path / f"N_{key:03d}.parquet",
            scan_result_cls=_FakeScanResult,
        )
        assert len(results) == 0

    def test_runs_pending_items(self, tmp_path: Path) -> None:
        items = [(1, 0.5), (1, 1.0), (2, 0.5)]
        results = run_pending_groups(
            items_to_run=items,
            checkpoint_dir=tmp_path,
            worker_fn=_worker_fn,
            build_result_from_dict=_build_from_dict,
            group_key_fn=lambda item: item[0],
            checkpoint_name_fn=lambda key: tmp_path / f"N_{key:03d}.parquet",
            scan_result_cls=_FakeScanResult,
        )
        assert len(results) == 3
        assert results[0].N == 1 or results[0].N == 2

    def test_skips_existing_checkpoint(self, tmp_path: Path) -> None:
        """If checkpoint file exists, the group is skipped."""
        # Create a pre-existing checkpoint for N=1
        _make_fake_ckpt(
            tmp_path / "N_001.parquet",
            [{"N": 1, "omega": 0.5, "delta_omega_opt": 0.05, "sql": 0.1}],
        )
        items = [(1, 0.5), (1, 1.0), (2, 0.5)]
        results = run_pending_groups(
            items_to_run=items,
            checkpoint_dir=tmp_path,
            worker_fn=_worker_fn,
            build_result_from_dict=_build_from_dict,
            group_key_fn=lambda item: item[0],
            checkpoint_name_fn=lambda key: tmp_path / f"N_{key:03d}.parquet",
            scan_result_cls=_FakeScanResult,
        )
        # Only N=2 should run
        assert len(results) == 1
        assert results[0].N == 2

    def test_skips_non_finite_delta(self, tmp_path: Path) -> None:
        items = [(1, 0.5)]
        results = run_pending_groups(
            items_to_run=items,
            checkpoint_dir=tmp_path,
            worker_fn=_worker_inf_fn,
            build_result_from_dict=_build_from_dict,
            group_key_fn=lambda item: item[0],
            checkpoint_name_fn=lambda key: tmp_path / f"N_{key:03d}.parquet",
            scan_result_cls=_FakeScanResult,
        )
        assert len(results) == 0

    def test_saves_checkpoint_file(self, tmp_path: Path) -> None:
        items = [(1, 0.5)]
        run_pending_groups(
            items_to_run=items,
            checkpoint_dir=tmp_path,
            worker_fn=_worker_fn,
            build_result_from_dict=_build_from_dict,
            group_key_fn=lambda item: item[0],
            checkpoint_name_fn=lambda key: tmp_path / f"N_{key:03d}.parquet",
            scan_result_cls=_FakeScanResult,
        )
        assert (tmp_path / "N_001.parquet").exists()

    def test_deterministic_with_same_input(self, tmp_path: Path) -> None:
        items = [(1, 0.5), (2, 1.0)]
        r1 = run_pending_groups(
            items_to_run=items,
            checkpoint_dir=tmp_path,
            worker_fn=_worker_fn,
            build_result_from_dict=_build_from_dict,
            group_key_fn=lambda item: item[0],
            checkpoint_name_fn=lambda key: tmp_path / f"N_{key:03d}.parquet",
            scan_result_cls=_FakeScanResult,
        )
        # Run again with fresh dir
        tmp_path2 = tmp_path / "redo"
        tmp_path2.mkdir()
        r2 = run_pending_groups(
            items_to_run=items,
            checkpoint_dir=tmp_path2,
            worker_fn=_worker_fn,
            build_result_from_dict=_build_from_dict,
            group_key_fn=lambda item: item[0],
            checkpoint_name_fn=lambda key: tmp_path2 / f"N_{key:03d}.parquet",
            scan_result_cls=_FakeScanResult,
        )
        assert len(r1) == len(r2)
        for r_a, r_b in zip(r1, r2, strict=False):
            assert r_a.delta_omega_opt == r_b.delta_omega_opt
