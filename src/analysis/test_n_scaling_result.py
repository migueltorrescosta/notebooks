"""Tests for the shared N-scaling result dataclasses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from src.analysis.n_scaling_result import NScalingResult, NScalingScanResult

if TYPE_CHECKING:
    from pathlib import Path


class TestNScalingResult:
    @pytest.fixture
    def make_result(self) -> NScalingResult:
        return NScalingResult(
            N=5,
            omega=0.2,
            delta_omega_opt=0.015,
            sql=0.0447,
            ratio=0.0447 / 0.015,
            a_x_opt=3.0,
            a_y_opt=-2.5,
            a_z_opt=1.0,
            a_zz_opt=4.0,
            expectation_Jz=0.5,
            variance_Jz=0.25,
            success=True,
            nfev=100,
        )

    def test_to_dataframe_columns(self, make_result: NScalingResult) -> None:
        df = make_result.to_dataframe()
        expected = set(NScalingResult._PARQUET_COLUMNS)
        assert set(df.columns) == expected

    def test_roundtrip(self, make_result: NScalingResult, tmp_path: Path) -> None:
        p = tmp_path / "test.parquet"
        make_result.save_parquet(p)
        loaded = NScalingResult.from_parquet(p)
        assert loaded.N == make_result.N
        assert loaded.omega == make_result.omega
        assert np.isclose(loaded.delta_omega_opt, make_result.delta_omega_opt)
        assert np.isclose(loaded.sql, make_result.sql)
        assert np.isclose(loaded.ratio, make_result.ratio)
        assert loaded.a_x_opt == make_result.a_x_opt
        assert loaded.a_y_opt == make_result.a_y_opt
        assert loaded.a_z_opt == make_result.a_z_opt
        assert loaded.a_zz_opt == make_result.a_zz_opt
        assert loaded.expectation_Jz == make_result.expectation_Jz
        assert loaded.variance_Jz == make_result.variance_Jz
        assert loaded.T_hold == make_result.T_hold
        assert loaded.fd_step == make_result.fd_step
        assert loaded.success == make_result.success
        assert loaded.nfev == make_result.nfev

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"N": [5], "omega": [0.2]})
        p = tmp_path / "missing.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            NScalingResult.from_parquet(p)

    def test_missing_a_x_opt_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "N": [5],
                "omega": [0.2],
                "delta_omega_opt": [0.015],
                "sql": [0.0447],
                "ratio": [2.98],
                "a_y_opt": [0.0],
                "a_z_opt": [0.0],
                "a_zz_opt": [0.0],
                "expectation_Jz": [0.0],
                "variance_Jz": [0.0],
                "T_hold": [10.0],
                "fd_step": [1e-6],
                "success": [1],
                "nfev": [0],
            },
        )
        p = tmp_path / "missing_ax.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            NScalingResult.from_parquet(p)


class TestNScalingScanResult:
    @pytest.fixture
    def make_results(self) -> list[NScalingResult]:
        return [
            NScalingResult(
                N=n,
                omega=w,
                delta_omega_opt=0.02 / (n**0.5),
                sql=1.0 / (np.sqrt(n) * 10.0),
                ratio=(1.0 / (np.sqrt(n) * 10.0)) / (0.02 / (n**0.5)),
                a_x_opt=float(n),
                a_y_opt=0.0,
                a_z_opt=0.0,
                a_zz_opt=float(n),
                success=True,
                nfev=50,
            )
            for n in [1, 2, 5, 10]
            for w in [0.1, 0.2, 0.5]
        ]

    def test_roundtrip(
        self,
        make_results: list[NScalingResult],
        tmp_path: Path,
    ) -> None:
        summary = NScalingScanResult(results=make_results)
        p = tmp_path / "scan.parquet"
        summary.save_parquet(p)
        loaded = NScalingScanResult.from_parquet(p)
        assert len(loaded.results) == len(make_results)
        for orig, loaded_r in zip(make_results, loaded.results, strict=False):
            assert orig.N == loaded_r.N
            assert orig.omega == loaded_r.omega

    def test_empty(self) -> None:
        summary = NScalingScanResult(results=[])
        df = summary.to_dataframe()
        assert df.empty

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"N": [5], "omega": [0.2]})
        p = tmp_path / "bad_scan.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            NScalingScanResult.from_parquet(p)

    def test_N_values(
        self,
        make_results: list[NScalingResult],
    ) -> None:
        summary = NScalingScanResult(results=make_results)
        assert len(summary.N_values) == 4
        assert len(summary.omega_values) == 3
