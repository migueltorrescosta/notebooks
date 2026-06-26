"""Tests for the coupling-sweep orchestration module."""

from __future__ import annotations

import numpy as np

from src.analysis.coupling_sweep import resolve_sweep_defaults, run_sweep_base


class TestResolveSweepDefaults:
    def test_both_provided(self) -> None:
        omega = np.array([0.1, 0.5, 1.0], dtype=float)
        N = np.array([1, 2, 3], dtype=int)
        omega_out, N_out = resolve_sweep_defaults(omega, N)
        assert np.array_equal(omega_out, omega)
        assert np.array_equal(N_out, N)

    def test_default_omega(self) -> None:
        N = np.array([1, 5], dtype=int)
        _, N_out = resolve_sweep_defaults(N_values=N, omega_values=None)
        assert np.array_equal(N_out, N)

    def test_default_N(self) -> None:
        omega = np.array([0.1, 1.0], dtype=float)
        omega_out, _ = resolve_sweep_defaults(omega_values=omega, N_values=None)
        assert np.array_equal(omega_out, omega)

    def test_both_default(self) -> None:
        omega_out, N_out = resolve_sweep_defaults(None, None)
        assert len(omega_out) > 0
        assert len(N_out) > 0
        assert N_out[0] == 1
        assert N_out[-1] == 20


class TestRunSweepBase:
    def test_simple_sweep(self) -> None:
        def per_point(N: int, omega: float) -> dict:
            return {"delta_omega": 1.0 / (N + 1), "sql": 1.0}

        omega_arr = np.array([0.1, 0.5], dtype=float)
        N_arr = np.array([1, 2], dtype=int)
        data = run_sweep_base(omega_arr, N_arr, per_point)

        assert len(data["omegas"]) == 4
        assert len(data["Ns"]) == 4
        assert len(data["delta_opts"]) == 4
        assert len(data["sqls"]) == 4
        assert len(data["ratio"]) == 4

        # Verify ordering: N=1 ω=0.1, N=1 ω=0.5, N=2 ω=0.1, N=2 ω=0.5
        assert data["Ns"][0] == 1
        assert data["omegas"][1] == 0.5
        assert data["Ns"][2] == 2

    def test_ratio_computed_when_missing(self) -> None:
        def per_point(N: int, omega: float) -> dict:
            return {"delta_omega": 0.5, "sql": 1.0}

        data = run_sweep_base(
            np.array([0.1], dtype=float),
            np.array([1], dtype=int),
            per_point,
        )
        assert np.isclose(data["ratio"][0], 0.5)

    def test_ratio_respected_when_provided(self) -> None:
        def per_point(N: int, omega: float) -> dict:
            return {"delta_omega": 0.5, "sql": 1.0, "ratio": 0.99}

        data = run_sweep_base(
            np.array([0.1], dtype=float),
            np.array([1], dtype=int),
            per_point,
        )
        assert np.isclose(data["ratio"][0], 0.99)

    def test_extra_keys_stored(self) -> None:
        def per_point(N: int, omega: float) -> dict:
            return {
                "delta_omega": 0.1,
                "sql": 1.0,
                "expectation_Jz": float(N) * 0.5,
                "variance_Jz": float(N) * 0.1,
                "d_expectation": 0.01,
                "alpha_xx_opt": float(N) * 0.3,
            }

        omega_arr = np.array([0.1, 1.0], dtype=float)
        N_arr = np.array([1, 2], dtype=int)
        data = run_sweep_base(omega_arr, N_arr, per_point)

        assert "expectation_Jz" in data
        assert "variance_Jz" in data
        assert "d_expectation" in data
        assert "alpha_xx_opt" in data
        assert np.isclose(data["expectation_Jz"][0], 0.5)
        assert np.isclose(data["variance_Jz"][2], 0.2)

    def test_progress_callback(self) -> None:
        calls: list[tuple[int, int]] = []

        def per_point(N: int, omega: float) -> dict:
            return {"delta_omega": 0.1, "sql": 1.0}

        def cb(current: int, total: int) -> None:
            calls.append((current, total))

        run_sweep_base(
            np.array([0.1, 0.5], dtype=float),
            np.array([1, 2], dtype=int),
            per_point,
            progress_callback=cb,
        )

        assert len(calls) == 4
        assert calls[-1] == (4, 4)

    def test_delta_opts_inf_initialised(self) -> None:
        def per_point(N: int, omega: float) -> dict:
            return {"delta_omega": 0.5, "sql": 1.0}

        data = run_sweep_base(
            np.array([0.1], dtype=float),
            np.array([1], dtype=int),
            per_point,
        )
        # Should be overwritten to finite value
        assert np.isfinite(data["delta_opts"][0])

    def test_inf_delta_yields_inf_ratio(self) -> None:
        def per_point(N: int, omega: float) -> dict:
            return {"delta_omega": np.inf, "sql": 1.0}

        data = run_sweep_base(
            np.array([0.1], dtype=float),
            np.array([1], dtype=int),
            per_point,
        )
        assert np.isinf(data["ratio"][0])

    def test_zero_sql_yields_inf_ratio(self) -> None:
        def per_point(N: int, omega: float) -> dict:
            return {"delta_omega": 0.5, "sql": 0.0}

        data = run_sweep_base(
            np.array([0.1], dtype=float),
            np.array([1], dtype=int),
            per_point,
        )
        assert np.isinf(data["ratio"][0])
