"""Tests for the shared sampling utilities."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.sampling import sample_6d_config, sample_ball_3d


class TestSampleBall3D:
    @pytest.mark.parametrize("radius", [0.0, 1.0, 5.0])
    def test_sample_ball_3d_within_radius(self, radius: float) -> None:
        rng = np.random.default_rng(42)
        for _ in range(100):
            vec = sample_ball_3d(rng, radius=radius)
            assert vec.shape == (3,)
            assert np.linalg.norm(vec) <= radius * (1 + 1e-12), (
                f"Vector norm {np.linalg.norm(vec)} exceeds radius {radius}"
            )

    def test_sample_ball_3d_zero_radius(self) -> None:
        rng = np.random.default_rng(42)
        vec = sample_ball_3d(rng, radius=0.0)
        assert np.allclose(vec, 0.0)

    def test_sample_ball_3d_reproducible(self) -> None:
        rng1 = np.random.default_rng(12345)
        rng2 = np.random.default_rng(12345)
        v1 = sample_ball_3d(rng1)
        v2 = sample_ball_3d(rng2)
        assert np.allclose(v1, v2), "Reproducibility failure"


class TestSample6DConfig:
    def test_sample_6d_config_shape(self) -> None:
        rng = np.random.default_rng(42)
        result = sample_6d_config(rng)
        assert len(result) == 6
        theta_A, phi_A, a_x, a_y, a_z, _a_zz = result
        assert 0.0 <= theta_A <= np.pi
        assert 0.0 <= phi_A <= 2.0 * np.pi
        assert np.linalg.norm([a_x, a_y, a_z]) <= 1.0 * (1 + 1e-12)

    def test_sample_6d_config_azz_bounds(self) -> None:
        rng = np.random.default_rng(42)
        _, _, _, _, _, a_zz = sample_6d_config(rng, azz_bounds=(0.0, 0.0))
        assert np.isclose(a_zz, 0.0)

        _, _, _, _, _, a_zz = sample_6d_config(rng, azz_bounds=(-1.0, 1.0))
        assert -1.0 <= a_zz <= 1.0

    def test_sample_6d_config_drive_radius(self) -> None:
        rng = np.random.default_rng(42)
        result = sample_6d_config(rng, drive_radius=2.0)
        _, _, a_x, a_y, a_z, _ = result
        assert np.linalg.norm([a_x, a_y, a_z]) <= 2.0 * (1 + 1e-12)

    def test_sample_6d_config_reproducible(self) -> None:
        rng1 = np.random.default_rng(999)
        rng2 = np.random.default_rng(999)
        c1 = sample_6d_config(rng1)
        c2 = sample_6d_config(rng2)
        for x, y in zip(c1, c2, strict=True):
            assert np.isclose(x, y), "Reproducibility failure"
