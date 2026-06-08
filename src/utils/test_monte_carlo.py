"""
Tests for the Monte Carlo sampling utilities (src.utils.monte_carlo).

Run with:
    uv run pytest src/utils/test_monte_carlo.py -q --tb=short
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import kstest

if TYPE_CHECKING:
    from collections.abc import Callable

from src.utils.monte_carlo import (
    StratifiedBallConfig,
    marsaglia_ball_sample,
    stratified_ball_sample,
)


def _uniform_cdf(lo: float, hi: float) -> Callable[[float], float]:
    """Return the CDF of Uniform(lo, hi) as a callable."""
    width = hi - lo

    def _cdf(x: float) -> float:
        return (x - lo) / width

    return _cdf


# ── Reusable constants ───────────────────────────────────────────────────────

R = 10.0
AZZ_LO = -5.0
AZZ_HI = 5.0


# ============================================================================
# StratifiedBallConfig
# ============================================================================


class TestStratifiedBallConfig:
    def test_default_values(self) -> None:
        cfg = StratifiedBallConfig()
        assert cfg.n_strata == 10
        assert cfg.n_per_stratum == 500

    def test_custom_values(self) -> None:
        cfg = StratifiedBallConfig(n_strata=20, n_per_stratum=100)
        assert cfg.n_strata == 20
        assert cfg.n_per_stratum == 100

    def test_is_dataclass(self) -> None:
        cfg = StratifiedBallConfig()
        assert hasattr(cfg, "__dataclass_fields__")


# ============================================================================
# Marsaglia Ball Sample
# ============================================================================


class TestMarsagliaBallSample:
    def test_shape(self) -> None:
        rng = np.random.default_rng(42)
        drive, azz = marsaglia_ball_sample(rng, 100, R, AZZ_LO, AZZ_HI)
        assert drive.shape == (100, 3)
        assert azz.shape == (100,)

    def test_norms_within_ball(self) -> None:
        rng = np.random.default_rng(42)
        drive, _ = marsaglia_ball_sample(rng, 1000, R, AZZ_LO, AZZ_HI)
        norms = np.sqrt(np.sum(drive**2, axis=1))
        assert np.all(norms <= R + 1e-12), "Samples must be within ball"
        assert np.all(norms >= 0.0), "Norms must be non-negative"

    def test_azz_within_bounds(self) -> None:
        rng = np.random.default_rng(42)
        _, azz = marsaglia_ball_sample(rng, 1000, R, AZZ_LO, AZZ_HI)
        assert np.all(azz >= AZZ_LO - 1e-12)
        assert np.all(azz <= AZZ_HI + 1e-12)

    def test_uniform_distribution(self) -> None:
        """KS test: P(||a|| ≤ r) = (r/R)³ for 3-ball."""
        rng = np.random.default_rng(42)
        drive, _ = marsaglia_ball_sample(rng, 5000, R, AZZ_LO, AZZ_HI)
        norms = np.sqrt(np.sum(drive**2, axis=1))
        stat, pval = kstest(norms, lambda x: (x / R) ** 3)
        assert stat < 0.05, (
            f"KS statistic {stat:.4f} too large (p={pval:.4f}); "
            "distribution may not be uniform in the ball"
        )

    def test_deterministic_with_seed(self) -> None:
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        d1, a1 = marsaglia_ball_sample(rng1, 100, R, AZZ_LO, AZZ_HI)
        d2, a2 = marsaglia_ball_sample(rng2, 100, R, AZZ_LO, AZZ_HI)
        assert np.allclose(d1, d2)
        assert np.allclose(a1, a2)

    def test_zero_radius(self) -> None:
        rng = np.random.default_rng(42)
        drive, azz = marsaglia_ball_sample(rng, 10, 0.0, AZZ_LO, AZZ_HI)
        assert np.allclose(drive, 0.0)
        assert azz.shape == (10,)


# ============================================================================
# Stratified Ball Sample
# ============================================================================


class TestStratifiedBallSample:
    def test_shape(self) -> None:
        rng = np.random.default_rng(42)
        drive, azz = stratified_ball_sample(rng, 100, 10, R, AZZ_LO, AZZ_HI)
        assert drive.shape == (1000, 3)  # 10 × 100
        assert azz.shape == (1000,)

    def test_norms_within_ball(self) -> None:
        rng = np.random.default_rng(42)
        drive, _ = stratified_ball_sample(rng, 200, 5, R, AZZ_LO, AZZ_HI)
        norms = np.sqrt(np.sum(drive**2, axis=1))
        assert np.all(norms <= R + 1e-12), "Samples must be within ball"
        assert np.all(norms >= 0.0), "Norms must be non-negative"

    def test_azz_within_bounds(self) -> None:
        rng = np.random.default_rng(42)
        _, azz = stratified_ball_sample(rng, 200, 5, R, AZZ_LO, AZZ_HI)
        assert np.all(azz >= AZZ_LO - 1e-12)
        assert np.all(azz <= AZZ_HI + 1e-12)

    def test_deterministic_with_seed(self) -> None:
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        d1, a1 = stratified_ball_sample(rng1, 100, 10, R, AZZ_LO, AZZ_HI)
        d2, a2 = stratified_ball_sample(rng2, 100, 10, R, AZZ_LO, AZZ_HI)
        assert np.allclose(d1, d2)
        assert np.allclose(a1, a2)

    def test_stratum_counts(self) -> None:
        """Each stratum produces exactly n_per_stratum samples."""
        rng = np.random.default_rng(42)
        n_per = 50
        n_strata = 8
        drive, _ = stratified_ball_sample(rng, n_per, n_strata, R, AZZ_LO, AZZ_HI)
        norms = np.sqrt(np.sum(drive**2, axis=1))
        r_bounds = np.linspace(0.0, R, n_strata + 1)
        for i in range(n_strata):
            r_lo = r_bounds[i]
            r_hi = r_bounds[i + 1]
            # Samples with norm in (r_lo, r_hi] — allow boundary tolerance
            mask = (norms > r_lo - 1e-12) & (norms <= r_hi + 1e-12)
            count = int(np.sum(mask))
            assert count == n_per, (
                f"Stratum [{r_lo:.2f}, {r_hi:.2f}] has {count} samples, "
                f"expected {n_per}"
            )

    def test_radius_uniform_within_stratum(self) -> None:
        """Within each stratum, radii should be uniform (KS test)."""
        rng = np.random.default_rng(99)
        n_per = 200
        n_strata = 5
        drive, _ = stratified_ball_sample(rng, n_per, n_strata, R, AZZ_LO, AZZ_HI)
        norms = np.sqrt(np.sum(drive**2, axis=1))
        r_bounds = np.linspace(0.0, R, n_strata + 1)
        for i in range(n_strata):
            r_lo = r_bounds[i]
            r_hi = r_bounds[i + 1]
            mask = (norms > r_lo - 1e-12) & (norms <= r_hi + 1e-12)
            stratum_norms = norms[mask]
            # CDF: Uniform(r_lo, r_hi)
            stat, pval = kstest(stratum_norms, _uniform_cdf(r_lo, r_hi))
            assert stat < 0.1, (
                f"Stratum [{r_lo:.2f}, {r_hi:.2f}] KS stat {stat:.4f} "
                f"(p={pval:.4f}) — radii may not be uniform"
            )

    def test_small_r_coverage(self) -> None:
        """With n_strata=100, the first stratum [0, 0.1] has exactly n_per_stratum."""
        rng = np.random.default_rng(42)
        n_per = 50
        n_strata = 100
        drive, _ = stratified_ball_sample(rng, n_per, n_strata, R, AZZ_LO, AZZ_HI)
        norms = np.sqrt(np.sum(drive**2, axis=1))
        # First stratum: [0, 0.1]
        count_r_le_01 = int(np.sum(norms <= 0.1 + 1e-12))
        assert count_r_le_01 == n_per, (
            f"First stratum [0, 0.1] has {count_r_le_01} samples, expected {n_per}"
        )

    def test_zero_radius(self) -> None:
        rng = np.random.default_rng(42)
        drive, azz = stratified_ball_sample(rng, 10, 5, 0.0, AZZ_LO, AZZ_HI)
        assert np.allclose(drive, 0.0)
        assert azz.shape == (50,)

    def test_single_stratum_equals_uniform_ball(self) -> None:
        """With n_strata=1, stratified_ball_sample should approximate uniform
        volume sampling (within tolerance)."""
        rng = np.random.default_rng(42)
        n_samp = 5000
        drive, _ = stratified_ball_sample(rng, n_samp, 1, R, AZZ_LO, AZZ_HI)
        norms = np.sqrt(np.sum(drive**2, axis=1))
        # Single stratum [0, R] with uniform radius → P(r) = 1/R, not (r/R)^3
        # KS test against Uniform(0, R)
        stat, pval = kstest(norms, _uniform_cdf(0.0, R))
        assert stat < 0.05, (
            f"Single-stratum KS stat {stat:.4f} (p={pval:.4f}) "
            "— radii should be uniform in [0, R]"
        )

    def test_stratified_vs_marsaglia_small_r_count(self) -> None:
        """Stratified sampling gives many more samples at small r than Marsaglia
        for the same total count."""
        rng_m = np.random.default_rng(42)
        rng_s = np.random.default_rng(42)
        n_total = 5000
        # Marsaglia: 5000 samples
        drive_m, _ = marsaglia_ball_sample(rng_m, n_total, R, AZZ_LO, AZZ_HI)
        norms_m = np.sqrt(np.sum(drive_m**2, axis=1))
        count_m = int(np.sum(norms_m <= 2.15))
        # Stratified: 50 strata × 100 per = 5000
        drive_s, _ = stratified_ball_sample(rng_s, 100, 50, R, AZZ_LO, AZZ_HI)
        norms_s = np.sqrt(np.sum(drive_s**2, axis=1))
        count_s = int(np.sum(norms_s <= 2.15))
        # At n_strata=50, the first (2.15/10)*50 ≈ 10.75 strata cover r ≤ 2.15
        # → about 10 or 11 strata × 100 = ~1050 samples
        assert count_s >= 5 * count_m, (
            f"Stratified count at r≤2.15: {count_s}, "
            f"Marsaglia count: {count_m} "
            f"(ratio {count_s / max(count_m, 1):.1f}x, expected ≥5x)"
        )
