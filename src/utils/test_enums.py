"""Smoke tests for enums module."""

from .enums import (
    BoundaryCondition,
    PotentialFunction,
    ProbabilityDistribution,
    WavePacket,
)


class TestWavePacket:
    def test_gaussian_exists(self) -> None:
        assert WavePacket.Gaussian is not None, (
            "Expected WavePacket.Gaussian to not be None"
        )
        assert WavePacket.Gaussian.value == "Gaussian", (
            'Expected WavePacket.Gaussian.value == "Gaussian"'
        )

    def test_step_exists(self) -> None:
        assert WavePacket.Step is not None, "Expected WavePacket.Step to not be None"
        assert WavePacket.Step.value == "Step function", (
            'Expected WavePacket.Step.value == "Step function"'
        )

    def test_all_are_strings(self) -> None:
        assert all(isinstance(wp.value, str) for wp in WavePacket), (
            "Expected all(isinstance(wp.value, str) for wp in WavePacket)"
        )


class TestPotentialFunction:
    def test_double_well_exists(self) -> None:
        assert PotentialFunction.DoubleWell is not None, (
            "Expected PotentialFunction.DoubleWell to not be None"
        )

    def test_quadratic_exists(self) -> None:
        assert PotentialFunction.Quadratic is not None, (
            "Expected PotentialFunction.Quadratic to not be None"
        )

    def test_quartic_exists(self) -> None:
        assert PotentialFunction.Quartic is not None, (
            "Expected PotentialFunction.Quartic to not be None"
        )

    def test_trigonometric_exists(self) -> None:
        assert PotentialFunction.Trigonometric is not None, (
            "Expected PotentialFunction.Trigonometric to not be None"
        )

    def test_uniform_exists(self) -> None:
        assert PotentialFunction.Uniform is not None, (
            "Expected PotentialFunction.Uniform to not be None"
        )


class TestBoundaryCondition:
    def test_cyclic_exists(self) -> None:
        assert BoundaryCondition.Cyclic is not None, (
            "Expected BoundaryCondition.Cyclic to not be None"
        )
        assert BoundaryCondition.Cyclic.value == "Cyclic", (
            'Expected BoundaryCondition.Cyclic.value == "Cyclic"'
        )

    def test_dirichlet_exists(self) -> None:
        assert BoundaryCondition.Dirichlet is not None, (
            "Expected BoundaryCondition.Dirichlet to not be None"
        )
        assert BoundaryCondition.Dirichlet.value == "Dirichlet", (
            'Expected BoundaryCondition.Dirichlet.value == "Dirichlet"'
        )


class TestProbabilityDistribution:
    def test_particle_decay_exists(self) -> None:
        assert ProbabilityDistribution.ParticleDecay is not None, (
            "Expected ProbabilityDistribution.ParticleDecay to not be None"
        )
