"""Smoke tests for enums module."""

from src.enums import (
    BoundaryCondition,
    PotentialFunction,
    ProbabilityDistribution,
    WavePacket,
)


class TestWavePacket:
    def test_gaussian_exists(self):
        assert WavePacket.Gaussian is not None
        assert WavePacket.Gaussian.value == "Gaussian"

    def test_step_exists(self):
        assert WavePacket.Step is not None
        assert WavePacket.Step.value == "Step function"

    def test_all_are_strings(self):
        assert all(isinstance(wp.value, str) for wp in WavePacket)


class TestPotentialFunction:
    def test_double_well_exists(self):
        assert PotentialFunction.DoubleWell is not None

    def test_quadratic_exists(self):
        assert PotentialFunction.Quadratic is not None

    def test_quartic_exists(self):
        assert PotentialFunction.Quartic is not None

    def test_trigonometric_exists(self):
        assert PotentialFunction.Trigonometric is not None

    def test_uniform_exists(self):
        assert PotentialFunction.Uniform is not None


class TestBoundaryCondition:
    def test_cyclic_exists(self):
        assert BoundaryCondition.Cyclic is not None
        assert BoundaryCondition.Cyclic.value == "Cyclic"

    def test_dirichlet_exists(self):
        assert BoundaryCondition.Dirichlet is not None
        assert BoundaryCondition.Dirichlet.value == "Dirichlet"


class TestProbabilityDistribution:
    def test_particle_decay_exists(self):
        assert ProbabilityDistribution.ParticleDecay is not None
