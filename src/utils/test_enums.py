"""Smoke tests for enums module."""

import pytest

from .enums import (
    BoundaryCondition,
    OperatorBasis,
    PotentialFunction,
    ProbabilityDistribution,
    WavePacket,
)


class TestWavePacket:
    @pytest.mark.parametrize(
        ("member", "expected_value"),
        [
            (WavePacket.Gaussian, "Gaussian"),
            (WavePacket.Step, "Step function"),
        ],
        ids=["Gaussian", "Step"],
    )
    def test_given_member_then_has_expected_value(
        self, member: WavePacket, expected_value: str
    ) -> None:
        assert member.value == expected_value

    def test_given_all_members_then_values_are_strings(self) -> None:
        assert all(isinstance(wp.value, str) for wp in WavePacket)


class TestPotentialFunction:
    @pytest.mark.parametrize(
        ("member", "expected_value"),
        [
            (PotentialFunction.DoubleWell, "Double-well"),
            (PotentialFunction.Quadratic, "Quadratic"),
            (PotentialFunction.Quartic, "Quartic"),
            (PotentialFunction.Trigonometric, "Trigonometric"),
            (PotentialFunction.Uniform, "Uniform"),
        ],
        ids=["DoubleWell", "Quadratic", "Quartic", "Trigonometric", "Uniform"],
    )
    def test_given_member_then_has_expected_value(
        self, member: PotentialFunction, expected_value: str
    ) -> None:
        assert member.value == expected_value


class TestBoundaryCondition:
    @pytest.mark.parametrize(
        ("member", "expected_value"),
        [
            (BoundaryCondition.Cyclic, "Cyclic"),
            (BoundaryCondition.Dirichlet, "Dirichlet"),
        ],
        ids=["Cyclic", "Dirichlet"],
    )
    def test_given_member_then_has_expected_value(
        self, member: BoundaryCondition, expected_value: str
    ) -> None:
        assert member.value == expected_value


class TestProbabilityDistribution:
    def test_given_particle_decay_then_has_expected_value(self) -> None:
        assert ProbabilityDistribution.ParticleDecay.value == "ParticleDecay"


class TestOperatorBasis:
    @pytest.mark.parametrize(
        ("member", "expected_value"),
        [
            (OperatorBasis.DICKE, "dicke"),
            (OperatorBasis.FOCK, "fock"),
        ],
        ids=["DICKE", "FOCK"],
    )
    def test_given_member_then_has_expected_value(
        self, member: OperatorBasis, expected_value: str
    ) -> None:
        assert member.value == expected_value
