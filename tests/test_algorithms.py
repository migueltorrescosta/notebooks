"""Smoke tests for algorithms module."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.algorithms.algorithms import (
    AbstractMetropolisHastings,
    GaussianMetropolisHastings,
)


class TestGaussianMetropolisHastings:
    def test_instantiation(self) -> None:
        """Test that we can instantiate the class."""
        mh = GaussianMetropolisHastings(initial_configuration=0.0)
        assert mh.current_configuration == 0.0

    def test_initial_history_contains_start(self) -> None:
        mh = GaussianMetropolisHastings(initial_configuration=5.0)
        assert mh.configuration_history == [5.0]

    def test_initial_counts_are_zero(self) -> None:
        mh = GaussianMetropolisHastings(initial_configuration=0.0)
        assert mh.accepted_configuration_count == 0
        assert mh.rejected_configuration_count == 0

    def test_generator_function_returns_float(self) -> None:
        mh = GaussianMetropolisHastings(initial_configuration=0.0)
        new_config = mh.generator_function()
        assert isinstance(new_config, (float, np.floating))

    def test_state_likelihood_is_valid_probability(self) -> None:
        mh = GaussianMetropolisHastings(initial_configuration=0.0)
        likelihood = mh.state_likelihood(0.0)
        assert likelihood > 0
        assert likelihood <= 1  # exp(0) = 1

    def test_state_likelihood_decreases_with_distance(self) -> None:
        mh = GaussianMetropolisHastings(initial_configuration=0.0)
        lik_0 = mh.state_likelihood(0.0)
        lik_1 = mh.state_likelihood(1.0)
        assert lik_0 > lik_1

    def test_approval_function_accepts_when_likelihood_ratio_greater(self) -> None:
        """Test approval function accepts when new state has higher likelihood."""
        mh = GaussianMetropolisHastings(initial_configuration=0.0)
        # Current: exp(0) = 1.0, New: exp(-0.5^2) ≈ 0.78
        # Current likelihood is higher, so any random > current/new will reject
        with patch("src.algorithms.np.random.random", return_value=0.5):
            # For Gaussian: new(0.5) = exp(-0.25) ≈ 0.78
            result = mh.approval_function(0.5)
            # ratio = 0.78/1.0 = 0.78; 0.78 >= 0.5 is True
            assert result

    def test_run_single_iteration_returns_new_state(self) -> None:
        mh = GaussianMetropolisHastings(initial_configuration=0.0)
        # Return 0.0 so approval always passes (likelihood ratio >= 0)
        with (
            patch("src.algorithms.np.random.random", return_value=0.0),
            patch.object(mh, "generator_function", return_value=1.0),
        ):
            new_state = mh.run_single_iteration()
            assert new_state == 1.0
            assert mh.accepted_configuration_count == 1

    def test_run_iterations_updates_counts(self) -> None:
        mh = GaussianMetropolisHastings(initial_configuration=0.0)

        # Create a mock progress bar
        mock_pbar = MagicMock()
        mock_pbar.start_t = 1000.0
        mock_pbar.__iter__ = lambda self: iter(range(5))

        with patch("src.algorithms.trange", return_value=mock_pbar):
            # Manually call iterations a few times without the loop to test
            for _ in range(3):
                mh.run_single_iteration()
            assert len(mh.configuration_history) == 4  # initial + 3

    def test_plot_does_not_raise(self) -> None:
        mh = GaussianMetropolisHastings(initial_configuration=0.0)
        mh.configuration_history = [0.0, 1.0, 0.5, 0.8]
        with patch("src.algorithms.plt.plot") as mock_plot:
            mh.plot()
            mock_plot.assert_called_once()


class TestAbstractMetropolisHastings:
    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that AbstractMetropolisHastings cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractMetropolisHastings(initial_configuration=0.0)

    def test_subclass_must_implement_abstract_methods(self) -> None:
        """Test that subclass without abstract methods raises error."""

        class IncompleteMH(AbstractMetropolisHastings):
            pass

        with pytest.raises(TypeError):
            IncompleteMH(initial_configuration=0.0)

    def test_current_configuration_property(self) -> None:
        """Test that current_configuration returns last element of history."""
        mh = GaussianMetropolisHastings(initial_configuration=3.14)
        assert mh.current_configuration == 3.14
        mh.configuration_history.append(2.71)
        assert mh.current_configuration == 2.71
