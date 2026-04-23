"""
Metropolis-Hastings algorithm implementations for statistical sampling.

This module provides abstract base classes and concrete implementations for
the Metropolis-Hastings MCMC algorithm, useful for sampling from complex
probability distributions in quantum state estimation problems.
"""

import datetime
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

T = TypeVar("T")


class AbstractMetropolisHastings(ABC, Generic[T]):
    """Abstract base class for Metropolis-Hastings MCMC sampling.

    Implements the core Metropolis-Hastings algorithm for generating
    correlated samples from a target distribution. Subclasses must define
    how to propose new configurations and compute their likelihoods.

    The algorithm works by proposing random perturbations to the current
    configuration and accepting/rejecting based on the likelihood ratio.
    This creates a Markov chain that asymptotically samples from the
    target distribution.

    Attributes:
        configuration_history: List of accepted configurations.
        accepted_configuration_count: Number of accepted proposals.
        rejected_configuration_count: Number of rejected proposals.
    """

    def __init__(self, initial_configuration: T) -> None:
        """Initialize the sampler with a starting configuration.

        Args:
            initial_configuration: Starting point for the Markov chain.
                Must have non-zero probability under the target distribution.
        """
        self.configuration_history: List[T] = [initial_configuration]
        self.accepted_configuration_count: int = 0
        self.rejected_configuration_count: int = 0

    @property
    def current_configuration(self) -> T:
        """Return the most recent accepted configuration."""
        return self.configuration_history[-1]

    @abstractmethod
    def generator_function(self) -> T:
        """Generate a proposal configuration from the current state.

        Returns:
            A new proposed configuration based on the current state.
            Common choices: Gaussian random walk, uniform perturbation, etc.
        """

    @abstractmethod
    def state_likelihood(self, configuration: T) -> float:
        """Compute unnormalized probability of a configuration.

        Args:
            configuration: The configuration to evaluate.

        Returns:
            Unnormalized probability (likelihood) under the target distribution.
            Only needs to be proportional to the true probability.
        """

    def approval_function(
        self, new_configuration: T, current_likelihood: float
    ) -> bool:
        """Determine whether to accept a proposed configuration.

        Uses the standard Metropolis acceptance ratio: min(1, P_new/P_curr).
        For symmetric proposal distributions, this simplifies to comparing
        likelihoods directly.

        Args:
            new_configuration: Proposed new configuration.
            current_likelihood: Likelihood of the current configuration.

        Returns:
            True if the proposal is accepted, False otherwise.
        """
        return (
            self.state_likelihood(new_configuration)
            >= current_likelihood * np.random.random()
        )

    def run_single_iteration(self, limit_tries: int = 10**5) -> T:
        """Execute one iteration of the Metropolis-Hastings algorithm.

        Proposes a new configuration and accepts/rejects based on likelihood.
        If rejected, increments the rejection counter and returns the current
        configuration.

        Args:
            limit_tries: Maximum number of proposal attempts before
                printing a warning (useful for debugging flat likelihoods).

        Returns:
            The newly accepted configuration (or current if rejected).
        """
        current_likelihood = self.state_likelihood(self.current_configuration)
        tries = 0
        while True:
            new_state = self.generator_function()
            if self.approval_function(new_state, current_likelihood):
                self.configuration_history.append(new_state)
                self.accepted_configuration_count += 1
                return new_state

            self.rejected_configuration_count += 1
            tries += 1
            if tries >= limit_tries:
                # Useful for debugging
                tries = 0
                limit_tries *= int(1.1)
                print(f"{new_state:e}", end=", ")

    def run_iterations(self, n: int) -> None:
        """Run the Metropolis-Hastings algorithm for n iterations.

        Executes n iterations while displaying a progress bar with the
        current rejection rate. The rejection rate is a useful diagnostic:
        too high (>99%) suggests the proposal distribution is poorly tuned,
        too low (<1%) may indicate the chain is mixing too slowly.

        Args:
            n: Number of iterations to run.
        """
        pbar = trange(n, desc="Bar desc", leave=True)
        for _ in pbar:
            self.run_single_iteration()

            # Update the progress bar roughly once a second
            seconds_passed = datetime.datetime.now().timestamp() - pbar.start_t
            n = self.rejected_configuration_count + self.accepted_configuration_count
            iterations_per_second = 1 + int(n / seconds_passed)
            update_frequency = 2 ** (int(np.log(iterations_per_second) / np.log(2)) - 1)

            if n % update_frequency == 0:
                rejection_rate = np.divide(
                    self.rejected_configuration_count,
                    self.accepted_configuration_count
                    + self.rejected_configuration_count,
                )
                pbar.set_description(
                    f"Rejected {100 * rejection_rate:.1f}%",
                    refresh=True,
                )

    def plot(self) -> None:
        """Plot the configuration history as a time series.

        Useful for visualizing the trajectory of the Markov chain and
        checking for convergence.
        """
        plt.plot(np.asarray(self.configuration_history))


class GaussianMetropolisHastings(AbstractMetropolisHastings[float]):
    """Metropolis-Hastings sampler for a standard normal distribution.

    A concrete implementation demonstrating the abstract base class.
    Uses a Gaussian random walk proposal centered on the current state,
    which targets a standard normal distribution N(0, 1) as the stationary
    distribution.

    The algorithm works because the proposal is symmetric, so the acceptance
    probability reduces to min(1, exp(-x_new²) / exp(-x_curr²)) = min(1, exp(-(x_new² - x_curr²))).
    This creates a Markov chain that converges to N(0, 1).

    Example:
        >>> sampler = GaussianMetropolisHastings(initial_configuration=0)
        >>> sampler.run_iterations(10**5)
        >>> samples = sampler.configuration_history[1000:]  # discard burn-in
        >>> np.mean(samples)  # should be close to 0
        >>> np.std(samples)   # should be close to 1
    """

    def generator_function(self) -> float:
        """Propose a new sample using Gaussian random walk.

        Adds a standard normal perturbation to the current configuration.
        This symmetric proposal simplifies the acceptance criterion.
        """
        return self.current_configuration + np.random.normal(0, 1)

    def state_likelihood(self, configuration: float) -> float:
        """Compute unnormalized probability under N(0, 1).

        Returns exp(-x²), which is proportional to the standard normal PDF
        up to a normalizing constant.
        """
        return np.exp(-1 * configuration**2)
