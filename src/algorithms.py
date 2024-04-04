import datetime
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

T = TypeVar("T")


class AbstractMetropolisHastings(ABC, Generic[T]):
    # Based on https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm#Description

    def __init__(self, initial_configuration: T):
        self.configuration_history: List[T] = [initial_configuration]
        self.accepted_configuration_count: int = 0
        self.rejected_configuration_count: int = 0

    @property
    def current_configuration(self) -> T:
        return self.configuration_history[-1]

    @abstractmethod
    def generator_function(self) -> T:
        pass

    @abstractmethod
    def state_likelihood(self, configuration: T) -> float:
        # This is proportional to the state probability
        pass

    def approval_function(self, new_configuration: T) -> bool:
        return (
            self.state_likelihood(new_configuration)
            >= self.state_likelihood(self.current_configuration) * np.random.random()
        )

    def run_single_iteration(self, limit_tries=10**5) -> T:
        tries = 0
        while True:
            new_state = self.generator_function()
            if self.approval_function(new_state):
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
                    f"Rejected {100*rejection_rate:.1f}%",
                    refresh=True,
                )

    def plot(self) -> None:
        plt.plot(self.configuration_history)
