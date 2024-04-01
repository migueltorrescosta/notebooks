from typing import TypeVar, Generic, List
from tqdm import trange
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize

T = TypeVar("T")


figsize(20, 5)


class AbstractMetropolisHastings(ABC, Generic[T]):
    def __init__(self, initial_configuration: T):
        self.configuration_history: List[T] = [
            initial_configuration
        ]  # This array evolves over time :D
        self.accepted_configuration_count: int = 0
        self.rejected_configuration_count: int = 0

    @property
    def current_configuration(self) -> T:
        return self.configuration_history[-1]

    @abstractmethod
    def generator_function(self) -> T:
        pass

    @abstractmethod
    def approval_function(self, new_configuration: T) -> bool:
        pass

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
            if (
                self.accepted_configuration_count + self.rejected_configuration_count
            ) % 10**3 == 0:
                pbar.set_description(
                    f"({self.rejected_configuration_count}/{self.accepted_configuration_count  +self.rejected_configuration_count}) {self.current_configuration:e} ",
                    refresh=True,
                )

    def plot(self) -> None:
        plt.plot(self.configuration_history)
