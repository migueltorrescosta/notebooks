"""
Bayesian updating for discrete parameter estimation.

This module provides utilities for performing Bayesian inference on discrete
parameter spaces. It handles prior distributions, likelihood functions, and
computes the posterior distribution using Bayes' rule.

Useful for quantum metrology problems where one wishes to infer unknown
parameters (like phase shifts) from measurement outcomes.
"""

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype


class BayesUpdate:
    """Bayesian posterior computation for discrete parameter spaces.

    Takes a DataFrame with prior and likelihood columns and computes the
    posterior using Bayes' rule: P(θ|data) ∝ P(data|θ) × P(θ).

    The class normalizes the posterior automatically and provides visualization
    methods for comparing prior, likelihood, and posterior distributions.

    Attributes:
        df: DataFrame with columns [prior, likelihood, posterior].
        is_categorical: True if the index is non-numeric (categorical parameters).

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {"prior": [0.25, 0.25, 0.25, 0.25], "likelihood": [0.9, 0.6, 0.3, 0.1]},
        ...     index=[0, 1, 2, 3]
        ... )
        >>> bayes = BayesUpdate(df)
        >>> bayes.df["posterior"]  # Posterior probabilities
        0    0.5625
        1    0.3750
        2    0.1875
        3    0.0625
        Name: posterior, dtype: float64
    """

    def __init__(self, df: DataFrame) -> None:
        """Initialize Bayesian update with prior and likelihood data.

        Args:
            df: DataFrame with exactly two columns: 'prior' and 'likelihood'.
                The index represents the parameter values being estimated.

        Raises:
            AssertionError: If columns are not exactly {prior, likelihood}.
        """
        assert set(df.columns) == {"prior", "likelihood"}
        self.df = df
        self.calculate_posterior()
        self.normalize_columns()
        self.is_categorical: bool = not is_numeric_dtype(df.index)

    def normalize_columns(self) -> None:
        """Normalize probability columns to sum to 1.

        Normalizes both prior and posterior columns so they represent
        valid probability distributions.
        """
        self.df = self.df / self.df.sum(axis=0)

    def calculate_posterior(self) -> None:
        """Compute posterior using Bayes' rule.

        Multiplies prior by likelihood and normalizes to produce the
        posterior distribution over parameter values.
        """
        posterior = self.df["prior"] * self.df["likelihood"]
        self.df["posterior"] = posterior / sum(posterior)

    def plot(self) -> Figure:
        """Plot prior, likelihood, and posterior distributions.

        Creates a three-panel figure comparing the three distributions.
        For categorical parameters (non-numeric index), uses bar charts.
        For numeric indices, uses area plots.

        Returns:
            Matplotlib Figure object containing the three subplots.
        """
        fig, ax = plt.subplots(figsize=(15, 5), ncols=3, sharey=True)
        if self.is_categorical:
            self.df.plot.bar(ax=ax, subplots=True)
        else:
            self.df.plot.area(ax=ax, subplots=True)
        return fig
