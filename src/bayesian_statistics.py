from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype


class BayesUpdate:
    def __init__(self, df: DataFrame) -> None:
        """Dataframe with a likelihood and prior column"""
        assert set(df.columns) == {"prior", "likelihood"}
        self.df = df
        self.calculate_posterior()
        self.normalize_columns()
        self.is_categorical: bool = not is_numeric_dtype(df.index)

    def normalize_columns(self) -> None:
        self.df = self.df / self.df.sum(axis=0)

    def calculate_posterior(self) -> None:
        posterior = self.df["prior"] * self.df["likelihood"]
        self.df["posterior"] = posterior / sum(posterior)

    def plot(self) -> Figure:
        fig, ax = plt.subplots(figsize=(15, 5), ncols=3, sharey=True)
        if self.is_categorical:
            self.df.plot.bar(ax=ax, subplots=True)
        else:
            self.df.plot.area(ax=ax, subplots=True)
        return fig
