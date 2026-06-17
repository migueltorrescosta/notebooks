"""
Abstract base class for Parquet-serializable dataclasses.

Provides ``save_parquet()`` and ``_validate_columns()`` scaffolding,
while leaving ``to_dataframe()``/``from_parquet()`` abstract per subclass.

Usage::

    from __future__ import annotations

    from dataclasses import dataclass
    from pathlib import Path
    from typing import ClassVar

    import pandas as pd

    from src.utils.serialization import ParquetSerializable


    @dataclass
    class MyResult(ParquetSerializable):
        x: float
        y: float

        _PARQUET_COLUMNS: ClassVar[list[str]] = ["x", "y"]

        def to_dataframe(self) -> pd.DataFrame:
            return pd.DataFrame({"x": [self.x], "y": [self.y]})

        @classmethod
        def from_parquet(cls, path: str | Path) -> MyResult:
            df = pd.read_parquet(path)
            cls._validate_columns(df)
            return cls(x=float(df["x"].iloc[0]), y=float(df["y"].iloc[0]))
"""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd


class ParquetSerializable(ABC):
    """Abstract base class for dataclasses with Parquet roundtrip.

    Subclasses **must** define:

    - ``_PARQUET_COLUMNS: ClassVar[list[str]]`` — all column names that
      must be present in the Parquet file.
    - ``to_dataframe() -> pd.DataFrame`` — serialise the instance.
    - ``from_parquet(cls, path) -> Self`` — deserialise from a Parquet file.

    Optional hooks for sidecar files:

    - ``_save_sidecars(self, path)`` — called at the end of ``save_parquet``.
    - ``_load_sidecars(cls, path) -> dict`` — override to return sidecar data
      that should be passed into the constructor in ``from_parquet``.
    """

    _PARQUET_COLUMNS: ClassVar[list[str]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Enforce that every concrete subclass defines ``_PARQUET_COLUMNS``."""
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "_PARQUET_COLUMNS"):
            raise TypeError(
                f"Class {cls.__name__} must define _PARQUET_COLUMNS ClassVar"
            )

    # ── Abstract interface ──────────────────────────────────────────────

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """Serialize this instance to a single DataFrame.

        Returns:
            DataFrame with one or more rows containing all required fields.
        """
        ...

    @classmethod
    @abstractmethod
    def from_parquet(cls, path: str | Path) -> ParquetSerializable:
        """Reconstruct an instance from a Parquet file written by ``save_parquet``.

        Args:
            path: Path to the Parquet file.

        Returns:
            Reconstructed instance.

        Raises:
            ValueError: If required columns are missing.
        """
        ...

    # ── Concrete scaffolding ────────────────────────────────────────────

    def save_parquet(self, path: str | Path) -> Path:
        """Save this instance to a Parquet file.

        Creates parent directories as needed, writes the DataFrame via
        ``to_dataframe()``, then calls ``_save_sidecars()`` if overridden.

        Args:
            path: Destination file path (``str`` or ``Path``).

        Returns:
            The ``Path`` that was written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        self._save_sidecars(path)
        return path

    @classmethod
    def _validate_columns(cls, df: pd.DataFrame) -> None:
        """Fail-fast column validation against ``_PARQUET_COLUMNS``.

        Args:
            df: DataFrame read from a Parquet file.

        Raises:
            ValueError: Listing every column that is missing from the
                Parquet file, directing the user to re-run the simulation.
        """
        missing = [c for c in cls._PARQUET_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Parquet file missing required columns: {sorted(missing)}. "
                "Re-run the simulation that generated it."
            )

    # ── Optional sidecar hooks ──────────────────────────────────────────

    def _save_sidecars(self, path: Path) -> None:
        """Override to save additional sidecar files alongside the main Parquet.

        Called at the end of ``save_parquet()``, after the main DataFrame
        has been written.

        Args:
            path: The main Parquet file path (already written to).
        """
        return

    @classmethod
    def _load_sidecars(cls, path: Path) -> dict:
        """Override to load sidecar data for use in ``from_parquet``.

        Args:
            path: The main Parquet file path.

        Returns:
            Dictionary of sidecar data (e.g. ``{"history": [...]}``).
            Defaults to an empty dict.
        """
        return {}


# ── Roundtrip test helper ──────────────────────────────────────────

_comparators: dict[str, Any] = {
    "eq": operator.eq,
    "isclose": lambda a, b: bool(np.isclose(a, b)),
    "allclose": lambda a, b: bool(np.allclose(a, b)),
    "array_eq": np.array_equal,
}


def assert_roundtrip_fields(
    loaded: object,
    original: object,
    field_specs: list[tuple[str, str]],
) -> None:
    """Assert that all specified fields survive a Parquet roundtrip.

    Args:
        loaded: Deserialized instance.
        original: Original instance.
        field_specs: Sequence of ``(field_name, comparator_key)`` tuples
            where *comparator_key* is ``'eq'``, ``'isclose'``,
            ``'allclose'``, or ``'array_eq'``.

    Raises:
        AssertionError: On the first field whose comparison fails,
            with ``original`` and ``loaded`` values in the message.
    """
    for field, comp_key in field_specs:
        orig_val = getattr(original, field)  # type: ignore[arg-type]
        loaded_val = getattr(loaded, field)  # type: ignore[arg-type]
        comp = _comparators[comp_key]
        assert comp(loaded_val, orig_val), (
            f"Field '{field}' roundtrip mismatch.\n"
            f"  original: {orig_val!r}\n"
            f"  loaded:   {loaded_val!r}"
        )
