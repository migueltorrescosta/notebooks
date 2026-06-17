"""Tests for ParquetSerializable ABC."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from src.utils.serialization import ParquetSerializable

# ── Concrete subclass for testing ─────────────────────────────────────


@dataclass
class _ConcreteResult(ParquetSerializable):
    """Minimal concrete subclass for testing the ABC scaffolding."""

    x: float = 1.0
    y: float = 2.0
    history: list[float] = field(default_factory=list)

    _PARQUET_COLUMNS: ClassVar[list[str]] = ["x", "y"]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"x": [self.x], "y": [self.y]})

    @classmethod
    def from_parquet(cls, path: str | Path) -> _ConcreteResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        sidecar = cls._load_sidecars(Path(path))
        return cls(
            x=float(df["x"].iloc[0]),
            y=float(df["y"].iloc[0]),
            history=sidecar.get("history", []),
        )

    def _save_sidecars(self, path: Path) -> None:
        if self.history:
            hist_path = path.with_stem(path.stem + "-history")
            pd.DataFrame({"history": [self.history]}).to_parquet(hist_path, index=False)

    @classmethod
    def _load_sidecars(cls, path: Path) -> dict:
        hist_path = path.with_stem(path.stem + "-history")
        if hist_path.exists():
            raw = pd.read_parquet(hist_path)["history"].iloc[0]
            return {"history": list(raw)}
        return {}


# ── Tests ─────────────────────────────────────────────────────────────


class TestParquetSerializableABC:
    """Tests for the abstract base class scaffolding."""

    def test_instantiation_ok(self) -> None:
        """Concrete subclass can be instantiated."""
        r = _ConcreteResult(x=3.0, y=4.0)
        assert r.x == 3.0
        assert r.y == 4.0

    def test_abstract_class_cannot_be_instantiated(self) -> None:
        """ParquetSerializable cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ParquetSerializable()  # type: ignore[abstract]

    def test_save_parquet_creates_file(self, tmp_path: Path) -> None:
        """save_parquet writes a Parquet file at the given path."""
        r = _ConcreteResult(x=5.0, y=6.0)
        out = tmp_path / "test.parquet"
        result_path = r.save_parquet(out)
        assert result_path == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_parquet_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_parquet creates parent directories automatically."""
        r = _ConcreteResult(x=1.0, y=2.0)
        out = tmp_path / "nested" / "deep" / "test.parquet"
        result_path = r.save_parquet(out)
        assert result_path == out
        assert out.exists()

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Data survives save_parquet -> from_parquet roundtrip."""
        r = _ConcreteResult(x=10.0, y=20.0)
        path = r.save_parquet(tmp_path / "roundtrip.parquet")
        loaded = _ConcreteResult.from_parquet(path)
        assert loaded.x == 10.0
        assert loaded.y == 20.0

    def test_validate_columns_ok(self) -> None:
        """_validate_columns passes when all columns present."""
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        # Should not raise
        _ConcreteResult._validate_columns(df)

    def test_validate_columns_fails(self) -> None:
        """_validate_columns raises ValueError when columns missing."""
        df = pd.DataFrame({"x": [1.0]})  # missing "y"
        with pytest.raises(ValueError, match="missing required columns"):
            _ConcreteResult._validate_columns(df)

    def test_from_parquet_fails_on_missing_columns(self, tmp_path: Path) -> None:
        """from_parquet fails fast when columns are missing."""
        # Write a bad Parquet with only one column
        bad_df = pd.DataFrame({"x": [1.0]})
        bad_path = tmp_path / "bad.parquet"
        bad_df.to_parquet(bad_path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            _ConcreteResult.from_parquet(bad_path)

    def test_sidecar_roundtrip(self, tmp_path: Path) -> None:
        """_save_sidecars / _load_sidecars roundtrip works."""
        r = _ConcreteResult(x=1.0, y=2.0, history=[0.5, 1.0, 1.5])
        path = r.save_parquet(tmp_path / "sidecar.parquet")
        loaded = _ConcreteResult.from_parquet(path)
        assert loaded.history == [0.5, 1.0, 1.5]

    def test_no_sidecar_when_empty(self, tmp_path: Path) -> None:
        """_save_sidecars does nothing when history is empty."""
        r = _ConcreteResult(x=1.0, y=2.0)
        path = r.save_parquet(tmp_path / "noside.parquet")
        loaded = _ConcreteResult.from_parquet(path)
        assert loaded.history == []

    def test_save_parquet_return_type(self, tmp_path: Path) -> None:
        """save_parquet returns a Path."""
        r = _ConcreteResult(x=1.0, y=2.0)
        result = r.save_parquet(tmp_path / "test.parquet")
        assert isinstance(result, Path)

    def test_save_parquet_accepts_str(self, tmp_path: Path) -> None:
        """save_parquet accepts str path."""
        r = _ConcreteResult(x=1.0, y=2.0)
        result = r.save_parquet(str(tmp_path / "str_path.parquet"))
        assert isinstance(result, Path)
        assert result.exists()


class TestParquetSerializableNoSubclass:
    """Tests that abstractness is enforced."""

    def test_missing_PARQUET_COLUMNS_raises(self) -> None:
        """Subclass without _PARQUET_COLUMNS raises TypeError at class definition."""
        with pytest.raises(TypeError):

            class _Bad(ParquetSerializable):  # type: ignore[abstract]
                def to_dataframe(self) -> pd.DataFrame:
                    return pd.DataFrame()

                @classmethod
                def from_parquet(cls, path: str | Path) -> _Bad:
                    return cls()

    def test_missing_to_dataframe_raises(self) -> None:
        """Subclass without to_dataframe cannot be instantiated."""

        class _Bad2(ParquetSerializable):  # type: ignore[abstract]
            _PARQUET_COLUMNS: ClassVar[list[str]] = ["a"]

            @classmethod
            def from_parquet(cls, path: str | Path) -> _Bad2:
                return cls()

        with pytest.raises(TypeError):
            _Bad2()

    def test_missing_from_parquet_raises(self) -> None:
        """Subclass without from_parquet cannot be instantiated."""

        class _Bad3(ParquetSerializable):  # type: ignore[abstract]
            _PARQUET_COLUMNS: ClassVar[list[str]] = ["a"]

            def to_dataframe(self) -> pd.DataFrame:
                return pd.DataFrame()

        with pytest.raises(TypeError):
            _Bad3()
