"""Tests for src.utils.parquet consolidation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.utils.parquet import consolidate_raw_parquet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_shard(
    path: Path,
    columns: dict[str, list],
) -> None:
    """Write a single-row Parquet file with the given columns."""
    pd.DataFrame(columns).to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Single-schema merging
# ---------------------------------------------------------------------------


class TestConsolidateSingleSchema:
    def test_given_three_shards_then_merged_file_has_all_rows(
        self, tmp_path: Path
    ) -> None:
        for i in range(3):
            _write_shard(
                tmp_path / f"shard-{i}.parquet",
                {"x": [float(i)], "y": [float(i * 10)]},
            )

        n = consolidate_raw_parquet(
            tmp_path, "shard-*.parquet", "merged.parquet"
        )

        assert n == 3
        out = tmp_path / "merged.parquet"
        assert out.exists()
        df = pd.read_parquet(out)
        assert list(df.columns) == ["x", "y"]
        assert len(df) == 3

    def test_given_delete_shards_then_originals_removed(
        self, tmp_path: Path
    ) -> None:
        for i in range(2):
            _write_shard(tmp_path / f"s{i}.parquet", {"a": [float(i)]})

        consolidate_raw_parquet(
            tmp_path, "s*.parquet", "out.parquet", delete_shards=True
        )

        assert (tmp_path / "out.parquet").exists()
        assert list(tmp_path.glob("s*.parquet")) == []

    def test_given_no_delete_shards_then_originals_kept(
        self, tmp_path: Path
    ) -> None:
        _write_shard(tmp_path / "s0.parquet", {"a": [1.0]})

        consolidate_raw_parquet(
            tmp_path, "s*.parquet", "out.parquet", delete_shards=False
        )

        assert (tmp_path / "s0.parquet").exists()
        assert (tmp_path / "out.parquet").exists()

    def test_given_no_matches_then_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No files matching"):
            consolidate_raw_parquet(
                tmp_path, "nonexistent-*.parquet", "out.parquet"
            )

    def test_given_single_file_then_output_has_one_row(
        self, tmp_path: Path
    ) -> None:
        _write_shard(tmp_path / "only.parquet", {"z": [42.0]})

        n = consolidate_raw_parquet(
            tmp_path, "only*.parquet", "single.parquet"
        )

        assert n == 1
        df = pd.read_parquet(tmp_path / "single.parquet")
        assert df["z"].iloc[0] == 42.0

    def test_given_metadata_columns_preserved(self, tmp_path: Path) -> None:
        """omega_value and sql metadata columns survive the merge."""
        for omega in [0.5, 1.0, 1.5]:
            pd.DataFrame(
                {
                    "delta_omega": [float(omega)],
                    "omega_value": [omega],
                    "sql": [0.1],
                }
            ).to_parquet(tmp_path / f"o{omega}.parquet", index=False)

        consolidate_raw_parquet(
            tmp_path, "o*.parquet", "all-omegas.parquet"
        )

        df = pd.read_parquet(tmp_path / "all-omegas.parquet")
        assert set(df["omega_value"]) == {0.5, 1.0, 1.5}
        assert all(df["sql"] == 0.1)


# ---------------------------------------------------------------------------
# Multi-schema merging
# ---------------------------------------------------------------------------


class TestConsolidateMultiSchema:
    def test_given_two_schemas_then_two_output_files(
        self, tmp_path: Path
    ) -> None:
        _write_shard(tmp_path / "a1.parquet", {"x": [1.0]})
        _write_shard(tmp_path / "a2.parquet", {"x": [2.0]})
        _write_shard(tmp_path / "b1.parquet", {"x": [1.0], "y": [10.0]})

        n = consolidate_raw_parquet(
            tmp_path, "*.parquet", "merged.parquet"
        )

        assert n == 3  # 2 + 1 rows total
        # Two output files
        outputs = sorted(tmp_path.glob("merged-*.parquet"))
        assert len(outputs) == 2

    def test_given_multi_schema_with_delete_then_only_outputs_remain(
        self, tmp_path: Path
    ) -> None:
        _write_shard(tmp_path / "s1.parquet", {"a": [1.0]})
        _write_shard(tmp_path / "s2.parquet", {"b": [1.0], "c": [2.0]})

        consolidate_raw_parquet(
            tmp_path, "*.parquet", "out.parquet", delete_shards=True
        )

        remaining = sorted(tmp_path.glob("*.parquet"))
        names = [p.name for p in remaining]
        # Only merged outputs remain, no s1/s2
        assert all(n.startswith("out-") for n in names)
        assert not any(n.startswith("s") for n in names)
