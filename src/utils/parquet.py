"""Shared Parquet consolidation utilities for report raw-data directories.

Parallel parameter-sweep pipelines write one Parquet file per parameter
value (e.g., per omega).  After all workers finish, the individual shards
can be merged into a single file per schema group to reduce file count
and speed up downstream loading.

Usage::

    from src.utils.parquet import consolidate_raw_parquet

    consolidate_raw_parquet(
        raw_data_dir=Path("reports/20260519/raw_data"),
        glob_pattern="20260519-phase-2d-slice-*-azz-omega*.parquet",
        output_name="20260519-phase-2d-slice-ax-azz.parquet",
        delete_shards=True,
    )
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def consolidate_raw_parquet(
    raw_data_dir: Path,
    glob_pattern: str,
    output_name: str,
    delete_shards: bool = False,
) -> int:
    """Merge all Parquet shards matching *glob_pattern* into a single file.

    Files are grouped by column schema (set of column names) so that
    files with incompatible schemas are never concatenated.  Each schema
    group produces one merged file.  If all files share the same schema,
    the output goes to ``raw_data_dir / output_name``.

    Parameters
    ----------
    raw_data_dir :
        Directory containing the shard Parquet files.
    glob_pattern :
        Glob pattern to select shards (e.g., ``"20260519-*-omega*.parquet"``).
    output_name :
        Filename for the merged output (written inside *raw_data_dir*).
    delete_shards :
        If True, remove the individual shard files after merging.

    Returns
    -------
    int
        Total number of rows written to the merged file.

    Raises
    ------
    FileNotFoundError
        If no files match *glob_pattern* in *raw_data_dir*.
    """
    raw_data_dir = Path(raw_data_dir)
    shard_paths = sorted(raw_data_dir.glob(glob_pattern))

    if not shard_paths:
        raise FileNotFoundError(
            f"No files matching '{glob_pattern}' in {raw_data_dir}"
        )

    # Group by column schema to avoid merging incompatible files
    schema_groups: dict[frozenset[str], list[Path]] = {}
    for path in shard_paths:
        meta = pq.read_metadata(path)
        key = frozenset(meta.schema.names)
        schema_groups.setdefault(key, []).append(path)

    total_rows = 0

    if len(schema_groups) == 1:
        # Single schema — write directly to the requested output name
        cols, paths = next(iter(schema_groups.items()))
        tables = [pq.read_table(p) for p in paths]
        merged = pa.concat_tables(tables, promote_options="default")
        out_path = raw_data_dir / output_name
        pq.write_table(merged, out_path)
        total_rows = len(merged)

        if delete_shards:
            for p in paths:
                p.unlink()

        print(
            f"[merge] {len(paths)} files -> {out_path.name} "
            f"({total_rows} rows)"
        )
    else:
        # Multiple schemas — write one file per schema group
        print(
            f"[merge] {len(schema_groups)} schema groups detected; "
            f"writing one file per group"
        )
        for cols, paths in sorted(
            schema_groups.items(), key=lambda x: -len(x[1])
        ):
            tables = [pq.read_table(p) for p in paths]
            merged = pa.concat_tables(tables, promote_options="default")

            # Derive output name from the first three column names
            col_slug = "-".join(sorted(cols)[:3])
            stem = Path(output_name).stem
            out_path = raw_data_dir / f"{stem}-{col_slug}.parquet"
            pq.write_table(merged, out_path)
            total_rows += len(merged)

            if delete_shards:
                for p in paths:
                    p.unlink()

            print(
                f"  [merge] {len(paths)} files -> {out_path.name} "
                f"({len(merged)} rows)"
            )

    return total_rows
