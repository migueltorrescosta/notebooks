"""Tests for the shared path helpers."""

from __future__ import annotations

from pathlib import Path

from src.utils.paths import fig_path, parquet_path, report_path_fn


def test_parquet_path_standard() -> None:
    """parquet_path constructs the expected path from (reports_dir, date, name)."""
    result = parquet_path(Path("/home/reports"), "20260616", "n-scaling-scan")
    expected = Path("/home/reports/20260616/raw_data/20260616-n-scaling-scan.parquet")
    assert result == expected


def test_parquet_path_without_dashes() -> None:
    """Works with compact date format (no dashes)."""
    result = parquet_path(Path("reports"), "20260519", "omega-scan")
    expected = Path("reports/20260519/raw_data/20260519-omega-scan.parquet")
    assert result == expected


def test_parquet_path_with_dashes() -> None:
    """Works with dash-separated date prefix."""
    result = parquet_path(Path("reports"), "2026-05-25", "random-search")
    expected = Path("reports/2026-05-25/raw_data/2026-05-25-random-search.parquet")
    assert result == expected


def test_fig_path_standard() -> None:
    """fig_path constructs the expected path from (reports_dir, date, name)."""
    result = fig_path(Path("/home/reports"), "20260616", "ratio-vs-n")
    expected = Path("/home/reports/20260616/figures/20260616-ratio-vs-n.svg")
    assert result == expected


def test_fig_path_with_dashes() -> None:
    """Works with dash-separated date prefix."""
    result = fig_path(Path("reports"), "2026-05-25", "2d-slice")
    expected = Path("reports/2026-05-25/figures/2026-05-25-2d-slice.svg")
    assert result == expected


def test_roundtrip_parquet_suffix() -> None:
    """Parquet path always ends with .parquet."""
    for date, name in [("20260616", "foo"), ("2026-05-25", "bar-baz")]:
        p = parquet_path(Path("/r"), date, name)
        assert p.suffix == ".parquet", f"Expected .parquet suffix, got {p.suffix}"


def test_roundtrip_fig_suffix() -> None:
    """Fig path always ends with .svg."""
    for date, name in [("20260616", "foo"), ("2026-05-25", "bar-baz")]:
        p = fig_path(Path("/r"), date, name)
        assert p.suffix == ".svg", f"Expected .svg suffix, got {p.suffix}"


def test_reports_dir_preserved() -> None:
    """The reports_dir argument is used as the first path component."""
    reports_dir = Path("/custom/path/to/reports")
    p = parquet_path(reports_dir, "20260601", "mzi-scan")
    assert str(p).startswith(str(reports_dir))


def test_report_path_fn_parquet() -> None:
    """report_path_fn parquet closure matches parquet_path(reports_dir, date, name)."""
    pq_fn, _ = report_path_fn(Path("/r"), "20260616")
    result = pq_fn("n-scaling-scan")
    expected = parquet_path(Path("/r"), "20260616", "n-scaling-scan")
    assert result == expected


def test_report_path_fn_fig() -> None:
    """report_path_fn fig closure matches fig_path(reports_dir, date, name)."""
    _, fig_fn = report_path_fn(Path("/r"), "20260616")
    result = fig_fn("ratio-vs-n")
    expected = fig_path(Path("/r"), "20260616", "ratio-vs-n")
    assert result == expected


def test_report_path_fn_different_dates() -> None:
    """Each date gets its own independent closure pair."""
    pq_a, fig_a = report_path_fn(Path("/r"), "20260601")
    pq_b, fig_b = report_path_fn(Path("/r"), "20260616")
    assert pq_a("foo") != pq_b("foo")
    assert fig_a("bar") != fig_b("bar")


def test_report_path_fn_different_dirs() -> None:
    """Each reports_dir gets its own independent closure pair."""
    pq_a, _ = report_path_fn(Path("/a"), "20260601")
    pq_b, _ = report_path_fn(Path("/b"), "20260616")
    assert pq_a("foo") != pq_b("foo")
