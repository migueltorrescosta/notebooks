# Miguel's Playground

Streamlit apps and Jupyter notebooks for quantum physics simulations.

## UI color scheme

The application uses a semantic color scheme for consistent visual language across all pages. Each color conveys meaning about the type of content being displayed.

### Semantic color mapping

| Color | Semantic purpose | Usage |
|-------|------------------|-------|
| `blue` | PRIMARY | Main page header, page title, key concepts |
| `gray` | SETUP/CONFIG | Sidebar settings, configuration sections, parameter inputs |
| `blue` | METHODOLOGY | Explanatory sections, methodology (consistent with PRIMARY) |
| `green` | RESULTS/SUCCESS | Energy levels, computation results, positive outcomes, potential V(x) |
| `orange` | DATA/WARNING | Raw data sections, intermediate steps, Likelihood, Ancillary sections |
| `red` | SUMMARY/CRITICAL | Final results, conclusions, summaries, Evolution sections |
| `violet` | SPECIAL | Ancilla systems, special quantum features (Density Matrix in MZI_Ancilla) |

### Usage in code

```python
# Page header (PRIMARY)
st.header("Page Title", divider="blue")

# Sidebar setup (SETUP/CONFIG)
with st.sidebar:
    st.header("Setup", divider="gray")

# Results section (RESULTS/SUCCESS)
st.header("Energy Levels", divider="green")

# Data section (DATA/WARNING)
st.header("Raw Data", divider="orange")

# Summary section (SUMMARY/CRITICAL)
st.header("Summary", divider="red")

# Special feature (SPECIAL)
st.header("Density Matrix", divider="violet")
```

## Testing conventions

### Slow tests

Tests that take longer than 5 seconds must be decorated with `@pytest.mark.slow`.
By default, slow tests are excluded via `addopts = "-m 'not slow'"` in `pyproject.toml`.
To run all the tests, including the slow ones:

```bash
uv run pytest . -m "not slow"          # default: fast tests only
uv run pytest . -m slow                # slow tests only
uv run pytest . -m ""                  # ALL tests (slow and fast)
```

### Parquet test fixtures

Tests that generate full-size Parquet datasets (for example, `reports/r20260625/`) use
pre-computed reference fixtures under `reports/*/tests/fixtures/` instead of
regenerating data from scratch. To regenerate the fixtures when the data schemas change:

```bash
uv run pytest . --regenerate-fixtures   # re-generates all fixtures
```

The fixture files are committed to the repository. They become stale if the data schemas
change. Run them with `--regenerate-fixtures` in CI or locally when the schemas are updated.

## Setup

```bash
# Python dependencies
uv sync

# Jupyter kernels
uv pip install ipykernel
python -m ipykernel install --user --name=playground
```

## Dev commands

| Command | Description |
|---------|-------------|
| `uv run streamlit run Home.py` | Start Streamlit app |
| `uv run pytest . --quiet --tb=short` | Run tests (fast subset, excludes slow) |
| `uv run pytest . --quiet --tb=short -m "not slow"` | Run only non-slow tests (default) |
| `uv run pytest . -m slow` | Run only slow tests (CI nightly) |
| `uv run pytest . --regenerate-fixtures` | Re-generate parquet test fixtures from scratch |
| `uv run mypy .` | Type check |
| `uv run ruff check . --fix` | Lint |
| `uv run ruff format .` | Format |
| `jupyter-book build .` | Build docs to `_build/` |

## Projects

- `pages/`: Streamlit pages (MZI simulations)
- `src/`: Core physics modules
- `jupyter/`: Notebooks by category
- `mathematica/`: Mathematica notebooks

## Notebooks

```bash
# Launch Jupyter
jupyter notebook
```

Add new kernels with `python -m ipykernel install --user --name=<name>`.

## Mathematica

1. Download from [julialang.org](https://julialang.org/downloads/)
2. Extract: `tar zxvf julia-*-linux-x86_64.tar.gz`
3. Run: `./julia`