# Miguel's Playground

Streamlit apps + Jupyter notebooks for quantum physics simulations.

## UI Colour Scheme

The application uses a **semantic colour scheme** for consistent visual language across all pages. Each colour conveys meaning about the type of content being displayed.

### Semantic Colour Mapping

| Colour | Semantic Purpose | Usage |
|--------|-------------------|-------|
| `blue` | **PRIMARY** | Main page header, page title, key concepts |
| `gray` | **SETUP/CONFIG** | Sidebar settings, configuration sections, parameter inputs |
| `blue` | **METHODOLOGY** | Explanatory sections, methodology (consistent with PRIMARY) |
| `green` | **RESULTS/SUCCESS** | Energy levels, computation results, positive outcomes, potential V(x) |
| `orange` | **DATA/WARNING** | Raw data sections, intermediate steps, Likelihood, Ancillary sections |
| `red` | **SUMMARY/CRITICAL** | Final results, conclusions, summaries, Evolution sections |
| `violet` | **SPECIAL** | Ancilla systems, special quantum features (Density Matrix in MZI_Ancilla) |

### Usage in Code

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

## Setup

```bash
# Python dependencies
uv sync

# Jupyter kernels
uv pip install ipykernel
python -m ipykernel install --user --name=playground
```

## Dev Commands

| Command | Description |
|---------|-------------|
| `uv run streamlit run Home.py` | Start Streamlit app |
| `uv run pytest . --quiet --tb=short` | Run tests |
| `uv run mypy .` | Type check |
| `uv run ruff check . --fix` | Lint |
| `uv run ruff format .` | Format |
| `jupyter-book build .` | Build docs to `_build/` |

## Projects

- `pages/` — Streamlit pages (MZI simulations)
- `src/` — Core physics modules
- `jupyter/` — Notebooks by category
- `mathematica/` — Mathematica notebooks

## Notebooks

```bash
# Launch Jupyter
jupyter notebook
```

Add new kernels via `python -m ipykernel install --user --name=<name>`.

## Mathematica

1. Download from [julialang.org](https://julialang.org/downloads/)
2. Extract: `tar zxvf julia-*-linux-x86_64.tar.gz`
3. Run: `./julia`