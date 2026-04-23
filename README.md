# Miguel's Playground

Streamlit apps + Jupyter notebooks for quantum physics simulations.

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