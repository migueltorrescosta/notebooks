# Miguel's Playground
I use these [jupyter](https://jupyter.org/) notebooks to and [streamlit](https://streamlit.io/) pages to play with different topics.

We use [poetry](https://python-poetry.org/) to manage dependencies. The local streamlit apps can be accessed on https://collapsedwave.strealit.app.

# Frequent commands
```bash
poetry install --no-root # Install all dependencies
```

```bash
poetry update # check for later versions of the used dependencies
```

```bash
poetry run python -m streamlit run Home.py # Run local streamlit app
```

```bash
jupyter-book build . # Build the jupyter notebook. I haven't used this command in a while, so it is likely to be broken
```

## Notes

Julia was installed from https://julialang.org/downloads/
After download, run the command below to have julia available as an executable.
```bash
tar zxvf julia-1.11.5-linux-x86_64.tar.gz # Replace the filename if using a different version
```