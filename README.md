# Picture Description Pipeline

Describe pictures in PDF documents using a local vision language model.

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Project Structure

```
pipeline/
  __init__.py        # public API re-exports
  config.py          # converter factory, convert wrapper
  output.py          # output builder, description extractor
streamlit_app.py     # Streamlit UI
tests/
  test_config.py     # config and convert tests
  test_output.py     # output builder tests
```
