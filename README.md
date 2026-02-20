# Picture Description Pipeline

Describe pictures in PDF documents with a local [IBM Granite Vision](https://huggingface.co/ibm-granite/granite-vision-3.3-2b) model.

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Project Structure

```
pipeline/
  __init__.py        # public API re-exports
  config.py          # constants, converter factory, convert wrapper
  output.py          # output builder
streamlit_app.py     # Streamlit UI
tests/
  test_config.py     # config and convert tests
  test_output.py     # output builder tests
```
