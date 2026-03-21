# Granite Vision Pipeline

Extract and describe pictures and tables in PDF documents using a local vision language model: [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b).

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
  output.py          # unified element builder, description and table extraction
streamlit_app.py     # Streamlit UI
tests/
  test_config.py     # converter factory and pipeline option tests
  test_output.py     # element builder, description, and table content tests
```
