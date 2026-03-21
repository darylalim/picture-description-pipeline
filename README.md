# Granite Vision Pipeline

Extract and describe pictures and tables in PDF documents, and segment objects in images using natural language prompts. Powered by [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b) with [SAM](https://huggingface.co/facebook/sam-vit-huge) refinement for segmentation.

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Features

**PDF Extraction** — Upload a PDF to extract pictures with AI-generated descriptions and tables with structured data. Results available as JSON download with per-element previews.

**Image Segmentation (Experimental)** — Upload an image and describe what to segment in natural language. Granite Vision generates a coarse mask, refined by SAM for pixel-accurate results. Available via the "Segmentation" page in the sidebar.

## Project Structure

```
pipeline/
  __init__.py        # public API re-exports
  config.py          # converter factory, convert wrapper
  output.py          # unified element builder, description and table extraction
  segmentation.py    # segmentation pipeline, SAM refinement, model loaders
pages/
  segmentation.py    # segmentation UI page
streamlit_app.py     # PDF extraction UI
tests/
  test_config.py     # converter factory and pipeline option tests
  test_output.py     # element builder, description, and table content tests
  test_segmentation.py # segmentation helper unit tests
```
