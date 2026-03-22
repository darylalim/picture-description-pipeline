# Granite Vision Pipeline

Streamlit web app with four capabilities:

1. **PDF Extraction** — extract and describe pictures and tables in PDF documents
2. **Image Segmentation** — segment objects in images using natural language prompts
3. **DocTags Generation** — parse document images and PDFs to structured text in doctags format
4. **Multipage QA** — answer questions across up to 8 document pages

Powered by [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b), [SAM](https://huggingface.co/facebook/sam-vit-huge), and [granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M). Navigate between features using the sidebar.

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Features

**PDF Extraction** — Upload a PDF to extract pictures with AI-generated descriptions and tables with structured data. Results available as JSON download with per-element previews.

**Image Segmentation (Experimental)** — Upload an image and describe what to segment in natural language. Granite Vision generates a coarse mask, refined by SAM for pixel-accurate results.

**DocTags Generation (Experimental)** — Upload a document image or PDF to generate structured doctags output. View raw doctags and converted Markdown side-by-side, with per-page results for multi-page PDFs.

**Multipage QA (Experimental)** — Upload a PDF or up to 8 images and ask questions about the content. Images are resized to 768px max dimension for GPU memory efficiency. Answers are displayed alongside page thumbnails.

## Project Structure

```
pipeline/
  __init__.py          # public API re-exports
  config.py            # converter factory, convert wrapper
  output.py            # unified element builder, description and table extraction
  segmentation.py      # segmentation pipeline, SAM refinement, model loaders
  doctags.py           # doctags generation, parsing, PDF rendering, model loaders
  qa.py                # multipage QA model loader, image resizing, inference
pages/
  segmentation.py      # segmentation UI page
  doctags.py           # doctags generation UI page
  qa.py                # multipage QA UI page
streamlit_app.py       # PDF extraction UI (main page)
tests/
  test_config.py       # converter factory and pipeline option tests
  test_output.py       # element builder, description, and table content tests
  test_segmentation.py # segmentation helper unit tests
  test_doctags.py      # doctags rendering, parsing, inference, and export tests
  test_qa.py           # QA resizing, model factory, and inference tests
```
