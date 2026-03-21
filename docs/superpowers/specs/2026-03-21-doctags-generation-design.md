# DocTags Generation — Design Spec

## Overview

Add a new Streamlit page for generating doctags from document images and PDFs using `ibm-granite/granite-docling-258M`. Doctags is IBM's markup format that represents document structure (text, tables, pictures, formulas, code) with bounding-box coordinates in a single token sequence. Users upload an image or PDF and receive raw doctags output plus converted Markdown.

## Architecture

Follows the established pattern: core logic in `pipeline/doctags.py`, UI in `pages/doctags.py`.

### New Files

- `pipeline/doctags.py` — model loading, inference, doctags parsing, PDF page rendering
- `pages/doctags.py` — Streamlit UI page
- `tests/test_doctags.py` — unit tests for pipeline logic

### Modified Files

- `pipeline/__init__.py` — re-export public API (`create_doctags_model`, `generate_doctags`, `parse_doctags`, `export_markdown`, `render_pdf_pages`)
- `pyproject.toml` — add `pypdfium2` as explicit dependency (currently transitive via docling)

## Pipeline Module (`pipeline/doctags.py`)

### Model Management

- `create_doctags_model(device: str | None = None) -> tuple[AutoModelForVision2Seq, AutoProcessor]`
  - Loads `ibm-granite/granite-docling-258M` via Transformers
  - Auto-detects device: CUDA if available, else CPU (skip MPS, consistent with segmentation)
  - Returns `(model, processor)` tuple
  - Cached via `st.cache_resource` at the page level

### Core Inference

- `generate_doctags(image: Image.Image, model: AutoModelForVision2Seq, processor: AutoProcessor, device: str) -> str`
  - Constructs chat template with prompt `"Convert this page to docling."`
  - Runs `model.generate()` with `max_new_tokens=8192`
  - Decodes output, strips input tokens, returns raw doctags string

### Parsing and Conversion

- `parse_doctags(doctags: str, image: Image.Image) -> DoclingDocument`
  - Uses `DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])`
  - Converts via `DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")`
  - Returns structured `DoclingDocument`

- `export_markdown(doc: DoclingDocument) -> str`
  - Wraps `doc.export_to_markdown()`

### PDF Rendering

- `render_pdf_pages(pdf_path: str) -> list[Image.Image]`
  - Uses `pypdfium2` to render each page to a PIL Image at 144 DPI
  - Returns list of page images

## UI Page (`pages/doctags.py`)

### Layout

- Title: "DocTags Generation (Experimental)"
- File uploader accepting `["png", "jpg", "jpeg", "pdf"]`
- "Generate" button (disabled until file uploaded)

### Single Image Flow

- Side-by-side columns: original image (left) | raw doctags in code block + rendered markdown (right)
- Download buttons for raw doctags (`.txt`) and markdown (`.md`)

### PDF Flow

- Metrics row: page count, total duration
- Expander per page (first expanded by default), each with side-by-side layout:
  - Left: rendered page image
  - Right: raw doctags in code block + rendered markdown
- Download buttons for per-page outputs plus combined document download

### Data Flow

```
Image upload -> generate_doctags() -> raw doctags string
                                   -> parse_doctags() -> DoclingDocument
                                                      -> export_markdown()

PDF upload -> render_pdf_pages() -> [Image per page]
           -> generate_doctags() per page -> same as above
```

## Testing (`tests/test_doctags.py`)

Unit tests that do not require model weights:

- `render_pdf_pages()` — renders test PDF (`tests/data/pdf/test_pictures.pdf`), verifies page count and image dimensions
- `parse_doctags()` — parses sample doctags strings into DoclingDocument, verifies structure
- `generate_doctags()` — mocked model inference, verifies prompt construction and output decoding
- `export_markdown()` — verifies markdown export from a DoclingDocument
- Edge cases: empty doctags, malformed tags

## Dependencies

- `pypdfium2` — added as explicit dependency in `pyproject.toml` (already transitive via docling)
- `docling-core` — already available (transitive via `docling[vlm]`), provides `DocTagsDocument` and `DoclingDocument`
- `transformers` — already a direct dependency, used for model loading
- `torch` — already a direct dependency, used for inference

## Model Details

- Model: `ibm-granite/granite-docling-258M`
- Architecture: Idefics3 with SigLIP2 vision encoder + Granite 165M LLM
- Parameters: 258M
- License: Apache 2.0
- Prompt: `"Convert this page to docling."`
- Output: doctags format wrapped in `<doctag>...</doctag>`
