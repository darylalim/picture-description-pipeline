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

### Imports

Key imports from `docling-core` (not `docling`):
- `from docling_core.types.doc.document import DocTagsDocument, DoclingDocument`

### Model Management

- `create_doctags_model(device: str | None = None) -> tuple[AutoProcessor, AutoModelForVision2Seq]`
  - Loads `ibm-granite/granite-docling-258M` via Transformers (`AutoModelForVision2Seq` — same auto-class used for granite-vision in segmentation)
  - Auto-detects device: CUDA if available, else CPU (skip MPS, consistent with segmentation)
  - Returns `(processor, model)` tuple — processor first, matching `create_granite_model` convention
  - Cached via `st.cache_resource` at the page level

### Core Inference

- `generate_doctags(image: Image.Image, processor: AutoProcessor, model: AutoModelForVision2Seq) -> str`
  - Infers device from model parameters (no separate `device` arg, matching `segment()` pattern)
  - Constructs chat template with prompt `"Convert this page to docling."`
  - Runs `model.generate()` with `max_new_tokens=8192` (sufficient for single pages; complex pages may need tuning)
  - Decodes output, strips input tokens, returns raw doctags string
  - Returns empty string on failure (empty model output)

### Parsing and Conversion

- `parse_doctags(doctags: str, image: Image.Image) -> DoclingDocument | None`
  - Returns `None` if doctags string is empty or missing `<doctag>` tags
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
- On parse failure: show raw doctags with `st.warning` that parsing failed

### PDF Flow

- Progress bar (`st.progress`) with per-page updates during processing
- Metrics row: page count, total duration
- Expander per page (first expanded by default), each with side-by-side layout:
  - Left: rendered page image
  - Right: raw doctags in code block + rendered markdown
- Download buttons for per-page outputs plus combined document download
- On per-page parse failure: show raw doctags with `st.warning` in that page's expander

### Data Flow

```
Image upload -> generate_doctags() -> raw doctags string
                                   -> parse_doctags() -> DoclingDocument | None
                                                      -> export_markdown()

PDF upload -> render_pdf_pages() -> [Image per page]
           -> generate_doctags() per page -> same as above
```

## Testing (`tests/test_doctags.py`)

Unit tests that do not require model weights:

- `render_pdf_pages()` — renders test PDF (`tests/data/pdf/test_pictures.pdf`), verifies returns list of PIL Images with non-zero dimensions
- `parse_doctags()` — parses sample doctags strings into DoclingDocument, verifies structure
- `parse_doctags()` — returns `None` for empty string and missing `<doctag>` tags
- `generate_doctags()` — mocked model inference, verifies prompt construction and output decoding
- `export_markdown()` — verifies markdown export from a DoclingDocument
- Edge cases: empty doctags, malformed tags

## Dependencies

- `pypdfium2` — added as explicit dependency in `pyproject.toml` (already transitive via docling; made explicit to ensure availability)
- `docling-core` — already available (transitive via `docling[vlm]`), provides `DocTagsDocument` and `DoclingDocument` from `docling_core.types.doc.document`
- `transformers` — already a direct dependency, used for model loading
- `torch` — already a direct dependency, used for inference

## Model Details

- Model: `ibm-granite/granite-docling-258M`
- Architecture: Idefics3 with SigLIP2 vision encoder + Granite 165M LLM
- Parameters: 258M
- License: Apache 2.0
- Prompt: `"Convert this page to docling."`
- Output: doctags format wrapped in `<doctag>...</doctag>`
