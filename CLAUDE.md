# CLAUDE.md

## Project Overview

A Streamlit web app that describes pictures in PDF documents using a local vision language model: [IBM Granite Vision 3.3 2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b)

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Commands

```bash
uv run ruff check .    # lint
uv run ruff format .   # format
uv run ty check .      # type check
uv run pytest          # run all tests
uv run pytest tests/test_file.py::test_name  # run single test
```

## Code Style

- `snake_case` for functions/variables, `PascalCase` for classes
- Type annotations on all parameters and returns

## Dependencies

Runtime (in `[project.dependencies]`):
- `docling[vlm]` — PDF parsing and VLM-based picture description
- `streamlit` — Web UI framework

Dev (in `[dependency-groups] dev`):
- `pytest` — Testing
- `ruff` — Linting and formatting
- `ty` — Type checking

## Architecture

`streamlit_app.py` — single-file app. Users upload a PDF, the app converts it with Docling, runs picture description inference, and outputs annotated results as downloadable JSON.

- `create_converter()` — cached `DocumentConverter` with picture description enabled via `granite_picture_description`. Limits: `MAX_PAGES=100`, `MAX_FILE_SIZE_BYTES=20MB`.
- `convert()` — converts a PDF file path to a `DoclingDocument`.
- `build_output()` — builds the output dict from a `DoclingDocument` and duration. Extracted for testability.
- **UI flow** — file upload → "Annotate" button → spinner → metrics (picture count, duration in seconds) → JSON download. Catches `ConversionError`. Temp file cleanup in `finally` block.

Output JSON contains `document_info` (count + timing) and a `pictures` array with each picture's reference, caption, and `description` (text + created_by) from `pic.meta.description`.

## Tests

`tests/test_output.py` — unit tests for `build_output()` using real Docling objects. Mocks `streamlit` in `sys.modules` to allow importing without the Streamlit runtime.
