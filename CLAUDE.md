# CLAUDE.md

## Project Overview

Streamlit web app that describes pictures in PDF documents using a local vision language model: [IBM Granite Vision 3.3 2B](https://huggingface.co/ibm-granite/granite-vision-3.3-2b).

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Commands

```bash
uv run ruff check .                          # lint
uv run ruff format .                         # format
uv run ty check .                            # type check
uv run pytest                                # run all tests
uv run pytest tests/test_file.py::test_name  # run single test
```

## Code Style

- `snake_case` for functions/variables, `PascalCase` for classes
- Type annotations on all parameters and returns

## Dependencies

Runtime (`[project.dependencies]`):
- `docling[vlm]` — PDF parsing and VLM-based picture description
- `streamlit` — web UI framework

Dev (`[dependency-groups] dev`):
- `pytest` — testing
- `ruff` — linting and formatting
- `ty` — type checking

## Architecture

- `src/pipeline/__init__.py` — re-exports public API (`convert`, `create_converter`, `build_output`, constants)
- `src/pipeline/config.py` — constants (`MAX_PAGES`, `MAX_FILE_SIZE_BYTES`), `create_converter()` factory, and `convert()` wrapper
- `src/pipeline/output.py` — `build_output()` pure function; builds output dict from a `DoclingDocument` and duration
- `streamlit_app.py` — UI only; imports from `pipeline`, applies `st.cache_resource`, handles file upload and download

Upload → "Annotate" button → spinner → metrics (picture count, duration) → JSON download. Catches `ConversionError`. Temp file cleanup in `finally` block.

Output JSON contains `document_info` (count, timing) and a `pictures` array with each picture's reference, caption, and description.

## Tests

- `tests/test_config.py` — constants and `create_converter()` factory
- `tests/test_output.py` — `build_output()` with real Docling objects

All tests import directly from `pipeline` — no Streamlit mocking needed.
