# Granite Vision Pipeline

## Project Overview

Streamlit web app that extracts and describes pictures and tables in PDF documents using a local vision language model: [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b).

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
- `docling[vlm]` — PDF parsing, table extraction, and VLM-based picture description
- `streamlit` — web UI framework

Dev (`[dependency-groups] dev`):
- `pytest` — testing
- `ruff` — linting and formatting
- `ty` — type checking

Overrides (`[tool.uv]`):
- `opencv-python` replaced with `opencv-python-headless` to avoid duplicate `libavdevice` dylib conflicts with `av` on macOS

## Architecture

- `pipeline/__init__.py` — re-exports public API (`convert`, `create_converter`, `build_output`, `get_description`, `get_table_content`)
- `pipeline/config.py` — `create_converter()` factory, `convert()` wrapper, warning filters for upstream docling/transformers deprecations
- `pipeline/output.py` — `build_output()` produces a unified `elements` array from pictures and tables via `build_element()`; `get_description()` extracts picture descriptions from `meta` with fallback to `annotations`; `get_table_content()` extracts table markdown and structured column/row data
- `streamlit_app.py` — UI only; caches the converter via `st.cache_resource`, passes it to `convert()`, handles file upload, download, per-picture preview in expanders, and per-table preview with interactive dataframes

Key details:
- `convert()` accepts an optional `converter` parameter to reuse a cached instance, avoiding model reload on each call
- `get_description()` falls back to `pic.annotations` because docling appends `DescriptionAnnotation` after `PictureItem` construction, so the `meta` migration validator doesn't run
- Upload flow: upload PDF, click "Annotate", spinner, metrics (picture count, table count, duration), JSON download, per-picture preview in expanders (image + description), and per-table preview in expanders (image + interactive dataframe)
- Output JSON contains `document_info` (picture count, table count, timing) and an `elements` array with `type` discriminator (`"picture"` or `"table"`) and type-specific `content`

## Tests

- `tests/test_config.py` — `create_converter()` factory with pipeline option verification, `convert()` with and without provided converter
- `tests/test_output.py` — `build_output()`, `build_element()`, `get_description()`, and `get_table_content()` with real Docling objects; covers pictures, tables, mixed documents, annotations fallback, meta priority

All tests import directly from `pipeline` — no Streamlit mocking needed.
