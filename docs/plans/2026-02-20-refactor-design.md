# Refactor: Modularize, Tighten, Optimize

## Goal

Refactor the single-file app into a small package for better separation of concerns, testability, conciseness, and runtime performance.

## Module Structure

```
picture-description-pipeline/
├── src/
│   └── pipeline/
│       ├── __init__.py      # empty
│       ├── config.py        # constants + converter factory
│       └── output.py        # build_output() pure function
├── streamlit_app.py         # UI only
├── tests/
│   ├── test_output.py       # no more Streamlit mock
│   └── data/pdf/
└── pyproject.toml           # updated with src layout + pytest config
```

## Module Responsibilities

### `src/pipeline/config.py`

- `MAX_PAGES` and `MAX_FILE_SIZE_BYTES` constants.
- `create_converter()` — builds `DocumentConverter` with picture description enabled. Uses constructor kwargs directly instead of multi-line attribute assignment. No Streamlit decorator here.

### `src/pipeline/output.py`

- `build_output(doc, duration_s)` — pure function, unchanged logic. Builds output dict from a `DoclingDocument`.

### `streamlit_app.py`

- UI-only code. Imports from `pipeline`.
- `@st.cache_resource` wraps the imported `create_converter` at this layer.
- Calls `convert()` from `pipeline.config` which wraps `converter.convert(...)` with limit args.
- Uses `NamedTemporaryFile(delete=False)` for temp files. (`SpooledTemporaryFile` was attempted but reverted — Docling requires a filesystem path, which `SpooledTemporaryFile.name` cannot provide.)

## Test Changes

- `test_output.py` imports from `pipeline.output` directly — eliminates the `sys.modules` Streamlit mock hack.
- Helpers and test cases unchanged.

## `pyproject.toml` Changes

- Add `[tool.pytest.ini_options]` with `pythonpath = ["src"]`.
