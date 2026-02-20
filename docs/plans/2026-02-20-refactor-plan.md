# Refactor: Modularize, Tighten, Optimize — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split the single-file app into `src/pipeline/` package (config + output), tighten code, and optimize runtime with SpooledTemporaryFile.

**Architecture:** Extract pure logic into `src/pipeline/config.py` (constants, converter factory) and `src/pipeline/output.py` (build_output). `streamlit_app.py` becomes UI-only, importing from the package. Tests import directly from the package — no Streamlit mocking needed.

**Tech Stack:** Python 3.12+, Streamlit, Docling, pytest, uv

---

### Task 1: Scaffold the package and update pyproject.toml

**Files:**
- Create: `src/pipeline/__init__.py`
- Modify: `pyproject.toml`

**Step 1: Create empty package**

Create `src/pipeline/__init__.py` as an empty file.

**Step 2: Update pyproject.toml**

Add pytest pythonpath so imports from `src/` resolve:

```toml
[project]
name = "picture-description-pipeline"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "docling[vlm]",
    "streamlit",
]

[dependency-groups]
dev = [
    "pytest",
    "ruff",
    "ty",
]

[tool.pytest.ini_options]
pythonpath = ["src"]
```

**Step 3: Verify structure**

Run: `ls src/pipeline/__init__.py`
Expected: file exists, no error

**Step 4: Commit**

```bash
git add src/pipeline/__init__.py pyproject.toml
git commit -m "Scaffold src/pipeline package and add pytest pythonpath"
```

---

### Task 2: Extract output.py with TDD

**Files:**
- Create: `src/pipeline/output.py`
- Modify: `tests/test_output.py`

**Step 1: Update tests to import from new location**

Rewrite `tests/test_output.py` — remove the `sys.modules` Streamlit mock, import `build_output` from `pipeline.output` at module level:

```python
"""Tests for the build_output function."""

import pytest
from docling_core.types.doc.document import (
    DescriptionMetaField,
    DoclingDocument,
    PictureItem,
    PictureMeta,
)

from pipeline.output import build_output


def _make_doc(pictures: list[PictureItem] | None = None) -> DoclingDocument:
    """Create a DoclingDocument with the given pictures."""
    doc = DoclingDocument(name="test")
    if pictures:
        doc.pictures = pictures
    return doc


def _make_picture(
    index: int,
    text: str | None = None,
    created_by: str | None = None,
) -> PictureItem:
    """Create a PictureItem with optional description metadata."""
    meta = None
    if text is not None:
        meta = PictureMeta(
            description=DescriptionMetaField(text=text, created_by=created_by)
        )
    return PictureItem(
        self_ref=f"#/pictures/{index}",
        meta=meta,
    )


def test_top_level_keys() -> None:
    doc = _make_doc()
    result = build_output(doc, 1.5)
    assert set(result.keys()) == {"document_info", "pictures"}


def test_document_info_fields() -> None:
    doc = _make_doc([_make_picture(0)])
    result = build_output(doc, 2.34)
    info = result["document_info"]
    assert isinstance(info, dict)
    assert info["num_pictures"] == 1
    assert info["total_duration_s"] == 2.34


def test_picture_entry_structure() -> None:
    pic = _make_picture(0, text="A test description.", created_by="test-model")
    doc = _make_doc([pic])
    result = build_output(doc, 0.5)
    pictures = result["pictures"]
    assert isinstance(pictures, list)
    assert len(pictures) == 1
    entry = pictures[0]
    assert entry["picture_number"] == 1
    assert entry["reference"] == "#/pictures/0"
    assert entry["caption"] == ""
    assert isinstance(entry["description"], dict)


def test_description_fields() -> None:
    pic = _make_picture(0, text="Describes an image.", created_by="granite-vision")
    doc = _make_doc([pic])
    result = build_output(doc, 1.0)
    description = result["pictures"][0]["description"]
    assert description["text"] == "Describes an image."
    assert description["created_by"] == "granite-vision"


def test_multiple_pictures() -> None:
    pics = [
        _make_picture(0, text="First pic.", created_by="model-a"),
        _make_picture(1, text="Second pic.", created_by="model-b"),
    ]
    doc = _make_doc(pics)
    result = build_output(doc, 3.0)
    pictures = result["pictures"]
    assert len(pictures) == 2
    assert pictures[0]["picture_number"] == 1
    assert pictures[1]["picture_number"] == 2
    assert pictures[0]["description"]["text"] == "First pic."
    assert pictures[1]["description"]["text"] == "Second pic."


def test_no_description() -> None:
    pic = _make_picture(0)
    doc = _make_doc([pic])
    result = build_output(doc, 0.5)
    assert result["pictures"][0]["description"] is None


def test_empty_document() -> None:
    doc = _make_doc()
    result = build_output(doc, 0.0)
    assert result["document_info"]["num_pictures"] == 0
    assert result["pictures"] == []
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_output.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.output'`

**Step 3: Create output.py**

Create `src/pipeline/output.py`:

```python
from docling_core.types.doc.document import DoclingDocument


def build_output(doc: DoclingDocument, duration_s: float) -> dict[str, object]:
    """Build the output dictionary from a converted document."""
    return {
        "document_info": {
            "num_pictures": len(doc.pictures),
            "total_duration_s": duration_s,
        },
        "pictures": [
            {
                "picture_number": idx,
                "reference": pic.self_ref,
                "caption": pic.caption_text(doc=doc) or "",
                "description": {
                    "created_by": pic.meta.description.created_by,
                    "text": pic.meta.description.text,
                }
                if pic.meta and pic.meta.description
                else None,
            }
            for idx, pic in enumerate(doc.pictures, 1)
        ],
    }
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_output.py -v`
Expected: all 7 tests PASS

**Step 5: Commit**

```bash
git add src/pipeline/output.py tests/test_output.py
git commit -m "Extract build_output into pipeline.output and simplify tests"
```

---

### Task 3: Extract config.py

**Files:**
- Create: `src/pipeline/config.py`

**Step 1: Create config.py**

Create `src/pipeline/config.py`:

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    granite_picture_description,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

MAX_PAGES: int = 100
MAX_FILE_SIZE_BYTES: int = 20 * 1024 * 1024


def create_converter() -> DocumentConverter:
    """Create a DocumentConverter with picture description enabled."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_picture_description = True
    pipeline_options.picture_description_options = granite_picture_description
    pipeline_options.picture_description_options.prompt = (
        "Describe the image in three sentences. Be concise and accurate."
    )
    pipeline_options.picture_description_options.generation_config = {
        "max_new_tokens": 200,
        "do_sample": False,
        "pad_token_id": 0,
        "bos_token_id": 0,
        "eos_token_id": 0,
    }
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
```

**Step 2: Verify import works**

Run: `uv run python -c "from pipeline.config import create_converter, MAX_PAGES, MAX_FILE_SIZE_BYTES; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/pipeline/config.py
git commit -m "Extract constants and converter factory into pipeline.config"
```

---

### Task 4: Rewrite streamlit_app.py as UI-only

**Files:**
- Modify: `streamlit_app.py`

**Step 1: Rewrite streamlit_app.py**

Replace `streamlit_app.py` with:

```python
import json
import tempfile
import time
from pathlib import Path

import streamlit as st
from docling.exceptions import ConversionError

from pipeline.config import MAX_FILE_SIZE_BYTES, MAX_PAGES, create_converter
from pipeline.output import build_output

converter = st.cache_resource(create_converter)

st.title("Picture Description Pipeline")
st.write("Describe pictures in a document with a local IBM Granite Vision model.")

uploaded_file = st.file_uploader("Upload file", type=["pdf"])

if st.button("Annotate", type="primary"):
    if uploaded_file is None:
        st.warning("Upload a PDF file.")
    else:
        tmp_path: str | None = None
        try:
            with st.spinner("Annotating..."):
                with tempfile.SpooledTemporaryFile(
                    max_size=5 * 1024 * 1024, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                start = time.perf_counter_ns()
                doc = (
                    converter()
                    .convert(
                        source=tmp_path,
                        max_num_pages=MAX_PAGES,
                        max_file_size=MAX_FILE_SIZE_BYTES,
                    )
                    .document
                )
                duration_s = (time.perf_counter_ns() - start) / 1e9

            st.success("Done.")

            col1, col2 = st.columns(2)
            col1.metric("Pictures", len(doc.pictures))
            col2.metric("Duration (s)", f"{duration_s:.2f}")

            st.download_button(
                label="Download JSON",
                data=json.dumps(build_output(doc, duration_s), indent=2),
                file_name=f"{uploaded_file.name}_annotations.json",
                mime="application/json",
            )

        except ConversionError as e:
            st.error(str(e))
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)
```

**Step 2: Run tests to verify nothing broke**

Run: `uv run pytest tests/test_output.py -v`
Expected: all 7 tests PASS

**Step 3: Run linter and formatter**

Run: `uv run ruff check . && uv run ruff format .`
Expected: clean

**Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "Rewrite streamlit_app.py as UI-only, use SpooledTemporaryFile"
```

---

### Task 5: Update CLAUDE.md and clean up

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md architecture section**

Update the Architecture section to reflect the new module structure:

- `src/pipeline/config.py` — constants (`MAX_PAGES`, `MAX_FILE_SIZE_BYTES`) and `create_converter()` factory.
- `src/pipeline/output.py` — `build_output()` pure function.
- `streamlit_app.py` — UI-only. Imports from `pipeline`, applies `@st.cache_resource`, handles file upload, conversion, and download.

Update the Tests section:

- `tests/test_output.py` — unit tests for `build_output()` using real Docling objects. Imports directly from `pipeline.output` — no Streamlit mocking needed.

**Step 2: Run full checks**

Run: `uv run ruff check . && uv run ruff format . && uv run pytest`
Expected: all clean, all tests pass

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "Update CLAUDE.md to reflect new module structure"
```
