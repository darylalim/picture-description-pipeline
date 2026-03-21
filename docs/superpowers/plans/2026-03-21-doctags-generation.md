# DocTags Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Streamlit page that generates doctags from document images and PDFs using `ibm-granite/granite-docling-258M`, with raw doctags and Markdown output.

**Architecture:** Core logic in `pipeline/doctags.py` (model loading, inference, parsing, PDF rendering), UI in `pages/doctags.py`. Follows the existing pattern where pipeline functions are pure Python with no Streamlit imports, and UI pages call pipeline functions. Tests mock model inference and use real docling-core objects.

**Tech Stack:** transformers (AutoModelForVision2Seq, AutoProcessor), docling-core (DocTagsDocument, DoclingDocument), pypdfium2, torch, Streamlit

**Spec:** `docs/superpowers/specs/2026-03-21-doctags-generation-design.md`

---

### Task 1: PDF Page Rendering

**Files:**
- Create: `pipeline/doctags.py`
- Create: `tests/test_doctags.py`

- [ ] **Step 1: Write tests for `render_pdf_pages`**

Create `tests/test_doctags.py` with tests that use the existing test fixture:

```python
"""Tests for the doctags module."""

from pathlib import Path

from PIL import Image

from pipeline.doctags import render_pdf_pages

TEST_PDF = str(Path(__file__).parent / "data" / "pdf" / "test_pictures.pdf")


# --- render_pdf_pages tests ---


def test_render_pdf_pages_returns_list_of_images() -> None:
    pages = render_pdf_pages(TEST_PDF)
    assert isinstance(pages, list)
    assert len(pages) > 0
    for page in pages:
        assert isinstance(page, Image.Image)


def test_render_pdf_pages_images_have_nonzero_dimensions() -> None:
    pages = render_pdf_pages(TEST_PDF)
    for page in pages:
        w, h = page.size
        assert w > 0
        assert h > 0


def test_render_pdf_pages_images_are_rgb() -> None:
    pages = render_pdf_pages(TEST_PDF)
    for page in pages:
        assert page.mode == "RGB"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_doctags.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.doctags'`

- [ ] **Step 3: Implement `render_pdf_pages`**

Create `pipeline/doctags.py`:

```python
"""DocTags generation using Granite Docling."""

import pypdfium2
from PIL import Image


def render_pdf_pages(pdf_path: str, dpi: int = 144) -> list[Image.Image]:
    """Render each page of a PDF to a PIL RGB Image.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering. Default 144.
    """
    pdf = pypdfium2.PdfDocument(pdf_path)
    pages: list[Image.Image] = []
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=dpi / 72)
        pil_image = bitmap.to_pil().convert("RGB")
        pages.append(pil_image)
    pdf.close()
    return pages
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_doctags.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/doctags.py tests/test_doctags.py
git commit -m "feat(doctags): add PDF page rendering with pypdfium2"
```

---

### Task 2: DocTags Parsing and Markdown Export

**Files:**
- Modify: `pipeline/doctags.py`
- Modify: `tests/test_doctags.py`

- [ ] **Step 1: Write tests for `parse_doctags` and `export_markdown`**

Append to `tests/test_doctags.py`:

```python
from docling_core.types.doc.document import DoclingDocument

from pipeline.doctags import export_markdown, parse_doctags


# --- parse_doctags tests ---


def test_parse_doctags_returns_docling_document() -> None:
    doctags = "<doctag><text><loc_50><loc_50><loc_450><loc_100>Hello world</text></doctag>"
    image = Image.new("RGB", (500, 500), (255, 255, 255))
    result = parse_doctags(doctags, image)
    assert isinstance(result, DoclingDocument)


def test_parse_doctags_returns_none_for_empty_string() -> None:
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    assert parse_doctags("", image) is None


def test_parse_doctags_returns_none_for_missing_doctag_tags() -> None:
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    assert parse_doctags("just some random text", image) is None


def test_parse_doctags_handles_malformed_content() -> None:
    doctags = "<doctag>this is not valid doctags content</doctag>"
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    # Should either return a DoclingDocument or None, but not raise
    result = parse_doctags(doctags, image)
    assert result is None or isinstance(result, DoclingDocument)


# --- export_markdown tests ---


def test_export_markdown_returns_string() -> None:
    doc = DoclingDocument(name="test")
    result = export_markdown(doc)
    assert isinstance(result, str)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_doctags.py::test_parse_doctags_returns_docling_document tests/test_doctags.py::test_parse_doctags_returns_none_for_empty_string tests/test_doctags.py::test_parse_doctags_returns_none_for_missing_doctag_tags tests/test_doctags.py::test_export_markdown_returns_string -v`
Expected: FAIL with `ImportError: cannot import name 'parse_doctags'`

- [ ] **Step 3: Implement `parse_doctags` and `export_markdown`**

Add to `pipeline/doctags.py`:

```python
from docling_core.types.doc.document import DocTagsDocument, DoclingDocument


def parse_doctags(doctags: str, image: Image.Image) -> DoclingDocument | None:
    """Parse raw doctags string into a DoclingDocument.

    Returns None if doctags is empty or missing <doctag> tags.
    """
    if not doctags or "<doctag>" not in doctags:
        return None
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    return DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")


def export_markdown(doc: DoclingDocument) -> str:
    """Export a DoclingDocument to Markdown."""
    return doc.export_to_markdown()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_doctags.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/doctags.py tests/test_doctags.py
git commit -m "feat(doctags): add doctags parsing and markdown export"
```

---

### Task 3: Model Factory

**Files:**
- Modify: `pipeline/doctags.py`
- Modify: `tests/test_doctags.py`

- [ ] **Step 1: Write test for `create_doctags_model`**

Append to `tests/test_doctags.py`:

```python
from unittest.mock import MagicMock, patch


# --- create_doctags_model tests ---


@patch("pipeline.doctags.AutoModelForVision2Seq")
@patch("pipeline.doctags.AutoProcessor")
def test_create_doctags_model_loads_correct_model(
    mock_processor_cls: MagicMock,
    mock_model_cls: MagicMock,
) -> None:
    from pipeline.doctags import create_doctags_model

    processor, model = create_doctags_model(device="cpu")

    mock_processor_cls.from_pretrained.assert_called_once_with(
        "ibm-granite/granite-docling-258M"
    )
    mock_model_cls.from_pretrained.assert_called_once_with(
        "ibm-granite/granite-docling-258M"
    )
    assert processor is mock_processor_cls.from_pretrained.return_value
    assert model is mock_model_cls.from_pretrained.return_value.to.return_value


@patch("pipeline.doctags.AutoModelForVision2Seq")
@patch("pipeline.doctags.AutoProcessor")
def test_create_doctags_model_moves_to_device(
    mock_processor_cls: MagicMock,
    mock_model_cls: MagicMock,
) -> None:
    from pipeline.doctags import create_doctags_model

    create_doctags_model(device="cpu")

    mock_model_cls.from_pretrained.return_value.to.assert_called_once_with("cpu")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_doctags.py::test_create_doctags_model_loads_correct_model tests/test_doctags.py::test_create_doctags_model_moves_to_device -v`
Expected: FAIL with `ImportError: cannot import name 'create_doctags_model'`

- [ ] **Step 3: Implement `create_doctags_model`**

Add imports and function to `pipeline/doctags.py`:

```python
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


def create_doctags_model(
    device: str | None = None,
) -> tuple[AutoProcessor, AutoModelForVision2Seq]:
    """Load Granite Docling 258M for doctags generation.

    When device is None, auto-detects: CUDA if available, else CPU.
    MPS is excluded for consistency with other pipeline models.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "ibm-granite/granite-docling-258M"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)
    return processor, model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_doctags.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/doctags.py tests/test_doctags.py
git commit -m "feat(doctags): add model factory for granite-docling-258M"
```

---

### Task 4: DocTags Inference

**Files:**
- Modify: `pipeline/doctags.py`
- Modify: `tests/test_doctags.py`

- [ ] **Step 1: Write tests for `generate_doctags`**

Append to `tests/test_doctags.py`:

```python
import torch


# --- generate_doctags tests ---


def test_generate_doctags_uses_correct_prompt() -> None:
    from pipeline.doctags import generate_doctags

    mock_processor = MagicMock()
    mock_model = MagicMock()

    # Set up device inference
    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    # Set up processor.apply_chat_template to return a string
    mock_processor.apply_chat_template.return_value = "formatted prompt"

    # Set up processor(...) to return input dict
    mock_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
    mock_processor.return_value = MagicMock()
    mock_processor.return_value.to.return_value = mock_inputs

    # Set up model.generate
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

    # Set up processor.batch_decode
    mock_processor.batch_decode.return_value = ["<doctag>content</doctag>"]

    result = generate_doctags(Image.new("RGB", (100, 100)), mock_processor, mock_model)

    # Verify the prompt contains "Convert this page to docling."
    call_args = mock_processor.apply_chat_template.call_args
    messages = call_args[0][0]
    text_content = [c for c in messages[0]["content"] if c["type"] == "text"]
    assert text_content[0]["text"] == "Convert this page to docling."

    assert result == "<doctag>content</doctag>"


def test_generate_doctags_returns_empty_on_empty_output() -> None:
    from pipeline.doctags import generate_doctags

    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    mock_processor.apply_chat_template.return_value = "prompt"
    mock_processor.return_value = MagicMock()
    mock_processor.return_value.to.return_value = {"input_ids": torch.tensor([[1, 2]])}
    mock_model.generate.return_value = torch.tensor([[1, 2]])
    mock_processor.batch_decode.return_value = [""]

    result = generate_doctags(Image.new("RGB", (10, 10)), mock_processor, mock_model)
    assert result == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_doctags.py::test_generate_doctags_uses_correct_prompt tests/test_doctags.py::test_generate_doctags_returns_empty_on_empty_output -v`
Expected: FAIL with `ImportError: cannot import name 'generate_doctags'`

- [ ] **Step 3: Implement `generate_doctags`**

Add to `pipeline/doctags.py`:

```python
def generate_doctags(
    image: Image.Image,
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
) -> str:
    """Generate doctags from a document image.

    Infers device from the model. Returns raw doctags string,
    or empty string if model produces no output.
    """
    device = next(model.parameters()).device

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Convert this page to docling."},
            ],
        },
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=8192)

    trimmed = output[:, inputs["input_ids"].shape[1] :]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=False)[0].lstrip()
    return decoded
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_doctags.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/doctags.py tests/test_doctags.py
git commit -m "feat(doctags): add doctags inference with granite-docling-258M"
```

---

### Task 5: Wire Up Exports and Dependencies

**Files:**
- Modify: `pipeline/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add re-exports to `pipeline/__init__.py`**

Add the doctags imports and update `__all__`:

```python
from pipeline.doctags import (
    create_doctags_model,
    export_markdown,
    generate_doctags,
    parse_doctags,
    render_pdf_pages,
)
```

Add to `__all__`:
```python
"create_doctags_model",
"export_markdown",
"generate_doctags",
"parse_doctags",
"render_pdf_pages",
```

- [ ] **Step 2: Sync lockfile**

Run: `uv sync`
Expected: Lockfile updated, `pypdfium2` resolved (likely already present as transitive dep)

- [ ] **Step 3: Add `pypdfium2` to `pyproject.toml`**

Add `"pypdfium2"` to the `dependencies` list:

```toml
dependencies = [
    "docling[vlm]",
    "pypdfium2",
    "streamlit",
    "torch",
    "transformers",
]
```

- [ ] **Step 4: Run all tests to verify nothing is broken**

Run: `uv run pytest -v`
Expected: All existing tests + new doctags tests pass

- [ ] **Step 5: Run linting**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add pipeline/__init__.py pyproject.toml uv.lock
git commit -m "feat(doctags): wire up exports and add pypdfium2 dependency"
```

---

### Task 6: Streamlit UI Page — Single Image Flow

**Files:**
- Create: `pages/doctags.py`

- [ ] **Step 1: Create the doctags page with single image support**

Create `pages/doctags.py`:

```python
import time

import streamlit as st
from PIL import Image

from pipeline import (
    create_doctags_model,
    export_markdown,
    generate_doctags,
    parse_doctags,
)

doctags_model = st.cache_resource(create_doctags_model)

st.title("DocTags Generation (Experimental)")
st.write(
    "Parse document images to structured text in doctags format. "
    "Powered by IBM Granite Docling."
)

uploaded_file = st.file_uploader(
    "Upload file", type=["png", "jpg", "jpeg", "pdf"]
)

is_pdf = uploaded_file is not None and uploaded_file.name.lower().endswith(".pdf")

if st.button("Generate", type="primary", disabled=not uploaded_file):
    assert uploaded_file is not None
    processor, model = doctags_model()

    if not is_pdf:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Generating doctags... This may take a few minutes."):
            start = time.perf_counter_ns()
            raw_doctags = generate_doctags(image, processor, model)
            duration_s = (time.perf_counter_ns() - start) / 1e9

        st.metric("Duration (s)", f"{duration_s:.2f}")

        col_img, col_output = st.columns(2)
        col_img.image(image, caption="Original")

        if raw_doctags:
            col_output.code(raw_doctags, language="xml")

            doc = parse_doctags(raw_doctags, image)
            if doc is not None:
                md = export_markdown(doc)
                col_output.markdown("**Markdown output:**")
                col_output.markdown(md)
                col_output.download_button(
                    label="Download Markdown",
                    data=md,
                    file_name="doctags_output.md",
                    mime="text/markdown",
                )
            else:
                col_output.warning("Could not parse doctags into structured document.")

            col_output.download_button(
                label="Download raw doctags",
                data=raw_doctags,
                file_name="doctags_output.txt",
                mime="text/plain",
            )
        else:
            col_output.warning("Model produced no output.")
```

- [ ] **Step 2: Manually verify single image flow**

Run: `uv run streamlit run streamlit_app.py`
Navigate to "DocTags Generation" in sidebar. Upload a PNG/JPG image, click "Generate". Verify:
- Side-by-side layout shows image and doctags output
- Raw doctags displayed in code block
- Markdown rendered below
- Download buttons work

- [ ] **Step 3: Commit**

```bash
git add pages/doctags.py
git commit -m "feat(doctags): add Streamlit UI page with single image flow"
```

---

### Task 7: Streamlit UI Page — PDF Flow

**Files:**
- Modify: `pages/doctags.py`

- [ ] **Step 1: Add PDF flow to the doctags page**

Add `render_pdf_pages` to the imports and import `DoclingDocument` for type annotation:

```python
from docling_core.types.doc.document import DoclingDocument

from pipeline import (
    create_doctags_model,
    export_markdown,
    generate_doctags,
    parse_doctags,
    render_pdf_pages,
)
```

Add the PDF branch inside the `if st.button(...)` block, before the `if not is_pdf:` branch. The full button block becomes:

```python
if st.button("Generate", type="primary", disabled=not uploaded_file):
    assert uploaded_file is not None
    processor, model = doctags_model()

    if is_pdf:
        import tempfile
        from pathlib import Path

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            with st.spinner("Rendering PDF pages..."):
                page_images = render_pdf_pages(tmp_path)

            num_pages = len(page_images)
            progress = st.progress(0, text="Generating doctags...")
            start = time.perf_counter_ns()

            all_doctags: list[str] = []
            all_markdown: list[str] = []
            all_docs: list[DoclingDocument | None] = []

            for i, page_image in enumerate(page_images):
                progress.progress(
                    (i + 1) / num_pages,
                    text=f"Processing page {i + 1} of {num_pages}...",
                )
                raw = generate_doctags(page_image, processor, model)
                all_doctags.append(raw)

                doc = parse_doctags(raw, page_image) if raw else None
                all_docs.append(doc)
                all_markdown.append(export_markdown(doc) if doc else "")

            duration_s = (time.perf_counter_ns() - start) / 1e9
            progress.empty()

            col1, col2 = st.columns(2)
            col1.metric("Pages", num_pages)
            col2.metric("Duration (s)", f"{duration_s:.2f}")

            combined_doctags = "\n\n".join(all_doctags)
            combined_markdown = "\n\n---\n\n".join(
                md for md in all_markdown if md
            )

            dl_col1, dl_col2 = st.columns(2)
            dl_col1.download_button(
                label="Download all doctags",
                data=combined_doctags,
                file_name=f"{uploaded_file.name}_doctags.txt",
                mime="text/plain",
            )
            dl_col2.download_button(
                label="Download all Markdown",
                data=combined_markdown,
                file_name=f"{uploaded_file.name}_doctags.md",
                mime="text/markdown",
            )

            for i, page_image in enumerate(page_images):
                with st.expander(f"Page {i + 1}", expanded=i == 0):
                    col_img, col_output = st.columns(2)
                    col_img.image(page_image, caption=f"Page {i + 1}")

                    if all_doctags[i]:
                        col_output.code(all_doctags[i], language="xml")

                        if all_docs[i] is not None:
                            col_output.markdown("**Markdown output:**")
                            col_output.markdown(all_markdown[i])
                        else:
                            col_output.warning(
                                "Could not parse doctags into structured document."
                            )
                    else:
                        col_output.warning("Model produced no output for this page.")

        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)

    else:
        # ... existing single image flow (unchanged from Task 6)
```

- [ ] **Step 2: Manually verify PDF flow**

Run: `uv run streamlit run streamlit_app.py`
Navigate to "DocTags Generation" in sidebar. Upload a PDF, click "Generate". Verify:
- Progress bar updates per page
- Metrics show page count and duration
- Combined download buttons work
- Per-page expanders with side-by-side layout
- First page expanded by default

- [ ] **Step 3: Commit**

```bash
git add pages/doctags.py
git commit -m "feat(doctags): add PDF flow with progress bar and per-page output"
```

---

### Task 8: Update Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `pyproject.toml` (description only)

- [ ] **Step 1: Update `pyproject.toml` description**

Change the description to include doctags:

```toml
description = "Extract and describe pictures and tables in PDF documents, segment objects in images, and generate doctags from document images using IBM Granite Vision"
```

- [ ] **Step 2: Update `CLAUDE.md`**

Add to the **Dependencies** runtime section:
```
- `pypdfium2` — PDF page rendering for doctags generation
```

Add to the **Architecture** section:
```
- `pipeline/doctags.py` — `create_doctags_model()` factory, `generate_doctags()` inference, `parse_doctags()` conversion to DoclingDocument, `export_markdown()` wrapper, `render_pdf_pages()` PDF-to-image rendering via pypdfium2
- `pages/doctags.py` — doctags generation UI page; image/PDF upload, raw doctags display, markdown preview, per-page expanders for PDFs; model cached via `st.cache_resource`
```

Add to the **Architecture** key details:
```
- DocTags generation uses `ibm-granite/granite-docling-258M` loaded directly via Transformers (not Docling's VlmPipeline), with prompt "Convert this page to docling."
- DocTags flow: upload image or PDF, click "Generate", model produces raw doctags, parsed via docling-core into DoclingDocument, exported to Markdown
- For PDFs, pages are rendered to images via pypdfium2 at 144 DPI, then each page is processed independently
```

Add to the **Tests** section:
```
- `tests/test_doctags.py` — `render_pdf_pages()` with real PDF fixture, `parse_doctags()` with sample doctags strings, `create_doctags_model()` and `generate_doctags()` with mocked model, `export_markdown()` verification; no model weights required
```

- [ ] **Step 3: Run linting on all changed files**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md pyproject.toml
git commit -m "docs: update CLAUDE.md and project description for doctags feature"
```
