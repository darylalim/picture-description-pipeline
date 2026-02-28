# UI Polish Design

## Context

The Streamlit app works but has rough edges: stale copy referencing the old model, no in-app preview of results, a generic spinner with no expectation-setting for large documents, and a button that lets you click before uploading a file.

Audience: technical colleagues. Typical workload: large PDFs (30+ pages, many pictures).

## Changes (all in streamlit_app.py)

### 1. Subtitle

Replace "Describe pictures in a document with a local IBM Granite Vision model." with "Describe pictures in PDF documents using a local vision model." — model-agnostic so it doesn't go stale on model swaps.

### 2. Button state

Disable the "Annotate" button when no file is uploaded (`disabled=not uploaded_file`). Remove the `st.warning("Upload a PDF file.")` branch.

### 3. Spinner text

Change "Annotating..." to "Describing pictures... This may take a few minutes for large documents." — sets expectations since `convert()` is a single blocking call with no progress callback.

### 4. In-app results preview

After conversion, below the existing metrics row and download button, add a "Pictures" section:

- One `st.expander` per picture, labeled "Picture 1", "Picture 2", etc.
- Inside each expander: two-column layout — picture image (`pic.get_image(doc)`) on the left, description text on the right.
- If a picture has a caption, show it under the image.
- First expander open by default, rest collapsed.

## Constraints

- No new dependencies.
- No changes to `pipeline/` — UI-only changes.
- `pic.get_image(doc)` returns a PIL Image when `generate_picture_images=True` (already enabled).
