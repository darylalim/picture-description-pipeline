import tempfile
import time
from pathlib import Path

import pypdfium2
import streamlit as st
from PIL import Image

from pipeline import create_qa_model, generate_qa_response, render_pdf_pages

qa_model = st.cache_resource(create_qa_model)

st.title("Multipage QA (Experimental)")
st.write(
    "Ask questions about document pages using IBM Granite Vision. "
    "Upload a PDF or up to 8 images, then type your question."
)

uploaded_files = st.file_uploader(
    "Upload file(s)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

page_images: list[Image.Image] = []
tmp_path: str | None = None
is_pdf = False
selected: list[int] = []

if uploaded_files:
    is_pdf = len(uploaded_files) == 1 and uploaded_files[0].name.lower().endswith(".pdf")

    if is_pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_files[0].read())
            tmp_path = tmp_file.name

        pdf = pypdfium2.PdfDocument(tmp_path)
        total_pages = len(pdf)
        pdf.close()

        default_pages = list(range(1, min(9, total_pages + 1)))
        selected = st.multiselect(
            "Select pages (up to 8)",
            options=list(range(1, total_pages + 1)),
            default=default_pages,
            max_selections=8,
        )
    else:
        if len(uploaded_files) > 8:
            st.warning("More than 8 images uploaded. Using the first 8.")
            uploaded_files = uploaded_files[:8]

question = st.text_input("Question", placeholder="e.g., What is shown on these pages?")

has_input = bool(uploaded_files) and bool(question)
if uploaded_files and len(uploaded_files) == 1 and uploaded_files[0].name.lower().endswith(".pdf"):
    has_input = has_input and bool(selected)

if st.button("Answer", type="primary", disabled=not has_input):
    assert uploaded_files is not None
    processor, model = qa_model()

    try:
        if is_pdf:
            assert tmp_path is not None
            with st.spinner("Rendering selected pages..."):
                all_pages = render_pdf_pages(tmp_path)
                page_images = [all_pages[i - 1] for i in selected]
        else:
            page_images = [Image.open(f).convert("RGB") for f in uploaded_files]

        with st.spinner("Generating answer..."):
            start = time.perf_counter_ns()
            answer = generate_qa_response(page_images, question, processor, model)
            duration_s = (time.perf_counter_ns() - start) / 1e9

        if not answer:
            st.warning("Model produced no output.")
        else:
            col_thumbs, col_answer = st.columns([1, 2])
            with col_thumbs:
                for i, img in enumerate(page_images, 1):
                    st.image(img, caption=f"Page {i}", use_container_width=True)
            with col_answer:
                st.markdown(answer)

            st.metric("Duration (s)", f"{duration_s:.2f}")

    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)
