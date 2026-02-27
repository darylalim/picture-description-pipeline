import json
import tempfile
import time
from pathlib import Path

import streamlit as st
from docling.exceptions import ConversionError

from pipeline import build_output, convert, create_converter

converter = st.cache_resource(create_converter)

st.title("Picture Description Pipeline")
st.write("Describe pictures in PDF documents using a local vision model.")

uploaded_file = st.file_uploader("Upload file", type=["pdf"])

if st.button("Annotate", type="primary", disabled=not uploaded_file):
    assert uploaded_file is not None
    tmp_path: str | None = None
    try:
        with st.spinner(
            "Describing pictures... This may take a few minutes for large documents."
        ):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            start = time.perf_counter_ns()
            doc = convert(tmp_path, converter=converter())
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
