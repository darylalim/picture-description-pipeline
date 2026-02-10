import json
import tempfile
import time
from pathlib import Path

import streamlit as st
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    granite_picture_description,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.exceptions import ConversionError
from docling_core.types.doc.document import DoclingDocument

MAX_PAGES: int = 100
MAX_FILE_SIZE_BYTES: int = 20 * 1024 * 1024


@st.cache_resource
def create_converter() -> DocumentConverter:
    """Create a cached DocumentConverter with picture description enabled."""
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


def convert(source: str) -> DoclingDocument:
    """Convert a source file to a Docling document."""
    return (
        create_converter()
        .convert(
            source=source,
            max_num_pages=MAX_PAGES,
            max_file_size=MAX_FILE_SIZE_BYTES,
        )
        .document
    )


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
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                start = time.perf_counter_ns()
                doc = convert(tmp_path)
                duration_s = (time.perf_counter_ns() - start) / 1e9

            st.success("Done.")

            col1, col2 = st.columns(2)
            col1.metric("Pictures", len(doc.pictures))
            col2.metric("Duration (s)", f"{duration_s:.2f}")

            download_data = build_output(doc, duration_s)

            st.download_button(
                label="Download JSON",
                data=json.dumps(download_data, indent=2),
                file_name=f"{uploaded_file.name}_annotations.json",
                mime="application/json",
            )

        except ConversionError as e:
            st.error(str(e))
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)
