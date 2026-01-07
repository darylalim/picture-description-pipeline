import json
import os
import tempfile
import time

import streamlit as st
from docling_core.types.doc.document import PictureDescriptionData
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, granite_picture_description
from docling.document_converter import DocumentConverter, PdfFormatOption

def convert(source, doc_converter):
    """Convert a source file to a Docling document."""
    result = doc_converter.convert(
        source=source,
        max_num_pages=100,
        max_file_size=20971520
    )
    doc = result.document
    return doc

pipeline_options = PdfPipelineOptions()
pipeline_options.do_picture_description = True
pipeline_options.picture_description_options = granite_picture_description
pipeline_options.picture_description_options.prompt = (
    "Describe the image in three sentences. Be concise and accurate."
)
pipeline_options.images_scale = 2.0
pipeline_options.generate_picture_images = True
                
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

st.title("Picture Description Pipeline")
st.write("Describe pictures in a document with a local IBM Granite Vision model.")

uploaded_file = st.file_uploader("Upload file", type=["pdf"])

if st.button("Annotate", type="primary"):
    if uploaded_file is not None:
        try:
            with st.spinner("Annotating..."):
                # Save uploaded file temporarily for Docling to process
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                start_time = time.time_ns()
                doc = convert(tmp_file_path, doc_converter)
                end_time = time.time_ns()
                total_duration_ns = end_time - start_time

                # Clean up temp file
                os.unlink(tmp_file_path)

            st.success("Done.")

            st.subheader("Metrics")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pictures", len(doc.pictures))
            with col2:
                st.metric("Total Duration (nanoseconds)", total_duration_ns)

            # Prepare JSON for download
            download_data = {
                "document_info": {
                    "num_pictures": len(doc.pictures),
                    "total_duration_ns": total_duration_ns
                },
                "pictures": []
            }
            
            for idx, pic in enumerate(doc.pictures, 1):
                pic_data = {
                    "picture_number": idx,
                    "reference": pic.self_ref,
                    "caption": pic.caption_text(doc=doc) or "",
                    "annotations": []
                }
                
                for annotation in pic.annotations:
                    if isinstance(annotation, PictureDescriptionData):
                        pic_data["annotations"].append(
                            {
                                "provenance": annotation.provenance,
                                "text": annotation.text
                            }
                        )
                
                download_data["pictures"].append(pic_data)
            
            json_str = json.dumps(download_data, indent=2)
            
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"{uploaded_file.name}_annotations.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"Syntax error: {str(e)}")
    else:
        st.warning("Upload a PDF file.")
