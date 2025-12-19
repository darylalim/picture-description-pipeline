import json
import os
import streamlit as st
import tempfile
import time
from docling_core.types.doc.document import PictureDescriptionData
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, granite_picture_description
from docling.document_converter import DocumentConverter, PdfFormatOption

st.set_page_config(page_title="Picture Description Pipeline", layout="wide")

st.title("Picture Description Pipeline")
st.write("Describe pictures in a document with a local IBM Granite Vision model.")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Display file info
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    st.info(f"📄 File uploaded: **{uploaded_file.name}** ({file_size_mb:.2f} MB)")
    
    # Convert button
    if st.button("🚀 Convert Document", type="primary"):
        with st.spinner("Processing document... This may take a few minutes."):
            try:
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Start timing
                start_time = time.time()
                
                # Configure pipeline
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_picture_description = True
                pipeline_options.picture_description_options = granite_picture_description
                pipeline_options.picture_description_options.prompt = (
                    "Describe the image in three sentences. Be concise and accurate."
                )
                pipeline_options.images_scale = 2.0
                pipeline_options.generate_picture_images = True
                
                # Create converter
                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options,
                        )
                    }
                )
                
                # Convert document
                result = converter.convert(tmp_file_path)
                doc = result.document
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Clean up temp file
                os.unlink(tmp_file_path)
                
                # Store results in session state
                st.session_state['doc'] = doc
                st.session_state['processing_time'] = processing_time
                st.session_state['file_size_mb'] = file_size_mb
                
                st.success(f"✅ Document processed successfully in {processing_time:.2f} seconds!")
                
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")
                if 'tmp_file_path' in locals():
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass

# Display results if document has been processed
if 'doc' in st.session_state:
    doc = st.session_state['doc']
    
    # Display metrics
    st.header("📊 Document Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Number of Pictures", len(doc.pictures))
    
    with col2:
        st.metric("Processing Time", f"{st.session_state['processing_time']:.2f}s")
    
    with col3:
        # Get number of pages
        num_pages = len(doc.pages) if hasattr(doc, 'pages') else "N/A"
        st.metric("Number of Pages", num_pages)
    
    with col4:
        st.metric("File Size", f"{st.session_state['file_size_mb']:.2f} MB")
    
    # Display pictures, captions, and annotations
    st.header("🖼️ Pictures and Descriptions")
    
    if len(doc.pictures) == 0:
        st.warning("No pictures found in the document.")
    else:
        # Display first 5 pictures
        num_to_display = min(5, len(doc.pictures))
        st.write(f"Displaying the first **{num_to_display}** picture(s):")
        
        for idx, pic in enumerate(doc.pictures[:5], 1):
            st.subheader(f"Picture {idx}: `{pic.self_ref}`")
            
            # Display image
            if pic.image and pic.image.uri:
                try:
                    st.image(str(pic.image.uri), use_container_width=True)
                except Exception as e:
                    st.error(f"Could not display image: {str(e)}")
            
            # Display caption
            caption = pic.caption_text(doc=doc)
            if caption:
                st.write("**Caption:**")
                st.write(caption)
            else:
                st.write("*No caption available*")
            
            # Display annotations
            annotations_found = False
            for annotation in pic.annotations:
                if isinstance(annotation, PictureDescriptionData):
                    if not annotations_found:
                        st.write("**Annotations:**")
                        annotations_found = True
                    st.write(f"*({annotation.provenance})*")
                    st.write(annotation.text)
            
            if not annotations_found:
                st.write("*No annotations available*")
            
            st.divider()
    
    # Prepare download data
    download_data = {
        "document_info": {
            "num_pictures": len(doc.pictures),
            "num_pages": len(doc.pages) if hasattr(doc, 'pages') else None,
            "processing_time_seconds": st.session_state['processing_time'],
            "file_size_mb": st.session_state['file_size_mb']
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
                pic_data["annotations"].append({
                    "provenance": annotation.provenance,
                    "text": annotation.text
                })
        
        download_data["pictures"].append(pic_data)
    
    # Download button
    st.header("💾 Download Results")
    json_str = json.dumps(download_data, indent=2)
    st.download_button(
        label="📥 Download Captions and Annotations (JSON)",
        data=json_str,
        file_name="picture_descriptions.json",
        mime="application/json"
    )
else:
    st.info("👆 Upload a PDF file and click 'Convert Document' to get started!")
