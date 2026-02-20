import os
import warnings

os.environ.setdefault("TRANSFORMERS_USE_FAST_IMAGE_PROCESSOR", "1")
warnings.filterwarnings(
    "ignore",
    message="The class `AutoModelForVision2Seq` is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="`torch_dtype` is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="'pin_memory' argument is set as true but not supported on MPS",
    category=UserWarning,
)

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    granite_picture_description,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument

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


def convert(source: str, converter: DocumentConverter | None = None) -> DoclingDocument:
    """Convert a PDF file to a DoclingDocument."""
    if converter is None:
        converter = create_converter()
    return (
        converter
        .convert(
            source=source,
            max_num_pages=MAX_PAGES,
            max_file_size=MAX_FILE_SIZE_BYTES,
        )
        .document
    )
