import warnings

from docling_core.types.doc.document import (
    DescriptionAnnotation,
    DoclingDocument,
    PictureItem,
)


def get_description(pic: PictureItem) -> dict[str, str] | None:
    """Extract description from meta or annotations fallback."""
    if pic.meta and pic.meta.description:
        return {
            "created_by": pic.meta.description.created_by,
            "text": pic.meta.description.text,
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        for ann in pic.annotations:
            if isinstance(ann, DescriptionAnnotation):
                return {"created_by": ann.provenance, "text": ann.text}
    return None


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
                "description": get_description(pic),
            }
            for idx, pic in enumerate(doc.pictures, 1)
        ],
    }
