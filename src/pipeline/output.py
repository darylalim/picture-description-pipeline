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
