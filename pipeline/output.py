import warnings
from typing import Literal

from docling_core.types.doc.document import (
    DescriptionAnnotation,
    DoclingDocument,
    PictureItem,
    TableItem,
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


def get_table_content(table: TableItem, doc: DoclingDocument) -> dict[str, object]:
    """Extract table content as markdown and structured data."""
    df = table.export_to_dataframe(doc=doc)
    return {
        "markdown": table.export_to_markdown(doc=doc),
        "data": {
            "columns": [str(c) for c in df.columns],
            "rows": df.values.tolist(),
        },
    }


def build_element(
    item: PictureItem | TableItem,
    doc: DoclingDocument,
    element_number: int,
    element_type: Literal["picture", "table"],
) -> dict[str, object]:
    """Build a unified element dict for a picture or table."""
    if element_type == "picture":
        assert isinstance(item, PictureItem)
        content: dict[str, object] = {"description": get_description(item)}
    else:
        assert isinstance(item, TableItem)
        content = get_table_content(item, doc)
    return {
        "element_number": element_number,
        "type": element_type,
        "reference": item.self_ref,
        "caption": item.caption_text(doc=doc) or "",
        "content": content,
    }


def build_output(doc: DoclingDocument, duration_s: float) -> dict[str, object]:
    """Build the output dictionary from a converted document."""
    elements: list[dict[str, object]] = []
    counter = 1
    for pic in doc.pictures:
        elements.append(build_element(pic, doc, counter, "picture"))
        counter += 1
    for table in doc.tables:
        elements.append(build_element(table, doc, counter, "table"))
        counter += 1
    return {
        "document_info": {
            "num_pictures": len(doc.pictures),
            "num_tables": len(doc.tables),
            "total_duration_s": duration_s,
        },
        "elements": elements,
    }
