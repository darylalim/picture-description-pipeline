"""Tests for the build_output function."""

import sys
from unittest.mock import MagicMock

import pytest
from docling_core.types.doc.document import (
    DescriptionMetaField,
    DoclingDocument,
    PictureItem,
    PictureMeta,
)


@pytest.fixture(autouse=True)
def _mock_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent Streamlit from running during import."""
    mock_st = MagicMock()
    mock_st.cache_resource = lambda f: f
    mock_st.button.return_value = False
    monkeypatch.setitem(sys.modules, "streamlit", mock_st)


def _make_doc(pictures: list[PictureItem] | None = None) -> DoclingDocument:
    """Create a DoclingDocument with the given pictures."""
    doc = DoclingDocument(name="test")
    if pictures:
        doc.pictures = pictures
    return doc


def _make_picture(
    index: int,
    text: str | None = None,
    created_by: str | None = None,
) -> PictureItem:
    """Create a PictureItem with optional description metadata."""
    meta = None
    if text is not None:
        meta = PictureMeta(
            description=DescriptionMetaField(text=text, created_by=created_by)
        )
    return PictureItem(
        self_ref=f"#/pictures/{index}",
        meta=meta,
    )


def test_top_level_keys() -> None:
    from streamlit_app import build_output

    doc = _make_doc()
    result = build_output(doc, 1.5)
    assert set(result.keys()) == {"document_info", "pictures"}


def test_document_info_fields() -> None:
    from streamlit_app import build_output

    doc = _make_doc([_make_picture(0)])
    result = build_output(doc, 2.34)
    info = result["document_info"]
    assert isinstance(info, dict)
    assert info["num_pictures"] == 1
    assert info["total_duration_s"] == 2.34


def test_picture_entry_structure() -> None:
    from streamlit_app import build_output

    pic = _make_picture(0, text="A test description.", created_by="test-model")
    doc = _make_doc([pic])
    result = build_output(doc, 0.5)
    pictures = result["pictures"]
    assert isinstance(pictures, list)
    assert len(pictures) == 1
    entry = pictures[0]
    assert entry["picture_number"] == 1
    assert entry["reference"] == "#/pictures/0"
    assert entry["caption"] == ""
    assert isinstance(entry["description"], dict)


def test_description_fields() -> None:
    from streamlit_app import build_output

    pic = _make_picture(0, text="Describes an image.", created_by="granite-vision")
    doc = _make_doc([pic])
    result = build_output(doc, 1.0)
    description = result["pictures"][0]["description"]
    assert description["text"] == "Describes an image."
    assert description["created_by"] == "granite-vision"


def test_multiple_pictures() -> None:
    from streamlit_app import build_output

    pics = [
        _make_picture(0, text="First pic.", created_by="model-a"),
        _make_picture(1, text="Second pic.", created_by="model-b"),
    ]
    doc = _make_doc(pics)
    result = build_output(doc, 3.0)
    pictures = result["pictures"]
    assert len(pictures) == 2
    assert pictures[0]["picture_number"] == 1
    assert pictures[1]["picture_number"] == 2
    assert pictures[0]["description"]["text"] == "First pic."
    assert pictures[1]["description"]["text"] == "Second pic."


def test_no_description() -> None:
    from streamlit_app import build_output

    pic = _make_picture(0)
    doc = _make_doc([pic])
    result = build_output(doc, 0.5)
    assert result["pictures"][0]["description"] is None


def test_empty_document() -> None:
    from streamlit_app import build_output

    doc = _make_doc()
    result = build_output(doc, 0.0)
    assert result["document_info"]["num_pictures"] == 0
    assert result["pictures"] == []
