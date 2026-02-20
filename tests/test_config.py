"""Tests for the pipeline config module."""

from unittest.mock import MagicMock, patch

from docling.document_converter import DocumentConverter

from pipeline.config import MAX_FILE_SIZE_BYTES, MAX_PAGES, convert, create_converter


def test_max_pages() -> None:
    assert MAX_PAGES == 100


def test_max_file_size_bytes() -> None:
    assert MAX_FILE_SIZE_BYTES == 20 * 1024 * 1024


def test_create_converter_returns_document_converter() -> None:
    converter = create_converter()
    assert isinstance(converter, DocumentConverter)


@patch("pipeline.config.create_converter")
def test_convert_calls_converter_with_limits(mock_create: MagicMock) -> None:
    mock_doc = MagicMock()
    mock_create.return_value.convert.return_value.document = mock_doc

    result = convert("test.pdf")

    mock_create.return_value.convert.assert_called_once_with(
        source="test.pdf",
        max_num_pages=MAX_PAGES,
        max_file_size=MAX_FILE_SIZE_BYTES,
    )
    assert result is mock_doc
