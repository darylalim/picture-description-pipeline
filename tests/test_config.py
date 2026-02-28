"""Tests for the pipeline config module."""

from unittest.mock import MagicMock, patch

from docling.document_converter import DocumentConverter

from pipeline.config import convert, create_converter


def test_create_converter_returns_document_converter() -> None:
    converter = create_converter()
    assert isinstance(converter, DocumentConverter)


@patch("pipeline.config.create_converter")
def test_convert_creates_converter_when_none_provided(mock_create: MagicMock) -> None:
    mock_doc = MagicMock()
    mock_create.return_value.convert.return_value.document = mock_doc

    result = convert("test.pdf")

    mock_create.assert_called_once()
    mock_create.return_value.convert.assert_called_once_with(source="test.pdf")
    assert result is mock_doc


def test_convert_uses_provided_converter() -> None:
    mock_converter = MagicMock(spec=DocumentConverter)
    mock_doc = MagicMock()
    mock_converter.convert.return_value.document = mock_doc

    result = convert("test.pdf", converter=mock_converter)

    mock_converter.convert.assert_called_once_with(source="test.pdf")
    assert result is mock_doc
