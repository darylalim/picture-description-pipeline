"""Tests for the pipeline config module."""

from docling.document_converter import DocumentConverter

from pipeline.config import MAX_FILE_SIZE_BYTES, MAX_PAGES, create_converter


def test_max_pages() -> None:
    assert MAX_PAGES == 100


def test_max_file_size_bytes() -> None:
    assert MAX_FILE_SIZE_BYTES == 20 * 1024 * 1024


def test_create_converter_returns_document_converter() -> None:
    converter = create_converter()
    assert isinstance(converter, DocumentConverter)
