"""Tests for the QA module."""

from unittest.mock import MagicMock, patch

from PIL import Image

from pipeline.qa import create_qa_model, resize_for_qa


# --- resize_for_qa tests ---


def test_resize_landscape_image() -> None:
    image = Image.new("RGB", (1024, 768))
    result = resize_for_qa(image)
    assert result.size == (768, 576)


def test_resize_portrait_image() -> None:
    image = Image.new("RGB", (768, 1024))
    result = resize_for_qa(image)
    assert result.size == (576, 768)


def test_small_image_unchanged() -> None:
    image = Image.new("RGB", (400, 300))
    result = resize_for_qa(image)
    assert result.size == (400, 300)


def test_exact_max_dim_unchanged() -> None:
    image = Image.new("RGB", (768, 500))
    result = resize_for_qa(image)
    assert result.size == (768, 500)


def test_resize_preserves_aspect_ratio() -> None:
    image = Image.new("RGB", (1600, 1200))
    result = resize_for_qa(image)
    w, h = result.size
    assert abs(w / h - 1600 / 1200) < 0.01


def test_resize_custom_max_dim() -> None:
    image = Image.new("RGB", (1024, 768))
    result = resize_for_qa(image, max_dim=512)
    assert result.size == (512, 384)


# --- create_qa_model tests ---


@patch("pipeline.qa.AutoModelForVision2Seq")
@patch("pipeline.qa.AutoProcessor")
def test_create_qa_model_loads_correct_model(
    mock_processor_cls: MagicMock,
    mock_model_cls: MagicMock,
) -> None:
    processor, model = create_qa_model(device="cpu")

    mock_processor_cls.from_pretrained.assert_called_once_with(
        "ibm-granite/granite-vision-3.3-2b"
    )
    mock_model_cls.from_pretrained.assert_called_once_with(
        "ibm-granite/granite-vision-3.3-2b"
    )
    assert processor is mock_processor_cls.from_pretrained.return_value
    assert model is mock_model_cls.from_pretrained.return_value.to.return_value


@patch("pipeline.qa.AutoModelForVision2Seq")
@patch("pipeline.qa.AutoProcessor")
def test_create_qa_model_moves_to_device(
    mock_processor_cls: MagicMock,
    mock_model_cls: MagicMock,
) -> None:
    create_qa_model(device="cpu")

    mock_model_cls.from_pretrained.return_value.to.assert_called_once_with("cpu")
