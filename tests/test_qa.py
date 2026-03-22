"""Tests for the QA module."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from pipeline.qa import create_qa_model, generate_qa_response, resize_for_qa


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


def test_resize_square_at_max_dim_unchanged() -> None:
    image = Image.new("RGB", (768, 768))
    result = resize_for_qa(image)
    assert result.size == (768, 768)


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


# --- generate_qa_response tests ---


def test_generate_qa_response_rejects_empty_images() -> None:
    with pytest.raises(ValueError, match="1 to 8"):
        generate_qa_response([], "What is this?", MagicMock(), MagicMock())


def test_generate_qa_response_rejects_more_than_8_images() -> None:
    images = [Image.new("RGB", (100, 100)) for _ in range(9)]
    with pytest.raises(ValueError, match="1 to 8"):
        generate_qa_response(images, "What is this?", MagicMock(), MagicMock())


def test_generate_qa_response_prompt_structure() -> None:
    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    mock_processor.apply_chat_template.return_value = MagicMock()
    mock_processor.apply_chat_template.return_value.to.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]])
    }

    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock_processor.decode.return_value = "The answer is 42."

    images = [Image.new("RGB", (100, 100)) for _ in range(3)]
    result = generate_qa_response(
        images, "What is the answer?", mock_processor, mock_model
    )

    # Verify conversation structure
    call_args = mock_processor.apply_chat_template.call_args
    conversation = call_args[0][0]
    content = conversation[0]["content"]

    # Should have 3 image entries + 1 text entry
    image_entries = [c for c in content if c["type"] == "image"]
    text_entries = [c for c in content if c["type"] == "text"]
    assert len(image_entries) == 3
    assert len(text_entries) == 1
    assert text_entries[0]["text"] == "What is the answer?"

    # Verify each image entry has an "image" key
    for entry in image_entries:
        assert "image" in entry

    # Verify apply_chat_template keyword arguments
    call_kwargs = mock_processor.apply_chat_template.call_args[1]
    assert call_kwargs["add_generation_prompt"] is True
    assert call_kwargs["tokenize"] is True
    assert call_kwargs["return_dict"] is True
    assert call_kwargs["return_tensors"] == "pt"

    assert result == "The answer is 42."


def test_generate_qa_response_trims_input_and_uses_skip_special_tokens() -> None:
    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    input_ids = torch.tensor([[1, 2]])
    mock_processor.apply_chat_template.return_value = MagicMock()
    mock_processor.apply_chat_template.return_value.to.return_value = {
        "input_ids": input_ids
    }

    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
    mock_processor.decode.return_value = "answer"

    generate_qa_response(
        [Image.new("RGB", (10, 10))], "question", mock_processor, mock_model
    )

    # Verify decode is called with only the new tokens (trimmed)
    mock_processor.decode.assert_called_once()
    decoded_tensor = mock_processor.decode.call_args[0][0]
    assert torch.equal(decoded_tensor, torch.tensor([3, 4]))
    assert mock_processor.decode.call_args[1]["skip_special_tokens"] is True


def test_generate_qa_response_returns_empty_on_no_new_tokens() -> None:
    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    input_ids = torch.tensor([[1, 2]])
    mock_processor.apply_chat_template.return_value = MagicMock()
    mock_processor.apply_chat_template.return_value.to.return_value = {
        "input_ids": input_ids
    }

    # Model generates no new tokens (output same length as input)
    mock_model.generate.return_value = torch.tensor([[1, 2]])
    mock_processor.decode.return_value = ""

    result = generate_qa_response(
        [Image.new("RGB", (10, 10))], "question", mock_processor, mock_model
    )
    assert result == ""
