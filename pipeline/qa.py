"""Multipage QA using Granite Vision."""

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def resize_for_qa(image: Image.Image, max_dim: int = 768) -> Image.Image:
    """Resize image so its longer dimension is at most max_dim pixels.

    Preserves aspect ratio using LANCZOS resampling.
    Returns the image unchanged if already within bounds.
    """
    w, h = image.size
    longer = max(w, h)
    if longer <= max_dim:
        return image
    scale = max_dim / longer
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def create_qa_model(
    device: str | None = None,
) -> tuple[AutoProcessor, AutoModelForVision2Seq]:
    """Load Granite Vision 3.3 2B for multipage QA.

    When device is None, auto-detects: CUDA if available, else CPU.
    MPS is excluded for consistency with other pipeline models.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "ibm-granite/granite-vision-3.3-2b"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)
    return processor, model


def generate_qa_response(
    images: list[Image.Image],
    question: str,
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
) -> str:
    """Answer a question about one or more page images.

    Accepts 1-8 images. Each image is converted to RGB and resized so the
    longer dimension is at most 768px. All images are passed to the model
    in a single conversation turn.

    Raises ValueError if images list has 0 or more than 8 items.
    Returns empty string if the model produces no output.
    """
    if not (1 <= len(images) <= 8):
        raise ValueError(f"Expected 1 to 8 images, got {len(images)}")

    prepared = [resize_for_qa(img.convert("RGB")) for img in images]

    device = next(model.parameters()).device

    content: list[dict] = [{"type": "image", "image": img} for img in prepared]
    content.append({"type": "text", "text": question})

    conversation = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(  # type: ignore[operator]
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=1024)

    trimmed = output[:, inputs["input_ids"].shape[1] :]
    decoded = processor.decode(trimmed[0], skip_special_tokens=True)  # type: ignore[operator]
    return decoded
