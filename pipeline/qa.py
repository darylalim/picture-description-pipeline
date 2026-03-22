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
