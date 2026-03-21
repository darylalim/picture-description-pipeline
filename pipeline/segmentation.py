"""Image segmentation using Granite Vision and SAM refinement."""

import re


def extract_segmentation(
    text: str,
    patch_h: int = 24,
    patch_w: int = 24,
) -> list[int] | None:
    """Parse <seg>...</seg> RLE output into a flat integer mask.

    Labels are mapped to 0 for "others" and 1 for any other label.
    Returns None if no <seg> tags found.
    """
    match = re.search(r"<seg>(.*?)</seg>", text, re.DOTALL)
    if match is None:
        return None
    rows = match.group(1).strip().split("\n")
    tokens = [token.split(" *") for row in rows for token in row.split("| ")]
    tokens = [x[0].strip() for x in tokens for _ in range(int(x[1]))]

    mask = [0 if item == "others" else 1 for item in tokens]

    total_size = patch_h * patch_w
    if len(mask) < total_size:
        mask = mask + [mask[-1]] * (total_size - len(mask))
    elif len(mask) > total_size:
        mask = mask[:total_size]
    return mask
