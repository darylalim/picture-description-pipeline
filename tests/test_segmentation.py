"""Tests for the segmentation module."""

from pipeline.segmentation import extract_segmentation


# --- extract_segmentation tests ---


def test_extract_segmentation_valid_text() -> None:
    text = "<seg>others *3\n dog *1| others *2\n others *3</seg>"
    result = extract_segmentation(text, patch_h=3, patch_w=3)
    assert result is not None
    assert len(result) == 9
    # Row 0: others others others -> 0 0 0
    # Row 1: dog others others -> 1 0 0
    # Row 2: others others others -> 0 0 0
    assert result == [0, 0, 0, 1, 0, 0, 0, 0, 0]


def test_extract_segmentation_no_seg_tags() -> None:
    result = extract_segmentation("no tags here")
    assert result is None


def test_extract_segmentation_pads_short_mask() -> None:
    text = "<seg>others *2</seg>"
    result = extract_segmentation(text, patch_h=2, patch_w=2)
    assert result is not None
    assert len(result) == 4
    # Pads with last value (0 for "others")
    assert result == [0, 0, 0, 0]


def test_extract_segmentation_truncates_long_mask() -> None:
    text = "<seg>cat *6</seg>"
    result = extract_segmentation(text, patch_h=2, patch_w=2)
    assert result is not None
    assert len(result) == 4
    assert result == [1, 1, 1, 1]


def test_extract_segmentation_multiple_labels() -> None:
    text = "<seg>others *1| cat *1| dog *1</seg>"
    result = extract_segmentation(text, patch_h=1, patch_w=3)
    assert result is not None
    assert result == [0, 1, 1]
