# Image Segmentation Design

## Overview

Add experimental image segmentation to the Granite Vision Pipeline. Users upload an image and provide a natural language prompt describing what to segment. The system produces a binary mask using Granite Vision's referring segmentation capability, refined with SAM (Segment Anything Model) for smooth edges.

This is a standalone feature, separate from the existing PDF extraction flow.

## Architecture

### New module: `pipeline/segmentation.py`

All `Image` references below refer to `PIL.Image.Image`. All `Tensor` references refer to `torch.Tensor`.

All model inference runs under `torch.inference_mode()` to avoid unnecessary gradient computation.

Input images are converted to RGB via `image.convert("RGB")` at the start of `segment()` to handle RGBA, palette, or grayscale inputs from PNG uploads.

**Public API:**

- `create_granite_model(device: str | None = None) -> tuple[AutoProcessor, AutoModelForVision2Seq]` — load Granite Vision 3.3 2B for segmentation. When `device` is None, auto-detects: CUDA if available, else CPU. (MPS is not used due to limited operator support in SAM/transformers.)
- `create_sam_model(device: str | None = None) -> tuple[SamProcessor, SamModel]` — load SAM ViT-Huge. Same device auto-detection as above.
- `segment(image: PIL.Image.Image, prompt: str, granite: tuple[AutoProcessor, AutoModelForVision2Seq], sam: tuple[SamProcessor, SamModel]) -> PIL.Image.Image | None` — run full segmentation pipeline. Converts input to RGB. Returns mask as PIL Image (mode "L", binary: 0=background, 255=foreground) or None if the model output contains no `<seg>` tags.
- `draw_mask(mask: PIL.Image.Image, image: PIL.Image.Image) -> PIL.Image.Image` — overlay mask on image as red semi-transparent composite (RGBA, color `(255, 0, 0, 50)` where mask is foreground).

**Internal helpers (not exported):**

- `extract_segmentation(text: str, patch_h: int = 24, patch_w: int = 24) -> list[int] | None` — parse `<seg>...</seg>` run-length encoded output into flat integer mask. Labels are mapped to 0 for `"others"` and 1 for any other label. Returns None if no `<seg>` tags found. Pads with last value if shorter than `patch_h * patch_w`, truncates if longer.
- `prepare_mask(mask: list[int], patch_h: int, patch_w: int, size: tuple[int, int]) -> torch.Tensor` — reshape flat mask to 2D, threshold (>0) to binary float32, nearest-neighbor interpolate to target image size.
- `sample_points(mask: torch.Tensor, num_pos: int = 15, num_neg: int = 10, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]` — sample positive (inside mask) and negative (outside mask) points. When `seed` is None, sampling is non-deterministic. Returns `(points, labels)` tensors. If the mask is all-zero (no foreground), returns zero positive points; if all-one (no background), returns zero negative points. The returned counts may be less than `num_pos`/`num_neg` in these edge cases.
- `compute_logits_from_mask(mask: torch.Tensor, eps: float = 1e-3, longest_side: int = 256) -> torch.Tensor` — convert binary mask to logits via `torch.logit`, resize to fit within `longest_side`, pad to exact `(1, longest_side, longest_side)`.
- `refine_with_sam(mask: torch.Tensor, image: PIL.Image.Image, sam: tuple[SamProcessor, SamModel]) -> torch.Tensor` — run SAM inference using sampled points and mask logits, return refined binary mask tensor at original image resolution.

**Segmentation flow (inside `segment()`):**

1. Convert input image to RGB
2. Build chat conversation with `"seg: "` prefix prompt
3. Run Granite inference under `torch.inference_mode()` with `max_new_tokens=8192`, decode output text
4. `extract_segmentation()` parses `<seg>` tags into flat mask list — returns None if not found
5. `prepare_mask()` upscales to original image dimensions
6. `sample_points()` + `compute_logits_from_mask()` prepare SAM inputs
7. `refine_with_sam()` produces pixel-accurate refined mask under `torch.inference_mode()`
8. Convert refined tensor to PIL Image (mode "L", 0/255 binary) and return

### Multipage Streamlit setup

The project currently uses a single `streamlit_app.py`. Adding a `pages/` directory activates Streamlit's multipage navigation. The existing `streamlit_app.py` remains as-is — Streamlit treats the root file as the default/home page and `pages/*.py` as additional pages. No renaming or restructuring of the existing app is needed.

This introduces a sidebar with page navigation that does not currently exist. Streamlit derives page names from filenames: `pages/segmentation.py` appears as "Segmentation" in the sidebar. The root page appears as "streamlit_app". To give it a cleaner label, add `st.set_page_config(page_title="Granite Vision Pipeline")` to `streamlit_app.py` if not already present.

### New UI page: `pages/segmentation.py`

- Page title and description noting this is experimental
- Image uploader (PNG, JPG, JPEG)
- Text input for segmentation prompt (e.g., "the dog on the left")
- "Segment" button (disabled until image + prompt provided)
- Spinner during inference
- Results: two columns — original image (left), mask overlay from `draw_mask()` (right)
- Download button for the binary mask image (the mode "L" PNG from `segment()`, not the overlay)
- Error message if `segment()` returns None (no `<seg>` tags in output)
- Models cached via `st.cache_resource` (Granite and SAM cached separately)

### Public API changes: `pipeline/__init__.py`

Add exports: `create_granite_model`, `create_sam_model`, `segment`, `draw_mask`

### Dependencies: `pyproject.toml`

Add to `[project.dependencies]`:
- `torch`
- `transformers`

Both `SamModel`/`SamProcessor` and `AutoProcessor`/`AutoModelForVision2Seq` are imported from `transformers`. No separate `segment-anything` package is needed.

These are currently indirect dependencies via docling but need to be explicit for direct use in segmentation.

Models downloaded on first use:
- `ibm-granite/granite-vision-3.3-2b` (~5GB) — note: this is a separate model instance from docling's internal copy. Both may be in memory simultaneously if the user runs PDF extraction and segmentation in the same session. This duplication is accepted as the cost of keeping the two features decoupled.
- `facebook/sam-vit-huge` (~2.5GB) — new download

## Tests: `tests/test_segmentation.py`

Unit tests for parsing and mask helpers (no model required):

- `extract_segmentation()` — valid text, missing tags returns None, short masks padded, long masks truncated
- `prepare_mask()` — correct output shape, binary thresholding, upscaling to target dimensions
- `sample_points()` — correct count of positive/negative points, coordinates within mask bounds, deterministic with seed, edge cases (all-zero and all-one masks return fewer points)
- `compute_logits_from_mask()` — output shape `(1, 256, 256)`, padding correct
- `draw_mask()` — output is RGBA, correct dimensions match input image, alpha channel varies with mask values (foreground pixels have nonzero alpha, background pixels have zero alpha)

`refine_with_sam()` and `segment()` require model weights and are excluded from the default test suite. Their internal logic (point sampling, logit computation) is covered by the helper tests above.

Integration tests requiring models/GPU are out of scope for the default test suite.

## Files changed

| File | Change |
|------|--------|
| `pipeline/segmentation.py` | New — segmentation logic |
| `pipeline/__init__.py` | Add segmentation exports |
| `pages/segmentation.py` | New — Streamlit UI page |
| `pyproject.toml` | Add torch, transformers deps |
| `tests/test_segmentation.py` | New — unit tests for helpers |
| `streamlit_app.py` | Add `st.set_page_config` if not present |
| `CLAUDE.md` | Update architecture docs |
