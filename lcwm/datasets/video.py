from collections.abc import Iterable
from typing import Any

from PIL import Image, ImageOps

from lcwm.datasets.protocol import legal_frame_count

TARGET_SIZE = (256, 256)
PADDING_RGB = (0, 0, 0)
RESAMPLING = Image.Resampling.LANCZOS
RESAMPLING_NAME = "PIL.Image.Resampling.LANCZOS"
CONTENT_BOX_INFO_KEY = "tiresias_letterbox_content_box"


def frame_indices_with_wan_padding(available: int, requested: int) -> list[int]:
    available = int(available)
    requested = int(requested)
    if available < 1 or requested < 1:
        raise ValueError("video frame counts must be positive")
    if requested > available and requested != legal_frame_count(available):
        raise ValueError(
            f"cannot pad {available} source frames to {requested}; "
            "only Wan 4k+1 padding is supported"
        )
    return [min(index, available - 1) for index in range(requested)]


def _positive_size(size: Iterable[int], label: str) -> tuple[int, int]:
    values = tuple(int(value) for value in size)
    if len(values) != 2 or values[0] <= 0 or values[1] <= 0:
        raise ValueError(f"{label} must be two positive integers, got {values}")
    return values


def contain_and_pad(
    image: Image.Image,
    target_size: Iterable[int] = TARGET_SIZE,
) -> Image.Image:
    target_width, target_height = _positive_size(target_size, "target_size")
    source = image.convert("RGB")
    contained = ImageOps.contain(
        source,
        (target_width, target_height),
        method=RESAMPLING,
    )
    left = (target_width - contained.width) // 2
    top = (target_height - contained.height) // 2
    canvas = Image.new("RGB", (target_width, target_height), PADDING_RGB)
    canvas.paste(contained, (left, top))
    canvas.info[CONTENT_BOX_INFO_KEY] = (
        left,
        top,
        left + contained.width,
        top + contained.height,
    )
    return canvas


def content_box_from_image(image: Image.Image) -> tuple[int, int, int, int]:
    raw = image.info.get(CONTENT_BOX_INFO_KEY)
    if raw is None:
        raise KeyError(
            f"image does not carry {CONTENT_BOX_INFO_KEY}; use contain_and_pad first"
        )
    box = tuple(int(value) for value in raw)
    if len(box) != 4:
        raise ValueError(f"invalid content box metadata: {raw}")
    return box


def mask_padding_tensor(frame: Any, content_box: Iterable[int]):
    left, top, right, bottom = (int(value) for value in content_box)
    if getattr(frame, "ndim", None) != 3:
        raise ValueError(f"frame must be rank 3, got {getattr(frame, 'shape', None)}")
    output = frame.clone()
    shape = tuple(int(value) for value in output.shape)
    if shape[0] in (1, 3, 4):
        height, width = shape[1], shape[2]
        output[:, :top, :] = 0
        output[:, bottom:, :] = 0
        output[:, :, :left] = 0
        output[:, :, right:] = 0
    elif shape[-1] in (1, 3, 4):
        height, width = shape[0], shape[1]
        output[:top, :, :] = 0
        output[bottom:, :, :] = 0
        output[:, :left, :] = 0
        output[:, right:, :] = 0
    else:
        raise ValueError(f"cannot infer channel axis from {shape}")
    if not (0 <= left <= right <= width and 0 <= top <= bottom <= height):
        raise ValueError(
            f"content box {(left, top, right, bottom)} outside frame {shape}"
        )
    return output


def preprocessing_contract() -> dict[str, Any]:
    return {
        "policy": "letterbox",
        "target_size_wh": list(TARGET_SIZE),
        "padding_rgb": list(PADDING_RGB),
        "resampling": RESAMPLING_NAME,
    }
