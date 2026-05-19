from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def infer_hwc_from_shape(shape: tuple[int, ...]) -> tuple[int, int, int]:
    if len(shape) == 4:
        if shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for image input, got shape {shape}")
        return int(shape[1]), int(shape[2]), int(shape[3])

    if len(shape) == 3:
        return int(shape[0]), int(shape[1]), int(shape[2])

    raise ValueError(
        f"Cannot convert image directly for unsupported input shape {shape}. "
        "Use a .npy input instead."
    )


def load_image_tensor(
    path: Path,
    expected_shape: tuple[int, ...],
    image_scale: float,
    image_mean: list[float] | None,
    image_std: list[float] | None,
) -> np.ndarray:
    h, w, c = infer_hwc_from_shape(expected_shape)
    if c not in (1, 3, 4):
        raise ValueError(
            f"Unsupported channel count for image input: {c}. "
            "Use a .npy input for non-image tensor layouts."
        )

    image = Image.open(path)

    if c == 1:
        image = image.convert("L")
    elif c == 3:
        image = image.convert("RGB")
    else:
        image = image.convert("RGBA")

    image = image.resize((w, h))
    arr = np.asarray(image, dtype=np.float32)

    if c == 1 and arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)

    if image_scale != 0:
        arr = arr / image_scale

    if image_mean is not None:
        if len(image_mean) != c:
            raise ValueError(f"image_mean length {len(image_mean)} does not match channels {c}")
        arr = arr - np.asarray(image_mean, dtype=np.float32)

    if image_std is not None:
        if len(image_std) != c:
            raise ValueError(f"image_std length {len(image_std)} does not match channels {c}")
        arr = arr / np.asarray(image_std, dtype=np.float32)

    if len(expected_shape) == 4:
        arr = np.expand_dims(arr, axis=0)

    return arr
