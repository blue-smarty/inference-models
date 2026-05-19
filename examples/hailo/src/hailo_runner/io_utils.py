from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_file(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def ensure_dir(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_csv_floats(value: str | None) -> list[float] | None:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return None
    return [float(x) for x in parts]


def load_npy_tensor(path: Path) -> np.ndarray:
    arr = np.load(path)
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy array in {path}, got {type(arr)!r}")
    return arr


def normalize_input_shape(
    tensor: np.ndarray,
    expected_shape: tuple[int, ...] | list[int] | Any,
    input_name: str,
) -> np.ndarray:
    expected = tuple(int(x) for x in expected_shape)

    if tensor.shape == expected:
        return tensor

    if len(expected) == tensor.ndim + 1 and expected[0] == 1 and tensor.shape == expected[1:]:
        return np.expand_dims(tensor, axis=0)

    raise ValueError(
        f"Input shape mismatch for '{input_name}'. "
        f"Expected {expected}, got {tensor.shape}."
    )


def to_serializable_shape(shape: Any) -> list[int]:
    return [int(x) for x in shape]


def format_stream_info(prefix: str, info: Any) -> str:
    shape = getattr(info, "shape", None)
    name = getattr(info, "name", "<unknown>")
    return json.dumps(
        {
            "type": prefix,
            "name": name,
            "shape": to_serializable_shape(shape) if shape is not None else None,
        },
        indent=2,
    )


def save_output_array(output_dir: Path | None, output_name: str, arr: np.ndarray) -> Path | None:
    if output_dir is None:
        return None
    out_path = output_dir / f"{output_name}.npy"
    np.save(out_path, arr)
    return out_path
