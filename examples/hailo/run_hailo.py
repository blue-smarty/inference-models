#!/usr/bin/env python3
"""
Generic Hailo-8 inference runner for HEF models.

Features:
- Loads a compiled Hailo `.hef` model
- Discovers model input/output streams
- Accepts one or more `.npy` inputs
- Optionally accepts images (`.jpg`, `.jpeg`, `.png`) for single-input models
- Runs inference on a Hailo-8 device
- Prints output tensor summaries
- Optionally saves outputs to `.npy`

Notes:
- Targets the HailoRT Python API.
- Some Hailo SDK versions expose slightly different Python APIs.
  This script includes a few compatibility fallbacks, but you may still
  need small adjustments depending on your installed version.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from hailo_platform import (
        HEF,
        VDevice,
        FormatType,
        HailoStreamInterface,
        InferVStreams,
        ConfigureParams,
        InputVStreamParams,
        OutputVStreamParams,
    )
except ImportError as exc:
    raise SystemExit(
        "Failed to import Hailo Python API from 'hailo_platform'. "
        "Make sure HailoRT and its Python bindings are installed."
    ) from exc


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on a Hailo-8 device using a compiled HEF model."
    )
    parser.add_argument(
        "--hef",
        required=True,
        help="Path to the compiled Hailo model (.hef).",
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help=(
            "Input source. Can be a .npy file, or for single-input image models, "
            "a .jpg/.jpeg/.png file. Pass multiple times for multi-input models."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to save output tensors as .npy files.",
    )
    parser.add_argument(
        "--print-full",
        action="store_true",
        help="Print full output tensors instead of only metadata and a small preview.",
    )
    parser.add_argument(
        "--input-format",
        choices=["float32", "uint8", "auto"],
        default="auto",
        help="Requested input vstream format. 'auto' defaults to float32.",
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=255.0,
        help="Scale divisor for image input. Common values: 255.0 or 1.0.",
    )
    parser.add_argument(
        "--image-mean",
        default=None,
        help=(
            "Optional comma-separated per-channel mean for image normalization, "
            "for example: 123.675,116.28,103.53"
        ),
    )
    parser.add_argument(
        "--image-std",
        default=None,
        help=(
            "Optional comma-separated per-channel std for image normalization, "
            "for example: 58.395,57.12,57.375"
        ),
    )
    return parser.parse_args()


def ensure_file(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
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


def to_serializable_shape(shape: Any) -> list[int]:
    return [int(x) for x in shape]


def print_stream_info(prefix: str, info: Any) -> None:
    shape = getattr(info, "shape", None)
    name = getattr(info, "name", "<unknown>")
    print(
        json.dumps(
            {
                "type": prefix,
                "name": name,
                "shape": to_serializable_shape(shape) if shape is not None else None,
            },
            indent=2,
        )
    )


def choose_format_type(name: str) -> Any:
    if name == "uint8":
        return FormatType.UINT8
    return FormatType.FLOAT32


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


def infer_hwc_from_shape(shape: tuple[int, ...]) -> tuple[int, int, int]:
    """
    Best-effort inference of image HWC layout from a model input shape.
    Supports common NHWC and HWC cases.
    """
    if len(shape) == 4:
        if shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for image input, got shape {shape}")
        # Assume NHWC
        return int(shape[1]), int(shape[2]), int(shape[3])

    if len(shape) == 3:
        # Assume HWC
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


def build_configure_params(hef: HEF) -> Any:
    try:
        return ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe
        )
    except AttributeError:
        return None


def build_vstream_params(network_group: Any, format_type: Any) -> tuple[Any, Any]:
    try:
        input_params = InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=format_type
        )
        output_params = OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32
        )
        return input_params, output_params
    except AttributeError:
        input_params = network_group.make_input_vstream_params(
            format_type=format_type
        )
        output_params = network_group.make_output_vstream_params(
            format_type=FormatType.FLOAT32
        )
        return input_params, output_params


def load_input_for_info(
    path: Path,
    info: Any,
    image_scale: float,
    image_mean: list[float] | None,
    image_std: list[float] | None,
    force_uint8: bool,
) -> np.ndarray:
    expected_shape = tuple(int(x) for x in info.shape)

    if path.suffix.lower() in IMAGE_SUFFIXES:
        tensor = load_image_tensor(
            path,
            expected_shape=expected_shape,
            image_scale=image_scale,
            image_mean=image_mean,
            image_std=image_std,
        )
    else:
        tensor = load_npy_tensor(path)
        tensor = normalize_input_shape(tensor, expected_shape, info.name)

    if force_uint8:
        tensor = tensor.astype(np.uint8, copy=False)
    else:
        tensor = tensor.astype(np.float32, copy=False)

    return tensor


def main() -> int:
    args = parse_args()

    hef_path = ensure_file(args.hef)
    input_paths = [ensure_file(p) for p in args.input]
    output_dir = Path(args.output_dir) if args.output_dir else None
    image_mean = parse_csv_floats(args.image_mean)
    image_std = parse_csv_floats(args.image_std)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    hef = HEF(str(hef_path))

    try:
        input_infos = hef.get_input_vstream_infos()
        output_infos = hef.get_output_vstream_infos()
    except AttributeError as exc:
        raise RuntimeError(
            "Unable to query input/output stream info from HEF. "
            "Your Hailo SDK version may use different APIs."
        ) from exc

    print(f"Loaded HEF: {hef_path}")
    print(f"Discovered {len(input_infos)} input stream(s) and {len(output_infos)} output stream(s).")

    for info in input_infos:
        print_stream_info("input", info)
    for info in output_infos:
        print_stream_info("output", info)

    if len(input_paths) != len(input_infos):
        raise ValueError(
            f"Number of --input files ({len(input_paths)}) does not match "
            f"number of model inputs ({len(input_infos)})."
        )

    requested_format = choose_format_type(args.input_format)
    force_uint8 = args.input_format == "uint8"

    with VDevice() as device:
        configure_params = build_configure_params(hef)

        if configure_params is None:
            network_groups = device.configure(hef)
        else:
            network_groups = device.configure(hef, configure_params)

        if not network_groups:
            raise RuntimeError("No network groups were configured from the HEF.")

        network_group = network_groups[0]

        input_vstream_params, output_vstream_params = build_vstream_params(
            network_group, requested_format
        )

        infer_inputs: dict[str, np.ndarray] = {}
        for info, path in zip(input_infos, input_paths):
            tensor = load_input_for_info(
                path,
                info,
                image_scale=args.image_scale,
                image_mean=image_mean,
                image_std=image_std,
                force_uint8=force_uint8,
            )
            infer_inputs[info.name] = tensor

        with network_group.activate():
            with InferVStreams(
                network_group,
                input_vstream_params,
                output_vstream_params,
            ) as infer_pipeline:
                results = infer_pipeline.infer(infer_inputs)

    print("\nInference completed.\n")

    for output_name, output_value in results.items():
        arr = np.asarray(output_value)
        print(f"Output: {output_name}")
        print(f"  shape={arr.shape}")
        print(f"  dtype={arr.dtype}")
        print(f"  min={arr.min() if arr.size else 'n/a'}")
        print(f"  max={arr.max() if arr.size else 'n/a'}")

        if args.print_full:
            print(arr)
        else:
            flat = arr.reshape(-1) if arr.size else arr
            preview = flat[: min(10, flat.size)] if arr.size else flat
            print(f"  preview={preview}")

        if output_dir is not None:
            out_path = output_dir / f"{output_name}.npy"
            np.save(out_path, arr)
            print(f"  saved={out_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
