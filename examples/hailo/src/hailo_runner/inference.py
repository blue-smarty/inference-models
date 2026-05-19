from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .image_utils import is_image_file, load_image_tensor
from .io_utils import load_npy_tensor, normalize_input_shape

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


def choose_format_type(name: str) -> Any:
    if name == "uint8":
        return FormatType.UINT8
    return FormatType.FLOAT32


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

    if is_image_file(path):
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


def inspect_hef(hef_path: str | Path) -> tuple[Any, list[Any], list[Any]]:
    hef = HEF(str(hef_path))
    input_infos = hef.get_input_vstream_infos()
    output_infos = hef.get_output_vstream_infos()
    return hef, list(input_infos), list(output_infos)


def run_inference(
    hef_path: str | Path,
    input_paths: list[Path],
    input_format: str = "auto",
    image_scale: float = 255.0,
    image_mean: list[float] | None = None,
    image_std: list[float] | None = None,
) -> tuple[list[Any], list[Any], dict[str, np.ndarray]]:
    hef, input_infos, output_infos = inspect_hef(hef_path)

    if len(input_paths) != len(input_infos):
        raise ValueError(
            f"Number of inputs ({len(input_paths)}) does not match "
            f"number of model inputs ({len(input_infos)})."
        )

    requested_format = choose_format_type(input_format)
    force_uint8 = input_format == "uint8"

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
                path=path,
                info=info,
                image_scale=image_scale,
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

    return input_infos, output_infos, {k: np.asarray(v) for k, v in results.items()}
