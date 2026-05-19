from __future__ import annotations

import argparse

from .inference import run_inference
from .io_utils import (
    ensure_dir,
    ensure_file,
    format_stream_info,
    parse_csv_floats,
    save_output_array,
)


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def main() -> int:
    args = build_parser().parse_args()

    hef_path = ensure_file(args.hef)
    input_paths = [ensure_file(p) for p in args.input]
    output_dir = ensure_dir(args.output_dir)
    image_mean = parse_csv_floats(args.image_mean)
    image_std = parse_csv_floats(args.image_std)

    input_infos, output_infos, results = run_inference(
        hef_path=hef_path,
        input_paths=input_paths,
        input_format=args.input_format,
        image_scale=args.image_scale,
        image_mean=image_mean,
        image_std=image_std,
    )

    print(f"Loaded HEF: {hef_path}")
    print(f"Discovered {len(input_infos)} input stream(s) and {len(output_infos)} output stream(s).")

    for info in input_infos:
        print(format_stream_info("input", info))
    for info in output_infos:
        print(format_stream_info("output", info))

    print("\nInference completed.\n")

    for output_name, arr in results.items():
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

        saved_path = save_output_array(output_dir, output_name, arr)
        if saved_path is not None:
            print(f"  saved={saved_path}")

    return 0
