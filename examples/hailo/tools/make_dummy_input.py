#!/usr/bin/env python3
"""
Generate dummy .npy inputs for a Hailo HEF model.

This helper inspects the HEF input stream shapes and writes one .npy file
per input stream, which is useful for testing `hailo_runner`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from hailo_platform import HEF
except ImportError as exc:
    raise SystemExit(
        "Failed to import Hailo Python API from 'hailo_platform'. "
        "Make sure HailoRT and its Python bindings are installed."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dummy .npy inputs for a compiled Hailo HEF model."
    )
    parser.add_argument(
        "--hef",
        required=True,
        help="Path to the compiled Hailo model (.hef).",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory to write generated .npy files into.",
    )
    parser.add_argument(
        "--mode",
        choices=["zeros", "ones", "random"],
        default="random",
        help="How to fill the generated inputs.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "uint8"],
        default="float32",
        help="Data type for generated arrays.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    hef = HEF(args.hef)
    input_infos = hef.get_input_vstream_infos()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    np_dtype = np.float32 if args.dtype == "float32" else np.uint8

    for idx, info in enumerate(input_infos):
        shape = tuple(int(x) for x in info.shape)

        if args.mode == "zeros":
            arr = np.zeros(shape, dtype=np_dtype)
        elif args.mode == "ones":
            arr = np.ones(shape, dtype=np_dtype)
        else:
            if np_dtype == np.uint8:
                arr = np.random.randint(0, 256, size=shape, dtype=np.uint8)
            else:
                arr = np.random.random(size=shape).astype(np.float32)

        path = outdir / f"input_{idx}.npy"
        np.save(path, arr)
        print(f"wrote {path} shape={arr.shape} dtype={arr.dtype}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
