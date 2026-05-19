from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import numpy as np

from hailo_runner.inference import run_inference


def parse_csv_floats(value: str):
    value = (value or "").strip()
    if not value:
        return None
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def summarize_results(results: dict[str, np.ndarray]) -> str:
    lines = []
    for name, arr in results.items():
        flat = arr.reshape(-1) if arr.size else arr
        preview = flat[: min(10, flat.size)] if arr.size else flat
        lines.append(
            "\n".join(
                [
                    f"Output: {name}",
                    f"  shape={arr.shape}",
                    f"  dtype={arr.dtype}",
                    f"  min={arr.min() if arr.size else 'n/a'}",
                    f"  max={arr.max() if arr.size else 'n/a'}",
                    f"  preview={preview}",
                ]
            )
        )
    return "\n\n".join(lines)


def run_ui_inference(
    hef_file,
    input_file,
    input_format,
    image_scale,
    image_mean,
    image_std,
):
    if hef_file is None:
        return "Please provide a .hef file.", None

    if input_file is None:
        return "Please provide an input file.", None

    hef_path = Path(hef_file.name)
    input_path = Path(input_file.name)

    image_mean_vals = parse_csv_floats(image_mean)
    image_std_vals = parse_csv_floats(image_std)

    input_infos, output_infos, results = run_inference(
        hef_path=hef_path,
        input_paths=[input_path],
        input_format=input_format,
        image_scale=float(image_scale),
        image_mean=image_mean_vals,
        image_std=image_std_vals,
    )

    summary_lines = []
    summary_lines.append(f"Loaded HEF: {hef_path}")
    summary_lines.append(f"Discovered {len(input_infos)} input(s), {len(output_infos)} output(s).")
    summary_lines.append("")
    summary_lines.append(summarize_results(results))

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        np.savez(tmp.name, **results)
        download_path = tmp.name

    return "\n".join(summary_lines), download_path


demo = gr.Interface(
    fn=run_ui_inference,
    inputs=[
        gr.File(label="HEF file"),
        gr.File(label="Input file (.npy, .jpg, .jpeg, .png)"),
        gr.Dropdown(["auto", "float32", "uint8"], value="auto", label="Input format"),
        gr.Number(value=255.0, label="Image scale"),
        gr.Textbox(label="Image mean (comma-separated)", value=""),
        gr.Textbox(label="Image std (comma-separated)", value=""),
    ],
    outputs=[
        gr.Textbox(label="Inference summary", lines=20),
        gr.File(label="Download outputs (.npz)"),
    ],
    title="Hailo-8 Inference Runner",
    description="Run a compiled Hailo HEF model with tensor or image input.",
)

if __name__ == "__main__":
    demo.launch()
