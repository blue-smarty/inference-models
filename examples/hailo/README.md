# Hailo-8 Python Inference Runner

This example provides a package-style Python runner for running a compiled Hailo model (`.hef`) on a Hailo-8 device.

## Layout

- `src/hailo_runner/` — package source
- `tools/make_dummy_input.py` — create dummy `.npy` inputs matching expected shapes
- `ui/app.py` — Gradio web UI
- `requirements.txt` — Python dependencies
- `pyproject.toml` — package metadata

## Requirements

You need:

- a Hailo-8 device
- HailoRT installed
- Hailo Python bindings installed (`hailo_platform`)
- Python 3.9+ recommended

> `hailo_platform` typically comes from Hailo's SDK/runtime installation rather than PyPI.

## Install dependencies

```bash
pip install -r examples/hailo/requirements.txt
```

## Run directly from source

```bash
PYTHONPATH=examples/hailo/src python -m hailo_runner \
  --hef model.hef \
  --input input.npy
```

## Save outputs

```bash
PYTHONPATH=examples/hailo/src python -m hailo_runner \
  --hef model.hef \
  --input input.npy \
  --output-dir outputs
```

## Run with image input

```bash
PYTHONPATH=examples/hailo/src python -m hailo_runner \
  --hef model.hef \
  --input image.jpg
```

## Generate dummy inputs

```bash
python examples/hailo/tools/make_dummy_input.py \
  --hef model.hef \
  --outdir sample_inputs
```

## Launch the web UI

```bash
PYTHONPATH=examples/hailo/src python examples/hailo/ui/app.py
```

Then open the local URL printed by Gradio.

## Optional local install

From `examples/hailo/`:

```bash
pip install -e .
python -m hailo_runner --hef model.hef --input input.npy
```

## Notes

This is a generic tensor runner. It does not include model-specific postprocessing such as classification labels, detection decoding, or segmentation rendering.
