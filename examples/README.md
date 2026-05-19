# Examples

This directory contains runnable examples for different inference backends, model formats, and hardware targets.

## Available examples

### Hailo

- [`hailo/`](./hailo/) — generic Hailo-8 `.hef` inference runner in Python with both CLI and Gradio UI

#### Hailo quick start

Install dependencies:

```bash
pip install -r examples/hailo/requirements.txt
```

Generate a sample input:

```bash
python examples/hailo/tools/make_dummy_input.py --hef model.hef --outdir sample_inputs
```

Run inference from source:

```bash
PYTHONPATH=examples/hailo/src python -m hailo_runner \
  --hef model.hef \
  --input sample_inputs/input_0.npy \
  --output-dir outputs
```

Launch the UI:

```bash
PYTHONPATH=examples/hailo/src python examples/hailo/ui/app.py
```

For full usage details, see [`examples/hailo/README.md`](./hailo/README.md).
