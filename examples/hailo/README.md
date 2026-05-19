# Hailo-8 Python Inference Runner

This example provides a generic Python program for running a compiled Hailo model (`.hef`) on a Hailo-8 device.

## Files

- `run_hailo.py` — generic inference runner
- `make_dummy_input.py` — create dummy `.npy` inputs matching expected shapes
- `requirements.txt` — Python dependencies for the helper scripts

## Requirements

You need:

- a Hailo-8 device
- HailoRT installed
- Hailo Python bindings installed (`hailo_platform`)
- Python 3.9+ recommended

> `hailo_platform` typically comes from Hailo's SDK/runtime installation rather than PyPI.

## Install Python dependencies

```bash
pip install -r examples/hailo/requirements.txt
```

## Basic usage

Run a compiled Hailo model with a `.npy` tensor input:

```bash
python examples/hailo/run_hailo.py \
  --hef model.hef \
  --input input.npy
```

## Save outputs

```bash
python examples/hailo/run_hailo.py \
  --hef model.hef \
  --input input.npy \
  --output-dir outputs
```

## Print full outputs

```bash
python examples/hailo/run_hailo.py \
  --hef model.hef \
  --input input.npy \
  --print-full
```

## Run with image input

For common single-input image models, you can pass an image directly:

```bash
python examples/hailo/run_hailo.py \
  --hef model.hef \
  --input image.jpg
```

The script will:

- inspect the model input shape
- resize the image to the expected width/height
- convert to grayscale/RGB/RGBA based on channel count
- add a batch dimension if needed

## Image normalization options

### Scale pixels to 0..1

```bash
python examples/hailo/run_hailo.py \
  --hef model.hef \
  --input image.jpg \
  --image-scale 255.0
```

### No scaling

```bash
python examples/hailo/run_hailo.py \
  --hef model.hef \
  --input image.jpg \
  --image-scale 1.0
```

### Mean/std normalization

```bash
python examples/hailo/run_hailo.py \
  --hef model.hef \
  --input image.jpg \
  --image-scale 1.0 \
  --image-mean 123.675,116.28,103.53 \
  --image-std 58.395,57.12,57.375
```

## Multi-input models

If the model has multiple inputs, pass `--input` multiple times in the same order as the discovered model input streams:

```bash
python examples/hailo/run_hailo.py \
  --hef multi_input_model.hef \
  --input input0.npy \
  --input input1.npy
```

## Creating dummy inputs

You can generate zero/random inputs that match the model input shapes:

```bash
python examples/hailo/make_dummy_input.py --hef model.hef --outdir sample_inputs
```

This creates one `.npy` file per input stream.

## Example workflow

### 1. Generate sample inputs

```bash
python examples/hailo/make_dummy_input.py --hef model.hef --outdir sample_inputs
```

### 2. Run inference

```bash
python examples/hailo/run_hailo.py \
  --hef model.hef \
  --input sample_inputs/input_0.npy \
  --output-dir outputs
```

## Notes on compatibility

Hailo's Python API may differ slightly between SDK versions. This runner includes some compatibility fallbacks, but if you see API errors, likely fixes include:

- adjusting imports from `hailo_platform`
- changing `ConfigureParams` creation
- changing vstream parameter creation helpers

## Limitations

This is a generic tensor runner. It does **not** automatically include model-specific postprocessing such as:

- ImageNet class labels
- object detection box decoding / NMS
- segmentation mask rendering
- OCR/token decoding

Those can be added on top once you know the specific model type.

## Troubleshooting

### `ImportError: No module named hailo_platform`

Install HailoRT and its Python bindings from Hailo's SDK/runtime package.

### Input shape mismatch

The script prints model input shapes at startup. Make sure your `.npy` input matches the expected shape, or use `make_dummy_input.py` to generate a valid template.

### Image input does not work

Direct image input only supports common HWC/NHWC image-shaped tensors. For unusual layouts, convert your preprocessed tensor into a `.npy` file and pass that instead.
