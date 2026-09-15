# PyTorch Playground

A small playground project for learning and experimenting with [PyTorch](https://pytorch.org/), GPU acceleration, and basic machine learning concepts.

The project currently focuses on:

* PyTorch fundamentals
* Tensor operations
* GPU acceleration
* AMD ROCm / HIP support
* GPU benchmarking
* Automatic differentiation
* Linear regression
* Basic model training

## Project Structure

```text
pytorch-playground/
├── .venv/
├── examples/
│   ├── 01_torch_basic.py
│   ├── 02_gpu_benchmark.py
│   └── 03_linear_regression.py
├── .gitignore
├── README.md
└── requirements.txt
```

### Examples

#### `01_torch_basic.py`

Basic PyTorch environment test.

It checks:

* PyTorch version
* HIP / ROCm availability
* GPU availability
* GPU name
* Tensor creation on the GPU

#### `02_gpu_benchmark.py`

Runs a matrix multiplication benchmark on the GPU.

It can be used to verify that PyTorch is actually executing computations on the GPU and to inspect GPU memory usage.

#### `03_linear_regression.py`

A minimal machine learning example that trains a linear model to learn the function:

```text
y = 2x + 1
```

It demonstrates:

* `torch.nn`
* Loss functions
* Optimizers
* Gradient computation
* Backpropagation
* Training loops
* Inference

## Requirements

* Python 3
* PyTorch
* A supported GPU is optional

For AMD GPU acceleration, a compatible ROCm environment is required.

> PyTorch uses the `torch.cuda` API for GPU operations even when running through AMD ROCm/HIP.

## Installation

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Examples

Basic PyTorch test:

```bash
python examples/01_torch_basic.py
```

GPU benchmark:

```bash
python examples/02_gpu_benchmark.py
```

Linear regression:

```bash
python examples/03_linear_regression.py
```

## GPU Detection

You can quickly check whether PyTorch detects the GPU with:

```python
import torch

print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

For an AMD ROCm installation:

```python
print(torch.version.hip)
```

should return the HIP version used by the installed PyTorch build.

## Purpose

This repository is intended as a learning and experimentation environment.

It may progressively include examples covering:

* Neural networks
* Classification
* Computer vision
* Model serialization
* Datasets and DataLoaders
* Mixed precision
* GPU performance
* Transformers
* LLM-related experiments
