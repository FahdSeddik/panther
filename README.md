<div align="center">
  <img src="docs/_static/panther-logo.png" alt="Panther Logo" width="200"/>
</div>

# Panther: Faster & Cheaper Computations with RandNLA

## Why Panther?

Panther makes it incredibly easy to accelerate your existing PyTorch models with minimal code changes. Simply replace standard layers with Panther's sketched equivalents to get significant speedups and memory reductions.

### 🚀 Quick Example: Drop-in Replacement

<table>
<tr>
<th>Standard PyTorch</th>
<th>With Panther (2-3x faster)</th>
</tr>
<tr>
<td>

```python
import torch.nn as nn

class StandardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8192, 8192)
```

</td>
<td>

```python
import torch.nn as nn
import panther as pr

class PantherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = pr.nn.SKLinear(8192, 8192, 
            num_terms=1, low_rank=16)
```

</td>
</tr>
</table>

**Result:** 2-3x speedup with minimal code changes on the example hyperparameters above.

### 🤖 Automatic Optimization with AutoTuner

For complex models like BERT, Panther can automatically find optimal configurations:

```python
from transformers import BertForMaskedLM
from panther.tuner import SKAutoTuner, LayerConfig, TuningConfigs

# Load pre-trained BERT
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Configure automatic layer discovery and tuning
config = LayerConfig(
    layer_names={"type": "Linear"},
    params="auto",  # Automatic search space
    separate=True,  # Optimize each layer independently
    copy_weights=True  # Preserve trained weights
)

# Create tuner with a quality metric constraint
tuner = SKAutoTuner(
    model=model,
    configs=TuningConfigs([config]),
    accuracy_eval_func=eval_quality,
    accuracy_threshold=thresh,  # Based on eval_quality
    optmization_eval_func=speed_eval_func,
    search_algorithm=OptunaSearch(n_trials=10)
)

# Search and apply optimal configuration
tuner.tune()
optimized_model = tuner.apply_best_params()
```

**Result:** Up to 75% memory reduction while maintaining model quality.

### 📊 Try It Yourself

- **Interactive Demo:** Run [tests/notebooks/demo_notebook.ipynb](tests/notebooks/demo_notebook.ipynb) with our Docker container
- **Benchmarks:** See detailed performance comparisons in the [documentation](docs/benchmarking/)

---

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Quick Start (Install & Use)](#quick-start-install--use)
  - [Using Docker](#using-docker)
- [Manual Setup (Optional)](#manual-setup-optional)
  - [Installing Dependencies](#installing-dependencies)
  - [Building the Native Backend (pawX)](#building-the-native-backend-pawx)
- [Running Panther](#running-panther)
- [Running Tests](#running-tests)
- [Generating Documentation (Optional)](#generating-documentation-optional)
- [Building a Python Wheel (Optional)](#building-a-python-wheel-optional)
- [Project Structure](#project-structure)
- [Pre-commit Hooks (Optional)](#pre-commit-hooks-optional)

---

## Getting Started

This guide explains how to build and run the Panther codebase, including the native backend (`pawX`), and how to generate a Python wheel for distribution.

---

## Prerequisites
- **Python 3.12+**: Panther is compatible with Python 3.12 and later.
- **Poetry** for dependency management
- **C++ Compiler**: GCC on Linux, MSVC on Windows
- **CUDA Toolkit** (Optional): For GPU acceleration, ensure you have the CUDA toolkit installed. **Panther fully supports CPU-only machines** - it will automatically build and run in CPU-only mode on systems without CUDA, providing the same functionality without GPU acceleration.

## Quick Start (Install & Use)

You can quickly install Panther using the powershell and Makefile scripts provided. This will set up the Python package and build the native backend.
Note: This sets up a venv environment, installs poetry, install dependencies, and builds the native backend.

**Note:** Panther works on both CPU-only and GPU-enabled machines. GPU users can use pip, while CPU-only users need to build from source using the installation scripts below.

If you have CUDA 12.4 installed and you're on a Windows machine with GPU, install using pip:

```bash
pip install --force-reinstall panther-ml==0.1.2 --extra-index-url https://download.pytorch.org/whl/cu124
```

### Using Docker

You can also use our pre-built Docker image with all dependencies included for GPU systems:

```bash
docker pull fahdseddik/panther-dev
```

### On Windows
1. **Open PowerShell** and run the following command:

    ```powershell
    .\install.ps1
    ```
### On Linux/macOS
1. **Open Terminal** and run the following command:

    ```bash
    make setup
    ```
---

## Manual Setup (Optional)
If you prefer to set up Panther manually, follow these steps:

### Installing Dependencies

To install Python dependencies, run:

```bash
poetry install
```

This will set up a virtual environment and install all required packages.

### Building the Native Backend (`pawX`)

#### On Linux

1. **Install Required System Libraries**

    ```sh
    sudo apt-get update
    sudo apt-get install liblapacke-dev libopenblas-dev
    ```

2. **Build and Install `pawX`**

    ```sh
    cd pawX
    make all
    cd ..
    ```

3. **Install Panther Package**

    ```sh
    pip install -e .
    ```

4. Confirm that `pawX.*.so` appears in the `pawX/` directory and panther is importable.

#### On Windows

1. **Build and Install `pawX`**

   ```powershell
    cd pawX
   .\build.ps1
   ```

2. Confirm that `pawX.*.pyd` appears in `pawX\` directory.

**Note:** The build process automatically detects whether CUDA is available on your system:
- If CUDA is available, it will build with GPU acceleration support
- If CUDA is not available, it will build a CPU-only version that works without CUDA installed
- The CPU-only build excludes CUDA-dependent features like `test_tensor_accessor` but maintains full functionality for core features

---

## Running Panther
To use panther in your python code, simply import the package:

```python
import torch
import panther as pr
# Example usage
A = torch.randn(1000, 1000)
Q, R, J = pr.linalg.cqrrpt(A)
print(Q.shape, R.shape, J.shape)
```

## Running Tests

Ensure your native backend is built, panther package is installed, and your Python environment is active. Then run:

```bash
pytest tests/
# or with poetry:
poetry run pytest tests/
```

This will execute unit tests and any Jupyter-based benchmarks.

---

## Generating Documentation (Optional)

Panther uses Sphinx for API docs located in `docs/`. To rebuild HTML docs:

```bash
cd docs
# On Windows:
.\make.bat clean
.\make.bat html
# On Linux/macOS:
make clean
make html
```

Open `docs/_build/html/index.html` in your browser.

---

## Building a Python Wheel (Optional)

Create a distributable wheel file:

```bash
poetry build
```

Find the resulting `.whl` under `dist/`.

---

## Project Structure

```
panther/          # Python package
├── linalg/       # Core linear algebra routines
├── nn/           # Neural network layers
├── sketch/       # Sketching algorithms
├── utils/        # AutoTuner & Helper functions
pawX/             # Native C++/CUDA backend
├── Makefile      # Linux build script
├── build.ps1     # Windows build script
tests/            # Unit tests, notebooks & benchmarks
docs/             # Sphinx documentation sources
```

---

## Pre-commit Hooks (Optional)

To enforce code style and formatting, install pre-commit hooks:

```bash
poetry run pre-commit install
```

---

## Acknowledgments

Panther's implementation of sparse sketching operators and the CQRRPT algorithm are derived from the [RandBLAS](https://github.com/BallisticLA/RandBLAS) and [RandLAPACK](https://github.com/BallisticLA/RandLAPACK) libraries, respectively. We gratefully acknowledge the RandBLAS and RandLAPACK teams for their foundational work in randomized numerical linear algebra. These libraries are distributed under the BSD-3-Clause license, and portions of our `pawX/` backend retain their copyright notices as required.

---

For more details, browse the source code and in-line documentation in each module.

