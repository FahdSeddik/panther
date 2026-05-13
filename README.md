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

- [Installation](#installation)
  - [Step 1 — Install PyTorch](#step-1--install-pytorch)
  - [Step 2 — Install panther-ml](#step-2--install-panther-ml)
- [Using Docker](#using-docker)
- [Running Panther](#running-panther)
- [Running Tests](#running-tests)
- [Building from Source (Contributors)](#building-from-source-contributors)
  - [Prerequisites](#prerequisites)
  - [Linux](#linux)
  - [Windows](#windows)
  - [macOS](#macos)
- [Generating Documentation (Optional)](#generating-documentation-optional)
- [Project Structure](#project-structure)
- [Pre-commit Hooks (Optional)](#pre-commit-hooks-optional)
- [Acknowledgements](#acknowledgments)
- [Citation](#-citation)

---

## Installation

Panther ships pre-built wheels for **Linux x86_64**, **Windows x64**, and **macOS Apple Silicon (arm64)** targeting Python 3.12. CPU and CUDA variants are available.

### Step 1 — Install PyTorch

Install PyTorch **before** panther-ml. Choose the variant that matches your hardware:

```bash
# CPU only (all platforms)
pip install torch==2.6.0 torchvision==0.21.0 --extra-index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8 (Linux / Windows)
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# CUDA 12.4 (Linux / Windows)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
```

### Step 2 — Install panther-ml

**CPU wheel (PyPI — Linux, Windows, macOS):**

```bash
pip install panther-ml
```

**CUDA wheels** are attached to each [GitHub Release](https://github.com/FahdSeddik/panther/releases) as direct-download assets. Install the wheel that matches your platform and CUDA version:

```bash
# Example: Linux x86_64, CUDA 12.4
pip install https://github.com/FahdSeddik/panther/releases/download/v0.1.3/panther_ml-0.1.3+cu124-cp312-cp312-manylinux_2_28_x86_64.whl

# Example: Linux x86_64, CUDA 11.8
pip install https://github.com/FahdSeddik/panther/releases/download/v0.1.3/panther_ml-0.1.3+cu118-cp312-cp312-manylinux_2_28_x86_64.whl

# Example: Windows x64, CUDA 12.4
pip install https://github.com/FahdSeddik/panther/releases/download/v0.1.3/panther_ml-0.1.3+cu124-cp312-cp312-win_amd64.whl

# Example: macOS Apple Silicon (arm64), CPU
pip install https://github.com/FahdSeddik/panther/releases/download/v0.1.3/panther_ml-0.1.3+cpu-cp312-cp312-macosx_12_0_arm64.whl
```

> **Platform matrix**
>
> | Platform | CPU | CUDA 11.8 | CUDA 12.4 |
> |----------|:---:|:---------:|:---------:|
> | Linux x86_64 | ✅ PyPI | ✅ Release | ✅ Release |
> | Windows x64 | ✅ PyPI | — | ✅ Release |
> | macOS arm64 | ✅ PyPI | — | — |

---

## Using Docker

A pre-built Docker image with all dependencies is available for GPU systems:

```bash
docker pull fahdseddik/panther-dev
```

---

## Building from Source (Contributors)

Use this path if you are contributing to panther-ml or need to modify the native backend.

### Prerequisites

- Python 3.12+
- Poetry (`pip install poetry`)
- C++ compiler (GCC on Linux, MSVC on Windows, Clang on macOS)
- CUDA Toolkit (optional — CPU-only build works without it)

### Linux

```bash
# System libraries
sudo apt-get update && sudo apt-get install -y libopenblas-dev liblapacke-dev

# Clone and install Python deps
git clone https://github.com/FahdSeddik/panther.git && cd panther
poetry install

# Build the native extension in-place
cd pawX && python setup.py build_ext --inplace && cd ..

# Install the panther package in editable mode
pip install -e .
```

### Windows

```powershell
# Clone and install Python deps
git clone https://github.com/FahdSeddik/panther.git; cd panther
poetry install

# Build the native extension in-place (bundled OpenBLAS is used automatically)
cd pawX; python setup.py build_ext --inplace; cd ..

pip install -e .
```

### macOS

```bash
# Homebrew OpenBLAS
brew install openblas

# Clone and install Python deps
git clone https://github.com/FahdSeddik/panther.git && cd panther
poetry install

# Build the native extension in-place
cd pawX && python setup.py build_ext --inplace && cd ..

pip install -e .
```

After building, verify the extension loaded correctly:

```bash
python -c "import torch; import pawX; t = pawX.scaled_sign_sketch(4, 4); print('OK:', t.shape)"
```

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

## Project Structure

```
panther/          # Python package
├── linalg/       # Core linear algebra routines
├── nn/           # Neural network layers
├── sketch/       # Sketching algorithms
├── utils/        # AutoTuner & Helper functions
pawX/             # Native C++/CUDA backend
├── setup.py      # Extension build script
scripts/          # Release utilities (wheel renaming)
tests/            # Unit tests, notebooks & benchmarks
docs/             # Sphinx documentation sources
.github/workflows/build-wheels.yml  # CI: multi-platform wheel build & publish
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


## 📜 Citation

If you use this code, please cite our paper:
```bibtex
@misc{seddik2026pantherfastercheapercomputations,
      title={Panther: Faster and Cheaper Computations with Randomized Numerical Linear Algebra}, 
      author={Fahd Seddik and Abdulrahman Elbedewy and Gaser Sami and Mohamed Abdelmoniem and Yahia Zakaria},
      year={2026},
      eprint={2601.15473},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.15473}, 
}
```

---

For more details, browse the source code and in-line documentation in each module.


