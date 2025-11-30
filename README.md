# Panther: Faster & Cheaper Computations with RandNLA

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

### On Windows
1. **Open PowerShell** and run the following command:

    ```powershell
    .\install.ps1
    ```
### On Linux/macOS
1. **Open Terminal** and run the following command:

    ```bash
    make install
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
    sudo apt-get install liblapacke-dev
    ```

2. **Build and Install `pawX`**

    ```sh
    cd pawX
    make all
    ```

3. Confirm that `pawX.*.so` appears in the `pawX/` directory.

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

Ensure your native backend is built and your Python environment is active. Then run:

```bash
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

For more details, browse the source code and in-line documentation in each module.
