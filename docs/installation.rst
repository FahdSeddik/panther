Installation Guide
==================

This guide explains how to install Panther and its dependencies.

System Requirements
-------------------

* **Python**: 3.12
* **PyTorch**: 2.6.0 or later
* **OS**: Linux x86_64, Windows x64, macOS Apple Silicon (arm64)
* **GPU**: NVIDIA GPU with CUDA 11.8 or 12.4 (optional — CPU-only is fully supported)

.. note::
   PyTorch must be installed **before** panther-ml. Choose the variant that matches your
   hardware (see Step 1 below). panther-ml itself declares ``torch>=2.6.0`` so it works
   with any compatible PyTorch flavor you install first.

Step 1 — Install PyTorch
------------------------

Choose the variant that matches your hardware:

.. code-block:: bash

   # CPU only (all platforms)
   pip install torch==2.6.0 torchvision==0.21.0 \
       --extra-index-url https://download.pytorch.org/whl/cpu

   # CUDA 11.8 (Linux / Windows)
   pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 \
       --extra-index-url https://download.pytorch.org/whl/cu118

   # CUDA 12.4 (Linux / Windows)
   pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
       --extra-index-url https://download.pytorch.org/whl/cu124

Step 2 — Install panther-ml
----------------------------

**CPU wheel (PyPI — Linux, Windows, macOS)**

.. code-block:: bash

   pip install panther-ml

This installs the CPU wheel for your platform directly from PyPI.

**CUDA wheels (GitHub Releases)**

CUDA variants are attached to each `GitHub Release <https://github.com/FahdSeddik/panther/releases>`_
as direct-download assets. Install the wheel for your platform and CUDA version:

.. code-block:: bash

   # Linux x86_64, CUDA 12.4
   pip install https://github.com/FahdSeddik/panther/releases/download/v0.1.3/panther_ml-0.1.3+cu124-cp312-cp312-manylinux_2_28_x86_64.whl

   # Linux x86_64, CUDA 11.8
   pip install https://github.com/FahdSeddik/panther/releases/download/v0.1.3/panther_ml-0.1.3+cu118-cp312-cp312-manylinux_2_28_x86_64.whl

   # Windows x64, CUDA 12.4
   pip install https://github.com/FahdSeddik/panther/releases/download/v0.1.3/panther_ml-0.1.3+cu124-cp312-cp312-win_amd64.whl

   # macOS Apple Silicon (arm64), CPU
   pip install https://github.com/FahdSeddik/panther/releases/download/v0.1.3/panther_ml-0.1.3+cpu-cp312-cp312-macosx_12_0_arm64.whl

**Platform matrix**

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 20

   * - Platform
     - CPU
     - CUDA 11.8
     - CUDA 12.4
   * - Linux x86_64
     - PyPI
     - GitHub Release
     - GitHub Release
   * - Windows x64
     - PyPI
     - —
     - GitHub Release
   * - macOS arm64
     - PyPI
     - —
     - —

Docker (Optional)
-----------------

A pre-built Docker image with all dependencies is available for GPU systems:

.. code-block:: bash

   docker pull fahdseddik/panther-dev

Verify Installation
-------------------

.. code-block:: python

   import torch
   import panther as pr

   print(f"CUDA available: {torch.cuda.is_available()}")

   A = torch.randn(100, 80)
   Q, R, J = pr.linalg.cqrrpt(A)
   print("Installation successful!")
   print(f"Q: {Q.shape}, R: {R.shape}, J: {J.shape}")

Building from Source (Contributors)
------------------------------------

Use this path if you are contributing to panther-ml or need to modify the native backend.

**Prerequisites**: Poetry, a C++ compiler (GCC / MSVC / Clang), and optionally the CUDA Toolkit.

**Linux**

.. code-block:: bash

   sudo apt-get update && sudo apt-get install -y libopenblas-dev liblapacke-dev

   git clone https://github.com/FahdSeddik/panther.git && cd panther
   poetry install

   # Build native extension in-place (required for relative import)
   cd pawX && python setup.py build_ext --inplace && cd ..

   pip install -e .

**Windows**

.. code-block:: powershell

   git clone https://github.com/FahdSeddik/panther.git; cd panther
   poetry install

   cd pawX; python setup.py build_ext --inplace; cd ..

   pip install -e .

**macOS**

.. code-block:: bash

   brew install openblas

   git clone https://github.com/FahdSeddik/panther.git && cd panther
   poetry install

   cd pawX && python setup.py build_ext --inplace && cd ..

   pip install -e .

After building, verify the extension:

.. code-block:: bash

   python -c "import torch; import pawX; t = pawX.scaled_sign_sketch(4, 4); print('OK:', t.shape)"

Common Issues
-------------

**"No module named 'panther'"**

Make sure the package is installed (editable mode for source builds):

.. code-block:: bash

   pip install -e .

**OpenBLAS not found (Linux)**

.. code-block:: bash

   sudo apt-get install libopenblas-dev liblapacke-dev

**CUDA not found**

.. code-block:: bash

   nvcc --version   # Check CUDA Toolkit is installed and on PATH

**Compilation errors on Windows**

Ensure Visual Studio Build Tools with C++ workload are installed.

Uninstallation
--------------

.. code-block:: bash

   pip uninstall panther-ml
