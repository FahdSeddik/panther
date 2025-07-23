Installation Guide
==================

This guide explains how to install Panther and its dependencies.

System Requirements
-------------------

**Python**: Panther requires Python 3.12 or later.

**Hardware Requirements**:

* **CPU**: Any modern x86-64 processor
* **GPU**: NVIDIA GPU with CUDA 12.4+ (optional, for GPU acceleration)
* **Memory**: Minimum 8GB RAM recommended

**Dependencies**:

* PyTorch 2.6.0+ with CUDA support
* CUDA Toolkit 12.4+ (for GPU features)
* C++ compiler (GCC on Linux, MSVC on Windows)

Quick Installation
------------------

**Option 1: Install from PyPI (Recommended)**

If you have CUDA 12.4 installed and you're on a Windows machine:

.. code-block:: bash

   pip install --force-reinstall panther-ml==0.1.2 --extra-index-url https://download.pytorch.org/whl/cu124

**Option 2: Automated Setup Scripts**

For a complete setup with all dependencies:

**Windows:**

.. code-block:: powershell

   # Open PowerShell as Administrator
   .\install.ps1

**Linux/macOS:**

.. code-block:: bash

   # Make sure you have build-essential installed
   make install

Manual Installation
-------------------

If you prefer to install Panther manually or need to customize the installation:

**Step 1: Install Poetry (if not already installed)**

.. code-block:: bash

   curl -sSL https://install.python-poetry.org | python3 -

**Step 2: Clone the Repository**

.. code-block:: bash

   git clone https://github.com/FahdSeddik/panther.git
   cd panther

**Step 3: Install Python Dependencies**

.. code-block:: bash

   poetry install

**Step 4: Build the Native Backend**

**On Linux:**

First, install required system libraries:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install liblapacke-dev

Then build the native backend:

.. code-block:: bash

   cd pawX
   make all

**On Windows:**

.. code-block:: powershell

   cd pawX
   .\build.ps1

**Step 5: Verify Installation**

.. code-block:: python

   import torch
   import panther as pr
   
   # Test basic functionality
   A = torch.randn(100, 100)
   Q, R, J = pr.linalg.cqrrpt(A)
   print("Installation successful!")
   print(f"Q shape: {Q.shape}, R shape: {R.shape}, J shape: {J.shape}")

Environment Setup
-----------------

**Virtual Environment (Recommended)**

It's recommended to use a virtual environment:

.. code-block:: bash

   python -m venv panther_env
   
   # On Windows:
   panther_env\Scripts\activate
   
   # On Linux/macOS:
   source panther_env/bin/activate

**CUDA Environment Variables**

For optimal GPU performance, ensure these environment variables are set:

.. code-block:: bash

   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

Development Installation
------------------------

For development work on Panther:

.. code-block:: bash

   git clone https://github.com/FahdSeddik/panther.git
   cd panther
   poetry install --with dev
   
   # Install pre-commit hooks
   poetry run pre-commit install

Common Issues
-------------

**Issue: CUDA not found**

Make sure CUDA Toolkit is properly installed and in your PATH:

.. code-block:: bash

   nvcc --version

**Issue: Compilation errors on Windows**

Ensure you have Visual Studio Build Tools installed with C++ support.

**Issue: Import errors**

Check that the native backend was compiled successfully:

.. code-block:: bash

   ls pawX/pawX.*.pyd  # Windows
   ls pawX/pawX.*.so   # Linux

**Issue: OpenBLAS not found**

The installation scripts should handle OpenBLAS automatically. If you encounter issues, OpenBLAS binaries are bundled with the package.

Verifying GPU Support
----------------------

To check if GPU acceleration is available:

.. code-block:: python

   import torch
   import panther as pr
   
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   
   if torch.cuda.is_available():
       # Test GPU functionality
       device = torch.device('cuda')
       A = torch.randn(1000, 1000, device=device)
       Q, R, J = pr.linalg.cqrrpt(A)
       print("GPU acceleration is working!")

Uninstallation
--------------

To uninstall Panther:

.. code-block:: bash

   pip uninstall panther-ml

If installed from source, simply delete the cloned directory after deactivating any virtual environment.
