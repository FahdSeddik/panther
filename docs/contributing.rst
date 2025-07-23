Contributing to Panther
=======================

We welcome contributions to Panther! This guide will help you get started with contributing to the project.

Getting Started
---------------

**1. Fork and Clone the Repository**

.. code-block:: bash

   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/panther.git
   cd panther
   
   # Add the original repository as upstream
   git remote add upstream https://github.com/FahdSeddik/panther.git

**2. Set Up Development Environment**

.. code-block:: bash

   # Install development dependencies
   poetry install --with dev
   
   # Install pre-commit hooks
   poetry run pre-commit install
   
   # Build the native backend
   cd pawX
   make all  # Linux/macOS
   # or
   .\build.ps1  # Windows

**3. Verify Installation**

.. code-block:: bash

   # Run tests to ensure everything works
   poetry run pytest tests/

Development Workflow
--------------------

**1. Create a Feature Branch**

.. code-block:: bash

   git checkout -b feature/your-feature-name

**2. Make Your Changes**

* Follow the coding standards (see below)
* Add tests for new functionality
* Update documentation as needed
* Ensure all tests pass

**3. Run Tests and Checks**

.. code-block:: bash

   # Run all tests
   poetry run pytest tests/
   
   # Run type checking
   poetry run mypy panther/
   
   # Run linting
   poetry run ruff check panther/
   
   # Run formatting
   poetry run ruff format panther/

**4. Commit and Push**

.. code-block:: bash

   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name

**5. Create a Pull Request**

* Go to GitHub and create a pull request
* Describe your changes clearly
* Link any related issues
* Wait for review and address feedback

Coding Standards
----------------

**Python Code Style**

We use Ruff for linting and formatting:

.. code-block:: python

   # Good: Clear variable names and docstrings
   def compute_sketched_linear(
       input_tensor: torch.Tensor,
       sketch_matrices: List[torch.Tensor],
       bias: Optional[torch.Tensor] = None
   ) -> torch.Tensor:
       """
       Compute sketched linear transformation.
       
       Args:
           input_tensor: Input tensor of shape (batch_size, in_features)
           sketch_matrices: List of sketching matrices
           bias: Optional bias tensor
           
       Returns:
           Output tensor of shape (batch_size, out_features)
       """
       # Implementation here
       pass

**Type Hints**

All functions should include type hints:

.. code-block:: python

   from typing import Optional, Tuple, List
   import torch
   
   def cqrrpt(
       matrix: torch.Tensor,
       gamma: float = 1.25,
       distribution: DistributionFamily = DistributionFamily.Gaussian
   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
       """Randomized QR with column pivoting."""
       # Implementation
       pass

**Documentation**

All public functions and classes must have docstrings:

.. code-block:: python

   class SKLinear(nn.Module):
       """
       Sketched linear layer using randomized low-rank approximation.
       
       This layer approximates a standard linear transformation using a sum
       of low-rank terms, reducing memory usage while maintaining performance.
       
       Args:
           in_features: Number of input features
           out_features: Number of output features
           num_terms: Number of low-rank terms in the approximation
           low_rank: Rank of each low-rank term
           bias: If True, adds a learnable bias
           
       Example:
           >>> layer = SKLinear(512, 256, num_terms=8, low_rank=64)
           >>> x = torch.randn(32, 512)
           >>> y = layer(x)  # Shape: (32, 256)
       """

**C++/CUDA Code Style**

For C++ and CUDA code:

.. code-block:: cpp

   // Good: Clear function documentation and naming
   /**
    * @brief Compute sketched linear forward pass on GPU.
    * 
    * @param input Input tensor with shape [batch_size, in_features]
    * @param S1s First sketch matrices [num_terms, in_features, low_rank]
    * @param S2s Second sketch matrices [num_terms, low_rank, out_features]
    * @param U1s Fixed random matrices [num_terms, low_rank, out_features]
    * @param U2s Fixed random matrices [num_terms, low_rank, in_features]
    * @param bias Optional bias tensor [out_features]
    * @return Output tensor [batch_size, out_features]
    */
   torch::Tensor sketched_linear_forward_cuda(
       const torch::Tensor& input,
       const torch::Tensor& S1s,
       const torch::Tensor& S2s,
       const torch::Tensor& U1s,
       const torch::Tensor& U2s,
       const torch::Tensor& bias
   );

Testing Guidelines
------------------

**Test Structure**

Organize tests by functionality:

.. code-block:: text

   tests/
   ├── test_linalg.py          # Linear algebra functions
   ├── test_nn.py              # Neural network layers
   ├── test_sketch.py          # Sketching operations
   ├── test_cuda.py            # CUDA kernel tests
   └── test_integration.py     # End-to-end tests

**Writing Tests**

.. code-block:: python

   import pytest
   import torch
   import panther as pr
   
   class TestSKLinear:
       """Test suite for SKLinear layer."""
       
       def test_forward_pass_shape(self):
           """Test that forward pass produces correct output shape."""
           layer = pr.nn.SKLinear(
               in_features=128,
               out_features=64,
               num_terms=4,
               low_rank=32
           )
           
           x = torch.randn(16, 128)
           y = layer(x)
           
           assert y.shape == (16, 64), f"Expected (16, 64), got {y.shape}"
       
       def test_backward_pass(self):
           """Test that backward pass computes gradients correctly."""
           layer = pr.nn.SKLinear(128, 64, num_terms=4, low_rank=32)
           x = torch.randn(16, 128, requires_grad=True)
           
           y = layer(x)
           loss = y.sum()
           loss.backward()
           
           # Check that gradients were computed
           assert x.grad is not None
           assert layer.S1s.grad is not None
           assert layer.S2s.grad is not None
       
       @pytest.mark.parametrize("device", ["cpu", "cuda"])
       def test_device_compatibility(self, device):
           """Test layer works on different devices."""
           if device == "cuda" and not torch.cuda.is_available():
               pytest.skip("CUDA not available")
           
           layer = pr.nn.SKLinear(64, 32, num_terms=2, low_rank=16)
           layer = layer.to(device)
           
           x = torch.randn(8, 64, device=device)
           y = layer(x)
           
           assert y.device.type == device

**Performance Tests**

.. code-block:: python

   import time
   import pytest
   
   def test_sketched_layer_performance():
       """Test that sketched layers provide memory benefits."""
       # Create layers
       standard = torch.nn.Linear(2048, 2048)
       sketched = pr.nn.SKLinear(2048, 2048, num_terms=8, low_rank=128)
       
       # Compare parameter counts
       standard_params = sum(p.numel() for p in standard.parameters())
       sketched_params = sum(p.numel() for p in sketched.parameters())
       
       assert sketched_params < standard_params, "Sketched layer should use fewer parameters"
       
       # Compare memory usage
       x = torch.randn(64, 2048)
       
       # Time forward pass
       start = time.time()
       for _ in range(100):
           _ = standard(x)
       standard_time = time.time() - start
       
       start = time.time()
       for _ in range(100):
           _ = sketched(x)
       sketched_time = time.time() - start
       
       # Sketched should be competitive (within 2x)
       assert sketched_time < 2 * standard_time

Documentation Guidelines
------------------------

**Building Documentation**

.. code-block:: bash

   cd docs
   
   # Clean previous build
   make clean  # Linux/macOS
   .\make.bat clean  # Windows
   
   # Build HTML documentation
   make html  # Linux/macOS
   .\make.bat html  # Windows
   
   # Open in browser
   open _build/html/index.html  # macOS
   xdg-open _build/html/index.html  # Linux
   start _build/html/index.html  # Windows

**Writing Documentation**

Use reStructuredText format with clear examples:

.. code-block:: rst

   Function Name
   =============
   
   Brief description of what the function does.
   
   Parameters
   ----------
   param1 : type
       Description of parameter 1
   param2 : type, optional
       Description of parameter 2 (default: value)
   
   Returns
   -------
   return_type
       Description of return value
   
   Examples
   --------
   >>> import panther as pr
   >>> result = pr.function_name(param1, param2)
   >>> print(result.shape)

Areas for Contribution
----------------------

**1. Core Algorithms**

* New sketching methods
* Improved randomized algorithms
* Memory optimization techniques

**2. Neural Network Layers**

* Sketched versions of more PyTorch layers
* Custom activation functions
* Attention mechanisms

**3. CUDA Kernels**

* Performance optimizations
* Support for new GPU architectures
* Memory-efficient implementations

**4. AutoTuner Improvements**

* Better hyperparameter search strategies
* Multi-objective optimization
* Integration with popular ML frameworks

**5. Documentation and Examples**

* Tutorial improvements
* Real-world examples
* Performance benchmarks

**6. Testing and CI**

* Expanded test coverage
* Performance regression tests
* Cross-platform compatibility

Submitting Issues
-----------------

When submitting bug reports or feature requests:

**Bug Reports**

Include:

* Clear description of the problem
* Minimal code to reproduce the issue
* Error messages and stack traces
* Environment information (OS, Python version, CUDA version)

.. code-block:: python

   # Example bug report template
   import torch
   import panther as pr
   
   # Environment
   print(f"Python: {sys.version}")
   print(f"PyTorch: {torch.__version__}")
   print(f"CUDA: {torch.version.cuda}")
   print(f"Panther: {pr.__version__}")
   
   # Minimal reproduction case
   layer = pr.nn.SKLinear(512, 256, num_terms=8, low_rank=64)
   x = torch.randn(32, 512)
   
   # This causes the error:
   y = layer(x)  # Error occurs here

**Feature Requests**

Include:

* Clear description of the desired feature
* Use cases and motivation
* Proposed API design (if applicable)
* Willingness to implement

Release Process
---------------

For maintainers:

**1. Version Bumping**

.. code-block:: bash

   # Update version in pyproject.toml
   poetry version patch  # or minor, major
   
   # Update version in __init__.py
   # Update CHANGELOG.md

**2. Testing**

.. code-block:: bash

   # Run full test suite
   poetry run pytest tests/
   
   # Run tests on different Python versions
   tox

**3. Building**

.. code-block:: bash

   # Build Python package
   poetry build
   
   # Build documentation
   cd docs && make html

**4. Release**

.. code-block:: bash

   # Tag release
   git tag v0.1.3
   git push origin v0.1.3
   
   # Publish to PyPI
   poetry publish

Recognition
-----------

Contributors will be recognized in:

* CONTRIBUTORS.md file
* Release notes
* Documentation acknowledgments

Thank you for contributing to Panther!
