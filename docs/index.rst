.. Panther documentation master file

Panther: Faster & Cheaper Computations with RandNLA
===================================================

.. raw:: html

   <script>
      /* Along with some CSS settings in style.css (look for `body:has(.hero)`)
         this will ensure that the menu sidebar is hidden on the main page. */
      if (window.innerWidth >= 960) {
         document.getElementById("__primary").checked = true;
      }
   </script>

.. raw:: html
   :file: hero.html

.. grid:: 3
   :class-container: product-offerings
   :margin: 0
   :padding: 0
   :gutter: 0

   .. grid-item-card:: Sketched Neural Networks
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      Replace standard linear and convolutional layers with memory-efficient sketched alternatives that maintain accuracy while reducing computational cost.

   .. grid-item-card:: Randomized Algorithms
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      Fast QR and SVD decompositions using randomized sketching, enabling efficient matrix operations on large-scale problems with theoretical guarantees.

   .. grid-item-card:: GPU Acceleration
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      Optimized CUDA kernels with Tensor Core support for maximum performance on modern GPUs, seamlessly integrated with PyTorch workflows.

.. grid:: 3
    :class-container: color-cards

    .. grid-item-card:: :material-regular:`download;2em` Installation
      :columns: 12 6 6 4
      :link: installation
      :link-type: doc
      :class-card: installation

      Get started with Panther installation instructions for CPU and GPU environments.

    .. grid-item-card:: :material-regular:`rocket_launch;2em` Quick Start
      :columns: 12 6 6 4
      :link: quickstart
      :link-type: doc
      :class-card: getting-started

      Learn the basics with our quick start guide and first examples.

    .. grid-item-card:: :material-regular:`library_books;2em` Tutorials
      :columns: 12 6 6 4
      :link: tutorials/index
      :link-type: doc
      :class-card: user-guides

      Comprehensive tutorials covering all aspects of Panther from basics to advanced topics.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:
   
   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:
   
   api/linalg
   api/nn
   api/sketch
   api/tuner
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Examples & Tutorials
   :hidden:
   
   examples/index
   examples/basic_usage
   examples/resnet_sketching
   examples/autotuner_guide
   examples/custom_sketching
   examples/performance_benchmarks

.. toctree::
   :maxdepth: 1
   :caption: Advanced Topics
   :hidden:
   
   advanced/cuda_kernels
   advanced/tensor_cores
   advanced/custom_sketching
   advanced/memory_optimization
   advanced/distributed_training

.. toctree::
   :maxdepth: 1
   :caption: Community
   :hidden:
   
   contributing
   changelog
   roadmap
   faq


Quick Summary
================== 

Panther is a PyTorch library that leverages randomized numerical linear algebra (RandNLA) techniques to create sketched neural network layers, enabling significant memory savings and performance improvements for large-scale machine learning models. It provides efficient implementations of sketched linear layers, matrix decompositions, and GPU-accelerated operations, making it ideal for resource-constrained environments.

🛠️ **Key Features**
-------------------

- **Sketched Linear Layers**: Memory-efficient alternatives to standard linear layers  
- **Randomized Matrix Decompositions**: Fast QR and SVD algorithms using sketching  
- **Neural Network Operations**: Optimized convolution and attention mechanisms  
- **GPU Acceleration**: CUDA kernels with Tensor Core support  
- **AutoTuner**: Automatic hyperparameter optimization for sketching parameters  

🎯 **Why Panther?**
-------------------

Panther enables you to:

- **Reduce Memory Usage**: Sketched layers use significantly less memory than standard layers  
- **Accelerate Training**: Faster forward and backward passes with optimized kernels  
- **Scale to Larger Models**: Handle bigger networks with limited GPU memory  
- **Maintain Accuracy**: Randomized algorithms with theoretical guarantees

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. raw:: html

   <style>
   .product-offerings {
       margin: 3rem 0;
   }
   
   .product-offerings .sd-card {
       text-align: center;
       padding: 2rem 1rem;
       border: none !important;
       box-shadow: none !important;
   }
   
   .product-offerings .sd-card-title {
       font-size: 1.25rem;
       font-weight: 600;
       color: #374151;
       margin-bottom: 1rem;
   }
   
   .product-offerings .sd-card-text {
       color: #6b7280;
       line-height: 1.6;
   }
   
   .color-cards {
       margin: 2rem 0;
       gap: 2rem;
   }
   
   .color-cards .sd-card {
       text-align: center;
       padding: 2rem 1rem;
       border-radius: 1rem;
       transition: transform 0.3s ease, box-shadow 0.3s ease;
       text-decoration: none;
       color: white;
       background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
   }
   
   .color-cards .sd-card:hover {
       transform: translateY(-5px);
       box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
       text-decoration: none;
       color: white;
   }
   
   .color-cards .installation {
       background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
   }
   
   .color-cards .getting-started {
       background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
   }
   
   .color-cards .user-guides {
       background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
   }
   
   .color-cards .sd-card-title {
       font-size: 1.1rem;
       font-weight: 600;
       margin-bottom: 0.5rem;
       color: white;
   }
   
   .color-cards .sd-card-text {
       font-size: 0.9rem;
       opacity: 0.9;
       color: white;
   }
   
   .color-cards .material-icons {
       font-size: 2em;
       margin-bottom: 1rem;
       display: block;
   }
   </style>
