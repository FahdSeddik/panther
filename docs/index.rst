.. Panther documentation master file

.. raw:: html

   <script>
      /* Mark this as the main page and hide sidebar by default on desktop */
      document.body.classList.add('main-index-page');
      
      // Only auto-hide sidebar on desktop screens
      //if (window.innerWidth >= 960) {
         // Wait for DOM to be ready
      //   document.addEventListener('DOMContentLoaded', function() {
      //      const primarySidebar = document.getElementById("__primary");
      //      if (primarySidebar) {
      //         primarySidebar.checked = false;
      //      }
      //   });
         
         // If DOM is already loaded
      //   if (document.readyState === 'loading') {
            // Document still loading, wait for DOMContentLoaded
      //   } else {
            // Document already loaded
      //    const primarySidebar = document.getElementById("__primary");
      //      if (primarySidebar) {
      //         primarySidebar.checked = false;
      //      }
      //   }
      //}
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
   benchmarking/index

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
