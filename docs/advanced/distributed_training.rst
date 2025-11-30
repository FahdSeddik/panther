Distributed Training Compatibility
===================================

This guide covers compatibility considerations for using Panther models with PyTorch's distributed training.

.. note::
   Panther layers (SKLinear, SKConv2d, RandMultiHeadAttention) are compatible with PyTorch's 
   DistributedDataParallel (DDP). Use them as you would any other PyTorch layer with DDP.

PyTorch DDP Compatibility
--------------------------

Panther's sketched layers work seamlessly with PyTorch's DistributedDataParallel:

**Sketching Matrix Synchronization**

The random sketching matrices (U1s, U2s, S1s, S2s) in SKLinear are registered as buffers and will automatically 
be broadcast to all processes during DDP initialization. No manual synchronization is needed.

**Using Panther with DDP**

Simply wrap your Panther-based model with DDP as you would with any PyTorch model:

.. code-block:: python

   import torch
   from torch.nn.parallel import DistributedDataParallel as DDP
   import panther as pr
   
   # Create model with sketched layers
   model = YourPantherModel().to(device)
   
   # Wrap with DDP
   model = DDP(model, device_ids=[local_rank])
   
   # Train as normal - sketching parameters sync automatically

**Important Notes**

1. **Buffer Synchronization**: Panther's sketching matrices are PyTorch buffers and sync automatically across processes
2. **Reproducibility**: Set the same random seed on all processes before model creation for consistent initialization
3. **Memory Benefits**: The memory savings from sketched layers apply per-GPU, multiplying benefits across all devices

Troubleshooting
---------------

**Issue: Inconsistent Results Across Ranks**

Ensure random seed is set identically on all ranks before model creation:

.. code-block:: python

   import torch
   
   def setup_model(rank):
       # Set seed for reproducibility
       torch.manual_seed(42)
       if torch.cuda.is_available():
           torch.cuda.manual_seed(42)
       
       # Now create model
       model = YourPantherModel()
       return model

**Issue: OOM on Some GPUs**

Panther's sketched layers help reduce memory, but ensure batch sizes are balanced across GPUs.

See Also
--------

* :doc:`../tutorials/performance_optimization` - Performance optimization techniques
* :doc:`memory_optimization` - Memory optimization strategies  
* `PyTorch DDP Documentation <https://pytorch.org/docs/stable/notes/ddp.html>`_
