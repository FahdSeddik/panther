Distributed Training
====================

This guide covers distributed training strategies for Panther models across multiple GPUs and nodes.

.. note::
   Panther layers (SKLinear, SKConv2d, RandMultiHeadAttention) are compatible with PyTorch's 
   DistributedDataParallel (DDP). Use them as you would any other PyTorch layer with DDP.

Data Parallel Training with DDP
--------------------------------

Panther's sketched layers work seamlessly with PyTorch's DistributedDataParallel (DDP):

**Basic DDP Setup**

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.distributed as dist
   from torch.nn.parallel import DistributedDataParallel as DDP
   import panther as pr
   
   def setup_distributed(rank, world_size, backend='nccl'):
       """Initialize distributed training."""
       import os
       os.environ['MASTER_ADDR'] = 'localhost'
       os.environ['MASTER_PORT'] = '12355'
       dist.init_process_group(backend, rank=rank, world_size=world_size)
       torch.cuda.set_device(rank)
   
   def cleanup_distributed():
       """Clean up distributed training."""
       dist.destroy_process_group()
   
   # Create a model with sketched layers
   class SketchedModel(nn.Module):
       def __init__(self, input_size, hidden_size, output_size):
           super().__init__()
           self.layer1 = pr.nn.SKLinear(input_size, hidden_size, 
                                        num_terms=4, low_rank=32)
           self.relu = nn.ReLU()
           self.layer2 = pr.nn.SKLinear(hidden_size, output_size,
                                        num_terms=2, low_rank=16)
       
       def forward(self, x):
           return self.layer2(self.relu(self.layer1(x)))
   
   def train_distributed(rank, world_size):
       """Distributed training function."""
       setup_distributed(rank, world_size)
       
       # Create model and wrap with DDP
       model = SketchedModel(1024, 512, 10).cuda(rank)
       model = DDP(model, device_ids=[rank])
       
       # Create dataset and distributed sampler
       dataset = torch.utils.data.TensorDataset(
           torch.randn(1000, 1024),
           torch.randint(0, 10, (1000,))
       )
       
       sampler = torch.utils.data.distributed.DistributedSampler(
           dataset, num_replicas=world_size, rank=rank
       )
       
       dataloader = torch.utils.data.DataLoader(
           dataset, batch_size=32, sampler=sampler
       )
       
       optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
       criterion = nn.CrossEntropyLoss()
       
       # Training loop
       model.train()
       for epoch in range(5):
           sampler.set_epoch(epoch)  # Important for proper shuffling
           
           for batch_idx, (data, target) in enumerate(dataloader):
               data, target = data.cuda(rank), target.cuda(rank)
               
               optimizer.zero_grad()
               output = model(data)
               loss = criterion(output, target)
               loss.backward()
               optimizer.step()
               
               if rank == 0 and batch_idx % 10 == 0:
                   print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
       
       cleanup_distributed()
   
   # Launch with torch.multiprocessing
   if __name__ == "__main__":
       import torch.multiprocessing as mp
       world_size = torch.cuda.device_count()
       mp.spawn(train_distributed, args=(world_size,), nprocs=world_size, join=True)

Best Practices
--------------

**1. Sketching Matrix Synchronization**

The random sketching matrices (U1s, U2s) in SKLinear are registered as buffers and will automatically 
be broadcast to all processes during DDP initialization. No manual synchronization is needed.

**2. Mixed Precision with DDP**

.. code-block:: python

   from torch.cuda.amp import autocast, GradScaler
   
   def train_with_amp_ddp(rank, world_size):
       setup_distributed(rank, world_size)
       
       model = SketchedModel(1024, 512, 10).cuda(rank)
       model = DDP(model, device_ids=[rank])
       
       optimizer = torch.optim.Adam(model.parameters())
       scaler = GradScaler()
       
       for data, target in dataloader:
           data, target = data.cuda(rank), target.cuda(rank)
           
           optimizer.zero_grad()
           
           with autocast():
               output = model(data)
               loss = torch.nn.functional.cross_entropy(output, target)
           
           scaler.scale(loss).backward()
           scaler.step(optimizer)
           scaler.update()

**3. Gradient Clipping**

.. code-block:: python

   # Clip gradients before optimizer step
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   optimizer.step()

Troubleshooting
---------------

**Issue: Inconsistent Results Across Ranks**

If you see different results across different ranks, ensure that:

1. The random seed is set before model creation
2. All ranks use the same initialization

.. code-block:: python

   def train_distributed(rank, world_size):
       setup_distributed(rank, world_size)
       
       # Set seed for reproducibility
       torch.manual_seed(42)
       
       model = SketchedModel(1024, 512, 10).cuda(rank)
       model = DDP(model, device_ids=[rank])

**Issue: OOM on Some GPUs**

Ensure batch size and model size are balanced across all GPUs.

For more information on PyTorch distributed training, see the `official PyTorch DDP tutorial <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_.

Performance Considerations
--------------------------

**Monitoring GPU Utilization**

.. code-block:: python

   import torch
   
   def print_gpu_stats():
       if torch.cuda.is_available():
           for i in range(torch.cuda.device_count()):
               print(f"GPU {i}:")
               print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
               print(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")

See Also
--------

* :doc:`../tutorials/performance_optimization` - Performance optimization techniques
* :doc:`memory_optimization` - Memory optimization strategies
* `PyTorch DDP Documentation <https://pytorch.org/docs/stable/notes/ddp.html>`_
