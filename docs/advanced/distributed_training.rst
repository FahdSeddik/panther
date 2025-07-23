Distributed Training
====================

This guide covers distributed training strategies for Panther models across multiple GPUs and nodes.

Data Parallel Training
----------------------

**Basic Data Parallel Setup**

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.distributed as dist
   import torch.multiprocessing as mp
   from torch.nn.parallel import DistributedDataParallel as DDP
   import panther as pr
   
   def setup_distributed(rank, world_size, backend='nccl'):
       """Initialize distributed training."""
       import os
       
       # Set up environment variables
       os.environ['MASTER_ADDR'] = 'localhost'
       os.environ['MASTER_PORT'] = '12355'
       
       # Initialize process group
       dist.init_process_group(backend, rank=rank, world_size=world_size)
       
       # Set device for this process
       torch.cuda.set_device(rank)
   
   def cleanup_distributed():
       """Clean up distributed training."""
       dist.destroy_process_group()
   
   class DistributedSketchedModel(nn.Module):
       """Sketched model optimized for distributed training."""
       
       def __init__(self, input_size, hidden_sizes, output_size, sketch_ratios):
           super().__init__()
           
           layers = []
           prev_size = input_size
           
           for i, (hidden_size, sketch_ratio) in enumerate(zip(hidden_sizes, sketch_ratios)):
               sketch_size = int(prev_size * sketch_ratio)
               layers.append(pr.nn.SKLinear(prev_size, hidden_size, sketch_size))
               layers.append(nn.ReLU())
               prev_size = hidden_size
           
           # Output layer
           final_sketch_size = int(prev_size * sketch_ratios[-1])
           layers.append(pr.nn.SKLinear(prev_size, output_size, final_sketch_size))
           
           self.model = nn.Sequential(*layers)
           
       def forward(self, x):
           return self.model(x)
   
   def train_distributed(rank, world_size):
       """Distributed training function."""
       
       setup_distributed(rank, world_size)
       
       # Create model
       model = DistributedSketchedModel(
           input_size=1024,
           hidden_sizes=[512, 256, 128],
           output_size=10,
           sketch_ratios=[0.5, 0.6, 0.7, 0.8]
       ).cuda(rank)
       
       # Wrap with DDP
       model = DDP(model, device_ids=[rank])
       
       # Create synthetic dataset
       dataset = torch.utils.data.TensorDataset(
           torch.randn(1000, 1024),
           torch.randint(0, 10, (1000,))
       )
       
       # Distributed sampler
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
           
           total_loss = 0
           for batch_idx, (data, target) in enumerate(dataloader):
               data, target = data.cuda(rank), target.cuda(rank)
               
               optimizer.zero_grad()
               output = model(data)
               loss = criterion(output, target)
               loss.backward()
               optimizer.step()
               
               total_loss += loss.item()
               
               if rank == 0 and batch_idx % 10 == 0:
                   print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
           
           if rank == 0:
               avg_loss = total_loss / len(dataloader)
               print(f"Epoch {epoch} completed, Average Loss: {avg_loss:.4f}")
       
       cleanup_distributed()
   
   # Launch distributed training
   def main():
       world_size = torch.cuda.device_count()
       if world_size < 2:
           print("Need at least 2 GPUs for distributed training")
           return
           
       mp.spawn(train_distributed, args=(world_size,), nprocs=world_size, join=True)
   
   if __name__ == "__main__":
       main()

**Advanced DDP with Gradient Synchronization**

.. code-block:: python

   class AdvancedDistributedTrainer:
       """Advanced distributed trainer with custom synchronization."""
       
       def __init__(self, model, rank, world_size):
           self.model = model
           self.rank = rank
           self.world_size = world_size
           
           # Wrap model with DDP
           self.ddp_model = DDP(
               model, 
               device_ids=[rank],
               find_unused_parameters=True,  # For complex model architectures
               gradient_as_bucket_view=True  # Memory optimization
           )
           
       def synchronize_sketching_matrices(self):
           """Synchronize random sketching matrices across processes."""
           
           # This ensures all processes use the same random sketching matrices
           for name, module in self.ddp_model.named_modules():
               if isinstance(module, pr.nn.SKLinear):
                   # Broadcast sketching matrix from rank 0
                   if hasattr(module, 'sketch_matrix'):
                       dist.broadcast(module.sketch_matrix, src=0)
       
       def train_epoch(self, dataloader, optimizer, criterion):
           """Train for one epoch with advanced synchronization."""
           
           self.ddp_model.train()
           total_loss = 0
           
           for batch_idx, (data, target) in enumerate(dataloader):
               data, target = data.cuda(self.rank), target.cuda(self.rank)
               
               optimizer.zero_grad()
               
               # Forward pass
               output = self.ddp_model(data)
               loss = criterion(output, target)
               
               # Backward pass with gradient synchronization
               loss.backward()
               
               # Optional: Gradient clipping before synchronization
               torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=1.0)
               
               optimizer.step()
               
               total_loss += loss.item()
               
               # Periodic sketching matrix synchronization
               if batch_idx % 100 == 0:
                   self.synchronize_sketching_matrices()
           
           return total_loss / len(dataloader)

Model Parallel Training
-----------------------

**Pipeline Parallelism with Sketched Layers**

.. code-block:: python

   class PipelineParallelSketchedModel(nn.Module):
       """Model split across devices using pipeline parallelism."""
       
       def __init__(self, layer_configs, devices):
           super().__init__()
           
           self.devices = devices
           self.num_stages = len(devices)
           
           # Create model stages
           self.stages = nn.ModuleList()
           
           layers_per_stage = len(layer_configs) // self.num_stages
           
           for stage_idx in range(self.num_stages):
               stage_layers = []
               
               start_layer = stage_idx * layers_per_stage
               end_layer = start_layer + layers_per_stage
               if stage_idx == self.num_stages - 1:  # Last stage gets remaining layers
                   end_layer = len(layer_configs)
               
               for layer_idx in range(start_layer, end_layer):
                   config = layer_configs[layer_idx]
                   layer = pr.nn.SKLinear(
                       config['in_features'],
                       config['out_features'],
                       config['sketch_size']
                   )
                   stage_layers.append(layer)
                   
                   # Add activation (except for last layer)
                   if layer_idx < len(layer_configs) - 1:
                       stage_layers.append(nn.ReLU())
               
               stage = nn.Sequential(*stage_layers).to(devices[stage_idx])
               self.stages.append(stage)
       
       def forward(self, x):
           """Forward pass through pipeline stages."""
           
           for stage_idx, stage in enumerate(self.stages):
               # Move input to current stage's device
               x = x.to(self.devices[stage_idx])
               x = stage(x)
           
           return x
   
   # Microbatch pipeline training
   class MicrobatchPipelineTrainer:
       """Train with microbatches for better pipeline utilization."""
       
       def __init__(self, model, num_microbatches=4):
           self.model = model
           self.num_microbatches = num_microbatches
           
       def forward_backward_pipeline(self, batch_data, batch_targets, criterion, optimizer):
           """Execute forward and backward passes with microbatching."""
           
           batch_size = batch_data.shape[0]
           microbatch_size = batch_size // self.num_microbatches
           
           total_loss = 0
           
           # Process microbatches
           for microbatch_idx in range(self.num_microbatches):
               start_idx = microbatch_idx * microbatch_size
               end_idx = start_idx + microbatch_size
               
               microbatch_data = batch_data[start_idx:end_idx]
               microbatch_targets = batch_targets[start_idx:end_idx]
               
               # Forward pass
               output = self.model(microbatch_data)
               
               # Move targets to output device
               microbatch_targets = microbatch_targets.to(output.device)
               
               # Compute loss
               loss = criterion(output, microbatch_targets) / self.num_microbatches
               
               # Backward pass (accumulate gradients)
               loss.backward()
               
               total_loss += loss.item()
           
           # Update parameters after processing all microbatches
           optimizer.step()
           optimizer.zero_grad()
           
           return total_loss

Tensor Parallelism
------------------

**Column-wise Tensor Parallel SKLinear**

.. code-block:: python

   class TensorParallelSKLinear(nn.Module):
       """SKLinear layer with tensor parallelism across columns."""
       
       def __init__(self, in_features, out_features, sketch_size, tensor_parallel_group):
           super().__init__()
           
           self.in_features = in_features
           self.out_features = out_features
           self.sketch_size = sketch_size
           self.tp_group = tensor_parallel_group
           self.tp_world_size = dist.get_world_size(tensor_parallel_group)
           self.tp_rank = dist.get_rank(tensor_parallel_group)
           
           # Split output features across tensor parallel ranks
           assert out_features % self.tp_world_size == 0
           self.local_out_features = out_features // self.tp_world_size
           
           # Each rank handles a slice of the output
           self.local_sklinear = pr.nn.SKLinear(
               in_features, self.local_out_features, sketch_size
           )
           
       def forward(self, x):
           """Forward pass with tensor parallelism."""
           
           # Each rank computes its portion of the output
           local_output = self.local_sklinear(x)
           
           # All-gather to collect results from all ranks
           output_list = [torch.zeros_like(local_output) for _ in range(self.tp_world_size)]
           dist.all_gather(output_list, local_output, group=self.tp_group)
           
           # Concatenate along feature dimension
           output = torch.cat(output_list, dim=-1)
           
           return output
   
   class TensorParallelModel(nn.Module):
       """Complete model with tensor parallelism."""
       
       def __init__(self, layer_configs, tensor_parallel_group):
           super().__init__()
           
           self.tp_group = tensor_parallel_group
           self.layers = nn.ModuleList()
           
           for config in layer_configs:
               layer = TensorParallelSKLinear(
                   config['in_features'],
                   config['out_features'],
                   config['sketch_size'],
                   tensor_parallel_group
               )
               self.layers.append(layer)
           
       def forward(self, x):
           for i, layer in enumerate(self.layers):
               x = layer(x)
               if i < len(self.layers) - 1:  # No activation after last layer
                   x = torch.relu(x)
           return x

Hybrid Parallelism
------------------

**Combining Data, Model, and Tensor Parallelism**

.. code-block:: python

   class HybridParallelTrainer:
       """Trainer supporting multiple parallelism strategies."""
       
       def __init__(self, model_config, parallelism_config):
           self.model_config = model_config
           self.parallelism_config = parallelism_config
           
           # Initialize process groups
           self.setup_process_groups()
           
           # Create model with appropriate parallelism
           self.model = self.create_hybrid_model()
           
       def setup_process_groups(self):
           """Set up different process groups for different parallelism types."""
           
           world_size = dist.get_world_size()
           rank = dist.get_rank()
           
           # Tensor parallel group (within node)
           tp_size = self.parallelism_config['tensor_parallel_size']
           self.tp_rank = rank % tp_size
           tp_ranks = list(range(rank - self.tp_rank, rank - self.tp_rank + tp_size))
           self.tp_group = dist.new_group(tp_ranks)
           
           # Data parallel group (across tensor parallel groups)
           dp_size = world_size // tp_size
           self.dp_rank = rank // tp_size
           dp_ranks = [i * tp_size + self.tp_rank for i in range(dp_size)]
           self.dp_group = dist.new_group(dp_ranks)
           
       def create_hybrid_model(self):
           """Create model with hybrid parallelism."""
           
           # Create base model
           layers = []
           for config in self.model_config['layers']:
               if config.get('tensor_parallel', False):
                   layer = TensorParallelSKLinear(
                       config['in_features'],
                       config['out_features'],
                       config['sketch_size'],
                       self.tp_group
                   )
               else:
                   layer = pr.nn.SKLinear(
                       config['in_features'],
                       config['out_features'],
                       config['sketch_size']
                   )
               layers.append(layer)
           
           model = nn.Sequential(*layers)
           
           # Apply data parallelism
           model = DDP(model, process_group=self.dp_group)
           
           return model
       
       def train_step(self, batch, optimizer, criterion):
           """Training step with hybrid parallelism."""
           
           data, targets = batch
           
           # Ensure data is on correct device
           device = next(self.model.parameters()).device
           data, targets = data.to(device), targets.to(device)
           
           optimizer.zero_grad()
           
           # Forward pass
           output = self.model(data)
           loss = criterion(output, targets)
           
           # Backward pass
           loss.backward()
           
           # Gradient synchronization happens automatically with DDP
           optimizer.step()
           
           return loss.item()

Performance Monitoring
-----------------------

**Distributed Training Metrics**

.. code-block:: python

   class DistributedMetricsCollector:
       """Collect and aggregate metrics across distributed processes."""
       
       def __init__(self, rank, world_size):
           self.rank = rank
           self.world_size = world_size
           self.metrics = {}
           
       def update_metric(self, name, value):
           """Update a metric value."""
           if name not in self.metrics:
               self.metrics[name] = []
           self.metrics[name].append(value)
       
       def compute_global_metrics(self):
           """Compute global metrics across all processes."""
           
           global_metrics = {}
           
           for name, values in self.metrics.items():
               local_mean = torch.tensor(sum(values) / len(values))
               
               # Gather means from all processes
               gathered_means = [torch.zeros_like(local_mean) for _ in range(self.world_size)]
               dist.all_gather(gathered_means, local_mean)
               
               # Compute global mean
               global_mean = sum(gathered_means) / self.world_size
               global_metrics[name] = global_mean.item()
           
           return global_metrics
       
       def print_metrics(self):
           """Print metrics (only on rank 0)."""
           if self.rank == 0:
               global_metrics = self.compute_global_metrics()
               print("\\nGlobal Metrics:")
               for name, value in global_metrics.items():
                   print(f"  {name}: {value:.6f}")

**Communication Profiling**

.. code-block:: python

   def profile_communication_overhead():
       """Profile communication overhead in distributed training."""
       
       import time
       
       # Test different message sizes
       message_sizes = [1024, 4096, 16384, 65536]  # bytes
       
       if dist.is_initialized():
           rank = dist.get_rank()
           world_size = dist.get_world_size()
           
           if rank == 0:
               print("Communication Profiling Results:")
               print("-" * 50)
           
           for size in message_sizes:
               # Create test tensor
               tensor = torch.randn(size // 4).cuda()  # 4 bytes per float
               
               # All-reduce timing
               torch.cuda.synchronize()
               start_time = time.time()
               
               for _ in range(10):  # Average over multiple runs
                   dist.all_reduce(tensor)
               
               torch.cuda.synchronize()
               end_time = time.time()
               
               avg_time = (end_time - start_time) / 10
               bandwidth = (size * world_size) / avg_time / 1e9  # GB/s
               
               if rank == 0:
                   print(f"Size: {size:6d} bytes, Time: {avg_time*1000:.2f}ms, "
                         f"Bandwidth: {bandwidth:.2f} GB/s")

Best Practices for Distributed Training
----------------------------------------

**Optimization Guidelines**

1. **Parallelism Strategy Selection**

.. code-block:: python

   def choose_parallelism_strategy(model_size, num_gpus, memory_per_gpu):
       """Recommend parallelism strategy based on resources."""
       
       strategies = []
       
       # Data parallelism: good for small to medium models
       if model_size * 4 < memory_per_gpu * 0.7:  # Model fits in 70% of GPU memory
           strategies.append("data_parallel")
       
       # Model parallelism: for very large models
       if model_size * 4 > memory_per_gpu:
           strategies.append("model_parallel")
       
       # Tensor parallelism: for transformer-like models
       if num_gpus >= 4:
           strategies.append("tensor_parallel")
       
       # Hybrid: best for large models with many GPUs
       if num_gpus >= 8 and model_size * 4 > memory_per_gpu * 0.5:
           strategies.append("hybrid_parallel")
       
       return strategies

2. **Communication Optimization**

- Use appropriate batch sizes to minimize communication frequency
- Enable gradient compression for bandwidth-limited environments
- Overlap computation and communication when possible
- Use efficient collective operations (all-reduce vs. all-gather)

3. **Load Balancing**

- Ensure even workload distribution across processes
- Monitor per-GPU utilization
- Adjust microbatch sizes for pipeline parallelism

This comprehensive distributed training guide enables efficient scaling of Panther models across multiple GPUs and nodes.
