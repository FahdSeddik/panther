Performance Optimization
========================

This tutorial covers techniques to maximize performance using Tensor Core usage.

Tensor Core Optimization
-------------------------

**Understanding Tensor Cores**

Modern NVIDIA GPUs (V100, A100, RTX series) include Tensor Cores that can dramatically accelerate matrix operations when certain conditions are met:

* Input dimensions are multiples of 8 (FP16) or 16 (INT8)
* Data types are FP16, BF16, or INT8
* Matrix operations are sufficiently large

.. code-block:: python

   def optimize_for_tensor_cores(in_features, out_features, num_terms, low_rank):
       """Optimize dimensions for Tensor Core usage."""
       
       # Round dimensions to multiples of 8 for FP16 Tensor Cores
       def round_to_multiple(x, multiple=8):
           return ((x + multiple - 1) // multiple) * multiple
       
       optimized_in = round_to_multiple(in_features)
       optimized_out = round_to_multiple(out_features)
       optimized_rank = round_to_multiple(low_rank)
       
       # Ensure we don't exceed original dimensions significantly
       if optimized_in > in_features * 1.2:
           optimized_in = in_features
       if optimized_out > out_features * 1.2:
           optimized_out = out_features
       
       return optimized_in, optimized_out, num_terms, optimized_rank
   
   # Example: Create Tensor Core optimized layers
   class TensorCoreOptimizedModel(nn.Module):
       def __init__(self, layer_configs):
           super().__init__()
           
           layers = []
           for config in layer_configs:
               in_feat, out_feat, terms, rank = optimize_for_tensor_cores(
                   config['in_features'], config['out_features'],
                   config['num_terms'], config['low_rank']
               )
               
               layer = pr.nn.SKLinear(in_feat, out_feat, num_terms=terms, low_rank=rank)
               layers.extend([layer, nn.ReLU()])
           
           self.network = nn.Sequential(*layers)
       
       def forward(self, x):
           return self.network(x)
   
   # Create optimized model
   configs = [
       {'in_features': 1000, 'out_features': 500, 'num_terms': 8, 'low_rank': 64},
       {'in_features': 500, 'out_features': 250, 'num_terms': 6, 'low_rank': 48},
       {'in_features': 250, 'out_features': 10, 'num_terms': 4, 'low_rank': 32}
   ]
   
   optimized_model = TensorCoreOptimizedModel(configs).to(device)

The next tutorial will cover custom sketching implementations.
