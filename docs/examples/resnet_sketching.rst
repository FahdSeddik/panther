ResNet with Sketched Layers
===========================

This example demonstrates how to use Panther's sketched layers in ResNet-style architectures.

Basic ResNet Block with Sketching
-----------------------------------

Here's how to build ResNet-style blocks using Panther's sketched layers:

.. code-block:: python

   import torch
   import torch.nn as nn
   import panther as pr
   
   class SketchedResidualBlock(nn.Module):
       """Residual block with sketched linear transformations."""
       
       def __init__(self, in_features, out_features, num_terms=4, low_rank=32):
           super().__init__()
           
           # Main path with sketched layers
           self.main_path = nn.Sequential(
               pr.nn.SKLinear(in_features, out_features, num_terms=num_terms, low_rank=low_rank),
               nn.BatchNorm1d(out_features),
               nn.ReLU(),
               pr.nn.SKLinear(out_features, out_features, num_terms=num_terms, low_rank=low_rank),
               nn.BatchNorm1d(out_features)
           )
           
           # Shortcut connection
           if in_features != out_features:
               self.shortcut = pr.nn.SKLinear(in_features, out_features, num_terms=2, low_rank=16)
           else:
               self.shortcut = nn.Identity()
           
           self.relu = nn.ReLU()
       
       def forward(self, x):
           main_out = self.main_path(x)
           shortcut_out = self.shortcut(x)
           return self.relu(main_out + shortcut_out)
   
   # Example: Build a ResNet-style model
   class SketchedResNetStyle(nn.Module):
       def __init__(self, input_dim, block_configs, num_classes):
           super().__init__()
           
           # Input layer
           self.input_layer = nn.Sequential(
               pr.nn.SKLinear(input_dim, block_configs[0]['features'], num_terms=8, low_rank=64),
               nn.BatchNorm1d(block_configs[0]['features']),
               nn.ReLU()
           )
           
           # Residual blocks
           self.blocks = nn.ModuleList()
           current_features = block_configs[0]['features']
           
           for config in block_configs:
               block = SketchedResidualBlock(
                   current_features, 
                   config['features'],
                   num_terms=config.get('num_terms', 4),
                   low_rank=config.get('low_rank', 32)
               )
               self.blocks.append(block)
               current_features = config['features']
           
           # Output layer
           self.output_layer = pr.nn.SKLinear(current_features, num_classes, num_terms=2, low_rank=16)
           
       def forward(self, x):
           x = self.input_layer(x)
           
           for block in self.blocks:
               x = block(x)
           
           x = self.output_layer(x)
           return x
   
   # Create model
   block_configs = [
       {'features': 256, 'num_terms': 6, 'low_rank': 48},
       {'features': 256, 'num_terms': 6, 'low_rank': 48},
       {'features': 512, 'num_terms': 8, 'low_rank': 64},
       {'features': 512, 'num_terms': 8, 'low_rank': 64},
   ]
   
   model = SketchedResNetStyle(input_dim=784, block_configs=block_configs, num_classes=10)
   
   # Test forward pass
   x = torch.randn(32, 784)
   output = model(x)
   print(f"Output shape: {output.shape}")  # (32, 10)

Converting Conv2D Layers
-------------------------

Panther provides ``SKConv2d.fromTorch()`` to convert existing Conv2d layers:

.. code-block:: python

   import torch
   import torch.nn as nn
   from panther.nn import SKConv2d
   
   # Original Conv2d layer
   original_conv = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
   
   # Convert to sketched version
   sketched_conv = SKConv2d.fromTorch(
       layer=original_conv,
       num_terms=4,
       low_rank=16
   )
   
   # Test forward pass
   x = torch.randn(8, 64, 32, 32)
   output = sketched_conv(x)
   print(f"Output shape: {output.shape}")  # (8, 128, 32, 32)
   # Test forward pass
   x = torch.randn(8, 64, 32, 32)
   output = sketched_conv(x)
   print(f"Output shape: {output.shape}")  # (8, 128, 32, 32)

Parameter Selection Guidelines
------------------------------

**Choosing Sketch Parameters**

When selecting ``num_terms`` and ``low_rank`` for ResNet-style sketching:

* Start with conservative values and increase if needed for accuracy
* Use smaller sketches for shortcut connections  
* Use larger sketches for main transformation layers
* Monitor memory usage and adjust accordingly

.. code-block:: python

   # Example parameter choices for different layers
   
   # For main residual blocks
   residual_params = {'num_terms': 4, 'low_rank': 32}
   
   # For shortcut connections
   shortcut_params = {'num_terms': 2, 'low_rank': 16}
   
   # For final fully connected layer
   fc_params = {'num_terms': 2, 'low_rank': 16}

Memory Comparison
-----------------

Compare parameter counts between standard and sketched models:

.. code-block:: python

   import torch.nn as nn
   import panther as pr
   
   # Standard residual block
   standard_block = nn.Sequential(
       nn.Linear(512, 512),
       nn.ReLU(),
       nn.Linear(512, 512)
   )
   
   # Sketched residual block
   sketched_block = nn.Sequential(
       pr.nn.SKLinear(512, 512, num_terms=4, low_rank=32),
       nn.ReLU(),
       pr.nn.SKLinear(512, 512, num_terms=4, low_rank=32)
   )
   
   standard_params = sum(p.numel() for p in standard_block.parameters())
   sketched_params = sum(p.numel() for p in sketched_block.parameters())
   
   print(f"Standard parameters: {standard_params:,}")
   print(f"Sketched parameters: {sketched_params:,}")
   print(f"Reduction: {(1 - sketched_params/standard_params)*100:.1f}%")

Performance Benchmarking
------------------------

Benchmark the performance of standard vs sketched ResNet models:

.. code-block:: python

   import time
   import torch.nn.functional as F
   
   def benchmark_model(model, batch_size=32, num_runs=100):
       """Benchmark model inference and training speed."""
       model.eval()
       device = next(model.parameters()).device
       dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
       
       # Warmup
       with torch.no_grad():
           for _ in range(10):
               _ = model(dummy_input)
       
       if device.type == 'cuda':
           torch.cuda.synchronize()
       
       # Benchmark forward pass
       start_time = time.time()
       with torch.no_grad():
           for _ in range(num_runs):
               output = model(dummy_input)
               if device.type == 'cuda':
                   torch.cuda.synchronize()
       
       forward_time = (time.time() - start_time) / num_runs
       
       # Benchmark backward pass
       model.train()
       optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
       
       start_time = time.time()
       for _ in range(num_runs):
           optimizer.zero_grad()
           output = model(dummy_input)
           loss = F.cross_entropy(output, torch.randint(0, 1000, (batch_size,), device=device))
           loss.backward()
           if device.type == 'cuda':
               torch.cuda.synchronize()
       
       backward_time = (time.time() - start_time) / num_runs - forward_time
       
       return {
           'forward_time_ms': forward_time * 1000,
           'backward_time_ms': backward_time * 1000,
           'total_time_ms': (forward_time + backward_time) * 1000
       }
   
   # Benchmark both models
   print("Benchmarking inference speed...")
   
   standard_perf = benchmark_model(standard_resnet)
   sketched_perf = benchmark_model(sketched_resnet)
   
   print("\\n" + "="*60)
   print("PERFORMANCE COMPARISON")
   print("="*60)
   print(f"Standard ResNet-50:")
   print(f"  Forward: {standard_perf['forward_time_ms']:.2f} ms")
   print(f"  Backward: {standard_perf['backward_time_ms']:.2f} ms")
   print(f"  Total: {standard_perf['total_time_ms']:.2f} ms")
   
   print(f"\\nSketched ResNet-50:")
   print(f"  Forward: {sketched_perf['forward_time_ms']:.2f} ms")
   print(f"  Backward: {sketched_perf['backward_time_ms']:.2f} ms")
   print(f"  Total: {sketched_perf['total_time_ms']:.2f} ms")
   
   speed_change = (sketched_perf['total_time_ms'] / standard_perf['total_time_ms'] - 1) * 100
   print(f"\\nSpeed change: {speed_change:+.1f}%")

Training on CIFAR-10
--------------------

**Complete Training Script**

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   import torchvision
   import torchvision.transforms as transforms
   from torch.utils.data import DataLoader
   
   def train_sketched_resnet():
       """Train sketched ResNet on CIFAR-10."""
       
       # Data preparation
       transform_train = transforms.Compose([
           transforms.RandomCrop(32, padding=4),
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
       ])
       
       transform_test = transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
       ])
       
       trainset = torchvision.datasets.CIFAR10(
           root='./data', train=True, download=True, transform=transform_train
       )
       trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
       
       testset = torchvision.datasets.CIFAR10(
           root='./data', train=False, download=True, transform=transform_test
       )
       testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
       
       # Model setup
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
       # Create sketched ResNet for CIFAR-10 (smaller version)
       class SketchedResNetCIFAR(SketchedResNet):
           def __init__(self, num_classes=10):
               # Smaller sketch parameters for CIFAR-10
               sketch_config = {
                   'conv_sketch': {'num_terms': 4, 'low_rank': 12},
                   'fc_sketch': {'num_terms': 6, 'low_rank': 32}
               }
               super().__init__(SketchedBasicBlock, [2, 2, 2, 2], num_classes, sketch_config)
               
               # Modify first conv for CIFAR-10 (32x32 images)
               self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
               self.maxpool = nn.Identity()  # Remove maxpool for small images
       
       model = SketchedResNetCIFAR().to(device)
       
       # Training setup
       criterion = nn.CrossEntropyLoss()
       optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
       scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
       
       # Training loop
       num_epochs = 10  # Reduced for example
       
       for epoch in range(num_epochs):
           model.train()
           running_loss = 0.0
           correct = 0
           total = 0
           
           for batch_idx, (inputs, targets) in enumerate(trainloader):
               inputs, targets = inputs.to(device), targets.to(device)
               
               optimizer.zero_grad()
               outputs = model(inputs)
               loss = criterion(outputs, targets)
               loss.backward()
               optimizer.step()
               
               running_loss += loss.item()
               _, predicted = outputs.max(1)
               total += targets.size(0)
               correct += predicted.eq(targets).sum().item()
               
               if batch_idx % 100 == 0:
                   print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                         f'Loss: {running_loss/(batch_idx+1):.3f}, '
                         f'Acc: {100.*correct/total:.2f}%')
           
           scheduler.step()
           
           # Test evaluation
           model.eval()
           test_loss = 0
           test_correct = 0
           test_total = 0
           
           with torch.no_grad():
               for inputs, targets in testloader:
                   inputs, targets = inputs.to(device), targets.to(device)
                   outputs = model(inputs)
                   loss = criterion(outputs, targets)
                   
                   test_loss += loss.item()
                   _, predicted = outputs.max(1)
                   test_total += targets.size(0)
                   test_correct += predicted.eq(targets).sum().item()
           
           print(f'Epoch {epoch}: Test Acc: {100.*test_correct/test_total:.2f}%\\n')
       
       return model
   
   # Run training
   if __name__ == "__main__":
       trained_model = train_sketched_resnet()

Parameter Selection Guidelines
------------------------------

**Choosing Sketch Parameters**

When selecting ``num_terms`` and ``low_rank`` for ResNet sketching:

* Start with conservative values and increase if accuracy drops
* Use smaller sketches for 1x1 convolutions  
* Use larger sketches for the final fully connected layer
* Monitor memory usage and adjust accordingly

.. code-block:: python

   # Example parameter choices for different ResNet layers
   
   # For residual blocks (3x3 convolutions)
   residual_params = {'num_terms': 4, 'low_rank': 16}
   
   # For 1x1 shortcut convolutions
   shortcut_params = {'num_terms': 2, 'low_rank': 8}
   
   # For final fully connected layer
   fc_params = {'num_terms': 8, 'low_rank': 64}

Production Deployment Tips
--------------------------

**Model Export and Optimization**

.. code-block:: python

   def export_for_deployment(model, example_input):
       """Export sketched ResNet for production deployment."""
       
       # Convert to evaluation mode
       model.eval()
       
       # TorchScript compilation
       with torch.no_grad():
           traced_model = torch.jit.trace(model, example_input)
       
       # Optimize for inference
       traced_model = torch.jit.optimize_for_inference(traced_model)
       
       # Save model
       traced_model.save("sketched_resnet.pt")
       
       # Test loading
       loaded_model = torch.jit.load("sketched_resnet.pt")
       
       print("Model exported successfully!")
       return loaded_model
   
   # Example usage
   model = sketched_resnet50()
   example_input = torch.randn(1, 3, 224, 224)
   deployed_model = export_for_deployment(model, example_input)

This example demonstrates how to effectively use Panther's sketched layers in a real-world architecture like ResNet, achieving significant memory savings while maintaining competitive performance.
