ResNet with Sketched Layers
===========================

This example demonstrates how to replace standard layers in ResNet with Panther's sketched layers for memory efficiency.

Standard ResNet vs Sketched ResNet
-----------------------------------

We'll modify a ResNet-50 architecture to use sketched linear and convolution layers, comparing memory usage and performance.

**Basic ResNet Block with Sketching**

.. code-block:: python

   import torch
   import torch.nn as nn
   import panther as pr
   
   class StandardBasicBlock(nn.Module):
       """Standard ResNet basic block."""
       expansion = 1
       
       def __init__(self, in_channels, out_channels, stride=1):
           super().__init__()
           
           self.conv1 = nn.Conv2d(in_channels, out_channels, 
                                 kernel_size=3, stride=stride, padding=1, bias=False)
           self.bn1 = nn.BatchNorm2d(out_channels)
           
           self.conv2 = nn.Conv2d(out_channels, out_channels,
                                 kernel_size=3, stride=1, padding=1, bias=False)
           self.bn2 = nn.BatchNorm2d(out_channels)
           
           self.relu = nn.ReLU(inplace=True)
           
           # Shortcut connection
           self.shortcut = nn.Sequential()
           if stride != 1 or in_channels != out_channels:
               self.shortcut = nn.Sequential(
                   nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                   nn.BatchNorm2d(out_channels)
               )
       
       def forward(self, x):
           residual = self.shortcut(x)
           
           out = self.relu(self.bn1(self.conv1(x)))
           out = self.bn2(self.conv2(out))
           
           out += residual
           out = self.relu(out)
           
           return out
   
   class SketchedBasicBlock(nn.Module):
       """ResNet basic block with sketched convolutions."""
       expansion = 1
       
       def __init__(self, in_channels, out_channels, stride=1, sketch_params=None):
           super().__init__()
           
           if sketch_params is None:
               sketch_params = {'num_terms': 4, 'low_rank': 16}
           
           # Use sketched convolutions
           self.conv1 = pr.nn.SKConv2d(
               in_channels, out_channels,
               kernel_size=3, stride=stride, padding=1, bias=False,
               **sketch_params
           )
           self.bn1 = nn.BatchNorm2d(out_channels)
           
           self.conv2 = pr.nn.SKConv2d(
               out_channels, out_channels,
               kernel_size=3, stride=1, padding=1, bias=False,
               **sketch_params
           )
           self.bn2 = nn.BatchNorm2d(out_channels)
           
           self.relu = nn.ReLU(inplace=True)
           
           # Shortcut connection (can also be sketched)
           self.shortcut = nn.Sequential()
           if stride != 1 or in_channels != out_channels:
               self.shortcut = nn.Sequential(
                   pr.nn.SKConv2d(in_channels, out_channels, 
                                 kernel_size=1, stride=stride, bias=False,
                                 num_terms=2, low_rank=8),  # Smaller sketch for 1x1
                   nn.BatchNorm2d(out_channels)
               )
       
       def forward(self, x):
           residual = self.shortcut(x)
           
           out = self.relu(self.bn1(self.conv1(x)))
           out = self.bn2(self.conv2(out))
           
           out += residual
           out = self.relu(out)
           
           return out

**Complete ResNet Architecture**

.. code-block:: python

   class SketchedResNet(nn.Module):
       """ResNet with sketched layers for memory efficiency."""
       
       def __init__(self, block, layers, num_classes=1000, sketch_config=None):
           super().__init__()
           
           if sketch_config is None:
               sketch_config = {
                   'conv_sketch': {'num_terms': 4, 'low_rank': 16},
                   'fc_sketch': {'num_terms': 8, 'low_rank': 64}
               }
           
           self.in_channels = 64
           self.sketch_config = sketch_config
           
           # Initial convolution (keep standard for first layer)
           self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
           self.bn1 = nn.BatchNorm2d(64)
           self.relu = nn.ReLU(inplace=True)
           self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
           
           # ResNet layers with progressive sketching
           self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
           self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
           self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
           self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
           
           self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
           
           # Sketched fully connected layer
           self.fc = pr.nn.SKLinear(
               512 * block.expansion, num_classes,
               **sketch_config['fc_sketch']
           )
           
           self._initialize_weights()
       
       def _make_layer(self, block, out_channels, blocks, stride=1):
           layers = []
           
           # First block (might have stride > 1)
           layers.append(block(
               self.in_channels, out_channels, stride,
               sketch_params=self.sketch_config['conv_sketch']
           ))
           
           self.in_channels = out_channels * block.expansion
           
           # Remaining blocks
           for _ in range(1, blocks):
               layers.append(block(
                   self.in_channels, out_channels,
                   sketch_params=self.sketch_config['conv_sketch']
               ))
           
           return nn.Sequential(*layers)
       
       def _initialize_weights(self):
           for m in self.modules():
               if isinstance(m, (nn.Conv2d, pr.nn.SKConv2d)):
                   nn.init.kaiming_normal_(m.weight if hasattr(m, 'weight') else m.S1s,
                                          mode='fan_out', nonlinearity='relu')
               elif isinstance(m, nn.BatchNorm2d):
                   nn.init.constant_(m.weight, 1)
                   nn.init.constant_(m.bias, 0)
       
       def forward(self, x):
           x = self.conv1(x)
           x = self.bn1(x)
           x = self.relu(x)
           x = self.maxpool(x)
           
           x = self.layer1(x)
           x = self.layer2(x)
           x = self.layer3(x)
           x = self.layer4(x)
           
           x = self.avgpool(x)
           x = torch.flatten(x, 1)
           x = self.fc(x)
           
           return x
   
   def sketched_resnet50(num_classes=1000, **kwargs):
       """ResNet-50 with sketched layers."""
       sketch_config = {
           'conv_sketch': {'num_terms': 6, 'low_rank': 24},
           'fc_sketch': {'num_terms': 12, 'low_rank': 128}
       }
       return SketchedResNet(SketchedBasicBlock, [3, 4, 6, 3], 
                            num_classes, sketch_config, **kwargs)

Memory Comparison and Benchmarking
-----------------------------------

**Memory Usage Analysis**

.. code-block:: python

   import torch
   import torchvision.models as models
   import psutil
   import os
   
   def measure_model_memory(model, input_size=(3, 224, 224), batch_size=32):
       """Measure model memory usage."""
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       model = model.to(device)
       
       # Reset memory stats
       if device.type == 'cuda':
           torch.cuda.reset_peak_memory_stats(device)
           torch.cuda.empty_cache()
       
       # Measure parameter memory
       param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
       
       # Create dummy input
       dummy_input = torch.randn(batch_size, *input_size, device=device)
       
       # Forward pass
       with torch.no_grad():
           output = model(dummy_input)
       
       if device.type == 'cuda':
           peak_memory = torch.cuda.max_memory_allocated(device)
           current_memory = torch.cuda.memory_allocated(device)
       else:
           process = psutil.Process(os.getpid())
           peak_memory = process.memory_info().rss
           current_memory = peak_memory
       
       return {
           'parameters_mb': param_memory / (1024**2),
           'peak_memory_mb': peak_memory / (1024**2),
           'current_memory_mb': current_memory / (1024**2)
       }
   
   # Compare standard vs sketched ResNet
   print("Creating models...")
   
   # Standard ResNet-50
   standard_resnet = models.resnet50(pretrained=False)
   
   # Sketched ResNet-50
   sketched_resnet = sketched_resnet50()
   
   print("\\nMeasuring memory usage...")
   
   # Measure memory
   standard_memory = measure_model_memory(standard_resnet)
   sketched_memory = measure_model_memory(sketched_resnet)
   
   print("\\n" + "="*60)
   print("MEMORY COMPARISON")
   print("="*60)
   print(f"Standard ResNet-50:")
   print(f"  Parameters: {standard_memory['parameters_mb']:.1f} MB")
   print(f"  Peak Memory: {standard_memory['peak_memory_mb']:.1f} MB")
   
   print(f"\\nSketched ResNet-50:")
   print(f"  Parameters: {sketched_memory['parameters_mb']:.1f} MB")
   print(f"  Peak Memory: {sketched_memory['peak_memory_mb']:.1f} MB")
   
   param_reduction = (1 - sketched_memory['parameters_mb'] / standard_memory['parameters_mb']) * 100
   memory_reduction = (1 - sketched_memory['peak_memory_mb'] / standard_memory['peak_memory_mb']) * 100
   
   print(f"\\nReductions:")
   print(f"  Parameter reduction: {param_reduction:.1f}%")
   print(f"  Memory reduction: {memory_reduction:.1f}%")

**Performance Benchmarking**

.. code-block:: python

   import time
   import torch.nn.functional as F
   
   def benchmark_model(model, input_size=(3, 224, 224), batch_size=32, num_runs=100):
       """Benchmark model inference speed."""
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       model = model.to(device).eval()
       
       # Create dummy input
       dummy_input = torch.randn(batch_size, *input_size, device=device)
       
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

AutoTuning Sketching Parameters
-------------------------------

**Automatic Parameter Optimization**

.. code-block:: python

   from panther.tuner import SkAutoTuner
   import torch.nn.functional as F
   
   def create_tuned_resnet():
       """Create ResNet with auto-tuned sketching parameters."""
       
       def evaluate_resnet_config(conv_terms, conv_rank, fc_terms, fc_rank):
           \"\"\"Evaluate ResNet configuration on validation set.\"\"\""
           
           # Create model with given parameters
           sketch_config = {
               'conv_sketch': {'num_terms': int(conv_terms), 'low_rank': int(conv_rank)},
               'fc_sketch': {'num_terms': int(fc_terms), 'low_rank': int(fc_rank)}
           }
           
           model = SketchedResNet(SketchedBasicBlock, [1, 1, 1, 1], 10, sketch_config)
           device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
           model = model.to(device)
           
           # Quick training (just a few batches for tuning)
           optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
           criterion = nn.CrossEntropyLoss()
           
           model.train()
           total_loss = 0
           num_batches = 0
           
           # Simulate training on a few batches
           for _ in range(10):  # Limited batches for speed
               inputs = torch.randn(32, 3, 32, 32, device=device)
               targets = torch.randint(0, 10, (32,), device=device)
               
               optimizer.zero_grad()
               outputs = model(inputs)
               loss = criterion(outputs, targets)
               loss.backward()
               optimizer.step()
               
               total_loss += loss.item()
               num_batches += 1
           
           avg_loss = total_loss / num_batches
           
           # Return negative loss (tuner maximizes)
           return -avg_loss
       
       # Set up AutoTuner
       tuner = SkAutoTuner(
           parameter_bounds={
               'conv_terms': (2, 8),
               'conv_rank': (8, 32),
               'fc_terms': (4, 16),
               'fc_rank': (16, 128)
           },
           objective_function=evaluate_resnet_config,
           n_initial_points=8,
           n_iterations=20
       )
       
       # Find optimal parameters
       best_params, best_score = tuner.optimize()
       
       print(f"Best parameters: {best_params}")
       print(f"Best score: {best_score}")
       
       # Create final model with best parameters
       best_sketch_config = {
           'conv_sketch': {
               'num_terms': int(best_params['conv_terms']),
               'low_rank': int(best_params['conv_rank'])
           },
           'fc_sketch': {
               'num_terms': int(best_params['fc_terms']),
               'low_rank': int(best_params['fc_rank'])
           }
       }
       
       return SketchedResNet(SketchedBasicBlock, [3, 4, 6, 3], 1000, best_sketch_config)
   
   # Create optimized model
   optimized_model = create_tuned_resnet()

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
