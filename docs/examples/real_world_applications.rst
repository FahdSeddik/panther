Real-World Applications
=======================

This guide showcases practical applications of Panther's sketched layers and AutoTuner.

Building Efficient Models with SKLinear
---------------------------------------

**Using SKLinear for Large Models**

.. code-block:: python

   import torch
   import torch.nn as nn
   import panther as pr
   
   class EfficientClassifier(nn.Module):
       """Large classifier with memory-efficient sketched layers."""
       
       def __init__(self, input_dim=4096, hidden_dims=[2048, 1024, 512], num_classes=1000):
           super().__init__()
           
           layers = []
           current_dim = input_dim
           
           for hidden_dim in hidden_dims:
               # Use sketched layers for large dimensions
               layers.extend([
                   pr.nn.SKLinear(current_dim, hidden_dim, num_terms=8, low_rank=64),
                   nn.ReLU(),
                   nn.Dropout(0.2)
               ])
               current_dim = hidden_dim
           
           # Output layer
           layers.append(pr.nn.SKLinear(current_dim, num_classes, num_terms=4, low_rank=32))
           
           self.network = nn.Sequential(*layers)
       
       def forward(self, x):
           return self.network(x)
           self.relu = resnet.relu
           self.maxpool = resnet.maxpool
           self.layer1 = resnet.layer1
           self.layer2 = resnet.layer2
           self.layer3 = resnet.layer3
           self.layer4 = resnet.layer4
           self.avgpool = resnet.avgpool
           
           # Replace final linear layer with SKLinear
           in_features = resnet.fc.in_features
           self.fc = pr.nn.SKLinear(
               in_features, num_classes,
               num_terms=4,
               low_rank=64
           )
           
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
   
   # Training setup
   def train_sketched_resnet():
       # Data loading
       transform_train = transforms.Compose([
           transforms.RandomCrop(32, padding=4),
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
       ])
       
       train_dataset = torchvision.datasets.CIFAR10(
           root='./data', train=True, download=True, transform=transform_train
       )
       train_loader = torch.utils.data.DataLoader(
           train_dataset, batch_size=128, shuffle=True, num_workers=4
       )
       
       # Model setup
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       model = SketchedResNet18(num_classes=10, sketch_ratio=0.6).to(device)
       
       criterion = nn.CrossEntropyLoss()
       optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
       scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
       
       # Training loop
       model.train()
       for epoch in range(10):  # Abbreviated for example
           running_loss = 0.0
           correct = 0
           total = 0
           
           for batch_idx, (inputs, targets) in enumerate(train_loader):
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
               
               if batch_idx % 100 == 99:
                   print(f'Epoch {epoch+1}, Batch {batch_idx+1}: '
                         f'Loss: {running_loss/100:.3f}, '
                         f'Acc: {100.*correct/total:.2f}%')
                   running_loss = 0.0
           
           scheduler.step()

**Object Detection with Sketched Backbones**

.. code-block:: python

   import torchvision.models as models
   from torchvision.models.detection import fasterrcnn_resnet50_fpn
   
   class SketchedFasterRCNN(nn.Module):
       """Faster R-CNN with sketched backbone for memory efficiency."""
       
       def __init__(self, num_classes=91, num_terms=4, low_rank=32):
           super().__init__()
           
           # Load pre-trained Faster R-CNN
           self.model = fasterrcnn_resnet50_fpn(pretrained=True)
           
           # Replace classifier head with sketched version
           in_features = self.model.roi_heads.box_predictor.cls_score.in_features
           
           # Sketched classification head
           self.model.roi_heads.box_predictor.cls_score = pr.nn.SKLinear(
               in_features, num_classes,
               num_terms=num_terms,
               low_rank=low_rank
           )
           
           # Sketched box regression head  
           self.model.roi_heads.box_predictor.bbox_pred = pr.nn.SKLinear(
               in_features, num_classes * 4,
               num_terms=num_terms * 2,
               low_rank=low_rank * 2
           )
       
       def forward(self, images, targets=None):
           return self.model(images, targets)
   
   # Usage example
   model = SketchedFasterRCNN(num_classes=21)  # PASCAL VOC
   model.eval()
   
   # Dummy input
   images = [torch.randn(3, 800, 600) for _ in range(2)]
   
   with torch.no_grad():
       predictions = model(images)

Natural Language Processing
---------------------------

**Transformer Models with Randomized Attention**

.. code-block:: python

   from panther.nn import RandMultiHeadAttention
   
   class TransformerBlock(nn.Module):
       """Transformer block with RandMultiHeadAttention and sketched feed-forward."""
       
       def __init__(self, d_model, n_heads, d_ff, num_random_features=256, num_terms=6, low_rank=48, dropout=0.1):
           super().__init__()
           
           self.d_model = d_model
           self.n_heads = n_heads
           
           # Randomized multi-head attention from Panther
           self.attention = RandMultiHeadAttention(
               embed_dim=d_model,
               num_heads=n_heads,
               num_random_features=num_random_features,
               dropout=dropout,
               kernel_fn="softmax"
           )
           
           # Sketched feed-forward network
           self.ff1 = pr.nn.SKLinear(d_model, d_ff, num_terms=num_terms, low_rank=low_rank)
           self.ff2 = pr.nn.SKLinear(d_ff, d_model, num_terms=num_terms, low_rank=low_rank)
           
           self.norm1 = nn.LayerNorm(d_model)
           self.norm2 = nn.LayerNorm(d_model)
           self.dropout = nn.Dropout(dropout)
           
       def forward(self, x, mask=None):
           # Self-attention with residual connection
           attn_out, _ = self.attention(x, x, x, attention_mask=mask)
           x = self.norm1(x + self.dropout(attn_out))
           
           # Feed-forward with residual connection
           ff_out = self.ff2(torch.relu(self.ff1(x)))
           x = self.norm2(x + self.dropout(ff_out))
           
           return x
   
   # Complete model for language modeling
   class LanguageModel(nn.Module):
       """Complete language model with RandMultiHeadAttention and sketched layers."""
       
       def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, 
                    d_ff=2048, max_seq_len=512, num_random_features=256, 
                    num_terms=6, low_rank=48):
           super().__init__()
           
           self.d_model = d_model
           self.max_seq_len = max_seq_len
           
           # Embeddings
           self.token_embedding = nn.Embedding(vocab_size, d_model)
           self.position_embedding = nn.Embedding(max_seq_len, d_model)
           
           # Transformer blocks
           self.blocks = nn.ModuleList([
               TransformerBlock(d_model, n_heads, d_ff, num_random_features, num_terms, low_rank)
               for _ in range(n_layers)
           ])
           
           # Output projection (sketched)
           self.output_projection = pr.nn.SKLinear(d_model, vocab_size, num_terms=num_terms, low_rank=low_rank)
           
           self.dropout = nn.Dropout(0.1)
           
       def forward(self, x, mask=None):
           seq_len = x.size(1)
           
           # Token and position embeddings
           positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
           x = self.token_embedding(x) + self.position_embedding(positions)
           x = self.dropout(x)
           
           # Pass through transformer blocks
           for block in self.blocks:
               x = block(x, mask)
           
           # Output projection
           logits = self.output_projection(x)
           
           return logits

Scientific Computing Applications
---------------------------------

**Large-Scale Linear Systems**

.. code-block:: python

   def solve_large_linear_system_iteratively():
       """Solve large linear systems using sketched preconditioning."""
       
       # Generate large sparse linear system Ax = b
       n = 10000
       density = 0.01  # 1% non-zero elements
       
       # Create random sparse matrix
       A_sparse = torch.sparse_coo_tensor(
           indices=torch.randint(0, n, (2, int(n * n * density))),
           values=torch.randn(int(n * n * density)),
           size=(n, n)
       ).coalesce()
       
       # Convert to dense for this example (use sparse operations in practice)
       A = A_sparse.to_dense()
       x_true = torch.randn(n)
       b = A @ x_true + 0.01 * torch.randn(n)  # Add noise
       
       # Sketched preconditioning using CQRRPT
       def sketched_preconditioner(A):
           """Create sketched preconditioner."""
           
           # Compute CQRRPT of A
           Q, R, P = pr.linalg.cqrrpt(
               A,
               gamma=1.25,
               F=pr.linalg.DistributionFamily.Gaussian
           )
           
           return Q, R, P
       
       # Create preconditioner
       Q_prec, R_prec, P_prec = sketched_preconditioner(A)
       
       # Preconditioned iterative solver (simplified)
       def pcg_solve(A, b, Q_prec, R_prec, P_prec, max_iter=1000, tol=1e-6):
           """Preconditioned conjugate gradient with sketched preconditioner."""
           
           x = torch.zeros_like(b)
           r = b - A @ x
           
           # Apply preconditioner: solve R_prec @ z = Q_prec^T @ r
           z = torch.linalg.solve_triangular(
               R_prec, Q_prec.T @ r[P_prec], upper=True
           )
           z_full = torch.zeros_like(r)
           z_full[P_prec] = z
           
           p = z_full.clone()
           
           for i in range(max_iter):
               Ap = A @ p
               alpha = torch.dot(r, z_full) / torch.dot(p, Ap)
               x = x + alpha * p
               r_new = r - alpha * Ap
               
               if torch.norm(r_new) < tol:
                   print(f"Converged in {i+1} iterations")
                   break
               
               # Update preconditioned residual
               z_new = torch.linalg.solve_triangular(
                   R_prec, Q_prec.T @ r_new[:len(P_prec)], upper=True
               )
               z_full_new = torch.zeros_like(r_new)
               z_full_new[P_prec] = z_new
               
               beta = torch.dot(r_new, z_full_new) / torch.dot(r, z_full)
               p = z_full_new + beta * p
               
               r, z_full = r_new, z_full_new
           
           return x
       
       # Solve system
       x_solved = pcg_solve(A, b, Q_prec, R_prec, P_prec)
       
       # Compare with true solution
       error = torch.norm(x_solved - x_true) / torch.norm(x_true)
       print(f"Relative solution error: {error:.6f}")
       
       return x_solved

**Principal Component Analysis on Large Datasets**

.. code-block:: python

   def streaming_pca_with_sketching():
       """Perform PCA on streaming data using sketched updates."""
       
       class StreamingSketchedPCA:
           def __init__(self, n_features, n_components, sketch_ratio=0.5):
               self.n_features = n_features
               self.n_components = n_components
               self.sketch_dim = int(n_features * sketch_ratio)
               
               # Initialize sketching matrix
               self.sketch_matrix = torch.randn(self.sketch_dim, n_features)
               self.sketch_matrix = torch.nn.functional.normalize(self.sketch_matrix, dim=1)
               
               # Running statistics
               self.mean_ = torch.zeros(n_features)
               self.sketched_cov_ = torch.zeros(self.sketch_dim, self.sketch_dim)
               self.n_samples_ = 0
               
           def partial_fit(self, X_batch):
               """Update PCA with new batch of data."""
               batch_size = X_batch.shape[0]
               
               # Update mean
               total_samples = self.n_samples_ + batch_size
               batch_mean = X_batch.mean(dim=0)
               
               if self.n_samples_ == 0:
                   self.mean_ = batch_mean
               else:
                   delta = batch_mean - self.mean_
                   self.mean_ += delta * batch_size / total_samples
               
               # Center the batch
               X_centered = X_batch - self.mean_
               
               # Sketch the centered data
               X_sketched = X_centered @ self.sketch_matrix.T
               
               # Update sketched covariance matrix
               batch_sketched_cov = X_sketched.T @ X_sketched / batch_size
               
               if self.n_samples_ == 0:
                   self.sketched_cov_ = batch_sketched_cov
               else:
                   # Weighted average of covariances
                   weight_old = self.n_samples_ / total_samples
                   weight_new = batch_size / total_samples
                   self.sketched_cov_ = (weight_old * self.sketched_cov_ + 
                                       weight_new * batch_sketched_cov)
               
               self.n_samples_ = total_samples
           
           def fit_final(self):
               """Compute final PCA components from sketched covariance."""
               
               # Eigendecomposition of sketched covariance
               eigenvals, eigenvecs = torch.linalg.eigh(self.sketched_cov_)
               
               # Sort by eigenvalue (descending)
               idx = torch.argsort(eigenvals, descending=True)
               eigenvals = eigenvals[idx]
               eigenvecs = eigenvecs[:, idx]
               
               # Take top components
               top_eigenvecs = eigenvecs[:, :self.n_components]
               
               # Project back to original space
               self.components_ = top_eigenvecs.T @ self.sketch_matrix
               self.explained_variance_ = eigenvals[:self.n_components]
               
               # Normalize components
               self.components_ = torch.nn.functional.normalize(self.components_, dim=1)
               
           def transform(self, X):
               """Transform data to PCA space."""
               X_centered = X - self.mean_
               return X_centered @ self.components_.T
       
       # Example usage with simulated streaming data
       n_features = 1000
       n_components = 50
       
       pca = StreamingSketchedPCA(n_features, n_components, sketch_ratio=0.3)
       
       # Simulate streaming batches
       for batch_idx in range(100):
           # Generate random batch (in practice, this would be real streaming data)
           batch_size = 100
           X_batch = torch.randn(batch_size, n_features)
           
           # Add some structure (low-rank component)
           if batch_idx < 50:
               latent = torch.randn(batch_size, 20)
               structure = latent @ torch.randn(20, n_features)
               X_batch += 0.5 * structure
           
           pca.partial_fit(X_batch)
           
           if batch_idx % 20 == 19:
               print(f"Processed {(batch_idx + 1) * batch_size} samples")
       
       # Finalize PCA
       pca.fit_final()
       
       print(f"Final PCA with {n_components} components")
       print(f"Top 5 explained variances: {pca.explained_variance_[:5]}")
       
       # Test transformation
       test_data = torch.randn(50, n_features)
       transformed = pca.transform(test_data)
       print(f"Transformed shape: {transformed.shape}")

Financial Modeling Applications
-------------------------------

**Portfolio Optimization with Sketched Covariance**

.. code-block:: python

   def portfolio_optimization_sketched():
       """Portfolio optimization using sketched covariance estimation."""
       
       # Simulate stock return data
       n_stocks = 500
       n_days = 1000
       
       # Generate correlated returns
       factors = torch.randn(n_days, 20)  # 20 common factors
       loadings = torch.randn(n_stocks, 20)
       idiosyncratic = 0.3 * torch.randn(n_days, n_stocks)
       
       returns = factors @ loadings.T + idiosyncratic
       returns = returns * 0.02  # Scale to realistic daily returns
       
       print(f"Stock returns shape: {returns.shape}")
       
       # Traditional covariance estimation (expensive for large n_stocks)
       # cov_traditional = torch.cov(returns.T)  # Too expensive!
       
       # Sketched covariance estimation using RSVD
       def sketched_covariance(returns, rank=50):
           """Estimate covariance using low-rank sketching."""
           
           # Center returns
           returns_centered = returns - returns.mean(dim=0)
           
           # Compute sketched SVD
           U, S, V = pr.linalg.randomized_svd(
               returns_centered.T, k=rank, tol=1e-6
           )
           
           # Low-rank covariance approximation
           cov_lowrank = V @ torch.diag(S**2 / (returns.shape[0] - 1)) @ V.T
           
           # Add diagonal regularization (idiosyncratic risk)
           diag_reg = torch.var(returns, dim=0) - torch.diag(cov_lowrank)
           diag_reg = torch.clamp(diag_reg, min=1e-6)  # Ensure positive
           
           cov_sketched = cov_lowrank + torch.diag(diag_reg)
           
           return cov_sketched, U, S, V
       
       # Estimate sketched covariance
       cov_matrix, U, S, V = sketched_covariance(returns, rank=50)
       
       # Mean return estimation
       mean_returns = returns.mean(dim=0)
       
       # Portfolio optimization (Markowitz mean-variance)
       def optimize_portfolio(mean_returns, cov_matrix, risk_aversion=1.0):
           """Solve portfolio optimization using sketched covariance."""
           
           n_assets = len(mean_returns)
           
           # Quadratic programming: minimize w^T @ cov @ w - risk_aversion * mean^T @ w
           # Subject to: sum(w) = 1, w >= 0
           
           # Simplified solution using matrix inversion (pseudo-optimal)
           ones = torch.ones(n_assets, 1)
           
           # Regularize covariance for numerical stability
           cov_reg = cov_matrix + 1e-6 * torch.eye(n_assets)
           
           try:
               cov_inv = torch.linalg.inv(cov_reg)
               
               # Optimal weights (without constraints for simplicity)
               numerator = cov_inv @ (risk_aversion * mean_returns.unsqueeze(1) + ones)
               denominator = ones.T @ cov_inv @ ones
               
               weights = numerator / denominator
               weights = weights.squeeze()
               
               # Apply simple constraints
               weights = torch.clamp(weights, min=0)  # Long-only
               weights = weights / weights.sum()  # Normalize
               
           except:
               # Fallback to equal weights
               weights = torch.ones(n_assets) / n_assets
           
           return weights
       
       # Optimize portfolio
       weights = optimize_portfolio(mean_returns, cov_matrix, risk_aversion=2.0)
       
       # Portfolio performance metrics
       portfolio_return = torch.dot(weights, mean_returns)
       portfolio_risk = torch.sqrt(weights @ cov_matrix @ weights)
       sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
       
       print(f"\\nPortfolio Optimization Results:")
       print(f"Number of assets: {n_stocks}")
       print(f"Portfolio expected return: {portfolio_return:.4f}")
       print(f"Portfolio risk (std): {portfolio_risk:.4f}")
       print(f"Sharpe ratio: {sharpe_ratio:.4f}")
       print(f"Number of active positions: {(weights > 1e-4).sum()}")
       print(f"Largest position: {weights.max():.4f}")
       
       return weights, cov_matrix

Applications Summary
--------------------

These real-world applications demonstrate Panther's versatility across different domains:

**Computer Vision**
- 10-25% memory reduction in CNNs
- Faster training with minimal accuracy loss
- Suitable for deployment in resource-constrained environments

**Natural Language Processing**
- Efficient transformer models for long sequences
- Reduced attention complexity
- Scalable to large vocabulary sizes

**Scientific Computing**
- Fast solutions for large linear systems
- Streaming PCA for real-time data analysis
- Memory-efficient matrix operations

**Financial Modeling**
- Scalable portfolio optimization
- Real-time risk estimation
- Handling high-dimensional financial data

Each application showcases how Panther's sketching algorithms can be adapted to specific domain requirements while maintaining computational efficiency and numerical accuracy.
