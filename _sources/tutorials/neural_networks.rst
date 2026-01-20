Neural Networks with Sketching
==============================

This tutorial demonstrates how to build complete neural networks using Panther's sketched layers, from simple MLPs to complex architectures.

Building Your First Sketched Network
-------------------------------------

**From Standard to Sketched**

Let's start by converting a standard neural network to use sketched layers:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import panther as pr
    from torch.utils.data import DataLoader, TensorDataset

    # Standard neural network
    class StandardMLP(nn.Module):
        def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
            super().__init__()

            layers = []
            current_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                current_dim = hidden_dim

            layers.append(nn.Linear(current_dim, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    # Sketched neural network
    class SketchedMLP(nn.Module):
        def __init__(self, input_dim, hidden_dims, output_dim,
                    num_terms_schedule, low_rank_schedule, dropout=0.1):
            super().__init__()

            layers = []
            current_dim = input_dim

            for i, hidden_dim in enumerate(hidden_dims):
                layers.extend([
                    pr.nn.SKLinear(
                        current_dim, hidden_dim,
                        num_terms=num_terms_schedule[i],
                        low_rank=low_rank_schedule[i]
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                current_dim = hidden_dim

            # Output layer
            layers.append(pr.nn.SKLinear(
                current_dim, output_dim,
                num_terms=num_terms_schedule[-1],
                low_rank=low_rank_schedule[-1]
            ))

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    # Compare models
    input_dim, hidden_dims, output_dim = 784, [512, 256, 128], 10

    standard_model = StandardMLP(input_dim, hidden_dims, output_dim)
    sketched_model = SketchedMLP(input_dim, hidden_dims, output_dim,num_terms_schedule=[3,3,2],low_rank_schedule=[32,16,8])

    # Parameter comparison
    standard_params = sum(p.numel() for p in standard_model.parameters())
    sketched_params = sum(p.numel() for p in sketched_model.parameters())

    print(f"Standard model parameters: {standard_params:,}")
    print(f"Sketched model parameters: {sketched_params:,}")
    print(f"Parameter reduction: {(1 - sketched_params/standard_params)*100:.1f}%")

**Training Comparison**

.. code-block:: python

    def create_synthetic_dataset(n_samples=10000, input_dim=784, n_classes=10):
        """Create synthetic classification dataset."""

        # Generate structured data
        class_centers = torch.randn(n_classes, input_dim)

        X = []
        y = []

        for class_idx in range(n_classes):
            n_class_samples = n_samples // n_classes

            # Generate samples around class center
            samples = class_centers[class_idx] + 0.5 * torch.randn(n_class_samples, input_dim)
            labels = torch.full((n_class_samples,), class_idx)

            X.append(samples)
            y.append(labels)

        X = torch.cat(X, dim=0)
        y = torch.cat(y, dim=0)

        # Shuffle data
        perm = torch.randperm(len(X))
        X, y = X[perm], y[perm]

        return X, y

    def train_and_evaluate(model, train_loader, test_loader, num_epochs=10, lr=0.001):
        """Train and evaluate a model."""

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training
        model.train()
        train_losses = []

        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)

            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = correct / total
        return train_losses, accuracy

    # Create dataset
    X, y = create_synthetic_dataset(n_samples=5000, input_dim=784, n_classes=10)

    # Split into train/test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Train both models
    print("Training standard model:")
    standard_losses, standard_acc = train_and_evaluate(
        StandardMLP(784, [512, 256, 128], 10),
        train_loader, test_loader
    )

    print("\\nTraining sketched model:")
    sketched_losses, sketched_acc = train_and_evaluate(
        SketchedMLP(784, [512, 256, 128], 10,num_terms_schedule=[3,3,2],low_rank_schedule=[32,16,8]),
        train_loader, test_loader
    )

    print(f"\\nFinal Results:")
    print(f"Standard model accuracy: {standard_acc:.4f}")
    print(f"Sketched model accuracy: {sketched_acc:.4f}")
    print(f"Accuracy difference: {abs(standard_acc - sketched_acc):.4f}")

Advanced Network Architectures
-------------------------------

**Attention Mechanisms with Randomized Features**

.. code-block:: python

    from panther.nn import RandMultiHeadAttention

    # Create randomized multi-head attention layer
    attention_layer = RandMultiHeadAttention(
        embed_dim=512,
        num_heads=8,
        num_random_features=256,  # Number of random features for approximation
        dropout=0.1,
        kernel_fn="softmax",      # Can be "softmax" or "relu"
        iscausal=False             # Set True for autoregressive tasks
    )

    # Forward pass
    x = torch.randn(32, 100, 512)  # (batch, seq_len, embed_dim)
    output, _ = attention_layer(x, x, x)
    print(f"Output shape: {output.shape}")  # (32, 100, 512)

    # Example: Transformer block with sketched feed-forward
    class TransformerBlock(nn.Module):
        """Transformer block with RandMultiHeadAttention and sketched feed-forward."""

        def __init__(self, d_model, n_heads, d_ff, num_random_features=256, num_terms=2, low_rank=16):
            super().__init__()

            # Use RandMultiHeadAttention from Panther
            self.attention = RandMultiHeadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                num_random_features=num_random_features,
                dropout=0.1,
                kernel_fn="softmax"
            )
            self.norm1 = nn.LayerNorm(d_model)

            # Feed-forward with sketched layers
            self.feed_forward = nn.Sequential(
                pr.nn.SKLinear(d_model, d_ff, num_terms=num_terms*2, low_rank=low_rank*2),
                nn.ReLU(),
                nn.Dropout(0.1),
                pr.nn.SKLinear(d_ff, d_model, num_terms=num_terms, low_rank=low_rank)
            )
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(0.1)

        def forward(self, x, mask=None):
            # Self-attention with residual connection
            attn_out, _ = self.attention(x, x, x, attention_mask=mask)
            x = self.norm1(x + self.dropout(attn_out))

            # Feed-forward with residual connection
            ff_out = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_out))

            return x

Real-World Application: Document Classification
-----------------------------------------------

**Complete Document Classification Pipeline**

.. code-block:: python

   class DocumentClassifier(nn.Module):
       """Complete document classifier using sketched layers."""
       
       def __init__(self, vocab_size, embed_dim=128, hidden_dims=[512, 256], 
                    num_classes=10, max_seq_len=512):
           super().__init__()
           
           # Embedding layer
           self.embedding = nn.Embedding(vocab_size, embed_dim)
           self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))
           
           # Transformer-style encoder with sketched layers
           self.encoder = TransformerBlock(
               d_model=embed_dim, 
               n_heads=8, 
               d_ff=embed_dim*4,
               num_terms=6, 
               low_rank=48
           )
           
           # Global pooling
           self.global_pool = nn.AdaptiveAvgPool1d(1)
           
           # Classification head with sketched layers
           classifier_layers = []
           current_dim = embed_dim
           
           for hidden_dim in hidden_dims:
               classifier_layers.extend([
                   pr.nn.SKLinear(current_dim, hidden_dim, num_terms=8, low_rank=64),
                   nn.ReLU(),
                   nn.Dropout(0.3)
               ])
               current_dim = hidden_dim
           
           classifier_layers.append(pr.nn.SKLinear(current_dim, num_classes, num_terms=4, low_rank=32))
           
           self.classifier = nn.Sequential(*classifier_layers)
       
       def forward(self, input_ids, attention_mask=None):
           # Embedding with positional encoding
           seq_len = input_ids.size(1)
           embeddings = self.embedding(input_ids)
           embeddings = embeddings + self.pos_encoding[:seq_len]
           
           # Encoder
           encoded = self.encoder(embeddings, attention_mask)
           
           # Global pooling
           if attention_mask is not None:
               # Masked average pooling
               masked_encoded = encoded * attention_mask.unsqueeze(-1)
               pooled = masked_encoded.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
           else:
               # Simple average pooling
               pooled = encoded.mean(dim=1)
           
           # Classification
           logits = self.classifier(pooled)
           
           return logits
   
   # Training function for document classification
   def train_document_classifier():
       # Hyperparameters
       vocab_size = 10000
       max_seq_len = 512
       num_classes = 20
       batch_size = 32
       
       # Model
       model = DocumentClassifier(
           vocab_size=vocab_size,
           embed_dim=128,
           hidden_dims=[512, 256],
           num_classes=num_classes,
           max_seq_len=max_seq_len
       )
       
       # Print model statistics
       total_params = sum(p.numel() for p in model.parameters())
       trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
       
       print(f"Model Parameters:")
       print(f"  Total: {total_params:,}")
       print(f"  Trainable: {trainable_params:,}")
       
       # Calculate memory usage for sketched vs standard layers
       sketched_params = sum(p.numel() for name, p in model.named_parameters() 
                            if any(layer_type in name for layer_type in ['S1s', 'S2s']))
       
       print(f"  Sketched layer parameters: {sketched_params:,}")
       
       return model

This comprehensive tutorial covers building neural networks with Panther's sketched layers. The next tutorial will focus on performance optimization techniques.
