Randomized Multi-Head Attention
================================

These benchmarks compare :class:`panther.nn.RandMultiHeadAttention` against
:class:`torch.nn.MultiheadAttention` across four embedding dimensions (128, 256, 512, 1024) and two
activation kernels (ReLU and Softmax). Timings are averaged over 200 repeated trials on NVIDIA Tesla
T4 and P100 GPUs. Each configuration measures:

* **Forward time** — wall-clock time (ms) for a single forward pass
* **Backward time** — wall-clock time (ms) for a backward pass
* **Forward memory** — peak GPU memory allocated (MB) during the forward pass
* **Backward memory** — peak GPU memory allocated (MB) during the backward pass

Random feature dimensions tested: :math:`\{64, 128, 256\}`. Head counts tested: :math:`\{4, 8, 16\}`.
Sequence lengths: up to 8 192 tokens. Lower is better for all metrics.

----

Embedding dimension 128 — ReLU activation
------------------------------------------

Forward time for attention embed=128 ReLU

.. image:: ../_static/67.png
   :alt: Forward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=128, ReLU kernel
   :width: 600px
   :align: center

Backward time for attention embed=128 ReLU

.. image:: ../_static/68.png
   :alt: Backward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=128, ReLU kernel
   :width: 600px
   :align: center

Forward memory for attention embed=128 ReLU

.. image:: ../_static/69.png
   :alt: Forward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=128, ReLU kernel
   :width: 600px
   :align: center

Backward memory for attention embed=128 ReLU

.. image:: ../_static/70.png
   :alt: Backward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=128, ReLU kernel
   :width: 600px
   :align: center

Embedding dimension 128 — Softmax activation
---------------------------------------------

Forward time for attention embed=128 Softmax

.. image:: ../_static/71.png
   :alt: Forward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=128, Softmax kernel
   :width: 600px
   :align: center

Backward time for attention embed=128 Softmax

.. image:: ../_static/72.png
   :alt: Backward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=128, Softmax kernel
   :width: 600px
   :align: center

Forward memory for attention embed=128 Softmax

.. image:: ../_static/73.png
   :alt: Forward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=128, Softmax kernel
   :width: 600px
   :align: center

Backward memory for attention embed=128 Softmax

.. image:: ../_static/74.png
   :alt: Backward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=128, Softmax kernel
   :width: 600px
   :align: center

Embedding dimension 256 — ReLU activation
------------------------------------------

Forward time for attention embed=256 ReLU

.. image:: ../_static/75.png
   :alt: Forward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=256, ReLU kernel
   :width: 600px
   :align: center

Backward time for attention embed=256 ReLU

.. image:: ../_static/76.png
   :alt: Backward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=256, ReLU kernel
   :width: 600px
   :align: center

Forward memory for attention embed=256 ReLU

.. image:: ../_static/77.png
   :alt: Forward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=256, ReLU kernel
   :width: 600px
   :align: center

Backward memory for attention embed=256 ReLU

.. image:: ../_static/78.png
   :alt: Backward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=256, ReLU kernel
   :width: 600px
   :align: center

Embedding dimension 256 — Softmax activation
---------------------------------------------

Forward time for attention embed=256 Softmax

.. image:: ../_static/79.png
   :alt: Forward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=256, Softmax kernel
   :width: 600px
   :align: center

Backward time for attention embed=256 Softmax

.. image:: ../_static/80.png
   :alt: Backward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=256, Softmax kernel
   :width: 600px
   :align: center

Forward memory for attention embed=256 Softmax

.. image:: ../_static/81.png
   :alt: Forward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=256, Softmax kernel
   :width: 600px
   :align: center

Backward memory for attention embed=256 Softmax

.. image:: ../_static/82.png
   :alt: Backward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=256, Softmax kernel
   :width: 600px
   :align: center

Embedding dimension 512 — ReLU activation
------------------------------------------

Forward time for attention embed=512 ReLU

.. image:: ../_static/83.png
   :alt: Forward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=512, ReLU kernel
   :width: 600px
   :align: center

Backward time for attention embed=512 ReLU

.. image:: ../_static/84.png
   :alt: Backward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=512, ReLU kernel
   :width: 600px
   :align: center

Forward memory for attention embed=512 ReLU

.. image:: ../_static/85.png
   :alt: Forward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=512, ReLU kernel
   :width: 600px
   :align: center

Backward memory for attention embed=512 ReLU

.. image:: ../_static/86.png
   :alt: Backward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=512, ReLU kernel
   :width: 600px
   :align: center

Embedding dimension 512 — Softmax activation
---------------------------------------------

Forward time for attention embed=512 Softmax

.. image:: ../_static/87.png
   :alt: Forward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=512, Softmax kernel
   :width: 600px
   :align: center

Backward time for attention embed=512 Softmax

.. image:: ../_static/88.png
   :alt: Backward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=512, Softmax kernel
   :width: 600px
   :align: center

Forward memory for attention embed=512 Softmax

.. image:: ../_static/89.png
   :alt: Forward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=512, Softmax kernel
   :width: 600px
   :align: center

Backward memory for attention embed=512 Softmax

.. image:: ../_static/90.png
   :alt: Backward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=512, Softmax kernel
   :width: 600px
   :align: center

Embedding dimension 1024 — ReLU activation
-------------------------------------------

Forward time for attention embed=1024 ReLU

.. image:: ../_static/91.png
   :alt: Forward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=1024, ReLU kernel
   :width: 600px
   :align: center

Backward time for attention embed=1024 ReLU

.. image:: ../_static/92.png
   :alt: Backward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=1024, ReLU kernel
   :width: 600px
   :align: center

Forward memory for attention embed=1024 ReLU

.. image:: ../_static/93.png
   :alt: Forward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=1024, ReLU kernel
   :width: 600px
   :align: center

Backward memory for attention embed=1024 ReLU

.. image:: ../_static/94.png
   :alt: Backward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=1024, ReLU kernel
   :width: 600px
   :align: center

Embedding dimension 1024 — Softmax activation
----------------------------------------------

Forward time for attention embed=1024 Softmax

.. image:: ../_static/95.png
   :alt: Forward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=1024, Softmax kernel
   :width: 600px
   :align: center

Backward time for attention embed=1024 Softmax

.. image:: ../_static/96.png
   :alt: Backward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=1024, Softmax kernel
   :width: 600px
   :align: center

Forward memory for attention embed=1024 Softmax

.. image:: ../_static/97.png
   :alt: Forward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=1024, Softmax kernel
   :width: 600px
   :align: center

Backward memory for attention embed=1024 Softmax

.. image:: ../_static/98.png
   :alt: Backward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=1024, Softmax kernel
   :width: 600px
   :align: center

----

Key Takeaways
-------------

* **Up to 75% memory reduction on BERT**: The paper demonstrates up to 75% peak memory savings when
  replacing standard attention with ``RandMultiHeadAttention`` in BERT, while maintaining comparable
  masked language modelling loss (4.601 vs. 4.594).
* **Memory savings grow with embedding dimension**: The :math:`O(n^2)` attention matrix is replaced
  by random feature projections of dimension :math:`r \ll d_{embed}`. At larger embed sizes the
  absolute memory advantage is more pronounced.
* **Softmax vs. ReLU kernel**: The Softmax kernel approximates standard scaled dot-product
  attention; the ReLU kernel corresponds to the Performer linearized attention. Compare both to
  choose the appropriate trade-off for your architecture.
* **Time overhead at small embed**: At embed=128 the projection overhead can exceed the attention
  matrix savings. ``RandMultiHeadAttention`` is most beneficial at embed :math:`\geq` 256.
* **Backward memory**: The memory advantage is especially pronounced in the backward pass, where
  storing the full attention map for gradient computation is replaced by projected gradients.
