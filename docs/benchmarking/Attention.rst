Randomized Multi-Head Attention
================================

These benchmarks compare :class:`panther.nn.RandMultiHeadAttention` against the standard
:class:`torch.nn.MultiheadAttention` layer across four embedding dimensions (128, 256, 512, 1024)
and two activation functions (ReLU and Softmax). Each configuration measures:

* **Forward time** — wall-clock time (ms) for a single forward pass
* **Backward time** — wall-clock time (ms) for a backward pass
* **Forward memory** — peak GPU memory allocated (MB) during the forward pass
* **Backward memory** — peak GPU memory allocated (MB) during the backward pass

The *x*-axis in all plots is the sequence length (or batch size — check the chart axis labels).
Lower is better for all metrics.

----

Embedding dimension 128 — ReLU activation
------------------------------------------

Forward time for attention embed=128 ReLU

.. image:: ../_static/67.png
   :alt: Forward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=128, ReLU
   :width: 600px
   :align: center

Backward time for attention embed=128 ReLU

.. image:: ../_static/68.png
   :alt: Backward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=128, ReLU
   :width: 600px
   :align: center

Forward memory for attention embed=128 ReLU

.. image:: ../_static/69.png
   :alt: Forward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=128, ReLU
   :width: 600px
   :align: center

Backward memory for attention embed=128 ReLU

.. image:: ../_static/70.png
   :alt: Backward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=128, ReLU
   :width: 600px
   :align: center

Embedding dimension 128 — Softmax activation
---------------------------------------------

Forward time for attention embed=128 Softmax

.. image:: ../_static/71.png
   :alt: Forward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=128, Softmax
   :width: 600px
   :align: center

Backward time for attention embed=128 Softmax

.. image:: ../_static/72.png
   :alt: Backward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=128, Softmax
   :width: 600px
   :align: center

Forward memory for attention embed=128 Softmax

.. image:: ../_static/73.png
   :alt: Forward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=128, Softmax
   :width: 600px
   :align: center

Backward memory for attention embed=128 Softmax

.. image:: ../_static/74.png
   :alt: Backward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=128, Softmax
   :width: 600px
   :align: center

Embedding dimension 256 — ReLU activation
------------------------------------------

Forward time for attention embed=256 ReLU

.. image:: ../_static/75.png
   :alt: Forward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=256, ReLU
   :width: 600px
   :align: center

Backward time for attention embed=256 ReLU

.. image:: ../_static/76.png
   :alt: Backward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=256, ReLU
   :width: 600px
   :align: center

Forward memory for attention embed=256 ReLU

.. image:: ../_static/77.png
   :alt: Forward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=256, ReLU
   :width: 600px
   :align: center

Backward memory for attention embed=256 ReLU

.. image:: ../_static/78.png
   :alt: Backward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=256, ReLU
   :width: 600px
   :align: center

Embedding dimension 256 — Softmax activation
---------------------------------------------

Forward time for attention embed=256 Softmax

.. image:: ../_static/79.png
   :alt: Forward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=256, Softmax
   :width: 600px
   :align: center

Backward time for attention embed=256 Softmax

.. image:: ../_static/80.png
   :alt: Backward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=256, Softmax
   :width: 600px
   :align: center

Forward memory for attention embed=256 Softmax

.. image:: ../_static/81.png
   :alt: Forward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=256, Softmax
   :width: 600px
   :align: center

Backward memory for attention embed=256 Softmax

.. image:: ../_static/82.png
   :alt: Backward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=256, Softmax
   :width: 600px
   :align: center

Embedding dimension 512 — ReLU activation
------------------------------------------

Forward time for attention embed=512 ReLU

.. image:: ../_static/83.png
   :alt: Forward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=512, ReLU
   :width: 600px
   :align: center

Backward time for attention embed=512 ReLU

.. image:: ../_static/84.png
   :alt: Backward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=512, ReLU
   :width: 600px
   :align: center

Forward memory for attention embed=512 ReLU

.. image:: ../_static/85.png
   :alt: Forward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=512, ReLU
   :width: 600px
   :align: center

Backward memory for attention embed=512 ReLU

.. image:: ../_static/86.png
   :alt: Backward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=512, ReLU
   :width: 600px
   :align: center

Embedding dimension 512 — Softmax activation
---------------------------------------------

Forward time for attention embed=512 Softmax

.. image:: ../_static/87.png
   :alt: Forward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=512, Softmax
   :width: 600px
   :align: center

Backward time for attention embed=512 Softmax

.. image:: ../_static/88.png
   :alt: Backward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=512, Softmax
   :width: 600px
   :align: center

Forward memory for attention embed=512 Softmax

.. image:: ../_static/89.png
   :alt: Forward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=512, Softmax
   :width: 600px
   :align: center

Backward memory for attention embed=512 Softmax

.. image:: ../_static/90.png
   :alt: Backward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=512, Softmax
   :width: 600px
   :align: center

Embedding dimension 1024 — ReLU activation
-------------------------------------------

Forward time for attention embed=1024 ReLU

.. image:: ../_static/91.png
   :alt: Forward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=1024, ReLU
   :width: 600px
   :align: center

Backward time for attention embed=1024 ReLU

.. image:: ../_static/92.png
   :alt: Backward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=1024, ReLU
   :width: 600px
   :align: center

Forward memory for attention embed=1024 ReLU

.. image:: ../_static/93.png
   :alt: Forward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=1024, ReLU
   :width: 600px
   :align: center

Backward memory for attention embed=1024 ReLU

.. image:: ../_static/94.png
   :alt: Backward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=1024, ReLU
   :width: 600px
   :align: center

Embedding dimension 1024 — Softmax activation
----------------------------------------------

Forward time for attention embed=1024 Softmax

.. image:: ../_static/95.png
   :alt: Forward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=1024, Softmax
   :width: 600px
   :align: center

Backward time for attention embed=1024 Softmax

.. image:: ../_static/96.png
   :alt: Backward pass time (ms) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=1024, Softmax
   :width: 600px
   :align: center

Forward memory for attention embed=1024 Softmax

.. image:: ../_static/97.png
   :alt: Forward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=1024, Softmax
   :width: 600px
   :align: center

Backward memory for attention embed=1024 Softmax

.. image:: ../_static/98.png
   :alt: Backward pass peak GPU memory (MB) for RandMultiHeadAttention vs nn.MultiheadAttention, embed=1024, Softmax
   :width: 600px
   :align: center

----

Key Takeaways
-------------

* **Memory savings scale with embedding dimension**: At larger embed sizes (512, 1024), the
  randomized projection in ``RandMultiHeadAttention`` reduces the size of the intermediate QK
  attention matrix, yielding progressively larger memory savings — consistent with the up-to-75%
  peak memory reduction reported for BERT-scale models.
* **Softmax vs. ReLU**: Both activation paths show similar trends. The Softmax path is the
  standard transformer attention; the ReLU path corresponds to performer-style linearized
  attention. Compare the two to understand which best suits your architecture.
* **Time overhead at small embed**: At embed=128, the overhead of the random projection may be
  visible. ``RandMultiHeadAttention`` is most beneficial at embed ≥ 256, where attention matrix
  memory dominates.
* **Backward memory**: The memory advantage is especially pronounced in the backward pass, where
  the full attention matrix gradient is replaced by projected gradients through the sketch.
