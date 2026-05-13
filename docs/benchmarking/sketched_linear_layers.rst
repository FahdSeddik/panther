Sketched Linear Layers
======================

These benchmarks compare :class:`panther.nn.SKLinear` against :class:`torch.nn.Linear` for forward
and backward pass wall-clock time. Timings are averaged over 200 repeated trials on NVIDIA Tesla T4
and P100 GPUs. Results shown only for configurations where :math:`2lk(d_{in} + d_{out}) < d_{in} \cdot d_{out}`
(i.e., where SKLinear actually uses fewer parameters than ``nn.Linear``).

Hardware:

* **NVIDIA T4** — Turing architecture with Tensor Core support (FP16). Panther uses a custom Tensor
  Core kernel when ``in_features``, ``out_features``, and ``low_rank`` are all multiples of 16.
* **NVIDIA P100** — Pascal architecture without Tensor Core support, using the standard CUDA path.

The *x*-axis is the layer width (square :math:`d_{in} = d_{out}`). The *y*-axis is wall-clock time
in milliseconds. Lower is better.

----

Backward time for linear on NVIDIA T4 with Tensor Core implementation

.. image:: ../_static/63.png
   :alt: Backward pass time (ms) for SKLinear vs nn.Linear on NVIDIA T4 (Tensor Cores), varying layer width
   :width: 600px
   :align: center

Backward time for linear on NVIDIA P100 without Tensor Cores

.. image:: ../_static/64.png
   :alt: Backward pass time (ms) for SKLinear vs nn.Linear on NVIDIA P100 (no Tensor Cores), varying layer width
   :width: 600px
   :align: center

Forward time for linear on NVIDIA T4 with Tensor Core implementation

.. image:: ../_static/65.png
   :alt: Forward pass time (ms) for SKLinear vs nn.Linear on NVIDIA T4 (Tensor Cores), varying layer width
   :width: 600px
   :align: center

Forward time for linear on NVIDIA P100 without Tensor Cores

.. image:: ../_static/66.png
   :alt: Forward pass time (ms) for SKLinear vs nn.Linear on NVIDIA P100 (no Tensor Cores), varying layer width
   :width: 600px
   :align: center

----

Key Takeaways
-------------

* **2–3× speedup at large width**: The paper reports 2–3× forward/backward speedup for
  :math:`d_{in} = d_{out} = 8192` with representative sketch parameters. This is the regime where
  the parameter-reduction condition is most easily satisfied.
* **Tensor Core path (T4)**: The T4 results show a larger absolute speedup than P100 because
  Panther's custom kernel exploits FP16 Tensor Cores. Ensure ``in_features``, ``out_features``,
  and ``low_rank`` are all multiples of 16 to activate this path.
* **Standard CUDA path (P100)**: Speedup still materializes but is more moderate. The benefit
  increases with :math:`d_{in}` as the matrix-multiply cost dominates the sketching overhead.
* **Backward > forward savings**: The gradient with respect to the sketched weight involves fewer
  flops than the full :math:`d_{out} \times d_{in}` outer product of ``nn.Linear``, so backward
  speedups tend to exceed forward speedups.
* See :doc:`../api/nn` for ``SKLinear`` parameters and :class:`panther.tuner.SKAutoTuner` for
  automated parameter search.
