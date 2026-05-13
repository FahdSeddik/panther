Sketched Conv2d
===============

These benchmarks compare :class:`panther.nn.SKConv2d` against :class:`torch.nn.Conv2d` for forward
and backward wall-clock time. Timings are averaged over 200 repeated trials on NVIDIA Tesla T4 and
P100 GPUs. Three kernel sizes and three feature-map sizes are covered:

* **Kernel sizes**: 3×3, 5×5, 9×9
* **Feature-map sizes (spatial resolution)**: 64×64, 128×128, 256×256
* **Sketch parameters**: :math:`l \in \{1, 2, 3\}` terms, :math:`k \in \{8, 16, 32\}` rank
* **Channel counts**: 64–2048

Results are shown only for configurations where the sketched parameterisation uses fewer parameters
than the standard weight tensor (fair comparison condition).

The *x*-axis is the number of input/output channels (equal). The *y*-axis is wall-clock time in
milliseconds. Lower is better.

.. note::
   Image 123 (last chart, kernel 9×9, feature-map 256) shows the **backward** pass; its preceding
   label in the original plots was duplicated as "Forward time" — this is a known labelling error
   in the benchmark figures.

----

Kernel 3×3
----------

Forward time for convolution kernel 3, image 64

.. image:: ../_static/106.png
   :alt: Forward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=3x3, feature-map=64
   :width: 600px
   :align: center

Backward time for convolution kernel 3, image 64

.. image:: ../_static/107.png
   :alt: Backward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=3x3, feature-map=64
   :width: 600px
   :align: center

Forward time for convolution kernel 3, image 128

.. image:: ../_static/108.png
   :alt: Forward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=3x3, feature-map=128
   :width: 600px
   :align: center

Backward time for convolution kernel 3, image 128

.. image:: ../_static/109.png
   :alt: Backward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=3x3, feature-map=128
   :width: 600px
   :align: center

Forward time for convolution kernel 3, image 256

.. image:: ../_static/110.png
   :alt: Forward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=3x3, feature-map=256
   :width: 600px
   :align: center

Backward time for convolution kernel 3, image 256

.. image:: ../_static/111.png
   :alt: Backward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=3x3, feature-map=256
   :width: 600px
   :align: center

Kernel 5×5
----------

Forward time for convolution kernel 5, image 64

.. image:: ../_static/112.png
   :alt: Forward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=5x5, feature-map=64
   :width: 600px
   :align: center

Backward time for convolution kernel 5, image 64

.. image:: ../_static/113.png
   :alt: Backward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=5x5, feature-map=64
   :width: 600px
   :align: center

Forward time for convolution kernel 5, image 128

.. image:: ../_static/114.png
   :alt: Forward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=5x5, feature-map=128
   :width: 600px
   :align: center

Backward time for convolution kernel 5, image 128

.. image:: ../_static/115.png
   :alt: Backward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=5x5, feature-map=128
   :width: 600px
   :align: center

Forward time for convolution kernel 5, image 256

.. image:: ../_static/116.png
   :alt: Forward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=5x5, feature-map=256
   :width: 600px
   :align: center

Backward time for convolution kernel 5, image 256

.. image:: ../_static/117.png
   :alt: Backward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=5x5, feature-map=256
   :width: 600px
   :align: center

Kernel 9×9
----------

Forward time for convolution kernel 9, image 64

.. image:: ../_static/118.png
   :alt: Forward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=9x9, feature-map=64
   :width: 600px
   :align: center

Backward time for convolution kernel 9, image 64

.. image:: ../_static/119.png
   :alt: Backward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=9x9, feature-map=64
   :width: 600px
   :align: center

Forward time for convolution kernel 9, image 128

.. image:: ../_static/120.png
   :alt: Forward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=9x9, feature-map=128
   :width: 600px
   :align: center

Backward time for convolution kernel 9, image 128

.. image:: ../_static/121.png
   :alt: Backward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=9x9, feature-map=128
   :width: 600px
   :align: center

Forward time for convolution kernel 9, image 256

.. image:: ../_static/122.png
   :alt: Forward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=9x9, feature-map=256
   :width: 600px
   :align: center

Backward time for convolution kernel 9, image 256

.. image:: ../_static/123.png
   :alt: Backward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=9x9, feature-map=256
   :width: 600px
   :align: center

----

Key Takeaways
-------------

* **30% size reduction demonstrated on ResNet-50**: Applying ``SKConv2d`` throughout ResNet-50
  achieves 30% fewer parameters, with accuracy on CIFAR-10 going from 89% to 86% — a small
  accuracy cost for a substantial compression gain (per paper Section IV).
* **Larger kernels benefit more**: The sketching approach reduces the dominant cost in the
  im2col + GEMM pipeline for large kernels (9×9). Relative speedup over ``nn.Conv2d`` increases
  with kernel size.
* **Larger feature maps amplify savings**: At 256×256 input, the number of sliding-window positions
  is large, amplifying the per-element savings from sketching the weight tensor.
* **Kernel 3×3 at small feature maps**: This is the cheapest Conv2d configuration. SKConv2d
  overhead may be visible here; it is best suited to wide (many channels) or large-kernel layers.
* See :doc:`../api/nn` for ``SKConv2d`` parameters and :class:`panther.tuner.SKAutoTuner` for
  automated parameter search.
