Sketched Conv2d
===============

These benchmarks compare :class:`panther.nn.SKConv2d` against the standard :class:`torch.nn.Conv2d`
layer for forward and backward wall-clock time. Three kernel sizes and three feature-map (image)
sizes are covered:

* **Kernel sizes**: 3×3, 5×5, 9×9
* **Feature-map sizes**: 64×64, 128×128, 256×256 (spatial resolution of the input feature map)

The *x*-axis is the number of input/output channels (equal). The *y*-axis is wall-clock time in
milliseconds. Lower is better.

.. note::
   Line 126 ("Forward time for convolution kernel 9 image 256") is a labeling error in the original
   plots — that chart actually shows the **backward** pass for kernel=9, image=256.

----

Kernel 3×3
----------

Forward time for convolution kernel 3, image 64

.. image:: ../_static/106.png
   :alt: Forward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=3, feature-map=64
   :width: 600px
   :align: center

Backward time for convolution kernel 3, image 64

.. image:: ../_static/107.png
   :alt: Backward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=3, feature-map=64
   :width: 600px
   :align: center

Forward time for convolution kernel 3, image 128

.. image:: ../_static/108.png
   :alt: Forward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=3, feature-map=128
   :width: 600px
   :align: center

Backward time for convolution kernel 3, image 128

.. image:: ../_static/109.png
   :alt: Backward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=3, feature-map=128
   :width: 600px
   :align: center

Forward time for convolution kernel 3, image 256

.. image:: ../_static/110.png
   :alt: Forward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=3, feature-map=256
   :width: 600px
   :align: center

Backward time for convolution kernel 3, image 256

.. image:: ../_static/111.png
   :alt: Backward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=3, feature-map=256
   :width: 600px
   :align: center

Kernel 5×5
----------

Forward time for convolution kernel 5, image 64

.. image:: ../_static/112.png
   :alt: Forward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=5, feature-map=64
   :width: 600px
   :align: center

Backward time for convolution kernel 5, image 64

.. image:: ../_static/113.png
   :alt: Backward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=5, feature-map=64
   :width: 600px
   :align: center

Forward time for convolution kernel 5, image 128

.. image:: ../_static/114.png
   :alt: Forward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=5, feature-map=128
   :width: 600px
   :align: center

Backward time for convolution kernel 5, image 128

.. image:: ../_static/115.png
   :alt: Backward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=5, feature-map=128
   :width: 600px
   :align: center

Forward time for convolution kernel 5, image 256

.. image:: ../_static/116.png
   :alt: Forward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=5, feature-map=256
   :width: 600px
   :align: center

Backward time for convolution kernel 5, image 256

.. image:: ../_static/117.png
   :alt: Backward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=5, feature-map=256
   :width: 600px
   :align: center

Kernel 9×9
----------

Forward time for convolution kernel 9, image 64

.. image:: ../_static/118.png
   :alt: Forward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=9, feature-map=64
   :width: 600px
   :align: center

Backward time for convolution kernel 9, image 64

.. image:: ../_static/119.png
   :alt: Backward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=9, feature-map=64
   :width: 600px
   :align: center

Forward time for convolution kernel 9, image 128

.. image:: ../_static/120.png
   :alt: Forward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=9, feature-map=128
   :width: 600px
   :align: center

Backward time for convolution kernel 9, image 128

.. image:: ../_static/121.png
   :alt: Backward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=9, feature-map=128
   :width: 600px
   :align: center

Forward time for convolution kernel 9, image 256

.. image:: ../_static/122.png
   :alt: Forward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=9, feature-map=256
   :width: 600px
   :align: center

Backward time for convolution kernel 9, image 256

.. image:: ../_static/123.png
   :alt: Backward pass time (ms) for SKConv2d vs nn.Conv2d, kernel=9, feature-map=256
   :width: 600px
   :align: center

----

Key Takeaways
-------------

* **Larger kernels benefit more**: The sketching approach reduces the cost of the expensive
  im2col + GEMM pipeline for large kernels (9×9). The relative speedup over ``nn.Conv2d`` tends
  to increase with kernel size.
* **Larger feature maps increase savings**: At 256×256 input, the number of sliding-window
  positions is large, amplifying the savings from sketching the weight matrix.
* **Kernel 3×3 at small feature maps**: This is the cheapest Conv2d configuration. SKConv2d
  overhead may be visible here; it is best suited to wide (many channels) or large-kernel layers.
* **Backward pass**: Similar to linear layers, backward savings can exceed forward savings because
  gradient accumulation through the sketch is cheaper than the full im2col gradient computation.
* See :doc:`../api/nn` for ``SKConv2d`` parameters and :class:`panther.tuner.SKAutoTuner` for
  automatic configuration search.
