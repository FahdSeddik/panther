CQRRPT and Randomized SVD
=========================

These benchmarks measure the runtime, memory usage, and numerical accuracy of Panther's randomized
linear algebra primitives versus deterministic baselines. Timings are averaged over 200 repeated
trials on NVIDIA Tesla T4 and P100 GPUs.

**CQRRPT** (Cholesky QR with Randomization and Pivoting for Tall matrices) is evaluated at two
matrix heights (:math:`m` = 8 192 and :math:`m` = 16 768) while varying the column count :math:`n`.
The baseline is a standard column-pivoted QR factorization.

**Randomized SVD (RSVD)** is evaluated for runtime, GPU memory, and reconstruction error relative
to the exact truncated SVD, as a function of the target rank :math:`k`.

Units: seconds for runtime plots, GB for memory plots, relative Frobenius norm error
(:math:`\|A - U_k S_k V_k^\top\|_F / \|A\|_F`) for accuracy plots. Lower is better in all cases.

----

CQRRPT runtime on m = 16 768

.. image:: ../_static/99.png
   :alt: CQRRPT runtime (s) vs number of columns n for m=16768: Panther CQRRPT vs standard pivoted QR
   :width: 600px
   :align: center

CQRRPT runtime on m = 8 192

.. image:: ../_static/100.png
   :alt: CQRRPT runtime (s) vs number of columns n for m=8192: Panther CQRRPT vs standard pivoted QR
   :width: 600px
   :align: center

CQRRPT reconstruction error on m = 8 192

.. image:: ../_static/101.png
   :alt: CQRRPT relative reconstruction error vs number of columns n for m=8192
   :width: 600px
   :align: center

CQRRPT reconstruction error on m = 16 768

.. image:: ../_static/102.png
   :alt: CQRRPT relative reconstruction error vs number of columns n for m=16768
   :width: 600px
   :align: center

RSVD runtime

.. image:: ../_static/103.png
   :alt: Randomized SVD runtime (s) vs target rank k: Panther RSVD vs exact truncated SVD
   :width: 600px
   :align: center

RSVD memory

.. image:: ../_static/104.png
   :alt: Randomized SVD GPU memory usage (GB) vs target rank k: Panther RSVD vs exact truncated SVD
   :width: 600px
   :align: center

RSVD error

.. image:: ../_static/105.png
   :alt: Randomized SVD relative reconstruction error vs target rank k
   :width: 600px
   :align: center

----

Key Takeaways
-------------

* **CQRRPT runtime**: CQRRPT is substantially faster than standard pivoted QR for tall matrices
  (:math:`m \gg n`). The randomized sketch replaces the expensive column-norm pass in classic QR
  pivoting, and the advantage widens as :math:`n` grows.
* **CQRRPT accuracy**: Reconstruction error is numerically comparable to deterministic pivoted QR,
  confirming that randomization does not sacrifice stability. The ``gamma`` oversampling parameter
  (default 1.25) controls the accuracy–speed trade-off.
* **RSVD runtime**: RSVD scales much more favorably than exact SVD with rank :math:`k`. Because
  RSVD only computes the leading :math:`k` components, it avoids the full cubic cost of exact SVD.
* **RSVD memory**: Significant savings arise because intermediate factors are never materialized
  at full rank.
* **RSVD error**: The relative error :math:`\|A - U_k S_k V_k^\top\|_F / \|A\|_F` decreases as
  :math:`k` increases. Use the ``tol`` parameter in :func:`panther.linalg.randomized_svd` to
  control the accuracy budget.
