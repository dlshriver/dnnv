Examples
========

In this section, we will go over several examples of properties,
and how to check them on a network. We will also discuss the
basics of the property DSL.

We also provide several example networks and properties
in an external archive,
`available here <http://cs.virginia.edu/~dls2fc/eran_benchmark.tar.gz>`_.
These networks and properties are from the benchmark of the `ERAN`_ verifier,
and are converted to the ONNX and property DSL formats required by DNNV.

.. _ERAN: https://github.com/eth-sri/eran


Local Robustness
----------------

Local robustness specifies that, given an input, :math:`x`,
to a DNN, :math:`\mathcal{N}`, any other input within
some distance, :math:`\epsilon`, of that input
will be classified to the same class. Formally:

.. math::

    \forall \delta \in [0, \epsilon]^n. \mathcal{N}(x) = \mathcal{N}(x \pm \delta)

This property can be specified in our DSL as follows:

.. code-block:: python

  from dnnv.properties import *
  import numpy as np

  N = Network("N")
  x = Image(Parameter("input", type=str))
  epsilon = Parameter("epsilon", float, default=1.0)

  output_class = np.argmax(N(x))

  Forall(
      x_,
      Implies(
          ((x - epsilon) < x_ < (x + epsilon)),
          np.argmax(N(x_)) == output_class,
      ),
  )



ACAS Xu
-------

Properties other than local robustness can also be specified in
the DNNV property DSL. For example, the properties for the ACAS Xu
aircraft collision avoidance network (as introduced in 
`this work <https://arxiv.org/pdf/1702.01135.pdf>`_) can easily be
encoded.

Here we write the specification for ACAS Xu Property :math:`\phi_3`.
The specification states that if an intruding aircraft is directly
ahead and moving towards our aircraft, then the score for a
Clear-of-Conflict classification (class 0) will not be minimal (this network
recommends the class with the minimal score).

In this property, we also see how inputs can be pre-processed.
The ACAS Xu networks expects inputs to be normalized by subtracting
a pre-computed mean value, and dividing by the given range. We
apply that normalization to the input bounds before bounding the
network input, ``x``.

.. code-block:: python

  from dnnv.properties import *
  import numpy as np

  # Setup Input Bounds
  x_mean = np.array([[19791.091, 0.0, 0.0, 650.0, 600.0]])
  x_range = np.array([[60261.0, 6.283, 6.283, 1100.0, 1200.0]])

  x_min = np.array([[12000, 0.7, -3.141592, 100, 0]])
  x_max = np.array([[62000, 3.141592, -3.136592, 1200, 1200]])
  x_min_1 = (x_min - x_mean) / x_range
  x_max_1 = (x_max - x_mean) / x_range

  x_min = np.array([[12000, -3.141592, -3.141592, 100, 0]])
  x_max = np.array([[62000, -0.7, -3.136592, 1200, 1200]])
  x_min_2 = (x_min - x_mean) / x_range
  x_max_2 = (x_max - x_mean) / x_range

  # Specify Property
  N = Network("ACAS")
  Forall(
      x,
      Implies(
          Or(
              x_min_1 <= x <= x_max_1,
              x_min_2 <= x <= x_max_2
          ),
          np.argmin(N(x)) == 0
      )
  )
