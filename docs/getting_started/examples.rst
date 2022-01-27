Examples
========

In this section, we will go over several examples of properties,
and how to check them on a network. We will also discuss the
basics of the property DSL.

We have made several DNN verification benchmarks available in DNNP and ONNX formats 
in `dlshriver/dnnv-benchmarks`_.
This benchmark repository includes both `ERAN-MNIST`_ and the `ACAS Xu`_ benchmark, 
ready to run with DNNV!


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

Properties other than local robustness can also be specified in DNNP. 
For example, the properties for the ACAS Xu
aircraft collision avoidance network (as introduced in 
the evaluation of `Reluplex <https://arxiv.org/pdf/1702.01135.pdf>`_) 
can easily be encoded in DNNP.

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
  
  N = Network("N")
  # x: $\rho$, $\theta$, $\psi$, $v_{own}$, $v_{int}$
  x_min = np.array([[1500.0, -0.06, 3.10, 980.0, 960.0]])
  x_max = np.array([[1800.0, 0.06, 3.141593, 1200.0, 1200.0]])
  
  x_mean = np.array([[1.9791091e04, 0.0, 0.0, 650.0, 600.0]])
  x_range = np.array([[60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]])
  
  x_min_normalized = (x_min - x_mean) / x_range
  x_max_normalized = (x_max - x_mean) / x_range
  
  Forall(
      x, Implies(x_min_normalized <= x <= x_max_normalized, argmin(N(x)) != 0),
  )

.. _dlshriver/dnnv-benchmarks: https://github.com/dlshriver/dnnv-benchmarks
.. _ERAN-MNIST: https://github.com/dlshriver/dnnv-benchmarks/tree/main/benchmarks/ERAN-MNIST
.. _ACAS Xu: https://github.com/dlshriver/dnnv-benchmarks/tree/main/benchmarks/ACAS_Xu
