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

