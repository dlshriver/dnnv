Networks
========

Networks are a key component to DNNP specifications. Without
a network, there is no DNN verification problem. As such, DNNP
provides several ways to facilitate working with networks
within property specifications. This includes network inference,
slicing, and composition. We explain each of these in more
detail below.

These operations on Network objects enable DNNP specifications
to be specified over production models, rather than models
designed specifically for a given specification. DNNP allows
models to be used, reduced, or combined to capture the exact
semantics of the desired property without having to customize
the network model for the specification being written.


Network Inference
-----------------

First, and probably most importantly, is that networks can
perform inference on inputs. Inference occurs when a network is 
applied to an input using the python call syntax, i.e., ``N(x)``.
Inference can be either concrete or symbolic.

Concrete inference occurs when the inputs to the network are
concrete values. This occurs when the exact value of the input
is known at runtime, such as when it is explicitly defined in 
the specification (e.g., ``x = np.array([[0.0, 1.0, -1.0]])``),
or parameterized at runtime 
(e.g., ``x = np.load(Parameter("inputpath", str))``).
In this case, inference will compute the concrete output value
for the given input.

Symbolic inference occurs when the input is non-concrete.
Currently, DNNV only supports symbolic inference when the input
is a :py:class:`Symbol` object. Instances of symbolic inference
are extracted during the property reduction process and
translated to appropriate constraints for the specified verifier.


Network Slicing
---------------

In DNNP and DNNV, networks are represented as computation graphs, 
where nodes in the graph are operations and directed edges between
nodes represent the data flow between operations.
Some of these operations are Inputs, and others are tagged as outputs.
DNNP introduces the concept of network slicing, which allows us to 
select a subgraph of the original network graph.
Slicing enables verification at any point in the network, 
while using the same full network model as input to DNNV.
For example, in a six layer network, we could define a property over 
the output of the third layer, such as some of the properties in 
`Property Inference for Deep Neural Networks`_ by Gopinath et al.

In DNNP, network slicing looks like ``N[start:end, output_index]``.
The second axis, ``output_index`` is optional, and can be used when 
a network has multiple tagged output operations to select a single 
operation as the output. This is not a common use case, and so in
many cases, slicing will more resemble ``N[start:end]``.
Slicing will take all operations that are at least ``start`` positions 
from an ``Input`` operation (i.e., there is a path with at least 
``start-1`` operations between the operation and an ``Input``) and 
at most ``end-1`` positions from an ``Input``, 
if both ``start`` and ``end`` are positive.
A negative index will count backwards from the network outputs, rather
than forwards from the inputs. A negative ``start`` value will select
operations that are at most ``-start`` positions from an output operation
(i.e., there is a path with at most ``-start-1`` operations between 
the operation and an output), while a negative ``end`` value will 
select operations at least ``-end`` operations from an output.
If ``start`` is not specified (e.g., ``N[:end]``), then slicing will
start from the input operations (i.e., ``N[0:end]``).
If ``end`` is not specified (e.g., ``N[start:]``), then slicing will
select the operations from the start index to the current output 
operations of the network.


Network Composition
-------------------

Another potentially useful technique when specifying properties
is network composition. This enables networks to be stacked such
that the input to one model is the output of another. The
following code composes the networks ``Ni`` and ``No`` into ``N``,
such that inference with ``N`` is equivalent to inference with 
``Ni`` followed by the inference with ``No``:

.. code-block:: python

  Ni = Network("Ni")
  No = Network("Ni")
  N = No.compose(Ni)

  Forall(x, N(x) > 0)

One use case for composition is the use of generative models
as input constraints to restrict inputs to a learned model of the
input space, as introduced by `Toledo et al`_.

.. _Property Inference for Deep Neural Networks: https://arxiv.org/pdf/1904.13215.pdf
.. _Toledo et al: https://davidshriver.me/files/publications/ASE21-DFV.pdf
