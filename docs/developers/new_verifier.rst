Adding a New Verifier
=====================

*TODO.* Sorry, this page is still in development.
As an example, the implementation for the planet verifier can be seen
`here <https://github.com/dlshriver/dnnv/tree/main/dnnv/verifiers/planet>`_.

In general a verifier will subclass the :py:class:`Verifier` base class 
and implement at least the methods 
``build_inputs(self, prop)``
and ``parse_results(self, prop, results)``.
When verifying a property, the base verifier implementation simplifies the network,
and reduces the property to a set of properties with 
hyper-rectangles in the input space
and a halfspace polytope in the output space.
Each property has methods to add a suffix to the network for reduction to robustness properties, 
as well as a method to add a prefix that modifies the input domain
to be a unit hyper cube.
