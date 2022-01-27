Adding a New Reduction
======================

*TODO.* Sorry, this page is still in development.
An example reduction can be seen
`here <https://github.com/dlshriver/dnnv/tree/develop/dnnv/verifiers/common/reductions/iopolytope>`_.

In general a reduction will subclass the :py:class:`Reduction` base class 
and implement the method
``reduce_property(self, phi: Expression) -> Iterator[Property]``, 
which takes in a property expression and returns an iterator of :py:class:`Property` objects.
A reduction will likely also require a custom :py:class:`Property` type
which must implement
``validate_counter_example(self, cex: Any) -> Tuple[bool, Optional[str]]``.
