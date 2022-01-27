Adding a New DNN Simplifier
===========================

*TODO.* Sorry, this page is still in development.
An example simplification can be seen
`here <https://github.com/dlshriver/dnnv/blob/develop/dnnv/nn/transformers/simplifiers/convert_matmul_to_gemm.py>`_.

In general a DNN simplification will subclass the :py:class:`Simplifier` base class.
This class implements a visitor pattern for an operation graph.
In general, a simplifier implementation will check if an operation sub-graph matches 
some pattern, and, if so, will replace it with a semantically equivalent sub-graph.
