DNN Simplification
==================

In order to allow verifiers to be applied to a wider range of real world networks, DNNV provides tools to simplify networks to more standard forms, while preserving the behavioral semantics of the network.

Network simplification takes in an operation graph and applies a set of semantics preserving transformations to the operation graph to remove unsupported structures, or  to transform sequences of operations into a single more commonly supported operation.

An operation graph :math:`G_\mathcal{N} = \langle V_\mathcal{N}, E_\mathcal{N} \rangle` is a directed graph where nodes, :math:`v \in V_\mathcal{N}` represent operations, and edges :math:`e \in E_\mathcal{N}` represent inputs to those operations.
Simplification, :math:`\mathit{simplify}: \mathcal{G} \rightarrow \mathcal{G}`, aims to transform an operation graph :math:`G_\mathcal{N} \in \mathcal{G}`, to an equivalent DNN with more commonly supported structure, :math:`\mathit{simplify}(G_\mathcal{N}) = G_{\mathcal{N}'}`, such that the resulting DNN has the same behavior as the original :math:`\forall x. \mathcal{N}(x) = \mathcal{N}'(x)`, and the resulting DNN has more commonly supported structures, :math:`support(G_{\mathcal{N}'}) \geq support(G_\mathcal{N})`, where :math:`\mathit{support}: \mathcal{G} \rightarrow \mathbb{R}` is a measure for the likelihood that a verifier supports a structure.


Available Simplifications
-------------------------

Here we list some of the available simplifications provided by DNNV.


Batch Normalization Simplification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Batch normalization simplification* removes batch normalization operations from a network by combining them with a preceeding convolution operation or generalized matrix multiplication (GEMM) operation.
This is possible since batch normalization, convolution, and GEMM operations are all affine operations.
If no applicable preceeding layer exists, the batch normalization layer is converted into an equivalent convolution operation.
