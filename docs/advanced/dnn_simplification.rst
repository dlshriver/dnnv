DNN Simplification
==================

In order to allow verifiers to be applied to a wider range of real world networks, DNNV provides tools to simplify networks to more standard forms, while preserving the behavioral semantics of the network.

Network simplification takes in an operation graph and applies a set of semantics preserving transformations to the operation graph to remove unsupported structures, or  to transform sequences of operations into a single more commonly supported operation.

An operation graph :math:`G_\mathcal{N} = \langle V_\mathcal{N}, E_\mathcal{N} \rangle` is a directed graph where nodes, :math:`v \in V_\mathcal{N}` represent operations, and edges :math:`e \in E_\mathcal{N}` represent inputs to those operations.
Simplification, :math:`\mathit{simplify}: \mathcal{G} \rightarrow \mathcal{G}`, aims to transform an operation graph :math:`G_\mathcal{N} \in \mathcal{G}`, to an equivalent DNN with more commonly supported structure, :math:`\mathit{simplify}(G_\mathcal{N}) = G_{\mathcal{N}'}`, such that the resulting DNN has the same behavior as the original :math:`\forall x. \mathcal{N}(x) = \mathcal{N}'(x)`, and the resulting DNN has more commonly supported structures, :math:`support(G_{\mathcal{N}'}) \geq support(G_\mathcal{N})`, where :math:`\mathit{support}: \mathcal{G} \rightarrow \mathbb{R}` is a measure for the likelihood that a verifier supports a structure.


Available Simplifications
-------------------------

Here we list some of the available simplifications provided by DNNV.


BatchNormalization Simplification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BatchNormalization simplification removes BatchNormalization operations from a network by combining them with a preceeding Conv operation or Gemm operation. If no applicable preceeding layer exists, the batch normalization layer is converted into an equivalent Conv operation. This simplification can decrease the number of operations in the model and increase verifier support, since many verifiers do not support BatchNormalization operations.

Identity Removal
^^^^^^^^^^^^^^^^

DNNV removes many types of identity operations from DNN models, including explicit Identity operations, Concat operations with a single input, and Flatten operations applied to flat tensors. Such operations can occur in DNN models due to user error, or through automated processes, and their removal does not affect model behavior.

Convert MatMul followed by Add to Gemm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DNNV converts instances of MatMul (matrix multiplication) operations, followed immediately by Add operations to an equivalent Gemm (generalized matrix multiplication) operation. The Gemm operation generalizes the matrix multiplication and addition, and can simplify subsequent processing and analysis of the DNN.

Combine Consecutive Gemm
^^^^^^^^^^^^^^^^^^^^^^^^

DNNV combines two consecutive Gemm operations into a single equivalent Gemm operation, reducing the number of operations in the DNN.

Combine Consecutive Conv
^^^^^^^^^^^^^^^^^^^^^^^^

In special cases, DNNV can combine consecutive Conv (convolution) operations into a single equivalent Conv operation, reducing the number of operations in the DNN.
Currently, DNNV can combine Conv operations when the first Conv uses a diagonal 1 by 1 kernel with a stride of 1 and no zero padding, and the second Conv has no zero padding. This case can occur after converting a normalization layer (such as BatchNormalization) to a Conv operation.

Bundle Pad
^^^^^^^^^^

DNNV can bundle explicit Pad operations with an immediately succeeding Conv or MaxPool operation. This both simplifies the DNN model, and increases support, since many verifiers do not support explicit Pad operations (but can support padding as part of a Conv or MaxPool operation).

Move Activations Backward
^^^^^^^^^^^^^^^^^^^^^^^^^

DNNV moves activation functions through reshaping operations to immediately succeed the most recent non-reshaping operation. This is possible since activation functions are element-wise operations. This transformation can simplify pattern matching in later analysis steps by reducing the number of possible patterns.

