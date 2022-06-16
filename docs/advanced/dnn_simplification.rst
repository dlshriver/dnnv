DNN Simplification
==================

In order to allow verifiers to be applied to a wider range of real world networks, DNNV provides tools to simplify networks to more standard forms, while preserving the behavioral semantics of the network.

Network simplification takes in an operation graph and applies a set of semantics preserving transformations to the operation graph to remove unsupported structures, or  to transform sequences of operations into a single more commonly supported operation.

An operation graph :math:`G_\mathcal{N} = \langle V_\mathcal{N}, E_\mathcal{N} \rangle` is a directed graph where nodes, :math:`v \in V_\mathcal{N}` represent operations, and edges :math:`e \in E_\mathcal{N}` represent inputs to those operations.
Simplification, :math:`\mathit{simplify}: \mathcal{G} \rightarrow \mathcal{G}`, aims to transform an operation graph :math:`G_\mathcal{N} \in \mathcal{G}`, to an equivalent DNN with more commonly supported structure, :math:`\mathit{simplify}(G_\mathcal{N}) = G_{\mathcal{N}'}`, such that the resulting DNN has the same behavior as the original :math:`\forall x. \mathcal{N}(x) = \mathcal{N}'(x)`, and the resulting DNN has more commonly supported structures, :math:`support(G_{\mathcal{N}'}) \geq support(G_\mathcal{N})`, where :math:`\mathit{support}: \mathcal{G} \rightarrow \mathbb{R}` is a measure for the likelihood that a verifier supports a structure.


Available Simplifications
-------------------------

Here we list some of the available simplifications provided by DNNV.
Currently all simplfiications are applied to a network, unless the simplification is marked as optional below.
Optional simplifications can be enabled with the ``DNNV_OPTIONAL_SIMPLIFICATIONS`` environment variable.
This variable accepts a colon separated list of optional simplifications.
For example, to include the optional ``ReluifyMaxPool`` simplification, set ``DNNV_OPTIONAL_SIMPLIFICATIONS=ReluifyMaxPool``.


BatchNormalization Simplification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BatchNormalization simplification removes BatchNormalization operations from a network by combining them with a preceeding Conv operation or Gemm operation. If no applicable preceeding layer exists, the batch normalization layer is converted into an equivalent Conv operation. This simplification can decrease the number of operations in the model and increase verifier support, since many verifiers do not support BatchNormalization operations.

Identity Removal
^^^^^^^^^^^^^^^^

DNNV removes many types of identity operations from DNN models, including explicit Identity operations, Concat operations with a single input, Flatten operations applied to flat tensors, and Relu operations applied to positive values. Such operations can occur in DNN models due to user error, or through automated processes, and their removal does not affect model behavior.

Convert Add
^^^^^^^^^^^

DNNV converts compatible instances of Add operations to an equivalent Gemm (generalized matrix multiplication) operation.

Convert MatMul Gemm
^^^^^^^^^^^^^^^^^^^

DNNV converts compatible instances of MatMul (matrix multiplication) operations to an equivalent Gemm (generalized matrix multiplication) operation. The Gemm operation generalizes the matrix multiplication and addition, and can simplify subsequent processing and analysis of the DNN.

Convert Reshape to Flatten
^^^^^^^^^^^^^^^^^^^^^^^^^^

DNNV converts the sequence of operations Shape, Gather, Unsqueeze, Concat, Reshape into an equivalent Flatten operation whenever possible. This replaces several, often unsupported operations, with a much more commonly supported operation.

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

Bundle Transpose
^^^^^^^^^^^^^^^^

DNNV can bundle Transpose operations followed by Flatten or Reshape operations with a successive Gemm operation. This effectively removes the Transpose operation from the network, since some verifiers do not support them.

Move Activations Backward
^^^^^^^^^^^^^^^^^^^^^^^^^

DNNV moves activation functions through reshaping operations to immediately succeed the most recent non-reshaping operation. This is possible since activation functions are element-wise operations. This transformation can simplify pattern matching in later analysis steps by reducing the number of possible patterns.

Reluify MaxPool
^^^^^^^^^^^^^^^

This is an optional simplification, which can be enabled by adding ``ReluifyMaxPool`` to the list of optional simplifiers. This simplification converts MaxPool operations into an equivalent set of Conv and Relu operations. We do this by encoding max ops as :math:`max(a, b) = relu(a - b) + relu(b) - relu(-b)`. This encoding is only a pairwise comparison, and so we cannot translate a single max pool operation into a single convolution operation, we must translate the max pool into a sequence of convolution and relu operations (using this encoding). In general, a MaxPool with a kernel size of :math:`n` will be converterted to a sequence of :math:`2*lg(n)` Conv operations, each followed by a Relu operation.

.. TODO: can we automatically generate this page? Maybe using docstrings?
