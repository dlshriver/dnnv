Usage
=====

DNNV can be used to run verification tools from the command line.
If DNNV is not yet installed, see our
:doc:`Installation guide <./installation>`
for more information.

DNNV Options
------------

DNNV can be run from the command line. Specifying the ``-h``
option will list the available options:

.. code-block:: console

  $ dnnv -h
  usage: dnnv [-h] [-V] [--seed SEED] [-v | -q] [-N NAME NETWORK] 
            [--save-violation PATH] [--vnnlib] [--bab]
            [--bab.reluify_maxpools RELUIFY_MAXPOOLS]
            [--bab.smart_branching SMART_BRANCHING] [--eran]
            [--eran.domain {deepzono,deeppoly,refinezono,refinepoly}]
            [--eran.timeout_lp TIMEOUT_LP] [--eran.timeout_milp TIMEOUT_MILP]
            [--eran.use_area_heuristic USE_AREA_HEURISTIC] [--marabou]
            [--mipverify] [--neurify] [--neurify.max_depth MAX_DEPTH]
            [--neurify.max_thread MAX_THREAD] [--nnenum]
            [--nnenum.num_processes NUM_PROCESSES] [--planet] [--reluplex]
            [--verinet] [--verinet.max_proc MAX_PROC]
            [--verinet.no_split NO_SPLIT]
            property

  dnnv - deep neural network verification

  positional arguments:
    property

  optional arguments:
    -h, --help            show this help message and exit
    -V, --version         show program's version number and exit
    --seed SEED           the random seed to use
    -v, --verbose         show messages with finer-grained information
    -q, --quiet           suppress non-essential messages
    -N, --network NAME NETWORK
    --save-violation PATH
                          the path to save a found violation
    --vnnlib              use the vnnlib property format
    --convert

  verifiers:
    --bab
    --eran
    --marabou
    --mipverify
    --neurify
    --nnenum
    --planet
    --reluplex
    --verinet

  bab parameters:
    --bab.reluify_maxpools RELUIFY_MAXPOOLS
    --bab.smart_branching SMART_BRANCHING
  
  convert parameters:
    --convert.to {vnnlib,rlv,nnet}
    --convert.dest DEST
    --convert.extended-vnnlib EXTENDED-VNNLIB

  eran parameters:
    --eran.domain {deepzono,deeppoly,refinezono,refinepoly}
                          The abstract domain to use.
    --eran.timeout_lp TIMEOUT_LP
                          Time limit for the LP solver.
    --eran.timeout_milp TIMEOUT_MILP
                          Time limit for the MILP solver.
    --eran.use_area_heuristic USE_AREA_HEURISTIC
                          Whether or not to use the ERAN area heuristic.

  neurify parameters:
    --neurify.max_depth MAX_DEPTH
                          Maximum search depth for neurify.
    --neurify.max_thread MAX_THREAD
                          Maximum number of threads to use.

  nnenum parameters:
    --nnenum.num_processes NUM_PROCESSES
                          Maximum number of processes to use.

  verinet parameters:
    --verinet.max_proc MAX_PROC
                          Maximum number of processes to use.
    --verinet.no_split NO_SPLIT
                          Whether or not to do splitting.

Only a single verifier can be specified.
If a network is specified without a property 
(i.e., ``dnnv --network N /path/to/model.onnx``),
then DNNV will print a brief description of the network.
This can be useful for understanding the structure of the network
to be verified.


Running DNNV
------------

DNNV can be used to check whether a given property holds
for a network. It accepts networks specified in the ONNX format,
and properties specified in our property DSL, DNNP, (explained
in more detail :doc:`here <../dnnp/introduction>`).
Networks can be converted to ONNX format by using native export
utilities, such as ``torch.onnx.export`` in `PyTorch`_, or by
using an external conversion tool, such as `MMDNN`_.

We provide several neural network verification benchmarks as example problems,
`available here <https://github.com/dlshriver/dnnv-benchmarks>`_.

One of these benchmarks, 
`ERAN-MNIST <https://github.com/dlshriver/dnnv-benchmarks/tree/main/benchmarks/ERAN-MNIST>`_, 
is from the evaluation of the `ERAN`_ verifier,
and have been converted to the DNNP and ONNX formats required by DNNV.

To check a property for a network, using the `ERAN`_ verifier, DNNV
can be run as::

  dnnv --eran --network N onnx/pyt/ffnnRELU__Point_6_500.onnx properties/pyt/property_7.py

This will check whether ``properties/pyt/property_7.py``, a local robustness
property, holds for the network ``ffnnRELU__Point_6_500.onnx``, a 6 layer,
3000 neuron fully connected network.

DNNV will first report a basic description of the network, followed
by the property to be verified. It will then run the specified
verifier and report the verification result and the total time to
translate and verify the property. The output of the property check
above should resemble the output below:

.. code-block:: console

  $ dnnv --eran --network N onnx/pyt/ffnnRELU__Point_6_500.onnx properties/pyt/property_7.py
  Verifying property:
  Forall(x_, ((([[[-0.008 -0.008 ... -0.008 -0.008] [-0.008 -0.008 ... -0.008 -0.008] ... [-0.008 -0.008 ... -0.008 -0.008] [-0.008 -0.008 ... -0.008 -0.008]]] < (0.1307 + (0.3081 * x_))) & ((0.1307 + (0.3081 * x_)) < [[[0.008 0.008 ... 0.008 0.008] [0.008 0.008 ... 0.008 0.008] ... [0.008 0.008 ... 0.008 0.008] [0.008 0.008 ... 0.008 0.008]]]) & (0 < (0.1307 + (0.3081 * x_))) & ((0.1307 + (0.3081 * x_)) < 1)) ==> (numpy.argmax(N(x_)) == 9)))

  Verifying Networks:
  N:
  Input_0                         : Input([ 1  1 28 28], dtype=float32)
  Transpose_0                     : Transpose(Input_0, permutation=[0 2 3 1])
  Reshape_0                       : Reshape(Transpose_0, [ -1 784])
  Gemm_0                          : Gemm(Reshape_0, ndarray(shape=(500, 784)), ndarray(shape=(500,)), transpose_a=0, transpose_b=1, alpha=1.000000, beta=1.000000)
  Relu_0                          : Relu(Gemm_0)
  Gemm_1                          : Gemm(Relu_0, ndarray(shape=(500, 500)), ndarray(shape=(500,)), transpose_a=0, transpose_b=1, alpha=1.000000, beta=1.000000)
  Relu_1                          : Relu(Gemm_1)
  Gemm_2                          : Gemm(Relu_1, ndarray(shape=(500, 500)), ndarray(shape=(500,)), transpose_a=0, transpose_b=1, alpha=1.000000, beta=1.000000)
  Relu_2                          : Relu(Gemm_2)
  Gemm_3                          : Gemm(Relu_2, ndarray(shape=(500, 500)), ndarray(shape=(500,)), transpose_a=0, transpose_b=1, alpha=1.000000, beta=1.000000)
  Relu_3                          : Relu(Gemm_3)
  Gemm_4                          : Gemm(Relu_3, ndarray(shape=(500, 500)), ndarray(shape=(500,)), transpose_a=0, transpose_b=1, alpha=1.000000, beta=1.000000)
  Relu_4                          : Relu(Gemm_4)
  Gemm_5                          : Gemm(Relu_4, ndarray(shape=(500, 500)), ndarray(shape=(500,)), transpose_a=0, transpose_b=1, alpha=1.000000, beta=1.000000)
  Relu_5                          : Relu(Gemm_5)
  Gemm_6                          : Gemm(Relu_5, ndarray(shape=(10, 500)), ndarray(shape=(10,)), transpose_a=0, transpose_b=1, alpha=1.000000, beta=1.000000)

  dnnv.verifiers.eran
    result: unsat
    time: 61.0565

Another common option is the ``--save-violation /path/to/array.npy`` which 
will save any violation found by a verifier as a `numpy`_ array at the path
specified. This can be useful for viewing counter-examples to properties
and enables performing additional debugging and analysis later.


.. _ERAN: https://github.com/eth-sri/eran
.. _MMDNN: https://github.com/microsoft/MMdnn
.. _numpy: https://numpy.org/
.. _PyTorch: https://pytorch.org/
