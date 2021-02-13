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

  $ python -m dnnv -h
  usage: dnnv [-h] [-V] [--seed SEED] [-v | -q] [-N NAME NETWORK] [--vnnlib]
            [--bab] [--bab.reluify_maxpools RELUIFY_MAXPOOLS]
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
    --vnnlib              use the vnnlib property format

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

DNNV requires a network and property be specified, and accepts
an optional list of verifiers to be run to check the network and
property.
Currently, verifiers are run sequentially, in the order that they
are specified on the command line.


Running DNNV
------------

DNNV can be used to check whether a given property holds
for a network. It accepts networks specified in the ONNX format,
and properties specified in our property DSL (which is explained
in more detail :doc:`here <../usage/specifying_properties>`).
Networks can be converted to ONNX format by using native export
utilities, such as ``torch.onnx.export`` in `PyTorch`_, or by
using an external conversion tool, such as `MMDNN`_.

We provide several example networks and properties,
`available here <http://cs.virginia.edu/~dls2fc/eran_benchmark.tar.gz>`_.
These networks and properties are from the benchmark of the `ERAN`_ verifier,
and are converted to the ONNX and property DSL formats required by DNNV.

To check a property for a network, using the `ERAN`_ verifier, DNNV
can be run as::

  python -m dnnv --eran --network N onnx/pyt/ffnnRELU__Point_6_500.onnx properties/pyt_property_7.py

This will check whether ``pyt_property_7``---a local robustness
property---holds for the network ``ffnnRELU__Point_6_500.onnx``---a 6 layer,
3000 neuron fully connected network.

DNNV will first report a basic description of the network, followed
by the property to be verified. It will then run each of the specified
verifiers and report the verification result and the total time to
translate and verify the property. The output of the property check
above should resemble the output below:

.. code-block:: console

  $ python -m dnnv --eran --network N onnx/pyt/ffnnRELU__Point_6_500.onnx properties/pyt_property_7.py
  Input_0                         : Input([ 1  1 28 28], dtype=float32)
  Reshape_0                       : Reshape(Input_0, ndarray_0)
  Gemm_0                          : Gemm(Reshape_0, ndarray_1, ndarray_2)
  Reshape_1                       : Reshape(Gemm_0, ndarray_3)
  Transpose_0                     : Transpose(Reshape_1, permutation=[0 2 3 1])
  Reshape_2                       : Reshape(Transpose_0, ndarray_4)
  Gemm_1                          : Gemm(Reshape_2, ndarray_5, ndarray_6)
  Relu_0                          : Relu(Gemm_1)
  Gemm_2                          : Gemm(Relu_0, ndarray_7, ndarray_8)
  Relu_1                          : Relu(Gemm_2)
  Gemm_3                          : Gemm(Relu_1, ndarray_9, ndarray_10)
  Relu_2                          : Relu(Gemm_3)
  Gemm_4                          : Gemm(Relu_2, ndarray_11, ndarray_12)
  Relu_3                          : Relu(Gemm_4)
  Gemm_5                          : Gemm(Relu_3, ndarray_13, ndarray_14)
  Relu_4                          : Relu(Gemm_5)
  Gemm_6                          : Gemm(Relu_4, ndarray_15, ndarray_16)
  Relu_5                          : Relu(Gemm_6)
  Gemm_7                          : Gemm(Relu_5, ndarray_17, ndarray_18)
  Verifying property:
  Forall(x_, (((x_ < 3.2457*Image("properties/image7.npy")-0.41637) & (3.2457*Image("properties/image7.npy")-0.432056 < x_)) ==> (numpy.argmax(N[4:](x_)) == numpy.argmax(N[4:](3.2457*Image("properties/image7.npy")-0.424213)))))

  dnnv.verifiers.eran
    result: unsat
    time: 2.4884


.. _MMDNN: https://github.com/microsoft/MMdnn
.. _PyTorch: https://pytorch.org/
.. _ERAN: https://github.com/eth-sri/eran
