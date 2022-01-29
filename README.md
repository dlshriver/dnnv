# Deep Neural Network Verification

A framework for verification and analysis of deep neural networks. You can read an overview of DNNV in our CAV 2021 paper [*DNNV: A Framework for Deep Neural Network Verification*](https://arxiv.org/abs/2105.12841), or watch our presentation on [YouTube](https://youtu.be/GhXlONbvx1Y).

## Getting Started

For detailed instructions on installing and using DNNV, see our [documentation](https://dnnv.readthedocs.io/en/stable/).

### Installation

DNNV requires python >=3.7,<3.10, and has been tested on linux. To install the latest stable version run:

```bash
$ pip install dnnv
```

or

```bash
$ pip install git+https://github.com/dlshriver/DNNV.git@main
```

We recommend installing DNNV into a [python virtual environment](https://docs.python.org/3/tutorial/venv.html).

Install any of the supported verifiers ([Reluplex](https://github.com/guykatzz/ReluplexCav2017), [planet](https://github.com/progirep/planet), [MIPVerify.jl](https://github.com/vtjeng/MIPVerify.jl), [Neurify](https://github.com/tcwangshiqi-columbia/Neurify), [ERAN](https://github.com/eth-sri/eran), [BaB](https://github.com/oval-group/PLNN-verification), [marabou](https://github.com/NeuralNetworkVerification/Marabou), [nnenum](https://github.com/stanleybak/nnenum), [verinet](https://vas.doc.ic.ac.uk/software/neural/)):

```bash
$ dnnv_manage install reluplex planet mipverify neurify eran bab marabou nnenum verinet
```

*Several verifiers make use of the [Gurobi solver](https://www.gurobi.com/).* This should be installed automatically, but requires a license to be manually activated and available on the host machine. Academic licenses can be obtained for free from the [Gurobi website](https://user.gurobi.com/download/licenses/free-academic).

> After installing a verifier that requires Gurobi, the grbgetkey command can be found at `.venv/opt/gurobi912/linux64/bin/grbgetkey`.

#### Source Installation

First create and activate a python virtual environment.

```bash
$ python -m venv .venv
$ . .venv/bin/activate
```

Then run the following commands to clone DNNV and install it into the virtual environment:

```bash
$ git clone https://github.com/dlshriver/DNNV.git
$ cd DNNV
$ pip install .
```

Verifiers can then be installed using the `dnnv_manage` tool as described above.

**Make sure that the project environment is activated** when using dnnv or the dnnv_manage tools.

#### Docker Installation

We provide a docker image with DNNV and all non-Gurobi dependent verifiers. To obtain and use the latest pre-built image of the main branch, run:

```bash
$ docker pull dlshriver/dnnv:latest
$ docker run --rm -it dlshriver/dnnv:latest
(.venv) dnnv@hostname:~$ dnnv -h
```

The latest version of the develop branch is available as `dlshriver/dnnv:develop`, and tagged releases are available as `dlshriver/dnnv:vX.X.X` where `vX.X.X` is the desired version number.

The docker image can also be built using the provided Dockerfile. The provided build file will install DNNV with all of the verifiers that do not require Gurobi. To build and run the docker image, run:

```bash
$ docker build . -t dlshriver/dnnv
$ docker run --rm -it dlshriver/dnnv
(.venv) dnnv@hostname:~$ dnnv -h
```

### Usage

Properties are specified in our Python-embedded DSL, [DNNP](https://dnnv.readthedocs.io/en/latest/usage/specifying_properties.html). A property specification can import python modules, and define variables. The only required component is the property expression, which must appear at the end of the file. An example of a local robustness property is shown below.

```python
from dnnv.properties import *

N = Network("N")
x = Image("path/to/image")
epsilon = Parameter("epsilon", float, default=1.0)

Forall(
    x_,
    Implies(
        ((x - epsilon) < x_ < (x + epsilon)),
        argmax(N(x_)) == argmax(N(x))),
    ),
)
```

To check whether property holds for some network using the ERAN verifier, run:

```bash
$ dnnv property.dnnp --network N network.onnx --eran
```

Additionally, if the property defines parameters, using the `Parameter` keyword, they can be specified on the command line using the option `--prop.PARAMETER_NAME`, where `PARAMETER_NAME` is the name of the parameter. For the property defined above, a value for `epsilon` can be provided with a command line option as follows:

```bash
$ dnnv property.dnnp --network N network.onnx --eran --prop.epsilon=2.0
```

To save any counter-example found by the verifier, use the option `--save-violation /path/to/array.npy` when running DNNV. This will save any violation found as a numpy array at the path specified, which is useful for viewing counter-examples to properties and enables additional debugging and analysis later.

### Example Problems

We have made several DNN verification benchmarks available in DNNP+ONNX format in [dlshriver/dnnv-benchmarks](https://github.com/dlshriver/dnnv-benchmarks).
This repo includes the [ACAS Xu](https://github.com/dlshriver/dnnv-benchmarks/tree/main/benchmarks/ACAS_Xu) benchmark, ready to run with DNNV!

## Acknowledgements

This material is based in part upon work supported by the National Science Foundation under grant number 1900676 and 2019239.
