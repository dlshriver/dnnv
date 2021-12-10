# Deep Neural Network Verification

A framework for verification and analysis of deep neural networks. You can read an overview of DNNV in our CAV 2021 paper [*DNNV: A Framework for Deep Neural Network Verification*](https://arxiv.org/abs/2105.12841), or watch our presentation on [YouTube](https://youtu.be/GhXlONbvx1Y).

## Getting Started

For detailed instructions on installing and using DNNV, see our [documentation](https://dnnv.readthedocs.io/en/stable/).

### Installation

DNNV requires python3.7 or python3.8, and has only been tested on linux. To install the latest stable version run:

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

**May 30, 2021**: The current version of nnenum can return errors for some problems, as reported [here](https://github.com/stanleybak/nnenum/issues/3), which will result in an `NnenumError(result:error)` result from DNNV. This can sometimes be avoided with the option `--nnenum.num_processes=1`, but this is not a general fix for the issue.

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

#### Full Installation Script

DNNV, with all supported verifiers can be installed using a provided installation script. We have tested this script on a fresh Ubuntu 20.04 system. **WARNING: This will install system packages. We recommend this only be run in a VM.**

```
$ wget https://raw.githubusercontent.com/dlshriver/DNNV/CAV2021/scripts/install_artifact.sh
$ chmod u+x install_artifact.sh
$ ./install_artifact.sh
```

### Usage

Properties are specified in our property DSL, extended from Python. A property specification can import python modules, and define variables. The only required component is the property expression, which must appear at the end of the file. An example of a local robustness property is shown below.

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
$ dnnv property.prop --network N network.onnx --eran
```

Additionally, if the property defines parameters, using the `Parameter` keyword, they can be specified on the command line using the option `--prop.PARAMETER_NAME`, where `PARAMETER_NAME` is the name of the parameter. For the property defined above, a value for `epsilon` can be provided with a command line option as follows:

```bash
$ dnnv property.prop --network N network.onnx --eran --prop.epsilon=2.0
```

To save any counter-example found by the verifier, use the option `--save-violation /path/to/array.npy` when running DNNV. This will save any violation found as a numpy array at the path specified, which is useful for viewing counter-examples to properties and enables additional debugging and analysis later.

### Examples

A set of example networks and properties that can be run with DNNV are available [here](http://cs.virginia.edu/~dls2fc/eranmnist_benchmark.tar.gz).

We also provide the [ACAS Xu benchmark](http://cs.virginia.edu/~dls2fc/acasxu_benchmark.tar.gz) in DNNP and ONNX format, along with a run script to run DNNV on all of the problems in the benchmark.

## Acknowledgements

This material is based in part upon work supported by the National Science Foundation under grant number 1900676 and 2019239.
