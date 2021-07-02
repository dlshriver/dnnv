# Deep Neural Network Verification

A framework for verification and analysis of deep neural networks. You can read an overview of DNNV in our CAV 2021 paper [*DNNV: A Framework for Deep Neural Network Verification*](https://arxiv.org/abs/2105.12841), or watch our presentation on [YouTube](https://youtu.be/GhXlONbvx1Y).

## Getting Started

For more detailed instructions, see our [documentation](https://dnnv.readthedocs.io/en/stable/).

### Installation

Clone this repository:

```bash
$ git clone https://github.com/dlshriver/DNNV.git
```

Create a python virtual environment for this project:

```bash
$ ./manage.sh init
```

To activate the virtual environment and set environment variables correctly for tools installed using the provided `manage.sh` script, run:

```bash
$ . .env.d/openenv.sh
```

Install any of the supported verifiers ([Reluplex](https://github.com/guykatzz/ReluplexCav2017), [planet](https://github.com/progirep/planet), [MIPVerify.jl](https://github.com/vtjeng/MIPVerify.jl), [Neurify](https://github.com/tcwangshiqi-columbia/Neurify), [ERAN](https://github.com/eth-sri/eran), [PLNN](https://github.com/oval-group/PLNN-verification), [marabou](https://github.com/NeuralNetworkVerification/Marabou), [nnenum](https://github.com/stanleybak/nnenum), [verinet](https://vas.doc.ic.ac.uk/software/neural/)):

```bash
$ ./manage.sh install reluplex planet mipverify neurify eran plnn marabou nnenum verinet
```

**Make sure that the project environment is activated** when installing verifiers with the `manage.sh` script. Otherwise some tools may not install correctly.

Additionally, several verifiers make use of the [Gurobi solver](https://www.gurobi.com/). This should be installed automatically, but requires a license to be manually activated and available on the host machine. Academic licenses can be obtained for free from the [Gurobi website](https://user.gurobi.com/download/licenses/free-academic).

*April 26, 2021:* The Marabou installation script may fail if boost is not installed. We use the original Marabou installation scripts which attempt to download boost if it cannot be found, but we have noticed at this time that this download can fail. To avoid this, try installing boost with `sudo apt install libboost1.71-all-dev` before installing marabou using `./manage.sh`.

Finally, planet has several additional requirements that currently must be installed separately before installation with `./manage.sh`: libglpk-dev, qt5-qmake, valgrind, libltdl-dev, protobuf-compiler.

#### Docker Installation

DNNV can also be built using the provided Docker build scripts. The provided build file will install DNNV with all of the verifiers that do not require Gurobi. To build and run the docker image, run:

```
$ docker build . -t dlshriver/dnnv
$ docker run --rm -it dlshriver/dnnv
(.venv) dnnv@hostname:~$ python -m dnnv -h
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
$ python -m dnnv property.prop --network N network.onnx --eran
```

Additionally, if the property defines parameters, using the `Parameter` keyword, they can be specified on the command line using the option `--prop.PARAMETER_NAME`, where `PARAMETER_NAME` is the name of the parameter. For the property defined above, a value for `epsilon` can be provided with a command line option as follows:

```bash
$ python -m dnnv property.prop --network N network.onnx --eran --prop.epsilon=2.0
```

### Examples

A set of example networks and properties that can be run with DNNV are available [here](http://cs.virginia.edu/~dls2fc/eranmnist_benchmark.tar.gz).

We also provide the [ACAS Xu benchmark](http://cs.virginia.edu/~dls2fc/acasxu_benchmark.tar.gz) in DNNP and ONNX format, along with a run script to run DNNV on all of the problems in the benchmark.

## Acknowledgements

This material is based in part upon work supported by the National Science Foundation under grant number 1900676 and 2019239.
