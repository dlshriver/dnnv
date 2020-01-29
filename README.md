# Deep Neural Network Verification Toolbox

Tools for verification and analysis of deep neural networks. 
Watch our [demo video](https://youtu.be/M5G_OWfCF2o) for a quick description!

## Getting Started

For more detailed instructions, see our [documentation](https://dnnv.readthedocs.io/en/latest/).

### Installation

Clone this network:

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

Install any of the supported verifiers ([Reluplex](https://github.com/guykatzz/ReluplexCav2017), [planet](https://github.com/progirep/planet), [MIPVerify.jl](https://github.com/vtjeng/MIPVerify.jl), [Neurify](https://github.com/tcwangshiqi-columbia/Neurify), [ERAN](https://github.com/eth-sri/eran)):

```bash
$ ./manage.sh install reluplex planet mipverify neurify eran
```

**Make sure that the project environment is activated** when installing verifiers with the `manage.sh` script. Otherwise some tools may not install correctly.

Additionally, several verifiers make use of the [Gruobi solver](https://www.gurobi.com/). This should be installed automatically, but requires a license to be manually activated and available on the host machine. Academic licenses can be obtained for free from the [Gurobi website](https://user.gurobi.com/download/licenses/free-academic).

Finally, planet has several additional requirements that currently must be installed separately before installation with `./manage.sh`: libglpk-dev, qt5-qmake, valgrind, libltdl-dev, protobuf-compiler.

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
$ python -m dnnv network.onnx property.prop --eran
```

Additionally, if the property defines parameters, using the `Parameter` keyword, they can be specified on the command line using the option `--prop.PARAMETER_NAME`, where `PARAMETER_NAME` is the name of the parameter. For the property defined above, a value for `epsilon` can be provided with a command line option as follows:

```bash
$ python -m dnnv network.onnx property.prop --eran --prop.epsilon=2.0
```

A set of example networks and properties that can be run with DNNV are available [here](http://cs.virginia.edu/~dls2fc/eran_benchmark.tar.gz).

## Contributing
