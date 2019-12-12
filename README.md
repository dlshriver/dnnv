# Deep Neural Network Verification Toolbox

Tools for verification and analysis of deep neural networks.

## Getting Started

### Installation


### Usage

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

## Contributing
