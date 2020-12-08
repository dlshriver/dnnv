.. _installation:

Installation
============

We currently recommend installing DNNV from source, since it
requires less manual effort to correctly set up environment
variables.

From Source
-----------

Requirements:

* Python 3.7+
.. * make
.. * gcc/g++
.. * gfortran
.. * lapack

Currently, the easiest way to use DNNV is to clone the
github repository, and use the provided ``manage.sh`` script
to initiallize a python virtual environment and install
verifiers. This requires Python 3.7 or above to be installed,
as well as the venv module, which may need to be installed
separately (e.g., ``sudo apt-get install python3-venv``).

To clone the source code, run::

  git clone https://github.com/dlshriver/DNNV.git

To create a python virtual environment, and install required
pacakges for this project, run::

  ./manage.sh init

Additionally, we provide a script to activate the virtual
environment and set up environment variables required to find
verification tools installed using ``manage.sh``. To activate
the environment, run::

  . .env.d/openenv.sh

Finally, any of the supported verifiers can be installed
using the ``install`` command to the ``manage.sh`` script, followed
by the name of the verifier.
For example, to install the Reluplex verifier, run::

  ./manage.sh install reluplex

DNNV supports the following verifiers:

* `Reluplex`_
* `planet`_
* `BaB`_
* `MIPVerify.jl`_
* `Neurify`_
* `ERAN`_ (deepzono, deeppoly, refinezono, refinepoly)
* `PLNN`_ (bab, babsb)
* `marabou`_
* `nnenum`_
* `VeriNet`_

.. _Reluplex: https://github.com/guykatzz/ReluplexCav2017
.. _planet: https://github.com/progirep/planet
.. _BaB: https://github.com/oval-group/PLNN-verification
.. _MIPVerify.jl: https://github.com/vtjeng/MIPVerify.jl
.. _Neurify: https://github.com/tcwangshiqi-columbia/Neurify
.. _ERAN: https://github.com/eth-sri/eran
.. _PLNN: https://github.com/oval-group/PLNN-verification
.. _marabou: https://github.com/NeuralNetworkVerification/Marabou
.. _nnenum: https://github.com/stanleybak/nnenum
.. _verinet: https://vas.doc.ic.ac.uk/software/neural/

DNNV can also be installed into an existing virtual environment.
To do so, we require the module ``flit`` be installed.
To install DNNV, ensure that the desired virtual environment is
activated, and then run::

  flit install

This method requires the user to manually configure environment
variables to point to installed verification tools. This can still
be done with the ``.env.d/openenv.sh`` script if tools were installed
with ``manage.sh``.

From Pip
--------

DNNV can also be installed using pip.
Currently, installing with pip does not provide access to the
``manage.sh`` script, so verification tools must be installed
separately. Additionally, they must be accessible on the ``PATH``.
To install DNNV using pip, run::

  pip install dnnv
