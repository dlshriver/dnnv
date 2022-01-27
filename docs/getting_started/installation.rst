.. _installation:

Installation
============

We provide several installation options for DNNV.
We recommend using pip for general usage.

With Pip
--------

Requirements:

* Python 3.7, 3.8, or 3.9

To install the latest released version of DNNV using pip, run::

  pip install dnnv

To install the latest and greatest version of DNNV using pip, run::

  pip install git+https://github.com/dlshriver/dnnv.git@develop


With Docker
-----------

Requirements:

* `Docker`_

To download and use the latest released version of DNNV in a docker image, run::

  docker pull dlshriver/dnnv:latest
  docker run --rm -it dlshriver/dnnv:latest

To use a specific version of DNNV, run::

  docker pull dlshriver/dnnv:vX.X.X
  docker run --rm -it dlshriver/dnnv:vX.X.X

Where ``X.X.X`` is replaced by the desired version of DNNV.

To use the latest development version of DNNV in a docker image, 
specify the ``develop`` tag::

  docker pull dlshriver/dnnv:develop
  docker run --rm -it dlshriver/dnnv:develop

The development version does not come with any verifiers pre-installed.
These can be installed using the ``dnnv_manage``, 
either in a custom Dockerfile or through an interactive prompt.


From Source
-----------

Requirements:

* Python 3.7 or 3.8
* Git

Installing DNNV from source is primarily recommended for 
development of DNNV itself. This requires Python 3.7 or above,
as well as the venv module, which may need to be installed
separately (e.g., ``sudo apt-get install python3-venv``).

To clone the source code, run::

  git clone https://github.com/dlshriver/dnnv.git
  cd DNNV

To create a python virtual environment, and install required
pacakges for this project, run::

  python -m venv .venv
  . .venv/bin/activate
  pip install --upgrade pip flit
  flit install -s


Verifier Installation
---------------------

After installing DNNV, any of the supported verifiers can be
installed using the ``install`` command to the ``dnnv_manage``
tool, followed by the name of the verifier.
For example, to install the Reluplex verifier, run::

  dnnv_manage install reluplex

DNNV supports the following verifiers:

* `Reluplex`_
* `planet`_
* `BaB`_
* `BaBSB`_
* `MIPVerify`_
* `Neurify`_
* `ERAN`_ (deepzono, deeppoly, refinezono, refinepoly)
* `marabou`_
* `nnenum`_
* `VeriNet`_

.. _Reluplex: https://github.com/guykatzz/ReluplexCav2017
.. _planet: https://github.com/progirep/planet
.. _BaB: https://github.com/oval-group/PLNN-verification
.. _BaBSB: https://github.com/oval-group/PLNN-verification
.. _MIPVerify: https://github.com/vtjeng/MIPVerify.jl
.. _Neurify: https://github.com/tcwangshiqi-columbia/Neurify
.. _ERAN: https://github.com/eth-sri/eran
.. _marabou: https://github.com/NeuralNetworkVerification/Marabou
.. _nnenum: https://github.com/stanleybak/nnenum
.. _verinet: https://vas.doc.ic.ac.uk/software/neural/

.. _Docker: https://www.docker.com/products/docker-desktop
