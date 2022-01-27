.. DNNV documentation master file, created by
   sphinx-quickstart on Fri Dec 13 15:52:14 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DNNV
====

DNNV is a framework for verifying deep neural networks (DNN).
DNN verification takes in a neural network, and a property over
that network, and checks whether the property is true or false.
DNNV standardizes the network and property input formats to enable
multiple verification tools to run on a single network and property.
This facilitates both verifier comparison, and artifact re-use.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :glob:

   getting_started/installation
   getting_started/*

.. toctree::
   :maxdepth: 1
   :caption: DNNP
   :glob:

   dnnp/*

.. toctree::
   :maxdepth: 1
   :caption: Advanced
   :glob:

   advanced/*

.. toctree::
   :maxdepth: 1
   :caption: Developers
   :glob:

   developers/*

.. toctree::
   :maxdepth: 1
   :caption: Publications
   
   publications


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
