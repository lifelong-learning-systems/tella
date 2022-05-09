Installing tella
================

Requirements
----------------

* Python 3.7 or greater

Install
-------------
#. Create a conda or virtual environment and activate it

#. We recommend that you update pip and wheel in your environment::

    pip install -U pip wheel

#. Install the tella package and its dependencies::

    pip install tella


There are optional packages that can also be installed if you want
to use atari environments or run the unit tests.
They can be installed using pip's extras syntax::

    pip install tella[atari]


For Developers
----------------
To install tella in editable mode with our development requirements,
clone the repository and then install tella with the dev extras::

    pip install -e ".[dev]"


To run unit tests::

    pytest

For running in conda environment::

    python -m pytest


To check for PEP8 compliance::

    black --check tella


To autoformat for PEP8 compliance::

    black tella
