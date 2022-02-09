Installing tella
================

Requirements
----------------

* Python 3.7 or greater

Install
-------------
#. Create a conda or virtual environment and activate it

#. Update pip and wheel in your environment::

    pip install -U pip wheel

#. Install `l2logger <https://github.com/darpa-l2m/l2logger>`_.
   If you have ssh keys configured for GitHub, install like so::

    pip install git+ssh://git@github.com/darpa-l2m/l2logger.git

   If this does not work, try::

    pip install git+https://github.com/darpa-l2m/l2logger.git

   Otherwise, clone the l2logger repository and install::

       git clone https://github.com/darpa-l2m/l2logger
       pip install ./l2logger

#. Clone this repository::

    git clone git@github.com:darpa-l2m/tella.git

   or::

    git clone https://github.com/darpa-l2m/tella.git

5. Install the tella package and its dependencies::

    pip install "./tella[minigrid]"


To update tella, pull the latest changes from the git repository and upgrade::

    pip install -U .


For Developers
----------------
To install tella in editable mode with our development requirements::

    pip install -e ".[dev]"


To run unit tests::

    pytest

For running in conda environment::

    python -m pytest


To check for PEP8 compliance::

    black --check tella


To autoformat for PEP8 compliance::

    black tella
