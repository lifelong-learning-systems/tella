tella
===========
tella stands for Training and Evaluating Lifelong Learning Agents.
It provides a standard API and tools for performing continual learning experiments.

Requirements
----------------
* Python 3.7 or greater

Install
-------------
Cython needs to be installed into your conda/virtual environment first:
```
pip install cython
```
Then install the tella package and its dependencies:
```
pip install .
```

API
-------------
tella defines an interface for agents that consists of callbacks.
tella calls the callbacks to run through training and evaluation blocks.
The callbacks are
 * block_start() and block_end()
 * task_start() and tast_end()
 * episode_start() and episode_end()
 * step and step_result()


Run
-------------

Bug Reports and Feature Requests
---------------------------------

For Developers
----------------
To install with development requirements in editable mode:
```
pip install -e .[dev]
```

To run unit tests:
```
pytest
```

To check for PEP8 compliance:
```
```

To autoformat for PEP8 compliance:
```
```
