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
 * step_action() and step_reward()

A learning block or evaluation block consists of 1 or more task.
The agent is notified of the start of the block and the start of each task.
The task start callback receives basic information about the task.
The agent is also notified of the end of the block and the end of each task.

A task consists of multiple episodes.
The agent is notified of the start and end of the episode.
During the episode the agent is called through step_action() with an observation and must return an action.
After the environment is updated with the action, the reward is passed through step_reward().
The step_reward() method also received the previous observation and new observation.
These calls continue until the episode is complete.


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
