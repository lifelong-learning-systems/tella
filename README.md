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
tella defines an event-based interface for agents.
tella calls methods of the agent to run through training and evaluation blocks.
The event handlers are
 * block_start() and block_end()
 * task_start() and task_end()
 * task_variant_start() and task_variant_end()
 * choose_action() and view_transition()

A learning block or evaluation block consists of 1 or more tasks.
The agent is notified of the start of the block and the start of each task.
The task_start() method receives basic information about the task.
The agent is also notified of the end of the block and the end of each task.

A task consists of multiple episodes.
The agent is notified of the start and end of the episode.
During the episode the agent is called through choose_action() with an observation and must return an action.
After the environment is updated with the action, the reward is passed to the agent by calling view_transition().
The view_transition() method also received the previous observation and new observation.
These calls continue until the episode is complete.


Run
-------------
To do

Bug Reports and Feature Requests
---------------------------------
Bug reports and feature requests should be made through issues on Github.

A bug report should contain:
 * descriptive title
 * environment (python version, operating system if install issue)
 * expected behavior
 * actual behavior
 * stack trace if applicable
 * any input parameters need to reproduce the bug

A feature request should describe what you want to do but cannot
and any recommendations for how this new feature should work.

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
