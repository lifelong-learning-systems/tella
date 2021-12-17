tella
===========
tella stands for Training and Evaluating Lifelong Learning Agents.
It provides a standard API and tools for performing continual learning experiments.

Requirements
----------------
* Python 3.7 or greater

Install
-------------
1. Create a conda or virtual environment and activate it

2. Update pip and wheel in your environment:
  ```
  pip install -U pip wheel
  ```
3. Install [l2logger](https://github.com/darpa-l2m/l2logger).
   If you have ssh keys configured for GitHub, install like so:
   ```
   pip install git+https://github.com/darpa-l2m/l2logger.git
   ```
   Otherwise, clone the l2logger repository and install:
   ```
   git clone https://github.com/darpa-l2m/l2logger
   pip install l2logger
   ```
4. Clone this repository:
   ```
   git clone git@github.com:darpa-l2m/tella.git
   ```
   or
   ```
   git clone https://github.com/darpa-l2m/tella.git
   ```
5. Install the tella package and its dependencies:
   ```
   pip install tella
   ```

To update tella, pull the latest changes from the git repository and upgrade:
```
pip install -U .
```

API
-------------
tella defines an event-based interface for agents.
tella calls methods of the agent to run through training and evaluation blocks.
The event handlers are
 * block_start() and block_end()
 * task_start() and task_end()
 * task_variant_start() and task_variant_end()
 * choose_action() and receive_transition()

A learning block or evaluation block consists of 1 or more tasks.
The agent is notified of the start of the block and the start of each task.
The task_start() method receives basic information about the task.
The agent is also notified of the end of the block and the end of each task.

A task consists of multiple episodes.
The agent is notified of the start and end of the episode.
During the episode the agent is called through choose_action() with an observation and must return an action.
After the environment is updated with the action, the reward is passed to the agent by calling receive_transition().
The receive_transition() method also received the previous observation and new observation.
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
To install tella in editable mode with our development requirements:
```
pip install -e .[dev]
```

To run unit tests:
```
pytest
```
For running in conda environment:
```
python -m pytest 
```

To check for PEP8 compliance:
```
black --check tella
```

To autoformat for PEP8 compliance:
```
black tella
```
