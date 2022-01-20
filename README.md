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
   pip install git+ssh://git@github.com/darpa-l2m/l2logger.git
   ```
   If this does not work, try:
   ```
   pip install git+https://github.com/darpa-l2m/l2logger.git
   ```
   Otherwise, clone the l2logger repository and install:
   ```
   git clone https://github.com/darpa-l2m/l2logger
   pip install ./l2logger
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
   pip install "./tella[minigrid]"
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

Here is a minimal agent that takes random agents:
```python
import tella


class MinimalRandomAgent(tella.ContinualRLAgent):
    def choose_action(self, observations):
        """Loop over the environments' observations and select action"""
        return [
            None if obs is None else self.action_space.sample() for obs in observations
        ]

    def receive_transition(self, transition):
        """Do nothing here since we are not learning"""
        pass


if __name__ == "__main__":
    # rl_cli() is tella's command line interface function.
    # It expects a constructor or factory function to create the agent.
    tella.rl_cli(MinimalRandomAgent)
    print("Done! Check logs for results.")
```


Run
-------------
Assuming your agent is defined in a file called `my_agent.py`,
run it through a curriculum like so:
```
python my_agent.py --curriculum SimpleCartPole
```

To see all the command line options, run:
```
python my_agent.py --help
```
All the curriculums registered with tella are listed in the help.

The l2logger output by default is stored in your current directory in `logs`.
This can be set with the `--log-dir` argument.

For reproducing behavior, use the `--agent-seed`  and `--curriculum-seed` arguments.
If a seed is not provided, a random seed is generated.
The seeds used will be logged using the python logging package.

To run an agent through multiple lifetimes of a curriculum, use the `--num-lifetimes`
flag. If you want to run a specific lifetime (useful for running on a cluster),
use the `--lifetime-idx` flag. Note that seeds must be provided to use `--lifetime-idx`.

To view a rendering of the agent learning, set the `--render` flag.
This will render the first environment in the list.

To pass a configuration file to the agent, set the `--agent-config` argument.
The format of the configuration file is determined by the specific agent.

Parallel environments is not currently implemented.


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
pip install -e ".[dev]"
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
