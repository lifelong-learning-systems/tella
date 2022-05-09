tella
===========

![apl logo](apl_small_logo.png)

tella stands for Training and Evaluating Lifelong Learning Agents.
It provides a standard API and tools for performing continual learning experiments.
Full documentation is available at [readthedocs](https://tella.readthedocs.io/).

Requirements
----------------
* Python 3.7 or greater

Install
-------------
1. Create a conda or virtual environment and activate it

2. We recommend that you update pip and wheel in your environment:
  ```
  pip install -U pip wheel
  ```
3. Install tella
   ```
   pip install tella
   ```

There are optional packages that can also be installed if you want
to use atari environments or run the unit tests.
They can be installed using pip's extras syntax:
```
pip install tella[atari]
```
or
```
pip install tella[dev]
```


API
-------------
tella defines an event-based interface for agents.
tella calls methods of the agent to run through training and evaluation blocks.
The event handlers are
 * block_start() and block_end()
 * task_start() and task_end()
 * task_variant_start() and task_variant_end()
 * choose_actions() and receive_transitions()

A learning block or evaluation block consists of 1 or more tasks.
The agent is notified of the start of the block and the start of each task.
The task_start() method receives basic information about the task.
The agent is also notified of the end of the block and the end of each task.

A task consists of multiple episodes.
The agent is notified of the start and end of the episode.
During the episode the agent is called through choose_actions() with an observation and must return an action.
After the environment is updated with the action, the reward is passed to the agent by calling receive_transitions().
The receive_transitions() method also received the previous observation and new observation.
These calls continue until the episode is complete.

Here is a minimal agent that takes random agents:
```python
import tella


class MinimalRandomAgent(tella.ContinualRLAgent):
    def choose_actions(self, observations):
        """Loop over the environments' observations and select action"""
        return [
            None if obs is None else self.action_space.sample() for obs in observations
        ]

    def receive_transitions(self, transitions):
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
```bash
> python my_agent.py --help
usage: my_agent.py [-h] [--lifetime-idx LIFETIME_IDX] [--num-lifetimes NUM_LIFETIMES]
                           [--num-parallel-envs NUM_PARALLEL_ENVS] [--log-dir LOG_DIR] [--render] [--seed SEED]
                           [--agent-seed AGENT_SEED] [--curriculum-seed CURRICULUM_SEED] [--agent-config AGENT_CONFIG]
                           --curriculum {...}

optional arguments:
  -h, --help            show this help message and exit
  --lifetime-idx LIFETIME_IDX
                        The index, starting at zero, of the first lifetime to run. Use this to skip lifetimes or run a
                        specific lifetime other than the first. (default: 0)
  --num-lifetimes NUM_LIFETIMES
                        Number of lifetimes to execute. (default: 1)
  --num-parallel-envs NUM_PARALLEL_ENVS
                        Number of environments to run in parallel inside of task variant blocks. This enables the use
                        of multiple CPUs at the same time for running environment logic, via vectorized environments.
                        (default: 1)
  --log-dir LOG_DIR     The root directory for the l2logger logs produced. (default: ./logs)
  --render              Whether to render the environment (default: False)
  --seed SEED           replaced by --agent-seed and --curriculum-seed (default: None)
  --agent-seed AGENT_SEED
                        The agent rng seed to use for reproducibility. (default: None)
  --curriculum-seed CURRICULUM_SEED
                        The curriculum rng seed to use for reproducibility. (default: None)
  --agent-config AGENT_CONFIG
                        Optional path to agent config file. (default: None)
  --curriculum {...}
                        Curriculum name for registry. (default: None)
```
All the curriculums registered with tella are listed in the help.

The l2logger output by default is stored in your current directory in `logs`.
This can be set with the `--log-dir` argument.

For reproducing behavior, use the `--agent-seed`  and `--curriculum-seed` arguments.
If a seed is not provided, a random seed is generated.
The seeds used will be logged using the python logging package.

For utilizing multiple cores, use the `--num-parallel-envs` flag.
When using `--num-parallel-envs` > 1, you may need to configure
python multiprocessing's start method via `mp.set_start_method("spawn")`
at the start of the program, depending on the underlying OS.

To run an agent through multiple lifetimes of a curriculum, use the `--num-lifetimes`
flag. If you want to run a specific lifetime (useful for running on a cluster),
use the `--lifetime-idx` flag. Note that the curriculum seed must be provided to use `--lifetime-idx`.

To view a rendering of the agent learning, set the `--render` flag.
This will render the first environment in the list.

To pass a configuration file to the agent, set the `--agent-config` argument.
The format of the configuration file is determined by the specific agent.


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
To install tella in editable mode with our development requirements,
clone the git repo and run:
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

License
-------

See [LICENSE](LICENSE) for license information.

Acknowledgments
----------------
![APL logo](https://raw.githubusercontent.com/lifelong-learning-systems/tella/master/apl_small_logo.png)

tella was created by the Johns Hopkins University Applied Physics Laboratory.

This software was funded by the DARPA Lifelong Learning Machines (L2M) Program.

The views, opinions, and/or findings expressed are those of the author(s) and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.
