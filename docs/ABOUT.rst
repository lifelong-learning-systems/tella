About tella
===========

tella stands for Training and Evaluating Lifelong Learning Agents.
It provides a standard API and tools for performing continual learning experiments.


Continual RL curriculums
------------------------
The experiences presented to a tella agent are structured in the following hierarchy::

    curriculum > block > task block > task variant block > individual experience

In tella, all task variants in a curriculum must share the same
observation space and action space.

.. glossary::

    Task Variant
        A specific MDP and objective, specified as an openai gym Environment, that
        the agent interacts with.
        Individual instances are randomized, but parameters are fixed.

    Task
        A set of task variants with varied parameters, but overall similar nature.

    Task Variant Block
        A sequence of one or more experiences of the same task variant.

    Task Block
        A sequence of one or more task variant blocks, all containing the same task.

    Block
        A sequence of one or more task blocks, potentially containing several different tasks.
        Blocks can be learning blocks or evaluation blocks.

    Curriculum
        A sequence of one or more blocks.

Several preset curriculums are included in tella.
It is also possible to define custom curriculums.

A small example curriculum could contain::

        Block 1, learning: 2 tasks
            Task 1, CartPoleEnv: 2 variants
                Task variant 1, CartPoleEnv - Default: 1 episode.
                Task variant 2, CartPoleEnv - Variant: 1 episode.
            Task 2, MountainCarEnv: 1 variant
                Task variant 1, MountainCarEnv - Default: 1 episode.

        Block 2, evaluation: 1 task
            Task 1, CartPoleEnv: 1 variant
                Task variant 1, CartPoleEnv - Default: 1 episode.

RL agent API
-------------
tella defines an event-based interface for agents.
The event handlers follow the curriculum structure described above:

* :meth:`ContinualRLAgent.block_start() <tella.agents.ContinualRLAgent.block_start>`
  and :meth:`ContinualRLAgent.block_end() <tella.agents.ContinualRLAgent.block_end>`
* :meth:`ContinualRLAgent.task_start() <tella.agents.ContinualRLAgent.task_start>`
  and :meth:`ContinualRLAgent.task_end() <tella.agents.ContinualRLAgent.task_end>`
* :meth:`ContinualRLAgent.task_variant_start() <tella.agents.ContinualRLAgent.task_variant_start>`
  and :meth:`ContinualRLAgent.task_variant_end() <tella.agents.ContinualRLAgent.task_variant_end>`
* :meth:`ContinualRLAgent.choose_actions() <tella.agents.ContinualRLAgent.choose_actions>`
* :meth:`ContinualRLAgent.receive_transitions() <tella.agents.ContinualRLAgent.receive_transitions>`

The agent is notified of the start and end of each block, task block, and task variant block.
During each task variant block, the agent is repeatedly called through
:meth:`ContinualRLAgent.choose_actions() <tella.agents.ContinualRLAgent.choose_actions()>`
and must return its actions based on the provided observations.
After each environment is updated with the action, the results are passed to the agent by calling
:meth:`ContinualRLAgent.receive_transitions() <tella.agents.ContinualRLAgent.receive_transitions()>`.
These calls continue until the task variant block is complete.

The abstract classes :class:`tella.agents.ContinualRLAgent`
implement the expected methods of an RL agent.
Here is a minimal agent subclass that takes random actions::

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


Running tella
-------------
tella defines a command line interface (CLI) for running continual RL experiments.
Assuming your agent is defined in a file called ``my_agent.py``,
and that file contains this block which directs calls to the tella CLI::

    if __name__ == "__main__":
        tella.rl_cli(<MyAgentClass>)

experiments with the agent can then be run by::

    python my_agent.py --curriculum SimpleCartPole

To see all the command line options, run::

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
        --curriculum-config CURRICULUM_CONFIG
                            Optional path to curriculum config file. (default: None)
        --curriculum {...}
                            Curriculum name for registry. (default: None)

All the curriculums registered with tella are listed in the help.

Experiments run in tella are monitored by `l2logger <https://github.com/darpa-l2m/l2logger>`_.
The l2logger output by default is stored relative your current directory in ``./logs/``.
This can be set with the ``--log-dir`` argument.

For reproducing behavior, use the ``--agent-seed``  and ``--curriculum-seed`` arguments.
If a seed is not provided, a random seed is generated.
The seeds used will be logged using the python logging package.

For utilizing multiple cores, use the ``--num-parallel-envs`` flag.
When using ``--num-parallel-envs`` > 1, you may need to configure
python multiprocessing's start method via ``mp.set_start_method("spawn")``
at the start of the program, depending on the underlying OS.

To run an agent through multiple lifetimes of a curriculum, use the ``--num-lifetimes``
flag. If you want to run a specific lifetime (useful for running on a cluster),
use the ``--lifetime-idx`` flag. Note that the curriculum seed must be provided to use ``--lifetime-idx``.
For example, two lifetimes can be run by::

    python my_agent.py --curriculum MiniGridCondensed --curriculum-seed 12345 --num-lifetimes 2

Or in parallel, assuming the same environments, by::

    python my_agent.py --curriculum MiniGridCondensed --curriculum-seed 12345 --num-lifetimes 1 --lifetime-idx 0
    python my_agent.py --curriculum MiniGridCondensed --curriculum-seed 12345 --num-lifetimes 1 --lifetime-idx 1

To view a rendering of the agent learning, set the ``--render`` flag.
This will render the first environment in the list when ``--num-parallel-envs`` > 1.

To pass a configuration file to the agent, use the ``--agent-config`` argument.
To pass a configuration file to the curriculum, use the ``--curriculum-config`` argument.
The format of the configuration file is determined by the specific object it is passed to.
