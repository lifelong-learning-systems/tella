"""
Copyright © 2021-2022 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import logging
import typing
import numpy as np
import gym
from l2logger import l2logger

from .agents import ContinualRLAgent
from .curriculum import (
    AbstractCurriculum,
    TaskVariant,
    validate_curriculum,
    Transition,
    Observation,
    Action,
)


logger = logging.getLogger(__name__)

AgentFactory = typing.Callable[[int, gym.Space, gym.Space, int, str], ContinualRLAgent]
"""
AgentFactory is a function or class that returns a 
:class:`ContinualRLAgent <tella.agents.ContinualRLAgent>`.

It takes 5 arguments, which are the same as 
:meth:`ContinualRLAgent.__init__() <tella.agents.ContinualRLAgent.__init__()>`:

    1. rng_seed, which is an integer to be used for repeatable random number generation
    2. observation_space, which is a :class:`gym.Space`
    3. action_space, which is a :class:`gym.Space`
    4. num_parallel_envs, which is an integer indicating how many environments will be run in parallel at the same time.
    5. config_file, which is a path as a string to a configuration file or None if no configuration.

A concrete subclass of :class:`ContinualRLAgent <tella.agents.ContinualRLAgent>` 
can be used as an AgentFactory::

    class MyAgent(ContinualRLAgent):
        ...

    agent_factory: AgentFactory = MyAgent
    agent = agent_factory(rng_seed, observation_space, action_space, num_parallel_envs, config_file)

A function can also be used as an AgentFactory::

    def my_factory(rng_seed, observation_space, action_space, num_parallel_envs, config_file):
        ...
        return my_agent(rng_seed, observation_space, action_space, num_parallel_envs, config_file)

    agent_factory: AgentFactory = my_factory
    agent = agent_factory(rng_seed, observation_space, action_space, num_parallel_envs, config_file)

"""

CurriculumFactory = typing.Callable[[int, typing.Optional[str]], AbstractCurriculum]
"""
CurriculumFactory is a function or class that returns an 
:class:`AbstractCurriculum <tella.curriculum.AbstractCurriculum>`.

It takes 2 arguments:

    1. an integer which is to be used for repeatable random number generation
    2. an option filepath to be loaded as a configuration dict.

A concrete subclass of :class:`AbstractCurriculum <tella.curriculum.AbstractCurriculum>` 
can be used as an CurriculumFactory::

    class MyCurriculum(AbstractCurriculum[AbstractRLTaskVariant]):
        ...

    curriculum_factory: CurriculumFactory = MyCurriculum
    curriculum = curriculum_factory(rng_seed, config_file)

A function can also be used as a CurriculumFactory::

    def my_factory(rng_seed, config_file):
        ...
        return my_curriculum(rng_seed, config_file)

    curriculum_factory: CurriculumFactory = my_factory
    curriculum = curriculum_factory(rng_seed, config_file)

"""


def rl_experiment(
    agent_factory: AgentFactory,
    curriculum_factory: CurriculumFactory,
    num_lifetimes: int,
    num_parallel_envs: int,
    log_dir: str,
    agent_seed: typing.Optional[int] = None,
    curriculum_seed: typing.Optional[int] = None,
    render: typing.Optional[bool] = False,
    agent_config: typing.Optional[str] = None,
    curriculum_config: typing.Optional[str] = None,
    lifetime_idx: int = 0,
) -> None:
    """
    Run an experiment with an RL agent and an RL curriculum.

    :param agent_factory: Function or class to produce agents.
    :param curriculum_factory: Function or class to produce curriculum.
    :param num_lifetimes: Number of times to call :func:`run()`.
    :param num_parallel_envs: Number of parallel environments.
    :param log_dir: The root log directory for l2logger.
    :param lifetime_idx: The index of the lifetime to start running with.
        This will skip the first N seeds of the RNGs, where N = `lifetime_idx`.
    :param agent_seed: The seed for the RNG for the agent or None for random seed.
    :param curriculum_seed: The seed for the RNG for the curriculum or None for random seed.
    :param render: Whether to render the environment for debugging or demonstrations.
    :param agent_config: Optional path to a configuration file for the agent.
    :param curriculum_config: Optional path to a configuration file for the curriculum.

    :return: None
    """
    if lifetime_idx < 0:
        raise ValueError(f"lifetime_idx must be >= 0, found {lifetime_idx}")

    if lifetime_idx > 0 and curriculum_seed is None:
        raise ValueError(
            "curriculum_seed must be specified when using lifetime_idx > 0."
            f"Found curriculum_seed={curriculum_seed}."
        )

    observation_space, action_space = _spaces(curriculum_factory)

    if agent_seed is None:
        logger.info("No agent seed provided; one will be generated randomly.")
        agent_seed = np.random.default_rng().bit_generator.random_raw()
    logger.info(f"Experiment RNG seed for agents: {agent_seed}")
    agent_rng = np.random.default_rng(agent_seed)
    if curriculum_seed is None:
        logger.info("No curriculum seed provided; one will be generated randomly.")
        curriculum_seed = np.random.default_rng().bit_generator.random_raw()
    logger.info(f"Experiment RNG seed for curriculums: {curriculum_seed}")
    curriculum_rng = np.random.default_rng(curriculum_seed)

    for i_lifetime in range(lifetime_idx):
        curriculum_seed = curriculum_rng.bit_generator.random_raw()
        agent_seed = agent_rng.bit_generator.random_raw()
        logger.info(
            f"Skipping lifetime #{i_lifetime + 1} (lifetime_idx={i_lifetime}), "
            f"curriculum_seed={curriculum_seed}, agent_seed={agent_seed}"
        )

    # FIXME: multiprocessing https://github.com/darpa-l2m/tella/issues/44
    for i_lifetime in range(lifetime_idx, lifetime_idx + num_lifetimes):
        logger.info(f"Starting lifetime #{i_lifetime + 1} (lifetime_idx={i_lifetime})")

        curriculum_seed = curriculum_rng.bit_generator.random_raw()
        curriculum = curriculum_factory(curriculum_seed, curriculum_config)
        logger.info(f"Constructed curriculum {curriculum} with seed {curriculum_seed}")

        # FIXME: check for RL task variant https://github.com/darpa-l2m/tella/issues/53
        validate_curriculum(curriculum.copy())
        logger.info("Validated curriculum")

        agent_seed = agent_rng.bit_generator.random_raw()
        agent = agent_factory(
            agent_seed, observation_space, action_space, num_parallel_envs, agent_config
        )
        logger.info(f"Constructed agent {agent} with seed {agent_seed}")

        # FIXME: pass num_parallel_envs to run https://github.com/darpa-l2m/tella/issues/32
        run(
            agent,
            curriculum,
            render=render,
            log_dir=log_dir,
            num_envs=num_parallel_envs,
        )


def _spaces(
    curriculum_factory: CurriculumFactory,
) -> typing.Tuple[gym.Space, gym.Space]:
    """
    Retrieves the observation & action space from the curriculum.

    :param curriculum_factory: The class of the curriculum. This function constructs an object using this class.
    :return: A tuple of (observation_space, action_space).
    """
    # FIXME: extract spaces based on solution in https://github.com/darpa-l2m/tella/issues/31
    curriculum_obj = curriculum_factory(0)
    for block in curriculum_obj.learn_blocks_and_eval_blocks():
        for task_block in block.task_blocks():
            for task_variant in task_block.task_variants():
                env = task_variant.make_env()
                observation_space = env.observation_space
                action_space = env.action_space
                env.close()
                del env
                return observation_space, action_space


def run(
    agent: ContinualRLAgent,
    curriculum: AbstractCurriculum,
    render: typing.Optional[bool],
    log_dir: str,
    num_envs: typing.Optional[int] = 1,
):
    """
    Run an agent through an entire curriculum.

    :param agent: Agent for this experiment.
    :param curriculum: Curriculum for this experiment.
    :param render: Bool flag to toggle environment rendering.
    :param log_dir: Directory for l2logger files.
    :param num_envs: Number of parallel environments.
    """
    scenario_dir = f"{curriculum.__class__.__name__}-{agent.__class__.__name__}"
    scenario_info = {
        "author": "JHU APL",
        "complexity": "1-low",
        "difficulty": "2-medium",
        "scenario_type": "custom",
        "curriculum_seed": curriculum.rng_seed,
        "agent_seed": agent.rng_seed,
        "curriculum_name": curriculum.__class__.__name__,
        "agent_name": agent.__class__.__name__,
    }
    data_logger = L2Logger(log_dir, scenario_dir, scenario_info, num_envs)
    for block in curriculum.learn_blocks_and_eval_blocks():
        is_learning_allowed = agent.is_learning_allowed = block.is_learning_allowed
        block_type = "Learning" if is_learning_allowed else "Evaluating"
        data_logger.block_start(is_learning_allowed)
        agent.block_start(is_learning_allowed)
        for task_block in block.task_blocks():
            agent.task_start(task_block.task_label)
            for task_variant in task_block.task_variants():
                logger.info(
                    f"{block_type} TaskVariant {task_variant.task_label}-{task_variant.variant_label}"
                )
                data_logger.task_variant_start(task_variant)
                agent.task_variant_start(
                    task_variant.task_label, task_variant.variant_label
                )
                for transitions in generate_transitions(
                    task_variant, agent.choose_actions, num_envs, render
                ):
                    data_logger.receive_transitions(transitions)
                    agent.receive_transitions(
                        transitions
                        if is_learning_allowed
                        else hide_rewards(transitions)
                    )
                agent.task_variant_end(
                    task_variant.task_label, task_variant.variant_label
                )
            agent.task_end(task_block.task_label)
        agent.block_end(block.is_learning_allowed)


class L2Logger:
    """
    A utility class for handling logging with the l2logger package.

    The l2logger package logs cumulative episode rewards when episodes finish,
    and all of this can be achieved by tracking data from transitions via
    the :meth:`L2Logger.receive_transitions()`.

    Additionally, l2logger needs to track the global block number and task id
    information, so :meth:`L2Logger.block_start()` and :meth:`L2Logger.task_variant_start()`
    should be called at the appropriate times to track this information.
    """

    def __init__(
        self,
        log_dir: str,
        scenario_dir: str,
        scenario_info: typing.Dict[str, str],
        num_envs: int,
    ) -> None:
        self.data_logger = l2logger.DataLogger(
            log_dir,
            scenario_dir,
            {"metrics_columns": ["reward"], "log_format_version": "1.0"},
            scenario_info,
        )
        self.block_num = -1
        self.block_type = None
        self.task_name = None
        self.task_params = None
        self.total_episodes = 0
        self.num_envs = num_envs
        self.cumulative_episode_rewards = [0.0] * num_envs
        self.episode_step_counts = [0] * num_envs

    def block_start(self, is_learning_allowed: bool) -> None:
        self.block_num += 1
        self.block_type = "train" if is_learning_allowed else "test"

    def task_variant_start(self, task_variant: TaskVariant):
        self.task_name = task_variant.task_label + "_" + task_variant.variant_label
        self.task_params = task_variant.params
        self.cumulative_episode_rewards = [0.0] * self.num_envs
        self.episode_step_counts = [0] * self.num_envs

    def receive_transitions(
        self, transitions: typing.List[typing.Optional[Transition]]
    ) -> None:
        for i, transition in enumerate(transitions):
            if transition is None:
                continue
            self.cumulative_episode_rewards[i] += transition.reward
            self.episode_step_counts[i] += 1
            if transition.done:
                self.data_logger.log_record(
                    {
                        "block_num": self.block_num,
                        "block_type": self.block_type,
                        "task_name": self.task_name,
                        "task_params": self.task_params,
                        "worker_id": "worker-default",
                        "exp_num": self.total_episodes,
                        "reward": self.cumulative_episode_rewards[i],
                        "exp_status": "complete",
                        "episode_step_count": self.episode_step_counts[i],
                    }
                )
                self.cumulative_episode_rewards[i] = 0
                self.episode_step_counts[i] = 0
                self.total_episodes += 1


ActionFn = typing.Callable[
    [typing.List[typing.Optional[Observation]]], typing.List[typing.Optional[Action]]
]
"""
A function that takes a list of Observations and returns a list of Actions, one
for each observation.
"""


def generate_transitions(
    task_variant: TaskVariant,
    action_fn: ActionFn,
    num_envs: int,
    render: bool = False,
) -> typing.Iterable[typing.List[typing.Optional[Transition]]]:
    """
    Yields markov transitions from the interaction between the `action_fn`
    and the :class:`gym.Env` contained in :class:`TaskVariant <tella.curriculum.TaskVariant>`.

    .. Note:: `None` transitions

        Extra data can be produced when using num_envs > 1, if the data limits in
        :class:`TaskVariant <tella.curriculum.TaskVariant>` % num_envs != 0. For an example if the limit
        is 4 episodes, and `num_envs` is 5, then this function will generate a whole
        extra episode worth of transitions. In order to prevent the leak of extra data,
        we mask out any transitions above the data limit by setting them to None.

    :param task_variant: The task variant containing environment and seed information.
    :param action_fn: Selects actions to take in the environment given an observation.
    :param num_envs: Controls the amount of parallelization to use for this episodic task variant.
        See :class:`gym.vector.VectorEnv` for more information.
    :param render: Whether to render the environment at each step.
    :return: A generator of transitions.
    """
    vector_env_cls = (
        gym.vector.SyncVectorEnv if num_envs == 1 else gym.vector.AsyncVectorEnv
    )

    env = vector_env_cls([task_variant.make_env for _ in range(num_envs)])

    env.seed(task_variant.rng_seed)
    num_episodes_finished = 0
    num_steps_finished = 0

    # data to keep track of which observations to mask out (set to None)
    episode_ids = list(range(num_envs))
    next_episode_id = episode_ids[-1] + 1

    observations = env.reset()
    if task_variant.num_episodes is not None:
        continue_task = num_episodes_finished < task_variant.num_episodes
    else:
        continue_task = num_steps_finished < task_variant.num_steps
    while continue_task:
        # mask out any environments that have episode id above max episodes or steps above max steps
        if task_variant.num_episodes is not None:
            mask = [ep_id >= task_variant.num_episodes for ep_id in episode_ids]
        else:
            mask = [
                task_variant.num_steps - (num_steps_finished + 1) < idx
                for idx in range(num_envs)
            ]

        # replace masked environment observations with None
        masked_observations = _where(mask, None, observations)

        # query for the actions
        actions = action_fn(masked_observations)

        # replace masked environment actions with random action
        unmasked_actions = _where(mask, env.single_action_space.sample(), actions)

        # step in the VectorEnv
        next_observations, rewards, dones, infos = env.step(unmasked_actions)
        if render:
            env.envs[0].render()

        # yield all the transitions of this step
        resulting_obs = [
            info["terminal_observation"] if done else next_obs
            for info, done, next_obs in zip(infos, dones, next_observations)
        ]
        unmasked_transitions = [
            Transition(*values)
            for values in zip(observations, actions, rewards, dones, resulting_obs)
        ]
        masked_transitions = _where(mask, None, unmasked_transitions)
        yield masked_transitions

        # increment episode ids if episode ended
        for i in range(num_envs):
            if not mask[i]:
                num_steps_finished += 1
                if dones[i]:
                    num_episodes_finished += 1
                    episode_ids[i] = next_episode_id
                    next_episode_id += 1

        # Decide if we should continue
        if task_variant.num_episodes is not None:
            continue_task = num_episodes_finished < task_variant.num_episodes
        else:
            continue_task = num_steps_finished < task_variant.num_steps
        observations = next_observations

    env.close()
    del env


def hide_rewards(
    transitions: typing.List[typing.Optional[Transition]],
) -> typing.List[typing.Optional[Transition]]:
    """
    Masks out any rewards in the transitions passed in by setting the reward
    field to None.

    :param transitions: The transitions to hide the reward in
    :return: The new list of transitions with all rewards set to None
    """
    return [
        None
        if t is None
        else Transition(t.observation, t.action, None, t.done, t.next_observation)
        for t in transitions
    ]


def _where(
    condition: typing.List[bool], replace_value: typing.Any, original_list: typing.List
) -> typing.List:
    """
    Replaces elements in `original_list[i]` with `replace_value` where the `condition[i]`
    is True.

    :param condition: List of booleans indicating where to put replace_value
    :param replace_value: The value to insert into the list
    :param original_list: The list of values to modify
    :return: A new list with replace_value inserted where condition elements are True
    """
    return [
        replace_value if condition[i] else original_list[i]
        for i in range(len(condition))
    ]
