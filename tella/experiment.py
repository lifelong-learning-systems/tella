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

from .agents import ContinualRLAgent, ContinualLearningAgent, AbstractRLTaskVariant
from .curriculum import (
    AbstractCurriculum,
    AbstractTaskVariant,
    validate_curriculum,
    EpisodicTaskVariant,
)


logger = logging.getLogger(__name__)

AgentFactory = typing.Callable[[int, gym.Space, gym.Space, int, str], ContinualRLAgent]
"""
AgentFactory is a function or class that returns a :class:`ContinualRLAgent`.

It takes 5 arguments, which are the same as :meth:`ContinualRLAgent.__init__()`:

    1. rng_seed, which is an integer to be used for repeatable random number generation
    2. observation_space, which is a :class:`gym.Space`
    3. action_space, which is a :class:`gym.Space
    4. num_parallel_envs, which is an integer indicating how many environments will be run in parallel at the same time.
    4. config_file, which is a path as a string to a configuration file or None if no configuration.

A concrete subclass of :class:`ContinualRLAgent` can be used as an AgentFactory:

    class MyAgent(ContinualRLAgent):
        ...

    agent_factory: AgentFactory = MyAgent
    agent = agent_factory(rng_seed, observation_space, action_space, num_parallel_envs, config_file)

A function can also be used as an AgentFactory:

    def my_factory(rng_seed, observation_space, action_space, num_parallel_envs, config_file):
        ...
        return my_agent

    agent_factory: AgentFactory = my_factory
    agent = agent_factory(rng_seed, observation_space, action_space, num_parallel_envs, config_file)
"""

CurriculumFactory = typing.Callable[[int], AbstractCurriculum[AbstractRLTaskVariant]]
"""
CurriculumFactory is a type alias for a function or class that returns a
:class:`AbstractCurriculum`.

It takes 1 argument, an integer which is to be used for repeatable random number generation.

A concrete subclass of :class:`AbstractCurriculum` can be used as an CurriculumFactory:

    class MyCurriculum(AbstractCurriculum[AbstractRLTaskVariant]):
        ...

    curriculum_factory: CurriculumFactory = MyCurriculum
    curriculum = curriculum_factory(rng_seed)

A function can also be used as an CurriculumFactory:

    def my_factory(rng_seed):
        ...
        return my_curriculum

    curriculum_factory: CurriculumFactory = my_factory
    curriculum = curriculum_factory(rng_seed)
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
    lifetime_idx: int = 0,
) -> None:
    """
    Run an experiment with an RL agent and an RL curriculum.

    :param agent_factory: Function or class to produce agents.
    :param curriculum_factory: Function or class to produce curriculum.
    :param num_lifetimes: Number of times to call :func:`run()`.
    :param num_parallel_envs: TODO
    :param log_dir: The root log directory for l2logger.
    :param lifetime_idx: The index of the lifetime to start running with.
        This will skip the first N seeds of the RNGs, where N = `lifetime_idx`.
    :param agent_seed: The seed for the RNG for the agent or None for random seed.
    :param curriculum_seed: The seed for the RNG for the curriculum or None for random seed.
    :param render: Whether to render the environment for debugging or demonstrations.
    :param agent_config: Optional path to a configuration file for the agent.
    :return: None
    """
    if lifetime_idx < 0:
        raise ValueError(f"lifetime_idx must be >= 0, found {lifetime_idx}")

    if lifetime_idx > 0 and (agent_seed is None or curriculum_seed is None):
        raise ValueError(
            "Both agent_seed and curriculum_seed must be specified when using lifetime_idx > 0."
            f"Found agent_seed={agent_seed}, curriculum_seed={curriculum_seed}."
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
        curriculum = curriculum_factory(curriculum_seed)
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
                env = task_variant.info()
                if isinstance(env, gym.vector.VectorEnv):
                    observation_space = env.single_observation_space
                    action_space = env.single_action_space
                else:
                    observation_space = env.observation_space
                    action_space = env.action_space
                env.close()
                del env
                return observation_space, action_space


def run(
    agent: ContinualLearningAgent[AbstractTaskVariant],
    curriculum: AbstractCurriculum[EpisodicTaskVariant],
    render: typing.Optional[bool],
    log_dir: str,
    num_envs: typing.Optional[int] = 1,
):
    """
    Run an agent through an entire curriculum. This assumes that the agent
    and the curriculum are both generic over the same type.

    I.e. the curriculum will be generating task variants of type T, and the agent
    will be consuming them via it's :meth:`ContinualLearningAgent.consume_task_variant`.
    """
    scenario_dir = curriculum.__class__.__name__
    scenario_info = {
        "author": "JHU APL",
        "complexity": "1-low",
        "difficulty": "2-medium",
        "scenario_type": "custom",
    }
    logger_info = {"metrics_columns": ["reward"], "log_format_version": "1.0"}
    data_logger = l2logger.DataLogger(log_dir, scenario_dir, logger_info, scenario_info)
    total_episodes = 0
    for i_block, block in enumerate(curriculum.learn_blocks_and_eval_blocks()):
        is_learning_allowed = agent.is_learning_allowed = block.is_learning_allowed
        block_type = "Learning" if is_learning_allowed else "Evaluating"
        agent.block_start(is_learning_allowed)
        for task_block in block.task_blocks():
            agent.task_start(task_block.task_label)
            for task_variant in task_block.task_variants():
                task_variant.set_show_rewards(is_learning_allowed)
                task_variant.set_num_envs(num_envs)
                # NOTE: assuming taskvariant has params
                task_variant.set_logger_info(
                    data_logger, i_block, is_learning_allowed, total_episodes
                )
                task_variant.set_render(render)
                logger.info(
                    f"{block_type} TaskVariant {task_variant.task_label}-{task_variant.variant_label}"
                )
                agent.task_variant_start(
                    task_variant.task_label, task_variant.variant_label
                )
                # FIXME: This run function should handle the learning and eval, not the agent.
                #   Move these methods out of the agent class. https://github.com/darpa-l2m/tella/issues/203
                if is_learning_allowed:
                    agent.learn_task_variant(task_variant)
                else:
                    agent.eval_task_variant(task_variant)
                agent.task_variant_end(
                    task_variant.task_label, task_variant.variant_label
                )
                total_episodes += task_variant.total_episodes
            agent.task_end(task_block.task_label)
        agent.block_end(block.is_learning_allowed)
