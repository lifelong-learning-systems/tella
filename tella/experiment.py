"""
Copyright © 2021 The Johns Hopkins University Applied Physics Laboratory LLC

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

import gym
from l2logger import l2logger

from .agents import ContinualRLAgent, ContinualLearningAgent, AbstractRLTaskVariant
from .curriculum import AbstractCurriculum, AbstractTaskVariant, validate_curriculum


logger = logging.getLogger(__name__)

AgentFactory = typing.Callable[[gym.Space, gym.Space, int], ContinualRLAgent]
"""
AgentFactory is a function or class that returns a :class:`ContinualRLAgent`.

It takes 3 arguments, which are the same as :meth:`ContinualRLAgent.__init__()`:

    1. observation_space, which is a :class:`gym.Space`
    2. action_space, which is a :class:`gym.Space
    3. num_parallel_envs, which is an integer indicating how many environments will be run in parallel at the same time.

A concrete subclass of :class:`ContinualRLAgent` can be used as an AgentFactory:

    class MyAgent(ContinualRLAgent):
        ...

    agent_factory: AgentFactory = MyAgent
    agent = agent_factory(observation_space, action_space, num_parallel_envs)

A function can also be used as an AgentFactory:

    def my_factory(observation_space, action_space, num_parallel_envs):
        ...
        return my_agent

    agent_factory: AgentFactory = my_factory
    agent = agent_factory(observation_space, action_space, num_parallel_envs)
"""

CurriculumFactory = typing.Callable[[], AbstractCurriculum[AbstractRLTaskVariant]]
"""
CurriculumFactory is a type alias for a function or class that returns a
:class:`AbstractCurriculum`.

It takes 0 arguments.

A concrete subclass of :class:`AbstractCurriculum` can be used as an CurriculumFactory:

    class MyCurriculum(AbstractCurriculum[AbstractRLTaskVariant]):
        ...

    curriculum_factory: CurriculumFactory = MyCurriculum
    curriculum = curriculum_factory()

A function can also be used as an CurriculumFactory:

    def my_factory():
        ...
        return my_curriculum

    curriculum_factory: CurriculumFactory = my_factory
    curriculum = curriculum_factory()
"""


def rl_experiment(
    agent_factory: AgentFactory,
    curriculum_factory: CurriculumFactory,
    num_lifetimes: int,
    num_parallel_envs: int,
    log_dir: str,
    render: typing.Optional[bool] = False,
) -> None:
    """
    Run an experiment with an RL agent and an RL curriculum.

    :param agent_factory: Function or class to produce agents.
    :param curriculum_factory: Function or class to produce curriculum.
    :param num_lifetimes: Number of times to call :func:`run()`.
    :param num_parallel_envs: TODO
    :param log_dir:TODO
    :return: None
    """
    observation_space, action_space = _spaces(curriculum_factory)

    # FIXME: multiprocessing https://github.com/darpa-l2m/tella/issues/44
    for i_lifetime in range(num_lifetimes):
        curriculum = curriculum_factory()
        logger.info(f"Constructed curriculum {curriculum}")
        # FIXME: seed the curriculum https://github.com/darpa-l2m/tella/issues/54

        # FIXME: check for RL task variant https://github.com/darpa-l2m/tella/issues/53
        validate_curriculum(curriculum)
        logger.info("Validated curriculum")

        agent = agent_factory(observation_space, action_space, num_parallel_envs)
        logger.info(f"Constructed agent {agent}")

        logger.info(f"Starting lifetime #{i_lifetime + 1}")
        # FIXME: pass num_parallel_envs to run https://github.com/darpa-l2m/tella/issues/32
        # FIXME: pass log_dir to run https://github.com/darpa-l2m/tella/issues/12
        run(agent, curriculum, render=render)


def _spaces(
    curriculum_factory: CurriculumFactory,
) -> typing.Tuple[gym.Space, gym.Space]:
    """
    Retrieves the observation & action space from the curriculum.

    :param curriculum_factory: The class of the curriculum. This function constructs an object using this class.
    :return: A tuple of (observation_space, action_space).
    """
    # FIXME: extract spaces based on solution in https://github.com/darpa-l2m/tella/issues/31
    curriculum_obj = curriculum_factory()
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
    curriculum: AbstractCurriculum[AbstractTaskVariant],
    render: typing.Optional[bool],
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
    # TODO change logs
    data_logger = l2logger.DataLogger("logs", scenario_dir, logger_info, scenario_info)
    total_episodes = 0
    for i_block, block in enumerate(curriculum.learn_blocks_and_eval_blocks()):
        is_learning_allowed = agent.is_learning_allowed = block.is_learning_allowed
        agent.block_start(is_learning_allowed)
        for task_block in block.task_blocks():
            agent.task_start(task_block.task_label)
            for task_variant in task_block.task_variants():
                # NOTE: assuming taskvariant has params
                task_variant.set_logger_info(
                    data_logger, i_block, is_learning_allowed, total_episodes
                )
                task_variant.set_render(render)
                agent.task_variant_start(
                    task_variant.task_label, task_variant.variant_label
                )
                if is_learning_allowed:
                    metrics = agent.learn_task_variant(task_variant)
                else:
                    metrics = agent.eval_task_variant(task_variant)
                logger.info(f"TaskVariant produced metrics: {metrics}")
                agent.task_variant_end(
                    task_variant.task_label, task_variant.variant_label
                )
                total_episodes += task_variant.total_episodes
            agent.task_end(task_block.task_label)
        agent.block_end(block.is_learning_allowed)
