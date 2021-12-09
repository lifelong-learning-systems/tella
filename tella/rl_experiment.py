import gym
import logging
import typing
from .curriculum import AbstractCurriculum
from .agents.continual_rl_agent import ContinualRLAgent, AbstractRLTaskVariant
from .validation import validate_curriculum
from .run import run


logger = logging.getLogger(__name__)

AgentFactory = typing.Callable[[gym.Space, gym.Space, int], ContinualRLAgent]
CurriculumFactory = typing.Callable[[], AbstractCurriculum[AbstractRLTaskVariant]]


def rl_experiment(
    agent_factory: AgentFactory,
    curriculum_factory: CurriculumFactory,
    num_runs: int,
    num_cores: int,
    log_dir: str,
) -> None:
    """
    Run an experiment with an RL agent and an RL curriculum.

    :param agent_factory: Function or class to produce agents.
    :param curriculum_factory: Function or class to produce curriculum.
    :param num_runs: Number of times to call :func:`run()`.
    :param num_cores: TODO
    :param log_dir:TODO
    :return: None
    """
    observation_space, action_space = _spaces(curriculum_factory)

    # FIXME: multiprocessing https://github.com/darpa-l2m/tella/issues/44
    for i_run in range(num_runs):
        curriculum = curriculum_factory()
        logger.info(f"Constructed curriculum {curriculum}")
        # FIXME: seed the curriculum https://github.com/darpa-l2m/tella/issues/54

        # FIXME: check for RL task variant https://github.com/darpa-l2m/tella/issues/53
        validate_curriculum(curriculum)
        logger.info("Validated curriculum")

        agent = agent_factory(observation_space, action_space, num_cores)
        logger.info(f"Constructed agent {agent}")

        logger.info(f"Starting run #{i_run + 1}")
        # FIXME: pass num_cores to run https://github.com/darpa-l2m/tella/issues/32
        # FIXME: pass log_dir to run https://github.com/darpa-l2m/tella/issues/12
        run(agent, curriculum)


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
