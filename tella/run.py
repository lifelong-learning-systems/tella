import logging
from .agents.continual_learning_agent import ContinualLearningAgent, GenericExperience
from .curriculum import Curriculum

logger = logging.getLogger(__name__)


def run(
    agent: ContinualLearningAgent[GenericExperience],
    curriculum: Curriculum[GenericExperience],
):
    """
    Run an agent through an entire curriculum. This assumes that the agent
    and the curriculum are both generic over the same type.

    I.e. the curriculum will be generating experiences of type T, and the agent
    will be consuming them via it's :meth:`ContinualLearningAgent.consume_experience`.
    """
    # TODO create l2logger here
    for i_block, block in enumerate(curriculum.blocks()):
        agent.is_learning_allowed = block.is_learning_allowed()
        agent.block_start(block.is_learning_allowed())
        for i_experience, experience in enumerate(block.experiences()):
            # NOTE: this assumes experience = task
            # FIXME: how to provide task & variant info?
            agent.task_start(None, None)
            metrics = agent.consume_experience(experience)
            logger.info(f"Experience produced metrics: {metrics}")
            agent.task_end(None, None)
            # TODO log metrics with l2logger
        agent.block_end(block.is_learning_allowed())
