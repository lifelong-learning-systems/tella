import logging
from .agents.continual_learning_agent import ContinualLearningAgent
from .curriculum import AbstractCurriculum, AbstractTaskVariant

logger = logging.getLogger(__name__)


def run(
    agent: ContinualLearningAgent[AbstractTaskVariant],
    curriculum: AbstractCurriculum[AbstractTaskVariant],
):
    """
    Run an agent through an entire curriculum. This assumes that the agent
    and the curriculum are both generic over the same type.

    I.e. the curriculum will be generating task variants of type T, and the agent
    will be consuming them via it's :meth:`ContinualLearningAgent.consume_task_variant`.
    """
    # TODO create l2logger here
    for i_block, block in enumerate(curriculum.learn_blocks_and_eval_blocks()):
        agent.is_learning_allowed = block.is_learning_allowed()
        agent.block_start(block.is_learning_allowed())
        for task_block in block.task_blocks():
            agent.task_start(None)
            for task_variant in task_block.task_variants():
                # FIXME: how to provide task & variant info?
                agent.task_variant_start(None, None)
                metrics = agent.consume_task_variant(task_variant)
                logger.info(f"TaskVariant produced metrics: {metrics}")
                agent.task_variant_end(None, None)
            agent.task_end(None)
            # TODO log metrics with l2logger
        agent.block_end(block.is_learning_allowed())
