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
from .agents.continual_learning_agent import ContinualLearningAgent
from .curriculum import AbstractCurriculum, AbstractTaskVariant
from l2logger import l2logger

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
        is_learning_allowed = agent.is_learning_allowed = block.is_learning_allowed()
        agent.block_start(is_learning_allowed)
        for task_block in block.task_blocks():
            agent.task_start(None)
            for task_variant in task_block.task_variants():
                # NOTE: assuming taskvariant has params
                task_variant.set_logger_info(
                    data_logger, i_block, is_learning_allowed, total_episodes
                )
                # FIXME: how to provide task & variant info?
                agent.task_variant_start(None, None)
                if is_learning_allowed:
                    metrics = agent.learn_task_variant(task_variant)
                else:
                    metrics = agent.eval_task_variant(task_variant)
                logger.info(f"TaskVariant produced metrics: {metrics}")
                agent.task_variant_end(None, None)
                total_episodes += task_variant.total_episodes()
            agent.task_end(None)
        agent.block_end(block.is_learning_allowed())
