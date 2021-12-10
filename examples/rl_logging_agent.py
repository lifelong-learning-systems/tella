import typing
import gym
from tella.curriculum.rl_task_variant import (
    StepData,
    Observation,
    Action,
    AbstractRLTaskVariant,
)
from tella.agents.continual_rl_agent import ContinualRLAgent
import logging

logger = logging.getLogger(__name__)


class LoggingAgent(ContinualRLAgent):
    def __init__(
        self, observation_space: gym.Space, action_space: gym.Space, num_envs: int
    ) -> None:
        super().__init__(observation_space, action_space, num_envs)
        logger.info(
            f"Constructed with {observation_space=} {action_space=} {num_envs=}"
        )

    def block_start(self, is_learning_allowed: bool) -> None:
        super().block_start(is_learning_allowed)
        if is_learning_allowed:
            logger.info("About to start a new learning block")
        else:
            logger.info("About to start a new evaluation block")

    def task_start(
        self,
        task_name: typing.Optional[str],
    ) -> None:
        logger.info(f"\tAbout to start interacting with a new task. {task_name=}")

    def task_variant_start(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        logger.info(
            f"\tAbout to start interacting with a new task variant. {task_name=} {variant_name=}"
        )

    def learn_task_variant(self, task_variant: AbstractRLTaskVariant):
        logger.info("\tConsuming task variant")
        return super().learn_task_variant(task_variant)

    def choose_action(
        self, observations: typing.List[typing.Optional[Observation]]
    ) -> typing.List[typing.Optional[Action]]:
        logger.info(f"\t\t\tReturn {len(observations)} random actions")
        return [
            None if obs is None else self.action_space.sample() for obs in observations
        ]

    def view_transition(self, step: StepData) -> None:
        obs, action, reward, done, next_obs = step
        logger.info(f"\t\t\tReceived step {done=}")

    def task_end(
        self,
        task_name: typing.Optional[str],
    ) -> None:
        logger.info(f"\tDone interacting with task. {task_name=}")

    def task_variant_end(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        logger.info(
            f"\tDone interacting with task variant. {task_name=} {variant_name=}"
        )

    def block_end(self, is_learning_allowed: bool) -> None:
        if is_learning_allowed:
            logger.info("Done with learning block")
        else:
            logger.info("Done with evaluation block")
