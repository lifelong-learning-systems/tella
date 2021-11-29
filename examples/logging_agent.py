import typing
import gym
from tella.agent import Agent, Observation, Action
import logging

logger = logging.getLogger(__name__)


class LoggingAgent(Agent):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space) -> None:
        super().__init__(observation_space, action_space)
        logger.info(f"Constructed with {observation_space=} {action_space=}")

    def block_start(self, is_learning_allowed: bool) -> None:
        if is_learning_allowed:
            logger.info("About to start a new learning block")
        else:
            logger.info("About to start a new evaluation block")

    def task_start(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        logger.info(
            f"\tAbout to start interacting with a new task. {task_name=} {variant_name=}"
        )

    def episode_start(self) -> None:
        logger.info("\t\tAbout to start a new episode")

    def step_observe(self, observation: Observation) -> Action:
        logger.info("\t\t\tReturn random action")
        return self.action_space.sample()

    def step_reward(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        done: bool,
        next_observation: Observation,
    ) -> bool:
        logger.info(f"\t\t\tReceived step result {done=}")
        return True

    def episode_end(self) -> None:
        logger.info("\t\tEpisode just ended")

    def task_end(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        logger.info(f"\tDone interacting with task. {task_name=} {variant_name=}")

    def block_end(self, is_learning_allowed: bool) -> None:
        if is_learning_allowed:
            logger.info("Done with learning block")
        else:
            logger.info("Done with evaluation block")
