import typing
import gym
from tella.agent import Agent, Observation, Action
import logging

logger = logging.getLogger(__name__)


class LoggingAgent(Agent):
    def __init__(self) -> None:
        self.action_space = None

    def block_start(self, is_learning_allowed: bool) -> None:
        if is_learning_allowed:
            logger.info("About to start a new learning block")
        else:
            logger.info("About to start a new evaluation block")

    def task_start(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        logger.info(
            f"\tAbout to start interacting with a new task. {observation_space=} {action_space=} {task_name=} {variant_name=}"
        )
        self.action_space = action_space

    def episode_start(self) -> None:
        logger.info("\t\tAbout to start a new episode")

    def step(self, observation: Observation) -> Action:
        logger.info("\t\t\tReturn random action")
        return self.action_space.sample()

    def step_result(
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
        observation_space: gym.Space,
        action_space: gym.Space,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        logger.info(
            f"\tDone interacting with task. {observation_space=} {action_space=} {task_name=} {variant_name=}"
        )

    def block_end(self, is_learning_allowed: bool) -> None:
        if is_learning_allowed:
            logger.info("Done with learning block")
        else:
            logger.info("Done with evaluation block")

    def save_internal_state(self, path: str) -> bool:
        logger.info("Saving internal state")
        return False

    def restore_internal_state(self, path: str) -> bool:
        logger.info("Restoring internal state")
        return False
