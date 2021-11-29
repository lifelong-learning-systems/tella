import typing
import gym
from tella.experiences.rl import (
    MDPTransition,
    Observation,
    Action,
    RLExperience,
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
        variant_name: typing.Optional[str],
    ) -> None:
        logger.info(
            f"\tAbout to start interacting with a new task. {task_name=} {variant_name=}"
        )

    def consume_experience(self, experience: RLExperience):
        logger.info("\tConsuming experience")
        return super().consume_experience(experience)

    def step_observe(
        self, observations: typing.List[Observation]
    ) -> typing.List[Action]:
        logger.info(f"\t\t\tReturn {len(observations)} random actions")
        return [self.action_space.sample() for obs in observations]

    def step_transition(self, transition: MDPTransition) -> bool:
        obs, action, reward, done, next_obs = transition
        logger.info(f"\t\t\tReceived transition {done=}")
        return True

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
