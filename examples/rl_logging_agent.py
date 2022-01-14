import logging
import typing

import gym
import tella

logger = logging.getLogger("Example Logging Agent")


class LoggingAgent(tella.ContinualRLAgent):
    def __init__(
        self,
        rng_seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_envs: int,
        config_file: typing.Optional[str] = None,
    ) -> None:
        super().__init__(rng_seed, observation_space, action_space, num_envs, config_file)
        logger.info(
            f"Constructed with observation_space={observation_space} "
            f"action_space={action_space} num_envs={num_envs}"
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
        logger.info(
            f"\tAbout to start interacting with a new task. task_name={task_name}"
        )

    def task_variant_start(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        logger.info(
            f"\tAbout to start interacting with a new task variant. "
            f"task_name={task_name} variant_name={variant_name}"
        )

    def learn_task_variant(self, task_variant: tella.AbstractRLTaskVariant):
        logger.info("\tConsuming task variant")
        return super().learn_task_variant(task_variant)

    def choose_action(
        self, observations: typing.List[typing.Optional[tella.Observation]]
    ) -> typing.List[typing.Optional[tella.Action]]:
        logger.debug(f"\t\t\tReturn {len(observations)} random actions")
        return [
            None if obs is None else self.action_space.sample() for obs in observations
        ]

    def receive_transition(self, transition: tella.Transition) -> None:
        obs, action, reward, done, next_obs = transition
        logger.debug(f"\t\t\tReceived transition done={done}")

    def task_end(
        self,
        task_name: typing.Optional[str],
    ) -> None:
        logger.info(f"\tDone interacting with task. task_name={task_name}")

    def task_variant_end(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        logger.info(
            f"\tDone interacting with task variant. "
            f"task_name={task_name} variant_name={variant_name}"
        )

    def block_end(self, is_learning_allowed: bool) -> None:
        if is_learning_allowed:
            logger.info("Done with learning block")
        else:
            logger.info("Done with evaluation block")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tella.rl_cli(LoggingAgent)
