import logging
import typing

import gym

import tella
from rl_logging_agent import LoggingAgent

logger = logging.getLogger("Manual Control Agent")


class ManualControlAgent(LoggingAgent):
    def __init__(
        self,
        rng_seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_envs: int,
    ) -> None:
        if num_envs != 1:
            raise NotImplementedError(
                "Manual Control Agent only makes sense for single environments"
            )
        super().__init__(rng_seed, observation_space, action_space, num_envs)

    def choose_actions(
        self, observations: typing.List[typing.Optional[tella.Observation]]
    ) -> typing.List[typing.Optional[tella.Action]]:
        logger.debug(f"\t\t\tReturn {len(observations)} player action")
        action = int(input(f"Input action int in range({self.action_space.n})"))
        return [action]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tella.rl_cli(ManualControlAgent)
